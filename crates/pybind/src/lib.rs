//! The `big2._core` Python extension module.
//!
//! The Python side sees only plain values:
//!
//! * a **set of cards** is an `int` bit mask (Big Two index: `rank * 4 + suit`,
//!   ranks `3 … 2`, suits `♦ ♣ ♥ ♠`);
//! * an **action** is the mask of the cards played, with `0` meaning *pass*;
//! * rule variants are keyword arguments with snake_case names
//!   (`straights="no_twos"`, `scoring="original"`, …).
//!
//! The friendlier Python API lives in `python/big2/games/big2.py`; this module is
//! its engine room.

use big2_big2::bots::{Greedy, StructuredGreedy, DEFAULT_PLAN_DEPTH};
use big2_big2::card::{format_mask, parse_mask};
use big2_big2::plan;
use big2_big2::pvnet::PvNet;
use big2_big2::value::ValueNet;
use big2_big2::{classify, Action, RuleSet, State};
use big2_games::{Game, Policy};
use big2_search::{
    ismcts_with, pimc, puct_with_rollout, Analysis, Belief, BeliefConfig, InfosetConfig,
    IsmctsConfig, Leaf, Objective, OracleSampler, PimcConfig, PolicyModel, PuctConfig, RulesMask,
    Sampler, UniformSampler,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

mod vecenv;

fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Loads a `.pknn` value network once per path and keeps it for later calls.
fn load_value_net(path: &str) -> PyResult<Arc<ValueNet>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<ValueNet>>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(net) = cache.get(path) {
        return Ok(net.clone());
    }
    let bytes = std::fs::read(path).map_err(|e| value_error(format!("{path}: {e}")))?;
    let net =
        Arc::new(ValueNet::from_bytes(&bytes).map_err(|e| value_error(format!("{path}: {e}")))?);
    cache.insert(path.to_string(), net.clone());
    Ok(net)
}

/// Loads a `.pknn` policy-value network (`big2 rl export` of a network v2
/// checkpoint) once per path and keeps it for later calls.
fn load_pvnet(path: &str) -> PyResult<Arc<PvNet>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<PvNet>>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(net) = cache.get(path) {
        return Ok(net.clone());
    }
    let bytes = std::fs::read(path).map_err(|e| value_error(format!("{path}: {e}")))?;
    let net = Arc::new(PvNet::from_bytes(&bytes).map_err(|e| value_error(format!("{path}: {e}")))?);
    cache.insert(path.to_string(), net.clone());
    Ok(net)
}

fn rng_from(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    }
}

/// The rollout policy the solvers play simulated games with (`rollout=` option).
#[derive(Clone, Copy)]
enum Rollout {
    Greedy(Greedy),
    Structured(StructuredGreedy),
}

impl Rollout {
    fn parse(name: &str, epsilon: f64, plan_depth: usize) -> PyResult<Self> {
        match name {
            "greedy" => Ok(Rollout::Greedy(Greedy { epsilon })),
            "structured" => Ok(Rollout::Structured(StructuredGreedy {
                epsilon,
                plan_depth,
            })),
            other => Err(value_error(format!(
                "unknown rollout {other:?} (use \"greedy\" or \"structured\")"
            ))),
        }
    }
}

impl Policy<State> for Rollout {
    fn choose<R: rand::Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        rng: &mut R,
    ) -> Action {
        match self {
            Rollout::Greedy(p) => p.choose(state, legal, rng),
            Rollout::Structured(p) => p.choose(state, legal, rng),
        }
    }
}

/// Builds a `RuleSet` from optional variant names.
pub(crate) fn rules_from(
    straights: Option<&str>,
    flush_order: Option<&str>,
    pass_rule: Option<&str>,
    scoring: Option<&str>,
    last_card_duty: Option<bool>,
) -> PyResult<RuleSet> {
    let mut r = RuleSet::default();
    if let Some(s) = straights {
        r.straights = s.parse().map_err(value_error)?;
    }
    if let Some(s) = flush_order {
        r.flush_order = s.parse().map_err(value_error)?;
    }
    if let Some(s) = pass_rule {
        r.pass_rule = s.parse().map_err(value_error)?;
    }
    if let Some(s) = scoring {
        r.scoring = s.parse().map_err(value_error)?;
    }
    if let Some(b) = last_card_duty {
        r.last_card_duty = b;
    }
    Ok(r)
}

/// Parses `"3d 4c Ah"` into a card mask.
#[pyfunction]
fn parse_cards(text: &str) -> PyResult<u64> {
    parse_mask(text).map_err(value_error)
}

/// Formats a card mask as `"3d 4c Ah"` (weakest first).
#[pyfunction]
fn format_cards(cards: u64) -> String {
    format_mask(cards)
}

/// Classifies cards as a combination: `(kind, key)` or `None` if not legal.
/// `key` orders combinations of the same size.
#[pyfunction]
#[pyo3(signature = (cards, *, straights=None, flush_order=None))]
fn classify_cards(
    cards: u64,
    straights: Option<&str>,
    flush_order: Option<&str>,
) -> PyResult<Option<(String, u32)>> {
    let rules = rules_from(straights, flush_order, None, None, None)?;
    Ok(classify(cards, &rules).map(|c| (c.kind.name().to_string(), c.key)))
}

/// The fewest plays that shed every card of `hand` (`crates/big2/src/plan.rs`).
#[pyfunction]
#[pyo3(signature = (hand, *, straights=None, flush_order=None))]
fn min_plays(hand: u64, straights: Option<&str>, flush_order: Option<&str>) -> PyResult<u32> {
    let rules = rules_from(straights, flush_order, None, None, None)?;
    Ok(plan::min_plays(hand, &rules))
}

/// Every way to shed `hand` in the fewest plays, best first, at most `max` of
/// them. Each decomposition is a list of card masks, strongest group first.
/// The order is the planner's fixed tiebreak: fewest singles, then the
/// strongest five-card groups first.
#[pyfunction]
#[pyo3(signature = (hand, max=8, *, straights=None, flush_order=None))]
fn decompositions(
    hand: u64,
    max: usize,
    straights: Option<&str>,
    flush_order: Option<&str>,
) -> PyResult<Vec<Vec<u64>>> {
    let rules = rules_from(straights, flush_order, None, None, None)?;
    Ok(plan::decompositions(hand, &rules, max)
        .iter()
        .map(|d| d.combos.iter().map(|c| c.cards).collect())
        .collect())
}

/// Every legal action of the player to act, labelled against the decompositions
/// of that player's hand (`plan::candidates` with up to `max_decompositions`
/// plans). Returns `[(cards, keeps, breaks)]` in the planner's order: `keeps` is
/// the index of the best decomposition that has the play as a group (`-1` if
/// none), `breaks` the mask of the group it breaks (`0` if none or a pass).
#[pyfunction]
#[pyo3(signature = (state, max_decompositions=8))]
fn candidates(state: &PyState, max_decompositions: usize) -> PyResult<Vec<(u64, i32, u64)>> {
    let s = &state.inner;
    let player = s
        .current_player()
        .ok_or_else(|| value_error("the deal is over"))?;
    let decomps = plan::decompositions(s.hand(player), s.rules(), max_decompositions);
    Ok(plan::candidates(s, &decomps)
        .iter()
        .map(|c| {
            (
                c.action.cards(),
                c.keeps.map_or(-1, |k| k as i32),
                c.breaks.map_or(0, |b| b.cards),
            )
        })
        .collect())
}

/// A Big Two deal. Holds the true state, including every player's hidden hand.
#[pyclass(name = "Big2State", module = "big2._core")]
#[derive(Clone)]
pub(crate) struct PyState {
    pub(crate) inner: State,
}

#[pymethods]
impl PyState {
    /// Shuffles and deals a new game.
    #[staticmethod]
    #[pyo3(signature = (seed=None, *, straights=None, flush_order=None, pass_rule=None, scoring=None, last_card_duty=None))]
    fn new_deal(
        seed: Option<u64>,
        straights: Option<&str>,
        flush_order: Option<&str>,
        pass_rule: Option<&str>,
        scoring: Option<&str>,
        last_card_duty: Option<bool>,
    ) -> PyResult<Self> {
        let rules = rules_from(straights, flush_order, pass_rule, scoring, last_card_duty)?;
        Ok(PyState {
            inner: State::new_deal(rules, &mut rng_from(seed)),
        })
    }

    /// Starts a game from four 13-card hands (masks) covering the deck.
    #[staticmethod]
    #[pyo3(signature = (hands, *, straights=None, flush_order=None, pass_rule=None, scoring=None, last_card_duty=None))]
    fn from_hands(
        hands: [u64; 4],
        straights: Option<&str>,
        flush_order: Option<&str>,
        pass_rule: Option<&str>,
        scoring: Option<&str>,
        last_card_duty: Option<bool>,
    ) -> PyResult<Self> {
        let rules = rules_from(straights, flush_order, pass_rule, scoring, last_card_duty)?;
        Ok(PyState {
            inner: State::from_hands(rules, hands).map_err(value_error)?,
        })
    }

    /// Sets up a mid-game position. `current_play` is `(player, cards)` or `None`.
    #[staticmethod]
    #[pyo3(signature = (hands, current, current_play=None, *, straights=None, flush_order=None, pass_rule=None, scoring=None, last_card_duty=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_position(
        hands: [u64; 4],
        current: usize,
        current_play: Option<(usize, u64)>,
        straights: Option<&str>,
        flush_order: Option<&str>,
        pass_rule: Option<&str>,
        scoring: Option<&str>,
        last_card_duty: Option<bool>,
    ) -> PyResult<Self> {
        let rules = rules_from(straights, flush_order, pass_rule, scoring, last_card_duty)?;
        Ok(PyState {
            inner: State::from_position(rules, hands, current, current_play)
                .map_err(value_error)?,
        })
    }

    /// The rule variants as a dict of names.
    fn rules<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.inner.rules();
        let d = PyDict::new(py);
        d.set_item("straights", r.straights.name())?;
        d.set_item("flush_order", r.flush_order.name())?;
        d.set_item("pass_rule", r.pass_rule.name())?;
        d.set_item("scoring", r.scoring.name())?;
        d.set_item("last_card_duty", r.last_card_duty)?;
        Ok(d)
    }

    /// Player to act, or `None` once the deal is over.
    #[getter]
    fn current_player(&self) -> Option<usize> {
        self.inner.current_player()
    }

    /// Every player's hand (hidden information included).
    #[getter]
    fn hands(&self) -> [u64; 4] {
        self.inner.hands()
    }

    /// One player's hand.
    fn hand(&self, player: usize) -> PyResult<u64> {
        if player >= 4 {
            return Err(value_error(format!("no player {player}")));
        }
        Ok(self.inner.hand(player))
    }

    /// Cards left per player.
    #[getter]
    fn hand_sizes(&self) -> [u32; 4] {
        self.inner.hand_sizes()
    }

    /// All cards played so far.
    #[getter]
    fn played(&self) -> u64 {
        self.inner.played()
    }

    /// `(player, cards, kind)` of the play to beat, or `None` when leading.
    #[getter]
    fn current_play(&self) -> Option<(usize, u64, String)> {
        self.inner
            .current_play()
            .map(|(p, c)| (p, c.cards, c.kind.name().to_string()))
    }

    /// `True` before the first play (which must include 3♦).
    #[getter]
    fn is_opening(&self) -> bool {
        self.inner.is_opening()
    }

    /// Players sitting out the current trick (`pass_rule="locked_out"` only).
    #[getter]
    fn locked_out(&self) -> [bool; 4] {
        self.inner.locked_out()
    }

    /// `[(player, cards)]` for every action so far; `cards == 0` is a pass.
    #[getter]
    fn history(&self) -> Vec<(u8, u64)> {
        self.inner
            .history()
            .iter()
            .map(|t| (t.player, t.action.cards()))
            .collect()
    }

    /// The player who went out, once the deal is over.
    #[getter]
    fn winner(&self) -> Option<usize> {
        self.inner.winner()
    }

    #[getter]
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// Final scores per player, or `None` while the deal is running.
    #[getter]
    fn scores(&self) -> Option<Vec<f64>> {
        self.inner.outcome().map(|o| o.scores)
    }

    /// Penalties if the deal ended now.
    #[getter]
    fn penalties(&self) -> [u32; 4] {
        self.inner.penalties()
    }

    /// Legal actions for the player to act, as card masks (`0` = pass).
    fn legal_actions(&self) -> Vec<u64> {
        self.inner
            .legal_actions_vec()
            .iter()
            .map(Action::cards)
            .collect()
    }

    /// Plays `cards` (a mask; `0` = pass) for the player to act.
    /// Raises `ValueError` with the reason if it is illegal.
    fn apply(&mut self, cards: u64) -> PyResult<()> {
        let a = self.inner.action_for_cards(cards).map_err(value_error)?;
        self.inner.apply(a);
        Ok(())
    }

    /// Like `apply`, but returns a new state.
    fn child(&self, cards: u64) -> PyResult<Self> {
        let mut c = self.clone();
        c.apply(cards)?;
        Ok(c)
    }

    /// A copy where everything hidden from `observer` is re-dealt at random.
    #[pyo3(signature = (observer, seed=None))]
    fn determinize(&self, observer: usize, seed: Option<u64>) -> Self {
        PyState {
            inner: self.inner.determinize(observer, &mut rng_from(seed)),
        }
    }

    /// The greedy baseline bot's choice (card mask). With `structured=True`, the
    /// planner-aware `StructuredGreedy` (keeps pairs, triples and five-card groups
    /// of the hand's best plan; `plan_depth` plays checked per decision).
    #[pyo3(signature = (epsilon=0.0, seed=None, structured=false, plan_depth=DEFAULT_PLAN_DEPTH))]
    fn greedy_action(
        &self,
        epsilon: f64,
        seed: Option<u64>,
        structured: bool,
        plan_depth: usize,
    ) -> PyResult<u64> {
        let legal = self.inner.legal_actions_vec();
        if legal.is_empty() {
            return Err(value_error("the deal is over"));
        }
        let name = if structured { "structured" } else { "greedy" };
        Ok(Rollout::parse(name, epsilon, plan_depth)?
            .choose(&self.inner, &legal, &mut rng_from(seed))
            .cards())
    }

    /// A uniformly random legal action (card mask).
    #[pyo3(signature = (seed=None))]
    fn random_action(&self, seed: Option<u64>) -> PyResult<u64> {
        let legal = self.inner.legal_actions_vec();
        if legal.is_empty() {
            return Err(value_error("the deal is over"));
        }
        Ok(legal[(rng_from(seed).next_u64() % legal.len() as u64) as usize].cards())
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Runs a solver for the player to act and returns a dict:
///
/// ```text
/// {"player": int, "simulations": int, "seconds": float,
///  "moves": [{"cards": int, "visits": int, "mean_score": float, "score_ci95": float,
///             "win_rate": float, "win_ci95": float}, ...]}   # best first
/// ```
///
/// `algorithm` is `"ismcts"` (default), `"pimc"` or `"puct"`. For PIMC, `iterations`
/// is the number of sampled deals; for PUCT, the simulations (each one network
/// evaluation), and `net` (a `.pknn` policy-value network from `big2 rl export`) is
/// required; `c_puct` (default 1.5) is its exploration constant and
/// `anytime_first_ms` records the move it would have played after that many
/// milliseconds as `first_answer`. `rollout` is `"greedy"` (default) or
/// `"structured"` (`StructuredGreedy` with `rollout_plan_depth`), both with
/// `rollout_epsilon`. PUCT plays no rollouts unless `leaf_network_weight` (λ,
/// default 1) is below 1: then a new leaf's value is `λ · network + (1 − λ) ·
/// rollout`, the rollout playing `rollout_moves` moves (default 0 = to the end of
/// the deal) with the same `rollout` policy. `belief="rules"` (ISMCTS and PUCT)
/// samples only deals the rules allow (`RulesMask`: the last-card duty) and reports
/// `belief_rejection_share` and `belief_fallbacks`; `"greedy"` and `"oracle"` are
/// ISMCTS only. `infoset_endgame=<threshold>` (ISMCTS and PUCT) solves the position
/// exactly over every deal of the hidden cards once at most that many are hidden
/// and at most `infoset_max_cards` are in all hands (`infoset_max_nodes` positions
/// at most; then the search runs) and reports `exact`
/// and `forced` (`(winner, plays)`). The GIL is released while searching.
#[pyfunction]
#[pyo3(signature = (state, *, algorithm="ismcts", iterations=10_000, time_limit=None, exploration=None, threads=1, seed=0, objective="score", endgame_threshold=0, endgame_max_nodes=20_000, rollout_epsilon=0.1, rollout="greedy", rollout_plan_depth=DEFAULT_PLAN_DEPTH, belief=None, belief_candidates=256, belief_epsilon=0.2, belief_min_ess=0.25, value=None, value_rollout_weight=0.0, net=None, c_puct=None, anytime_first_ms=None, leaf_network_weight=None, rollout_moves=None, infoset_endgame=None, infoset_max_cards=10, infoset_max_nodes=1_000_000))]
#[allow(clippy::too_many_arguments)]
fn analyze<'py>(
    py: Python<'py>,
    state: &PyState,
    algorithm: &str,
    iterations: u64,
    time_limit: Option<f64>,
    exploration: Option<f64>,
    threads: usize,
    seed: u64,
    objective: &str,
    endgame_threshold: usize,
    endgame_max_nodes: u64,
    rollout_epsilon: f64,
    rollout: &str,
    rollout_plan_depth: usize,
    belief: Option<&str>,
    belief_candidates: usize,
    belief_epsilon: f64,
    belief_min_ess: f64,
    value: Option<&str>,
    value_rollout_weight: f64,
    net: Option<&str>,
    c_puct: Option<f64>,
    anytime_first_ms: Option<u64>,
    leaf_network_weight: Option<f64>,
    rollout_moves: Option<u32>,
    infoset_endgame: Option<usize>,
    infoset_max_cards: usize,
    infoset_max_nodes: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let player = state
        .inner
        .current_player()
        .ok_or_else(|| value_error("the deal is over"))?;
    let objective = match objective {
        "score" => Objective::Score,
        "win" => Objective::Win,
        other => {
            return Err(value_error(format!(
                "unknown objective {other:?} (use \"score\" or \"win\")"
            )))
        }
    };
    let rollout = Rollout::parse(rollout, rollout_epsilon, rollout_plan_depth)?;
    let s = state.inner.clone();
    // "oracle" sees the true hidden cards: a diagnostic upper bound, not fair play.
    let oracle = belief == Some("oracle");
    // "rules": uniform deals minus those the rules exclude (last-card duty; Task 18).
    let rules_mask = (belief == Some("rules")).then(RulesMask::default);
    let model = match belief {
        None | Some("none") | Some("oracle") | Some("rules") => None,
        Some("greedy") => Some(PolicyModel {
            policy: Greedy { epsilon: 0.0 },
            epsilon: belief_epsilon,
        }),
        Some(other) => {
            return Err(value_error(format!(
                "unknown belief model {other:?} (use \"rules\", \"greedy\", \"oracle\" or none)"
            )))
        }
    };
    if algorithm != "ismcts" && (model.is_some() || oracle || value.is_some()) {
        return Err(value_error(
            "belief (other than \"rules\") and value are only supported with ismcts",
        ));
    }
    if algorithm == "pimc" && rules_mask.is_some() {
        return Err(value_error("belief=\"rules\" works with ismcts and puct"));
    }
    if algorithm != "puct"
        && (net.is_some()
            || c_puct.is_some()
            || anytime_first_ms.is_some()
            || leaf_network_weight.is_some()
            || rollout_moves.is_some())
    {
        return Err(value_error(
            "net, c_puct, anytime_first_ms, leaf_network_weight and rollout_moves are only \
             supported with puct",
        ));
    }
    if leaf_network_weight.is_some_and(|w| !(0.0..=1.0).contains(&w)) {
        return Err(value_error("leaf_network_weight must be between 0 and 1"));
    }
    if algorithm == "pimc" && infoset_endgame.is_some() {
        return Err(value_error("infoset_endgame works with ismcts and puct"));
    }
    let value_net = value.map(load_value_net).transpose()?;
    let mut belief_ess = None;
    let mut first_answer: Option<u64> = None;
    // Task 20: with few hidden cards, every deal is solved exactly at the root
    // (weighted by the rules mask with belief="rules", else uniform); the search
    // below runs only when that does not apply or its node budget runs out.
    let mut exact: Option<bool> = None;
    let mut forced: Option<(usize, u32)> = None;
    let solved: Option<Analysis<Action>> = match infoset_endgame {
        Some(threshold) if !oracle => {
            let ic = InfosetConfig {
                threshold,
                max_cards: infoset_max_cards,
                max_nodes: infoset_max_nodes,
            };
            let r = py.detach(|| match &rules_mask {
                Some(m) => big2_search::infoset_endgame(&s, player, &ic, objective, m),
                None => big2_search::infoset_endgame(&s, player, &ic, objective, &UniformSampler),
            });
            r.and_then(|r| {
                exact = Some(r.exact);
                forced = r.forced.map(|f| (f.winner, f.plays));
                r.exact.then_some(r.analysis)
            })
        }
        _ => None,
    };
    let result: Analysis<Action> = if let Some(a) = solved {
        a
    } else {
        match algorithm {
            "ismcts" => {
                let defaults = IsmctsConfig::default();
                let cfg = IsmctsConfig {
                    iterations,
                    time_limit: time_limit.map(Duration::from_secs_f64),
                    exploration: exploration.unwrap_or(defaults.exploration),
                    threads,
                    seed,
                    objective,
                    endgame_threshold,
                    endgame_max_nodes,
                    infoset_endgame: None, // solved above when asked for
                };
                let bcfg = BeliefConfig {
                    candidates: belief_candidates,
                    min_ess: belief_min_ess,
                };
                let leaf = value_net.as_deref().map(|n| Leaf {
                    evaluator: n,
                    rollout_weight: value_rollout_weight,
                });
                let (analysis, ess) = py.detach(|| {
                    let mut rng = SmallRng::seed_from_u64(seed ^ 0xB3_11EF);
                    let weighted = model.and_then(|m| Belief::new(&s, player, &m, &bcfg, &mut rng));
                    let ess = weighted.as_ref().map(|b| b.ess);
                    let a = match &weighted {
                        _ if oracle => {
                            ismcts_with(&s, player, &cfg, &rollout, &OracleSampler, leaf.as_ref())
                        }
                        _ if rules_mask.is_some() => ismcts_with(
                            &s,
                            player,
                            &cfg,
                            &rollout,
                            rules_mask.as_ref().unwrap(),
                            leaf.as_ref(),
                        ),
                        Some(b) => ismcts_with(&s, player, &cfg, &rollout, b, leaf.as_ref()),
                        None => {
                            ismcts_with(&s, player, &cfg, &rollout, &UniformSampler, leaf.as_ref())
                        }
                    };
                    (a, ess)
                });
                belief_ess = ess;
                analysis
            }
            "pimc" => {
                let cfg = PimcConfig {
                    determinizations: iterations,
                    threads,
                    seed,
                    objective,
                    endgame_threshold,
                    endgame_max_nodes,
                    ..Default::default()
                };
                py.detach(|| pimc(&s, player, &cfg, &rollout))
            }
            "puct" => {
                let net = load_pvnet(net.ok_or_else(|| {
                    value_error("puct needs net=<path to a .pknn policy-value network>")
                })?)?;
                let defaults = PuctConfig::default();
                let cfg = PuctConfig {
                    simulations: iterations,
                    time_limit: time_limit.map(Duration::from_secs_f64),
                    c_puct: c_puct.unwrap_or(defaults.c_puct),
                    threads,
                    seed,
                    objective,
                    endgame_threshold,
                    endgame_max_nodes,
                    anytime_first_ms,
                    leaf_network_weight: leaf_network_weight
                        .unwrap_or(defaults.leaf_network_weight),
                    rollout_moves: rollout_moves.unwrap_or(defaults.rollout_moves),
                    infoset_endgame: None, // solved above when asked for
                };
                // With λ = 1 (the default) the rollout is never played: same result as `puct`.
                let r = py.detach(|| match &rules_mask {
                    Some(m) => puct_with_rollout(&s, player, &cfg, &*net, m, &rollout),
                    None => puct_with_rollout(&s, player, &cfg, &*net, &UniformSampler, &rollout),
                });
                first_answer = r.first_answer.map(|a| a.cards());
                r.analysis
            }
            other => {
                return Err(value_error(format!(
                    "unknown algorithm {other:?} (use \"ismcts\", \"pimc\" or \"puct\")"
                )))
            }
        }
    };
    let d = PyDict::new(py);
    // infoset_endgame: None when it did not apply (too many hidden cards), else
    // whether every deal was solved; `forced` = (winner, plays) when they all agree.
    d.set_item("exact", exact)?;
    d.set_item("forced", forced)?;
    d.set_item("player", result.player)?;
    d.set_item("simulations", result.simulations)?;
    d.set_item("seconds", result.elapsed.as_secs_f64())?;
    d.set_item("belief_ess", belief_ess)?;
    // belief="rules": the share of drawn deals the rules excluded, and how many
    // samples gave up (after 1,000 rejections in a row) and used an excluded deal.
    let mask_stats = rules_mask.as_ref().map(RulesMask::stats);
    d.set_item(
        "belief_rejection_share",
        mask_stats.map(|m| m.rejection_share()),
    )?;
    d.set_item("belief_fallbacks", mask_stats.map(|m| m.fallbacks))?;
    d.set_item("first_answer", first_answer)?;
    let moves = result
        .moves
        .iter()
        .map(|m| {
            let md = PyDict::new(py);
            md.set_item("cards", m.action.cards())?;
            md.set_item("visits", m.visits)?;
            md.set_item("mean_score", m.mean_score)?;
            md.set_item("score_ci95", m.score_ci95)?;
            md.set_item("win_rate", m.win_rate)?;
            md.set_item("win_ci95", m.win_ci95)?;
            Ok(md)
        })
        .collect::<PyResult<Vec<_>>>()?;
    d.set_item("moves", moves)?;
    Ok(d)
}

/// Hidden-card samples as the search would draw them: `n` deals (each a list of
/// four hand masks) for the player to act, uniform (`belief=None`), uniform among
/// the deals the rules allow (`belief="rules"`) or weighted by the opponent model
/// (`belief="greedy"`). For studying belief models.
#[pyfunction]
#[pyo3(signature = (state, n, *, belief=None, candidates=256, epsilon=0.2, min_ess=0.25, seed=0))]
#[allow(clippy::too_many_arguments)]
fn sample_deals(
    state: &PyState,
    n: usize,
    belief: Option<&str>,
    candidates: usize,
    epsilon: f64,
    min_ess: f64,
    seed: u64,
) -> PyResult<Vec<[u64; 4]>> {
    let s = &state.inner;
    let player = s
        .current_player()
        .ok_or_else(|| value_error("the deal is over"))?;
    let mut rng = SmallRng::seed_from_u64(seed);
    if belief == Some("rules") {
        let mask = RulesMask::default();
        return Ok((0..n)
            .map(|_| mask.sample(s, player, &mut rng).hands())
            .collect());
    }
    let weighted = match belief {
        None => None,
        Some("greedy") => {
            let model = PolicyModel {
                policy: Greedy { epsilon: 0.0 },
                epsilon,
            };
            let cfg = BeliefConfig {
                candidates,
                min_ess,
            };
            Belief::new(s, player, &model, &cfg, &mut rng)
        }
        Some(other) => return Err(value_error(format!("unknown belief model {other:?}"))),
    };
    Ok((0..n)
        .map(|_| match &weighted {
            Some(b) => b.draw(&mut rng).hands(),
            None => s.determinize(player, &mut rng).hands(),
        })
        .collect())
}

/// A value network's estimate for the position (all hands visible to it):
/// `(scores, win_probabilities)` per absolute seat.
#[pyfunction]
fn value_estimate(state: &PyState, path: &str) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if state.inner.is_terminal() {
        return Err(value_error("the deal is over"));
    }
    let est = big2_games::Evaluator::evaluate(&*load_value_net(path)?, &state.inner);
    Ok((est.scores, est.wins))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyState>()?;
    m.add_function(wrap_pyfunction!(parse_cards, m)?)?;
    m.add_function(wrap_pyfunction!(format_cards, m)?)?;
    m.add_function(wrap_pyfunction!(classify_cards, m)?)?;
    m.add_function(wrap_pyfunction!(min_plays, m)?)?;
    m.add_function(wrap_pyfunction!(decompositions, m)?)?;
    m.add_function(wrap_pyfunction!(candidates, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(value_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(sample_deals, m)?)?;
    m.add_class::<vecenv::VecEnv>()?;
    m.add_function(wrap_pyfunction!(vecenv::encode, m)?)?;
    m.add_function(wrap_pyfunction!(vecenv::encode_full, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
