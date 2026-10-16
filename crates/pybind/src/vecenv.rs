//! `Big2VecEnv`: many Big Two games stepped in lockstep, for RL (Phase 3).
//!
//! Each call to `observe()` returns, for every game waiting on a learner decision,
//! the encoded state and the encoded legal actions (see `poker_big2::features`),
//! as flat `bytes` that Python views with `numpy.frombuffer`. `step(choices)`
//! applies one chosen action per observed game.
//!
//! Modes:
//! * `"selfplay"` — the learner plays all four seats.
//! * `"vs_greedy"` — the learner plays one seat, the greedy bot the others.
//!   Games come in groups of four sharing one deal, with the learner in seat
//!   `i % 4` of game `i`: a duplicate-deal evaluation.

use poker_big2::bots::Greedy;
use poker_big2::features::{encode_action, encode_state, ACTION_LEN, STATE_LEN};
use poker_big2::{Action, RuleSet, State};
use poker_games::{Game, Policy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    SelfPlay,
    VsGreedy,
}

/// One finished game: `(env_index, scores, winner, learner_seat)`; `learner_seat`
/// is `None` in self-play.
type Finished = (usize, Vec<f64>, usize, Option<usize>);

/// Observations: `(state_bytes, action_bytes, action_counts, env_ids, players)`.
type Observation<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Vec<u32>,
    Vec<u32>,
    Vec<u8>,
);

#[pyclass(name = "Big2VecEnv", module = "poker._core")]
pub struct VecEnv {
    rules: RuleSet,
    mode: Mode,
    auto_reset: bool,
    bot: Greedy,
    rng: SmallRng,
    envs: Vec<State>,
    done: Vec<bool>,
    legal: Vec<Vec<Action>>,
    observed: Vec<usize>,
}

impl VecEnv {
    fn learner_seat(&self, i: usize) -> Option<usize> {
        match self.mode {
            Mode::SelfPlay => None,
            Mode::VsGreedy => Some(i % 4),
        }
    }

    fn is_learner(&self, i: usize, player: usize) -> bool {
        self.learner_seat(i).is_none_or(|s| s == player)
    }

    /// Plays bot seats until the learner must act or the game ends.
    fn advance_bots(&mut self, i: usize) {
        let mut legal = Vec::new();
        while let Some(p) = self.envs[i].current_player() {
            if self.is_learner(i, p) {
                break;
            }
            self.envs[i].legal_actions(&mut legal);
            let a = self.bot.choose(&self.envs[i], &legal, &mut self.rng);
            self.envs[i].apply(a);
        }
    }

    fn deal_all(&mut self) {
        let n = self.envs.len();
        for i in 0..n {
            if self.mode == Mode::VsGreedy && i % 4 != 0 {
                self.envs[i] = self.envs[i - i % 4].clone();
            } else {
                self.envs[i] = State::new_deal(self.rules, &mut self.rng);
            }
            self.done[i] = false;
        }
        for i in 0..n {
            self.advance_bots(i);
        }
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, seed=0, mode="selfplay", auto_reset=true, bot_epsilon=0.0, *, straights=None, flush_order=None, pass_rule=None, scoring=None, last_card_duty=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        seed: u64,
        mode: &str,
        auto_reset: bool,
        bot_epsilon: f64,
        straights: Option<&str>,
        flush_order: Option<&str>,
        pass_rule: Option<&str>,
        scoring: Option<&str>,
        last_card_duty: Option<bool>,
    ) -> PyResult<Self> {
        let mode = match mode {
            "selfplay" => Mode::SelfPlay,
            "vs_greedy" => Mode::VsGreedy,
            m => {
                return Err(PyValueError::new_err(format!(
                    "unknown mode {m:?} (selfplay | vs_greedy)"
                )))
            }
        };
        if n == 0 || (mode == Mode::VsGreedy && !n.is_multiple_of(4)) {
            return Err(PyValueError::new_err(
                "n must be positive (and a multiple of 4 for vs_greedy)",
            ));
        }
        let rules = crate::rules_from(straights, flush_order, pass_rule, scoring, last_card_duty)?;
        let mut rng = SmallRng::seed_from_u64(seed);
        let placeholder = State::new_deal(rules, &mut rng);
        let mut env = VecEnv {
            rules,
            mode,
            auto_reset,
            bot: Greedy {
                epsilon: bot_epsilon,
            },
            rng,
            envs: vec![placeholder; n],
            done: vec![false; n],
            legal: vec![Vec::new(); n],
            observed: Vec::new(),
        };
        env.deal_all();
        Ok(env)
    }

    /// Self-play environment starting from given positions (no auto-reset), e.g.
    /// determinizations of one position for policy-rollout search.
    #[staticmethod]
    fn from_states(states: Vec<PyRef<'_, crate::PyState>>) -> PyResult<Self> {
        let first = states
            .first()
            .ok_or_else(|| PyValueError::new_err("no states"))?;
        let rules = *first.inner.rules();
        let envs: Vec<State> = states.iter().map(|s| s.inner.clone()).collect();
        let n = envs.len();
        Ok(VecEnv {
            rules,
            mode: Mode::SelfPlay,
            auto_reset: false,
            bot: Greedy::default(),
            rng: SmallRng::seed_from_u64(0),
            done: envs.iter().map(|s| s.is_terminal()).collect(),
            envs,
            legal: vec![Vec::new(); n],
            observed: Vec::new(),
        })
    }

    /// Final scores of game `i`, or `None` while it is running.
    fn scores(&self, i: usize) -> Option<Vec<f64>> {
        self.envs[i].outcome().map(|o| o.scores)
    }

    /// Winner of game `i`, once it is over.
    fn winner(&self, i: usize) -> Option<usize> {
        self.envs[i].winner()
    }

    /// Number of games.
    #[getter]
    fn n(&self) -> usize {
        self.envs.len()
    }

    /// Number of finished games (only grows when `auto_reset=False`).
    #[getter]
    fn num_done(&self) -> usize {
        self.done.iter().filter(|&&d| d).count()
    }

    /// Feature sizes: `(STATE_LEN, ACTION_LEN)`.
    #[staticmethod]
    fn feature_sizes() -> (usize, usize) {
        (STATE_LEN, ACTION_LEN)
    }

    /// New deals for every game.
    fn reset(&mut self) {
        self.deal_all();
    }

    /// Encodes every game waiting on a learner decision.
    ///
    /// Returns `(states, actions, counts, env_ids, players)`: `states` is
    /// `len(env_ids) × STATE_LEN` bytes, `actions` is `sum(counts) × ACTION_LEN`
    /// bytes (each game's legal actions in order), `players` is the seat to act.
    fn observe<'py>(&mut self, py: Python<'py>) -> Observation<'py> {
        let mut states = Vec::with_capacity(self.envs.len() * STATE_LEN);
        let mut actions = Vec::new();
        let (mut counts, mut ids, mut players) = (Vec::new(), Vec::new(), Vec::new());
        self.observed.clear();
        for i in 0..self.envs.len() {
            if self.done[i] {
                continue;
            }
            let Some(p) = self.envs[i].current_player() else {
                continue;
            };
            let mut legal = std::mem::take(&mut self.legal[i]);
            self.envs[i].legal_actions(&mut legal);
            encode_state(&self.envs[i], &mut states);
            for a in &legal {
                encode_action(a, &mut actions);
            }
            counts.push(legal.len() as u32);
            ids.push(i as u32);
            players.push(p as u8);
            self.legal[i] = legal;
            self.observed.push(i);
        }
        (
            PyBytes::new(py, &states),
            PyBytes::new(py, &actions),
            counts,
            ids,
            players,
        )
    }

    /// Card masks of the legal actions of game `i`, in the order `observe` used.
    fn legal_cards(&self, i: usize) -> Vec<u64> {
        self.legal[i].iter().map(Action::cards).collect()
    }

    /// Applies one action per observed game (`choices[k]` indexes game k's legal
    /// actions). Returns the games that finished: `(env, scores, winner, learner_seat)`.
    fn step(&mut self, choices: Vec<usize>) -> PyResult<Vec<Finished>> {
        if choices.len() != self.observed.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} choices (one per observed game), got {}",
                self.observed.len(),
                choices.len()
            )));
        }
        let mut finished = Vec::new();
        let observed = std::mem::take(&mut self.observed);
        for (&i, &c) in observed.iter().zip(&choices) {
            let a = *self.legal[i].get(c).ok_or_else(|| {
                PyValueError::new_err(format!("choice {c} out of range for game {i}"))
            })?;
            self.envs[i].apply(a);
            self.advance_bots(i);
            if let Some(o) = self.envs[i].outcome() {
                finished.push((
                    i,
                    o.scores,
                    o.winner.expect("Big Two has a winner"),
                    self.learner_seat(i),
                ));
                if self.auto_reset {
                    let seed = self.rng.next_u64();
                    self.envs[i] = State::new_deal(self.rules, &mut SmallRng::seed_from_u64(seed));
                    self.advance_bots(i);
                } else {
                    self.done[i] = true;
                }
            }
        }
        Ok(finished)
    }

    /// A copy of game `i` (for inspection and tests).
    fn state(&self, i: usize) -> crate::PyState {
        crate::PyState {
            inner: self.envs[i].clone(),
        }
    }
}

/// Encodes one state for its player to act: `(state_bytes, action_bytes, legal_cards)`.
#[pyfunction]
pub fn encode<'py>(
    py: Python<'py>,
    state: &crate::PyState,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<u64>)> {
    if state.inner.is_terminal() {
        return Err(PyValueError::new_err("the deal is over"));
    }
    let (mut s, mut a) = (Vec::with_capacity(STATE_LEN), Vec::new());
    encode_state(&state.inner, &mut s);
    let legal = state.inner.legal_actions_vec();
    for x in &legal {
        encode_action(x, &mut a);
    }
    Ok((
        PyBytes::new(py, &s),
        PyBytes::new(py, &a),
        legal.iter().map(Action::cards).collect(),
    ))
}
