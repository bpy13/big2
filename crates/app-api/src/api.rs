//! The calls of `docs/engine-interface.md` §3, over the Big Two engine.
//!
//! [`Api`] holds the running games by `game_id`. Every call takes and returns
//! JSON strings; see `README.md` for the shapes. Errors come back as
//! `{"api": 0, "error": {"code": "error.*", "detail": "..."}}`.

use crate::reasons::{position_reasons, solver_reasons};
use crate::shapes::{
    action_from_json, action_to_json, cards_to_json, current_play_to_json, hands_from_json, round2,
    rules_from_json, rules_to_json, variants_meta, ActionJson, ApiError, ApiResult,
    CurrentPlayJson, Event, GameConfig, PositionJson, Reason, StartJson, TurnJson, API_VERSION,
};
use crate::solver::{
    bot_iterations, grade_for, ismcts_analysis, pimc_budgeted, HINT_ITERATIONS, REVIEW_DEALS,
};
use big2_big2::bots::StructuredGreedy;
use big2_big2::qnet::{QNet, QNetPolicy};
use big2_big2::{Action, RuleSet, State, NUM_PLAYERS};
use big2_games::{Game, Policy};
use big2_search::{Analysis, MoveStats};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use web_time::Instant;

/// Bot levels, weakest first (`GameMeta.bot_levels`).
pub const BOT_LEVELS: [&str; 5] = ["random", "easy", "medium", "hard", "expert"];

/// The `easy` bot is the structured greedy policy with this much random play.
const EASY_EPSILON: f64 = 0.1;

/// Softmax temperature of the `medium` bot's network policy.
const MEDIUM_TEMPERATURE: f64 = 1.0;

/// The `medium` network, embedded because WebAssembly has no file I/O.
static QNET_BYTES: &[u8] = include_bytes!("../../../models/big2-qnet-duty.pknn");

/// The network is parsed once, on the first `medium` move, and shared by all games.
fn qnet() -> Arc<QNet> {
    static NET: OnceLock<Arc<QNet>> = OnceLock::new();
    NET.get_or_init(|| {
        Arc::new(QNet::from_bytes(QNET_BYTES).expect("the embedded network is valid"))
    })
    .clone()
}

/// Where a game started: a full deal, or a mid-game position (a drill).
#[derive(Clone, Debug)]
enum Start {
    Deal([u64; NUM_PLAYERS]),
    Position {
        hands: [u64; NUM_PLAYERS],
        current: usize,
        current_play: Option<(usize, u64)>,
    },
}

impl Start {
    fn state(&self, rules: RuleSet) -> ApiResult<State> {
        match self {
            Start::Deal(hands) => State::from_hands(rules, *hands),
            Start::Position {
                hands,
                current,
                current_play,
            } => State::from_position(rules, *hands, *current, *current_play),
        }
        .map_err(|e| ApiError::new("error.bad_position", e))
    }
}

/// One game: its start, its current state, and the settings the app gave it.
#[derive(Clone, Debug)]
struct GameEntry {
    start: Start,
    state: State,
    seed: u64,
    human_seats: Vec<usize>,
    /// Index in the history where each trick began (the first is 0).
    trick_starts: Vec<usize>,
}

impl GameEntry {
    /// Applies a legal action, keeping `trick_starts` in step.
    fn apply(&mut self, action: Action) {
        self.state.apply(action);
        // Everyone else passed: the next action opens a new trick.
        if self.state.current_play().is_none() && !self.state.is_terminal() {
            self.trick_starts.push(self.state.history().len());
        }
    }

    /// Replays `history` from the start (each turn is validated).
    fn replay(&mut self, history: &[TurnJson]) -> ApiResult<()> {
        for (i, turn) in history.iter().enumerate() {
            let Some(player) = self.state.current_player() else {
                return Err(ApiError::new(
                    "error.bad_position",
                    format!("history continues after the deal ended (turn {i})"),
                ));
            };
            if turn.player != player {
                return Err(ApiError::new(
                    "error.bad_position",
                    format!(
                        "turn {i}: player {} acted but player {player} was to move",
                        turn.player
                    ),
                ));
            }
            let action = action_from_json(&self.state, &turn.action).map_err(|e| {
                ApiError::new("error.bad_position", format!("turn {i}: {}", e.detail))
            })?;
            self.apply(action);
        }
        Ok(())
    }

    fn history_json(&self) -> Vec<TurnJson> {
        self.state
            .history()
            .iter()
            .map(|t| TurnJson {
                player: t.player as usize,
                action: action_to_json(t.action),
            })
            .collect()
    }

    /// The states before each decision of `seat`, with the history index of each.
    fn decisions_of(&self, seat: usize) -> ApiResult<Vec<(usize, State, Action)>> {
        let mut s = self.start.state(*self.state.rules())?;
        let mut out = Vec::new();
        for (i, t) in self.state.history().iter().enumerate() {
            if t.player as usize == seat {
                out.push((i, s.clone(), t.action));
            }
            s.apply(t.action);
        }
        Ok(out)
    }

    /// A seeded generator for the decision at this point of the game: the same
    /// position always draws the same numbers, so a restored game continues
    /// exactly as the original would have.
    fn rng(&self, salt: u64) -> SmallRng {
        SmallRng::seed_from_u64(self.decision_seed(salt))
    }

    fn decision_seed(&self, salt: u64) -> u64 {
        self.seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.state.history().len() as u64)
            .wrapping_mul(0xD1B5_4A32_D192_ED03)
            .wrapping_add(salt)
    }
}

/// The engine as the app sees it: games by id, JSON in and out.
#[derive(Default, Debug)]
pub struct Api {
    games: HashMap<String, GameEntry>,
    next_id: u64,
}

/// Serializes a successful result with `"api": 0`, or the error.
fn respond(result: ApiResult<Value>) -> String {
    let mut v = match result {
        Ok(v) => v,
        Err(e) => e.to_json(),
    };
    if let Value::Object(m) = &mut v {
        m.insert("api".into(), json!(API_VERSION));
    }
    v.to_string()
}

fn parse_json<T: serde::de::DeserializeOwned>(what: &str, text: &str) -> ApiResult<T> {
    serde_json::from_str(text).map_err(|e| ApiError::new("error.bad_json", format!("{what}: {e}")))
}

fn check_seat(seat: usize) -> ApiResult<()> {
    if seat < NUM_PLAYERS {
        Ok(())
    } else {
        Err(ApiError::new("error.bad_seat", format!("no seat {seat}")))
    }
}

fn budget(ms: u32) -> Option<Duration> {
    (ms > 0).then(|| Duration::from_millis(u64::from(ms)))
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

fn to_value<T: Serialize>(t: &T) -> Value {
    serde_json::to_value(t).expect("our shapes always serialize")
}

impl Api {
    pub fn new() -> Api {
        Api::default()
    }

    // -- the calls ------------------------------------------------------------

    /// `GameMeta` (§4).
    pub fn meta(&self) -> String {
        respond(Ok(json!({
            "game": "big2",
            "seats": [NUM_PLAYERS, NUM_PLAYERS],
            "variants": variants_meta(),
            "bot_levels": BOT_LEVELS,
            "has_hint": true,
            "has_review": true,
            "has_grade": true,
        })))
    }

    /// Deals a new game from `GameConfig` JSON; returns `{game_id}`.
    pub fn new_game(&mut self, config: &str) -> String {
        respond(self.new_game_inner(config))
    }

    fn new_game_inner(&mut self, config: &str) -> ApiResult<Value> {
        let cfg: GameConfig = parse_json("config", config)?;
        if cfg.seats != NUM_PLAYERS {
            return Err(ApiError::new(
                "error.bad_config",
                format!("big2 is played by {NUM_PLAYERS} seats, not {}", cfg.seats),
            ));
        }
        for &s in &cfg.human_seats {
            check_seat(s)?;
        }
        let rules = rules_from_json(cfg.variants.as_ref())?;
        let mut rng = SmallRng::seed_from_u64(cfg.seed);
        let state = State::new_deal(rules, &mut rng);
        let entry = GameEntry {
            start: Start::Deal(state.hands()),
            state,
            seed: cfg.seed,
            human_seats: cfg.human_seats,
            trick_starts: vec![0],
        };
        let id = self.register(entry);
        Ok(json!({ "game_id": id }))
    }

    fn register(&mut self, entry: GameEntry) -> String {
        self.next_id += 1;
        let id = format!("g{}", self.next_id);
        self.games.insert(id.clone(), entry);
        id
    }

    fn game(&self, game_id: &str) -> ApiResult<&GameEntry> {
        self.games
            .get(game_id)
            .ok_or_else(|| ApiError::new("error.unknown_game", format!("no game {game_id:?}")))
    }

    fn game_mut(&mut self, game_id: &str) -> ApiResult<&mut GameEntry> {
        self.games
            .get_mut(game_id)
            .ok_or_else(|| ApiError::new("error.unknown_game", format!("no game {game_id:?}")))
    }

    /// `SeatView`: what `seat` may see.
    pub fn view(&self, game_id: &str, seat: usize) -> String {
        respond((|| {
            check_seat(seat)?;
            Ok(seat_view(game_id, self.game(game_id)?, seat))
        })())
    }

    /// `{actions: [Action]}` for the seat to act (empty once the deal is over).
    pub fn legal_actions(&self, game_id: &str) -> String {
        respond((|| {
            let g = self.game(game_id)?;
            let actions: Vec<ActionJson> = g
                .state
                .legal_actions_vec()
                .into_iter()
                .map(action_to_json)
                .collect();
            Ok(json!({ "actions": actions, "seat": g.state.current_player() }))
        })())
    }

    /// Applies an `Action` for the seat to act; returns `{events, view}` where
    /// `view` is for the seat that acted.
    pub fn apply(&mut self, game_id: &str, action: &str) -> String {
        respond(self.apply_inner(game_id, action))
    }

    fn apply_inner(&mut self, game_id: &str, action: &str) -> ApiResult<Value> {
        let a: ActionJson = parse_json("action", action)?;
        let g = self.game_mut(game_id)?;
        let seat = g
            .state
            .current_player()
            .ok_or_else(|| ApiError::new("error.game_over", "the deal is already over"))?;
        let act = action_from_json(&g.state, &a)?;
        let events = apply_with_events(g, act);
        Ok(json!({ "events": events, "view": seat_view(game_id, g, seat) }))
    }

    /// The bot's move for the seat to act, at `level`, within `budget_ms`.
    /// The move is not applied; call `apply` with it.
    pub fn bot_action(&self, game_id: &str, level: &str, budget_ms: u32) -> String {
        respond(self.bot_action_inner(game_id, level, budget_ms))
    }

    fn bot_action_inner(&self, game_id: &str, level: &str, budget_ms: u32) -> ApiResult<Value> {
        let g = self.game(game_id)?;
        let legal = g.state.legal_actions_vec();
        if legal.is_empty() {
            return Err(ApiError::new("error.game_over", "the deal is already over"));
        }
        let level_index = BOT_LEVELS.iter().position(|l| *l == level).ok_or_else(|| {
            ApiError::new(
                "error.unknown_level",
                format!("unknown bot level {level:?}"),
            )
        })? as u64;
        let start = Instant::now();
        let action = match level {
            "random" => legal[g.rng(level_index).gen_range(0..legal.len())],
            "easy" => StructuredGreedy {
                epsilon: EASY_EPSILON,
                ..StructuredGreedy::default()
            }
            .choose(&g.state, &legal, &mut g.rng(level_index)),
            "medium" => QNetPolicy {
                net: qnet(),
                temperature: MEDIUM_TEMPERATURE,
            }
            .choose(&g.state, &legal, &mut g.rng(level_index)),
            _ => {
                let iterations = bot_iterations(level).expect("a search level");
                ismcts_analysis(
                    &g.state,
                    iterations,
                    budget(budget_ms),
                    g.decision_seed(level_index),
                )
                .best()
            }
        };
        let mut v = to_value(&action_to_json(action));
        v["budget_used_ms"] = json!(elapsed_ms(start));
        Ok(v)
    }

    /// `Hint` (§5) for `seat`, who must be the player to act.
    pub fn hint(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
        respond(self.hint_inner(game_id, seat, budget_ms, HINT_ITERATIONS))
    }

    /// [`Api::hint`] with an explicit simulation count (for measurements).
    pub fn hint_with_iterations(
        &self,
        game_id: &str,
        seat: usize,
        budget_ms: u32,
        iterations: u64,
    ) -> String {
        respond(self.hint_inner(game_id, seat, budget_ms, iterations))
    }

    fn hint_inner(
        &self,
        game_id: &str,
        seat: usize,
        budget_ms: u32,
        iterations: u64,
    ) -> ApiResult<Value> {
        check_seat(seat)?;
        let g = self.game(game_id)?;
        must_be_to_move(&g.state, seat)?;
        let start = Instant::now();
        let analysis = ismcts_analysis(
            &g.state,
            iterations,
            budget(budget_ms),
            g.decision_seed(0xA1),
        );
        let legal = g.state.legal_actions_vec();
        let moves: Vec<Value> = analysis
            .top(3)
            .iter()
            .map(|m| move_json(&g.state, &legal, m, &analysis))
            .collect();
        Ok(json!({
            "seat": seat,
            "moves": moves,
            "simulations": analysis.simulations,
            "budget_used_ms": elapsed_ms(start),
        }))
    }

    /// `{items: [ReviewItem]}` (§5): one item per decision `seat` has made so far,
    /// forced moves included (they carry `reason.forced` and no search).
    pub fn review(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
        respond(self.review_inner(game_id, seat, budget_ms))
    }

    fn review_inner(&self, game_id: &str, seat: usize, budget_ms: u32) -> ApiResult<Value> {
        check_seat(seat)?;
        let g = self.game(game_id)?;
        let start = Instant::now();
        let deadline = budget(budget_ms).map(|b| start + b);
        let decisions = g.decisions_of(seat)?;
        let mut open = decisions
            .iter()
            .filter(|(_, s, _)| s.legal_actions_vec().len() > 1)
            .count() as u32;
        let scoring = g.state.rules().scoring;
        let mut items = Vec::with_capacity(decisions.len());
        for (turn, state, played) in &decisions {
            let legal = state.legal_actions_vec();
            if legal.len() == 1 {
                let played_json = to_value(&action_to_json(*played));
                items.push(json!({
                    "turn": turn,
                    "player": seat,
                    "played": played_json,
                    "best": played_json,
                    "loss": 0.0,
                    "grade": "best",
                    "reasons": [Reason::new("reason.forced")],
                    "numbers": {},
                }));
                continue;
            }
            // Share the remaining time equally among the decisions still to review.
            let item_deadline = deadline.map(|d| {
                let left = d.saturating_duration_since(Instant::now());
                Instant::now() + left / open.max(1)
            });
            open -= 1;
            let analysis = pimc_budgeted(
                state,
                REVIEW_DEALS,
                item_deadline,
                g.seed.wrapping_add(*turn as u64),
            );
            let best = &analysis.moves[0];
            let mine = analysis
                .moves
                .iter()
                .find(|m| m.action == *played)
                .expect("the played move was legal");
            let loss = (best.mean_score - mine.mean_score).max(0.0);
            let is_best = best.action == *played;
            let mut reasons = position_reasons(state, *played, &legal);
            reasons.extend(solver_reasons(mine, best, analysis.moves.get(1)));
            items.push(json!({
                "turn": turn,
                "player": seat,
                "played": action_to_json(*played),
                "best": action_to_json(best.action),
                "loss": round2(loss),
                "grade": grade_for(is_best, loss, scoring),
                "reasons": reasons,
                "best_reasons": position_reasons(state, best.action, &legal),
                "numbers": {
                    "played_score": round2(mine.mean_score),
                    "best_score": round2(best.mean_score),
                    "played_win_rate": round3(mine.win_rate),
                    "best_win_rate": round3(best.win_rate),
                    "deals": mine.visits,
                    "simulations": analysis.simulations,
                },
            }));
        }
        Ok(json!({
            "seat": seat,
            "items": items,
            "budget_used_ms": elapsed_ms(start),
        }))
    }

    /// `Grade` (§6): how `action` compares with the solver's best move in
    /// `position`, for the player to act there.
    pub fn grade(&self, position: &str, action: &str, budget_ms: u32) -> String {
        respond(grade_inner(position, action, budget_ms))
    }

    /// `Position`: the whole hidden state, enough to `restore` an identical game.
    pub fn snapshot(&self, game_id: &str) -> String {
        respond(self.game(game_id).map(|g| to_value(&position_json(g))))
    }

    /// Rebuilds a game from a `Position`; returns `{game_id}`.
    pub fn restore(&mut self, position: &str) -> String {
        respond((|| {
            let entry = entry_from_position(position)?;
            let id = self.register(entry);
            Ok(json!({ "game_id": id }))
        })())
    }

    /// Forgets a game. Dropping an unknown id is not an error.
    pub fn drop_game(&mut self, game_id: &str) -> String {
        self.games.remove(game_id);
        respond(Ok(json!({})))
    }

    /// Number of games held (for tests).
    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }
}

fn must_be_to_move(state: &State, seat: usize) -> ApiResult<()> {
    match state.current_player() {
        None => Err(ApiError::new("error.game_over", "the deal is already over")),
        Some(p) if p == seat => Ok(()),
        Some(p) => Err(ApiError::new(
            "error.not_your_turn",
            format!("seat {seat} asked but seat {p} is to move"),
        )),
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// One hint move with its numbers and reasons.
fn move_json(
    state: &State,
    legal: &[Action],
    m: &MoveStats<Action>,
    analysis: &Analysis<Action>,
) -> Value {
    let best = &analysis.moves[0];
    let mut reasons = position_reasons(state, m.action, legal);
    reasons.extend(solver_reasons(m, best, analysis.moves.get(1)));
    let hand = state.hand(analysis.player);
    json!({
        "action": action_to_json(m.action),
        "score": round2(m.mean_score),
        "win_rate": round3(m.win_rate),
        "numbers": {
            "cards_left_after": hand.count_ones() - m.action.cards().count_ones(),
            "visits": m.visits,
            "share": round3(m.visits as f64 / analysis.simulations.max(1) as f64),
            "score_ci95": round2(m.score_ci95),
            "win_ci95": round3(m.win_ci95),
        },
        "reasons": reasons,
    })
}

/// Applies `action` and reports what happened.
fn apply_with_events(g: &mut GameEntry, action: Action) -> Vec<Event> {
    let player = g.state.current_player().expect("checked by the caller");
    g.apply(action);
    let mut events = vec![match action {
        Action::Pass => Event::Pass { player },
        Action::Play(_) => Event::Play {
            player,
            action: action_to_json(action),
            cards_left: g.state.hand_sizes()[player],
        },
    }];
    if let Some(winner) = g.state.winner() {
        let outcome = g.state.outcome().expect("finished deal");
        events.push(Event::DealOver {
            winner,
            scores: outcome.scores,
            penalties: g.state.penalties().to_vec(),
        });
    } else if g.state.current_play().is_none() {
        events.push(Event::TrickWon {
            player: g.state.current_player().expect("not over"),
        });
    }
    events
}

/// What `seat` may see of the game.
fn seat_view(game_id: &str, g: &GameEntry, seat: usize) -> Value {
    let s = &g.state;
    let over = s.is_terminal();
    let history = g.history_json();
    let trick_start = *g.trick_starts.last().unwrap_or(&0);
    let your_turn = s.current_player() == Some(seat);
    let legal: Vec<ActionJson> = if your_turn {
        s.legal_actions_vec()
            .into_iter()
            .map(action_to_json)
            .collect()
    } else {
        Vec::new()
    };
    let outcome = s.outcome();
    json!({
        "game_id": game_id,
        "seat": seat,
        "rules": rules_to_json(s.rules()),
        "seed": g.seed,
        "human_seats": g.human_seats,
        "hand": cards_to_json(s.hand(seat)),
        "hand_sizes": s.hand_sizes(),
        "current_player": s.current_player(),
        "your_turn": your_turn,
        "is_opening": s.is_opening(),
        "current_play": current_play_to_json(s),
        "trick": history[trick_start..].to_vec(),
        "history": history,
        "trick_starts": g.trick_starts,
        "legal": legal,
        "locked_out": s.locked_out(),
        "over": over,
        "winner": s.winner(),
        "scores": outcome.map(|o| o.scores),
        "penalties": over.then(|| s.penalties()),
        "final_hands": over.then(|| s.hands().iter().map(|&h| cards_to_json(h)).collect::<Vec<_>>()),
        "initial_hands": match (&g.start, over) {
            (Start::Deal(hands), true) => Some(hands.iter().map(|&h| cards_to_json(h)).collect::<Vec<_>>()),
            _ => None,
        },
    })
}

fn position_json(g: &GameEntry) -> PositionJson {
    let s = &g.state;
    let (initial_hands, start) = match &g.start {
        Start::Deal(hands) => (
            Some(hands.iter().map(|&h| cards_to_json(h)).collect()),
            None,
        ),
        Start::Position {
            hands,
            current,
            current_play,
        } => (
            None,
            Some(StartJson {
                hands: hands.iter().map(|&h| cards_to_json(h)).collect(),
                current_player: *current,
                current_play: current_play.map(|(p, c)| CurrentPlayJson {
                    player: p,
                    cards: cards_to_json(c),
                    combo: None,
                }),
            }),
        ),
    };
    PositionJson {
        api: API_VERSION,
        game: "big2".into(),
        rules: Some(rules_to_json(s.rules())),
        seed: g.seed,
        human_seats: g.human_seats.clone(),
        initial_hands,
        start,
        history: g.history_json(),
        hands: s.hands().iter().map(|&h| cards_to_json(h)).collect(),
        current_player: s.current_player(),
        current_play: current_play_to_json(s),
        trick_starts: g.trick_starts.clone(),
        winner: s.winner(),
    }
}

fn entry_from_position(position: &str) -> ApiResult<GameEntry> {
    let p: PositionJson = parse_json("position", position)?;
    if p.game != "big2" {
        return Err(ApiError::new(
            "error.bad_position",
            format!("this engine plays big2, not {:?}", p.game),
        ));
    }
    for &s in &p.human_seats {
        check_seat(s)?;
    }
    let rules = rules_from_json(p.rules.as_ref())?;
    let start = match (&p.initial_hands, &p.start) {
        (Some(hands), None) => Start::Deal(hands_from_json(hands)?),
        (None, Some(st)) => {
            check_seat(st.current_player)?;
            let current_play = match &st.current_play {
                None => None,
                Some(cp) => {
                    check_seat(cp.player)?;
                    Some((cp.player, crate::shapes::cards_from_json(&cp.cards)?))
                }
            };
            Start::Position {
                hands: hands_from_json(&st.hands)?,
                current: st.current_player,
                current_play,
            }
        }
        _ => {
            return Err(ApiError::new(
                "error.bad_position",
                "a position has exactly one of initial_hands and start",
            ))
        }
    };
    let state = start.state(rules)?;
    let mut entry = GameEntry {
        start,
        state,
        seed: p.seed,
        human_seats: p.human_seats,
        trick_starts: vec![0],
    };
    entry.replay(&p.history)?;
    Ok(entry)
}

fn grade_inner(position: &str, action: &str, budget_ms: u32) -> ApiResult<Value> {
    let g = entry_from_position(position)?;
    let a: ActionJson = parse_json("action", action)?;
    let seat = g
        .state
        .current_player()
        .ok_or_else(|| ApiError::new("error.game_over", "the position is a finished deal"))?;
    let played = action_from_json(&g.state, &a)?;
    let start = Instant::now();
    let legal = g.state.legal_actions_vec();
    let scoring = g.state.rules().scoring;
    if legal.len() == 1 {
        return Ok(json!({
            "seat": seat,
            "grade": "best",
            "loss": 0.0,
            "best": action_to_json(played),
            "reasons": [Reason::new("reason.forced")],
            "numbers": {},
            "budget_used_ms": elapsed_ms(start),
        }));
    }
    let analysis = ismcts_analysis(
        &g.state,
        HINT_ITERATIONS,
        budget(budget_ms),
        g.decision_seed(0xA1),
    );
    let best = &analysis.moves[0];
    let mine = analysis
        .moves
        .iter()
        .find(|m| m.action == played)
        .expect("the action was validated as legal");
    let is_best = best.action == played;
    let loss = (best.mean_score - mine.mean_score).max(0.0);
    let mut reasons = position_reasons(&g.state, played, &legal);
    reasons.extend(solver_reasons(mine, best, analysis.moves.get(1)));
    Ok(json!({
        "seat": seat,
        "grade": grade_for(is_best, loss, scoring),
        "loss": round2(loss),
        "best": action_to_json(best.action),
        "reasons": reasons,
        "best_reasons": position_reasons(&g.state, best.action, &legal),
        "numbers": {
            "played_score": round2(mine.mean_score),
            "best_score": round2(best.mean_score),
            "played_win_rate": round3(mine.win_rate),
            "best_win_rate": round3(best.win_rate),
            "simulations": analysis.simulations,
        },
        "budget_used_ms": elapsed_ms(start),
    }))
}
