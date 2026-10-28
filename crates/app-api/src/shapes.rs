//! The JSON shapes that cross the app boundary, and their conversions to and
//! from engine types. Every string the app sees is a card id (`"3d"`) or a key
//! (`combo.pair`, `action.pass`, `error.illegal_action`); never display text.

use big2_big2::card::{cards_in, parse_mask};
use big2_big2::{Action, FlushOrder, IllegalAction, PassRule, RuleSet, Scoring, State, Straights};
use big2_games::Game;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Version of `docs/engine-interface.md` this crate implements.
pub const API_VERSION: u32 = 0;

/// An error response: `{"error": {"code": "error.*", "detail": "..."}}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub code: &'static str,
    pub detail: String,
}

impl ApiError {
    pub fn new(code: &'static str, detail: impl Into<String>) -> ApiError {
        ApiError {
            code,
            detail: detail.into(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "api": API_VERSION,
            "error": { "code": self.code, "detail": self.detail },
        })
    }
}

impl From<IllegalAction> for ApiError {
    fn from(e: IllegalAction) -> ApiError {
        match e {
            IllegalAction::GameOver => ApiError::new("error.game_over", e.to_string()),
            _ => ApiError::new("error.illegal_action", e.to_string()),
        }
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

// -- cards ------------------------------------------------------------------

/// A card mask as a list of card ids, weakest first: `["3d", "Tc", "2s"]`.
pub fn cards_to_json(mask: u64) -> Vec<String> {
    cards_in(mask).iter().map(|c| c.to_string()).collect()
}

/// Parses a list of card ids into a mask. Any malformed or repeated card is
/// `error.bad_card`.
pub fn cards_from_json(cards: &[String]) -> ApiResult<u64> {
    parse_mask(&cards.join(" ")).map_err(|e| ApiError::new("error.bad_card", e.to_string()))
}

// -- actions ------------------------------------------------------------------

pub const ACTION_PLAY: &str = "action.play";
pub const ACTION_PASS: &str = "action.pass";

/// A move: `{"type": "action.play", "cards": [...], "combo": "combo.pair"}` or
/// `{"type": "action.pass"}`. On input `combo` is ignored and `type` may be left
/// out (an empty card list then means pass).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ActionJson {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cards: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo: Option<String>,
}

/// The `combo.*` key of a combination kind.
pub fn combo_key(kind: big2_big2::Kind) -> String {
    format!("combo.{}", kind.name())
}

pub fn action_to_json(action: Action) -> ActionJson {
    match action {
        Action::Pass => ActionJson {
            kind: ACTION_PASS.to_string(),
            cards: Vec::new(),
            combo: None,
        },
        Action::Play(c) => ActionJson {
            kind: ACTION_PLAY.to_string(),
            cards: cards_to_json(c.cards),
            combo: Some(combo_key(c.kind)),
        },
    }
}

/// Builds and validates an action for the player to act in `state`.
pub fn action_from_json(state: &State, a: &ActionJson) -> ApiResult<Action> {
    let cards = match a.kind.as_str() {
        ACTION_PASS => 0,
        ACTION_PLAY | "" => cards_from_json(&a.cards)?,
        other => {
            return Err(ApiError::new(
                "error.bad_action",
                format!("unknown action type {other:?}"),
            ))
        }
    };
    if a.kind == ACTION_PLAY && cards == 0 {
        return Err(ApiError::new(
            "error.bad_action",
            "action.play needs at least one card",
        ));
    }
    if state.is_terminal() {
        return Err(IllegalAction::GameOver.into());
    }
    Ok(state.action_for_cards(cards)?)
}

/// One entry of the public history.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct TurnJson {
    pub player: usize,
    pub action: ActionJson,
}

// -- rules --------------------------------------------------------------------

/// Rule variants as `{"straights": "with_low_twos", ..., "last_card_duty": "on"}`.
pub fn rules_to_json(r: &RuleSet) -> Value {
    json!({
        "straights": r.straights.name(),
        "flush_order": r.flush_order.name(),
        "pass_rule": r.pass_rule.name(),
        "scoring": r.scoring.name(),
        "last_card_duty": if r.last_card_duty { "on" } else { "off" },
    })
}

/// Every variant and its options, default first (for `GameMeta.variants`).
pub fn variants_meta() -> Value {
    json!({
        "straights": Straights::ALL.iter().map(|v| v.name()).collect::<Vec<_>>(),
        "flush_order": FlushOrder::ALL.iter().map(|v| v.name()).collect::<Vec<_>>(),
        "pass_rule": PassRule::ALL.iter().map(|v| v.name()).collect::<Vec<_>>(),
        "scoring": Scoring::ALL.iter().map(|v| v.name()).collect::<Vec<_>>(),
        "last_card_duty": ["on", "off"],
    })
}

/// Reads rule variants; keys left out keep the default. Unknown keys or values
/// are `error.unknown_variant`.
pub fn rules_from_json(v: Option<&Value>) -> ApiResult<RuleSet> {
    let mut r = RuleSet::default();
    let Some(v) = v else { return Ok(r) };
    if v.is_null() {
        return Ok(r);
    }
    let map = v
        .as_object()
        .ok_or_else(|| ApiError::new("error.unknown_variant", "variants must be an object"))?;
    fn parse<T: std::str::FromStr>(key: &str, v: &Value) -> ApiResult<T>
    where
        T::Err: std::fmt::Display,
    {
        let s = v.as_str().ok_or_else(|| {
            ApiError::new("error.unknown_variant", format!("{key}: expected a string"))
        })?;
        s.parse()
            .map_err(|e: T::Err| ApiError::new("error.unknown_variant", format!("{key}: {e}")))
    }
    for (key, value) in map {
        match key.as_str() {
            "straights" => r.straights = parse(key, value)?,
            "flush_order" => r.flush_order = parse(key, value)?,
            "pass_rule" => r.pass_rule = parse(key, value)?,
            "scoring" => r.scoring = parse(key, value)?,
            "last_card_duty" => {
                r.last_card_duty = match value {
                    Value::Bool(b) => *b,
                    Value::String(s) if s == "on" => true,
                    Value::String(s) if s == "off" => false,
                    other => {
                        return Err(ApiError::new(
                            "error.unknown_variant",
                            format!("last_card_duty: {other} (use \"on\" or \"off\")"),
                        ))
                    }
                }
            }
            other => {
                return Err(ApiError::new(
                    "error.unknown_variant",
                    format!("unknown rule variant {other:?}"),
                ))
            }
        }
    }
    Ok(r)
}

// -- game setup ---------------------------------------------------------------

/// `GameConfig` of the contract (§4).
#[derive(Deserialize, Clone, Debug)]
pub struct GameConfig {
    #[serde(default = "four")]
    pub seats: usize,
    #[serde(default)]
    pub variants: Option<Value>,
    pub seed: u64,
    #[serde(default)]
    pub human_seats: Vec<usize>,
}

fn four() -> usize {
    4
}

// -- positions ----------------------------------------------------------------

/// The play to beat in a position.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct CurrentPlayJson {
    pub player: usize,
    pub cards: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo: Option<String>,
}

/// A mid-game starting point for a position that did not begin with a full deal
/// (hand-written drills). The trick's pass count starts at zero.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct StartJson {
    pub hands: Vec<Vec<String>>,
    pub current_player: usize,
    #[serde(default)]
    pub current_play: Option<CurrentPlayJson>,
}

/// `Position` of the contract (§3, §6): the whole hidden state. Exactly one of
/// `initial_hands` (a full deal) and `start` (a mid-game position) is set; the
/// state is that start plus `history` replayed. The remaining fields describe
/// the resulting position and are ignored on restore.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PositionJson {
    #[serde(default = "api_version")]
    pub api: u32,
    #[serde(default = "big2")]
    pub game: String,
    #[serde(default)]
    pub rules: Option<Value>,
    pub seed: u64,
    #[serde(default)]
    pub human_seats: Vec<usize>,
    #[serde(default)]
    pub initial_hands: Option<Vec<Vec<String>>>,
    #[serde(default)]
    pub start: Option<StartJson>,
    #[serde(default)]
    pub history: Vec<TurnJson>,
    // Derived, informational:
    #[serde(default)]
    pub hands: Vec<Vec<String>>,
    #[serde(default)]
    pub current_player: Option<usize>,
    #[serde(default)]
    pub current_play: Option<CurrentPlayJson>,
    #[serde(default)]
    pub trick_starts: Vec<usize>,
    #[serde(default)]
    pub winner: Option<usize>,
}

fn api_version() -> u32 {
    API_VERSION
}

fn big2() -> String {
    "big2".to_string()
}

pub fn current_play_to_json(state: &State) -> Option<CurrentPlayJson> {
    state.current_play().map(|(p, c)| CurrentPlayJson {
        player: p,
        cards: cards_to_json(c.cards),
        combo: Some(combo_key(c.kind)),
    })
}

/// Parses four hands (any size) into masks.
pub fn hands_from_json(hands: &[Vec<String>]) -> ApiResult<[u64; 4]> {
    if hands.len() != 4 {
        return Err(ApiError::new(
            "error.bad_position",
            format!("expected 4 hands, got {}", hands.len()),
        ));
    }
    let mut out = [0u64; 4];
    for (i, h) in hands.iter().enumerate() {
        out[i] = cards_from_json(h)?;
    }
    Ok(out)
}

// -- events -------------------------------------------------------------------

/// What `apply` reports happened, in order.
#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Event {
    #[serde(rename = "event.play")]
    Play {
        player: usize,
        action: ActionJson,
        cards_left: u32,
    },
    #[serde(rename = "event.pass")]
    Pass { player: usize },
    /// Everyone else passed: `player` leads the next trick.
    #[serde(rename = "event.trick_won")]
    TrickWon { player: usize },
    #[serde(rename = "event.deal_over")]
    DealOver {
        winner: usize,
        scores: Vec<f64>,
        penalties: Vec<u32>,
    },
}

// -- reasons ------------------------------------------------------------------

/// A reason key with its arguments: `{"key": "reason.cards_left", "args": {"cards_left": 7}}`.
#[derive(Serialize, Clone, Debug)]
pub struct Reason {
    pub key: &'static str,
    pub args: Map<String, Value>,
}

impl Reason {
    pub fn new(key: &'static str) -> Reason {
        Reason {
            key,
            args: Map::new(),
        }
    }

    pub fn arg(mut self, name: &str, value: impl Into<Value>) -> Reason {
        self.args.insert(name.to_string(), value.into());
        self
    }
}

/// Rounds a solver number for the app (two decimals).
pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
