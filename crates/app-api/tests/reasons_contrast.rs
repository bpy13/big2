//! Causes and merits (`reasons::contrast_reasons`, plan Task 35): every cause
//! is true of the played move and false of the best move, and the stored
//! graded decisions that lost points mostly get one.
//!
//! `cargo test -p big2-app-api --test reasons_contrast -- --nocapture` prints
//! the coverage table recorded in `docs/study-results/build-phase4.md` §7.

use big2_app_api::api::state_from_position;
use big2_app_api::reasons::{contrast_reasons, position_reasons};
use big2_app_api::shapes::{action_from_json, ActionJson, Reason};
use big2_big2::{Action, State};
use big2_games::Game;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

/// Keys `contrast_reasons` may return as causes.
const CAUSE_KEYS: [&str; 10] = [
    "reason.lets_opponent_out",
    "reason.breaks_group",
    "reason.more_plays_to_out",
    "reason.can_be_beaten",
    "reason.misses_sure_out",
    "reason.spends_two",
    "reason.spends_high_card",
    "reason.leaves_loose_singles",
    "reason.keeps_low_single",
    "reason.pass_returns_lead",
];

/// Graded decisions the coverage bar is measured on.
const DECISIONS: usize = 200;

fn action(state: &State, v: &Value) -> Action {
    let a: ActionJson = serde_json::from_value(v.clone()).expect("action json");
    action_from_json(state, &a).expect("legal action")
}

fn causes_for(position: &Value, played: &Value, best: &Value) -> (Vec<Reason>, Vec<Reason>) {
    let state = state_from_position(&position.to_string()).expect("position restores");
    let p = action(&state, played);
    let b = action(&state, best);
    contrast_reasons(&state, p, b)
}

fn keys(rs: &[Reason]) -> Vec<&'static str> {
    rs.iter().map(|r| r.key).collect()
}

// The five positions behind the app's ten Big2 coach examples (hk-games
// `app/src/drills/big2.json`, `docs/coach-voice-examples.md`).
const P0001: &str = r#"{"game": "big2", "seed": 1001, "start": {"hands": [["3d", "4c", "5c", "8h", "8s", "Jd", "Kd", "Kh"], ["4s", "6d", "7s", "8c", "9h", "Ts", "Jc", "Qh"], ["3h", "4h", "6c", "7h", "Th", "Ah"], ["5s", "Td", "Tc", "Qd", "Qc", "Ac"]], "current_player": 0, "current_play": {"player": 3, "cards": ["Qs"], "combo": "combo.single"}}, "history": []}"#;
const P0002: &str = r#"{"game": "big2", "seed": 1002, "start": {"hands": [["3h", "4d", "9h", "Th", "Qd", "Kc", "Ks", "As"], ["4s", "5d", "5h", "6s", "9c", "Jd", "Ac"], ["4c", "5c", "6h", "7c", "7h", "7s", "Ts", "Js", "2s"], ["6d", "Qh", "Kd", "Kh", "Ad"]], "current_player": 0, "current_play": {"player": 3, "cards": ["Qc"], "combo": "combo.single"}}, "history": []}"#;
const P0004: &str = r#"{"game": "big2", "seed": 3001, "start": {"hands": [["3d", "3s", "Kd", "Ks"], ["7d", "Jc", "Qs"], ["7h", "8d", "8s", "Td"], ["Qd", "Ah"]], "current_player": 0, "current_play": null}, "history": []}"#;
const P0006: &str = r#"{"game": "big2", "seed": 3003, "start": {"hands": [["8h", "Jd", "Js"], ["3d", "4d", "9d", "Qc"], ["5s", "Th", "Ts"], ["7d", "Kd"]], "current_player": 0, "current_play": null}, "history": []}"#;
const P0009: &str = r#"{"game": "big2", "seed": 5001, "start": {"hands": [["7d", "8d", "8c", "Td"], ["3s", "5d", "7s", "Kd"], ["5s", "Ts", "Qs", "Ah"], ["5h", "8h", "9s", "2h"]], "current_player": 0, "current_play": null}, "history": []}"#;

fn play(cards: &[&str]) -> Value {
    serde_json::json!({"type": "action.play", "cards": cards})
}

fn pass() -> Value {
    serde_json::json!({"type": "action.pass"})
}

/// Examples 2-8 and 10: (example number, position, played, best).
fn coach_examples() -> Vec<(u32, &'static str, Value, Value)> {
    vec![
        (2, P0001, play(&["Kd"]), pass()),
        (3, P0001, play(&["Kh"]), pass()),
        (4, P0004, play(&["3d", "3s"]), play(&["Kd", "Ks"])),
        (5, P0004, play(&["Kd"]), play(&["Kd", "Ks"])),
        (6, P0006, play(&["Jd"]), play(&["Jd", "Js"])),
        (7, P0006, play(&["8h"]), play(&["Jd", "Js"])),
        (8, P0009, play(&["7d"]), play(&["8d", "8c"])),
        (10, P0002, play(&["Kc"]), pass()),
    ]
}

#[test]
fn each_coach_example_gets_a_cause() {
    for (n, pos, played, best) in coach_examples() {
        let position: Value = serde_json::from_str(pos).unwrap();
        let (causes, merits) = causes_for(&position, &played, &best);
        println!(
            "example {n}: causes {:?} merits {:?}",
            keys(&causes),
            keys(&merits)
        );
        assert!(
            !causes.is_empty(),
            "coach example {n} has no cause (played {played}, best {best})"
        );
    }
}

#[test]
fn breaking_the_pair_of_kings_is_the_cause() {
    let position: Value = serde_json::from_str(P0001).unwrap();
    let (causes, _) = causes_for(&position, &play(&["Kd"]), &pass());
    let b = causes
        .iter()
        .find(|r| r.key == "reason.breaks_group")
        .expect("breaks_group");
    assert_eq!(b.args["group"]["cards"], serde_json::json!(["Kd", "Kh"]));
    assert_eq!(b.args["group"]["combo"], "combo.pair");
    assert_eq!(b.args["extra_plays"], 1);
}

#[test]
fn a_beatable_lead_against_a_sure_out_is_the_cause() {
    // 8h Jd Js, leading: Jd Js cannot be beaten, then 8h goes out.
    let position: Value = serde_json::from_str(P0006).unwrap();
    let (causes, merits) = causes_for(&position, &play(&["8h"]), &play(&["Jd", "Js"]));
    let k = keys(&causes);
    assert!(k.contains(&"reason.can_be_beaten"), "{k:?}");
    assert!(k.contains(&"reason.misses_sure_out"), "{k:?}");
    assert!(keys(&merits).contains(&"reason.sure_out"));
    let sure = merits.iter().find(|r| r.key == "reason.sure_out").unwrap();
    assert_eq!(sure.args["plays"], 2);
}

#[test]
fn the_same_move_has_no_cause() {
    let position: Value = serde_json::from_str(P0004).unwrap();
    let kk = play(&["Kd", "Ks"]);
    let (causes, merits) = causes_for(&position, &kk, &kk);
    assert!(causes.is_empty() && merits.is_empty());
}

/// Swapping the moves never repeats a cause: a cause of `played` against `best`
/// is never also a cause of `best` against `played` with the same arguments.
#[test]
fn causes_are_never_shared_by_both_moves() {
    for (n, pos, played, best) in coach_examples() {
        let position: Value = serde_json::from_str(pos).unwrap();
        let (fwd, _) = causes_for(&position, &played, &best);
        let (back, _) = causes_for(&position, &best, &played);
        for k in keys(&fwd) {
            assert!(!keys(&back).contains(&k), "example {n}: {k} on both sides");
        }
    }
}

/// Played/best pairs from the stored solver-graded drills that lost points,
/// in file order.
fn graded_decisions() -> Vec<(String, Value, Value, Value, String)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../drills/big2.json");
    let text = std::fs::read_to_string(&path).expect("drills/big2.json");
    let drills: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut out = Vec::new();
    for d in &drills {
        let answers = d["answers"].as_array().unwrap();
        let Some(best) = answers.iter().find(|a| a["grade"] == "best") else {
            continue;
        };
        for a in answers {
            let g = a["grade"].as_str().unwrap();
            if matches!(g, "inaccuracy" | "mistake" | "blunder") {
                out.push((
                    d["id"].as_str().unwrap().to_string(),
                    d["position"].clone(),
                    a["action"].clone(),
                    best["action"].clone(),
                    g.to_string(),
                ));
            }
        }
    }
    out
}

#[test]
fn most_graded_losses_get_a_cause() {
    let all = graded_decisions();
    assert!(all.len() >= DECISIONS, "only {} graded losses", all.len());
    let mut covered = 0;
    let mut before = 0;
    let mut by_key: BTreeMap<&str, usize> = BTreeMap::new();
    let mut first: BTreeMap<&str, usize> = BTreeMap::new();
    let mut missed = Vec::new();
    let mut slowest = Duration::ZERO;
    for (id, pos, played, best, grade) in all.iter().take(DECISIONS) {
        let state = state_from_position(&pos.to_string()).unwrap();
        let p = action(&state, played);
        let b = action(&state, best);
        // Before: no existing key states a cost of the move (they are merits
        // or neutral facts), so none counts as a cause. For context, count the
        // decisions where some existing key other than `cards_left` at least
        // differs between the two moves.
        let legal = state.legal_actions_vec();
        let old_p = keys(&position_reasons(&state, p, &legal));
        let old_b = keys(&position_reasons(&state, b, &legal));
        if old_p
            .iter()
            .any(|k| *k != "reason.cards_left" && !old_b.contains(k))
        {
            before += 1;
        }
        let t = Instant::now();
        let (causes, _) = contrast_reasons(&state, p, b);
        slowest = slowest.max(t.elapsed());
        let (back, _) = contrast_reasons(&state, b, p);
        for k in keys(&causes) {
            assert!(!keys(&back).contains(&k), "{id}: {k} true of both moves");
        }
        for c in &causes {
            assert!(CAUSE_KEYS.contains(&c.key), "unknown cause {}", c.key);
            *by_key.entry(c.key).or_default() += 1;
        }
        if let Some(c) = causes.first() {
            *first.entry(c.key).or_default() += 1;
            covered += 1;
        } else {
            missed.push(format!("{id} {grade}: played {played} best {best}"));
        }
    }
    let share = covered as f64 / DECISIONS as f64;
    println!(
        "graded losses: {DECISIONS} (of {} in drills/big2.json)",
        all.len()
    );
    println!("with a cause before: 0 (0.0 %); an existing key differed in {before}");
    println!("with a cause now:    {covered} ({:.1} %)", 100.0 * share);
    println!("key | decisions carrying it | first (headline)");
    for (k, n) in &by_key {
        println!("{k} | {n} | {}", first.get(k).copied().unwrap_or(0));
    }
    println!("slowest contrast_reasons call: {slowest:?}");
    for m in missed.iter().take(20) {
        println!("no cause: {m}");
    }
    assert!(share >= 0.8, "only {:.1} % carry a cause", 100.0 * share);
}
