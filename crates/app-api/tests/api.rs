//! End-to-end tests of the JSON calls, as the app will use them.

use big2_app_api::Api;
use serde_json::{json, Value};

fn parse(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|e| panic!("bad JSON {e}: {s}"))
}

fn ok(s: &str) -> Value {
    let v = parse(s);
    assert!(v.get("error").is_none(), "unexpected error: {v}");
    assert_eq!(v["api"], 0, "every response carries api 0: {v}");
    v
}

fn error_code(s: &str) -> String {
    let v = parse(s);
    assert_eq!(v["api"], 0);
    v["error"]["code"]
        .as_str()
        .unwrap_or_else(|| panic!("expected an error, got {v}"))
        .to_string()
}

fn new_game(api: &mut Api, seed: u64) -> String {
    let v = ok(&api.new_game(&json!({ "seats": 4, "seed": seed, "human_seats": [0] }).to_string()));
    v["game_id"].as_str().unwrap().to_string()
}

/// Plays the deal to the end with `level` bots in every seat; returns the
/// number of actions and the last event list.
fn play_out(api: &mut Api, id: &str, level: &str) -> (usize, Vec<Value>) {
    let mut n = 0;
    let mut last = Vec::new();
    loop {
        let view = ok(&api.view(id, 0));
        if view["over"].as_bool().unwrap() {
            return (n, last);
        }
        let action = ok(&api.bot_action(id, level, 200));
        let r = ok(&api.apply(id, &action.to_string()));
        last = r["events"].as_array().unwrap().clone();
        n += 1;
        assert!(n < 400, "the deal never ends");
    }
}

#[test]
fn meta_lists_variants_and_levels() {
    let api = Api::new();
    let m = ok(&api.meta());
    assert_eq!(m["game"], "big2");
    assert_eq!(m["seats"], json!([4, 4]));
    assert_eq!(m["bot_levels"][0], "random");
    assert_eq!(m["variants"]["straights"][0], "with_low_twos");
    assert_eq!(m["variants"]["last_card_duty"], json!(["on", "off"]));
    assert_eq!(m["has_hint"], true);
}

#[test]
fn four_random_bots_finish_a_deal() {
    let mut api = Api::new();
    let id = new_game(&mut api, 1);
    let (n, events) = play_out(&mut api, &id, "random");
    assert!(n >= 13, "at least 13 plays are needed to go out");
    let over = events.last().unwrap();
    assert_eq!(over["type"], "event.deal_over");
    let scores: Vec<f64> = over["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_f64().unwrap())
        .collect();
    assert!(
        (scores.iter().sum::<f64>()).abs() < 1e-9,
        "zero-sum: {scores:?}"
    );
    let view = ok(&api.view(&id, 2));
    assert_eq!(view["over"], true);
    assert_eq!(view["final_hands"].as_array().unwrap().len(), 4);
    assert_eq!(view["initial_hands"][0].as_array().unwrap().len(), 13);
    assert_eq!(
        error_code(&api.bot_action(&id, "random", 10)),
        "error.game_over"
    );
}

#[test]
fn four_greedy_bots_finish_a_deal() {
    let mut api = Api::new();
    let id = new_game(&mut api, 2);
    let (_, events) = play_out(&mut api, &id, "easy");
    assert_eq!(events.last().unwrap()["type"], "event.deal_over");
}

#[test]
fn search_bots_answer_within_budget() {
    let mut api = Api::new();
    let id = new_game(&mut api, 3);
    for level in ["medium", "hard", "expert"] {
        let a = ok(&api.bot_action(&id, level, 150));
        assert!(a["type"].as_str().unwrap().starts_with("action."));
        assert!(
            a["budget_used_ms"].as_u64().unwrap() < 180 + 500,
            "{level}: {a}"
        );
    }
    assert_eq!(
        error_code(&api.bot_action(&id, "grandmaster", 10)),
        "error.unknown_level"
    );
}

#[test]
fn opening_view_hides_other_hands_and_lists_legal_moves() {
    let mut api = Api::new();
    let id = new_game(&mut api, 4);
    let view = ok(&api.view(&id, 0));
    assert_eq!(view["hand"].as_array().unwrap().len(), 13);
    assert_eq!(view["hand_sizes"], json!([13, 13, 13, 13]));
    assert!(view["final_hands"].is_null());
    assert_eq!(view["is_opening"], true);
    let opener = view["current_player"].as_u64().unwrap() as usize;
    let legal = ok(&api.legal_actions(&id));
    let actions = legal["actions"].as_array().unwrap();
    assert!(!actions.is_empty());
    assert!(actions
        .iter()
        .all(|a| a["cards"].as_array().unwrap().iter().any(|c| c == "3d")));
    assert!(actions
        .iter()
        .all(|a| a["combo"].as_str().unwrap().starts_with("combo.")));
    let opener_view = ok(&api.view(&id, opener));
    assert_eq!(opener_view["your_turn"], true);
    assert_eq!(opener_view["legal"], legal["actions"]);
    assert_eq!(ok(&api.view(&id, (opener + 1) % 4))["legal"], json!([]));
    assert_eq!(error_code(&api.view(&id, 4)), "error.bad_seat");
    assert_eq!(error_code(&api.view("nope", 0)), "error.unknown_game");
}

#[test]
fn illegal_actions_are_rejected_with_a_code() {
    let mut api = Api::new();
    let id = new_game(&mut api, 5);
    // Passing while leading the opening trick.
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.pass"}"#)),
        "error.illegal_action"
    );
    // A card the opener does not hold.
    let view = ok(&api.view(&id, 0));
    let opener = view["current_player"].as_u64().unwrap() as usize;
    let other = ok(&api.view(&id, (opener + 1) % 4));
    let foreign = other["hand"][0].as_str().unwrap();
    let a = json!({ "type": "action.play", "cards": [foreign] }).to_string();
    assert_eq!(error_code(&api.apply(&id, &a)), "error.illegal_action");
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.play","cards":["3x"]}"#)),
        "error.bad_card"
    );
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.fold"}"#)),
        "error.bad_action"
    );
    assert_eq!(error_code(&api.apply(&id, "not json")), "error.bad_json");
    // The deal has not moved.
    assert_eq!(ok(&api.view(&id, 0))["history"], json!([]));
}

#[test]
fn apply_reports_events() {
    let mut api = Api::new();
    let id = new_game(&mut api, 6);
    let legal = ok(&api.legal_actions(&id));
    let first = &legal["actions"][0];
    let r = ok(&api.apply(&id, &first.to_string()));
    let events = r["events"].as_array().unwrap();
    assert_eq!(events[0]["type"], "event.play");
    assert_eq!(
        events[0]["cards_left"],
        13 - first["cards"].as_array().unwrap().len() as u64
    );
    assert_eq!(r["view"]["history"].as_array().unwrap().len(), 1);
    // Three passes end the trick: the leader wins it.
    let leader = events[0]["player"].as_u64().unwrap();
    for i in 0..3 {
        let r = ok(&api.apply(&id, r#"{"type":"action.pass"}"#));
        let ev = r["events"].as_array().unwrap();
        assert_eq!(ev[0]["type"], "event.pass");
        if i == 2 {
            assert_eq!(ev[1]["type"], "event.trick_won");
            assert_eq!(ev[1]["player"], leader);
        } else {
            assert_eq!(ev.len(), 1);
        }
    }
    let view = ok(&api.view(&id, 0));
    assert_eq!(view["trick_starts"], json!([0, 4]));
    assert_eq!(view["trick"], json!([]));
    assert!(view["current_play"].is_null());
}

#[test]
fn snapshot_and_restore_give_identical_views() {
    let mut api = Api::new();
    let id = new_game(&mut api, 7);
    for _ in 0..9 {
        let a = ok(&api.bot_action(&id, "easy", 50));
        ok(&api.apply(&id, &a.to_string()));
    }
    let snap = ok(&api.snapshot(&id));
    assert_eq!(snap["history"].as_array().unwrap().len(), 9);
    assert_eq!(snap["initial_hands"].as_array().unwrap().len(), 4);
    let restored = ok(&api.restore(&snap.to_string()));
    let id2 = restored["game_id"].as_str().unwrap();
    assert_ne!(id2, id);
    let strip = |mut v: Value| {
        v.as_object_mut().unwrap().remove("game_id");
        v
    };
    for seat in 0..4 {
        assert_eq!(
            strip(ok(&api.view(&id, seat))),
            strip(ok(&api.view(id2, seat)))
        );
    }
    assert_eq!(strip(ok(&api.snapshot(&id))), strip(ok(&api.snapshot(id2))));
    // The two games now continue identically: seeded bots draw the same moves.
    let a1 = ok(&api.bot_action(&id, "random", 10));
    let a2 = ok(&api.bot_action(id2, "random", 10));
    assert_eq!(a1, a2);
    // Drop and the id is gone.
    ok(&api.drop_game(&id));
    assert_eq!(error_code(&api.view(&id, 0)), "error.unknown_game");
    assert_eq!(api.len(), 1);
}

#[test]
fn restore_rejects_bad_positions() {
    let mut api = Api::new();
    assert_eq!(
        error_code(&api.restore(r#"{"seed": 1}"#)),
        "error.bad_position"
    );
    let bad_game = json!({ "game": "mahjong", "seed": 1, "initial_hands": [[], [], [], []] });
    assert_eq!(
        error_code(&api.restore(&bad_game.to_string())),
        "error.bad_position"
    );
    let id = new_game(&mut api, 8);
    let mut snap = ok(&api.snapshot(&id));
    snap["history"] = json!([{ "player": 9, "action": { "type": "action.pass" } }]);
    assert_eq!(
        error_code(&api.restore(&snap.to_string())),
        "error.bad_position"
    );
}

#[test]
fn mid_game_positions_restore_and_play() {
    let mut api = Api::new();
    // Seat 0 leads with two cards left against three opponents; no deal behind it.
    let position = json!({
        "game": "big2", "seed": 0,
        "start": {
            "hands": [["3d", "3c"], ["4d", "5d", "6d"], ["7d", "8d"], ["9d", "Td"]],
            "current_player": 0,
            "current_play": null
        },
        "history": []
    });
    let id = ok(&api.restore(&position.to_string()))["game_id"]
        .as_str()
        .unwrap()
        .to_string();
    let view = ok(&api.view(&id, 0));
    assert_eq!(view["hand_sizes"], json!([2, 3, 2, 2]));
    assert_eq!(view["is_opening"], false);
    let r = ok(&api.apply(&id, r#"{"type":"action.play","cards":["3d","3c"]}"#));
    assert_eq!(r["events"][0]["action"]["combo"], "combo.pair");
    assert_eq!(r["events"][1]["type"], "event.deal_over");
    let snap = ok(&api.snapshot(&id));
    assert!(snap["initial_hands"].is_null());
    assert_eq!(snap["start"]["hands"][0], json!(["3d", "3c"]));
    let again = ok(&api.restore(&snap.to_string()));
    let v2 = ok(&api.view(again["game_id"].as_str().unwrap(), 0));
    assert_eq!(v2["over"], true);
    assert_eq!(v2["winner"], 0);
}

#[test]
fn hint_gives_moves_with_numbers_and_reasons() {
    let mut api = Api::new();
    let id = new_game(&mut api, 9);
    let view = ok(&api.view(&id, 0));
    let opener = view["current_player"].as_u64().unwrap() as usize;
    assert_eq!(
        error_code(&api.hint(&id, (opener + 1) % 4, 100)),
        "error.not_your_turn"
    );
    let h = ok(&api.hint(&id, opener, 300));
    let moves = h["moves"].as_array().unwrap();
    assert!(!moves.is_empty() && moves.len() <= 3);
    for m in moves {
        assert!(m["action"]["type"].as_str().unwrap().starts_with("action."));
        assert!(m["score"].is_number());
        assert!(m["win_rate"].is_number());
        assert!(m["numbers"]["cards_left_after"].is_number());
        let reasons = m["reasons"].as_array().unwrap();
        assert!(!reasons.is_empty(), "every hint move has a reason: {m}");
        assert!(reasons
            .iter()
            .all(|r| r["key"].as_str().unwrap().starts_with("reason.")));
    }
    assert!(h["budget_used_ms"].as_u64().unwrap() < 360 + 500, "{h}");
    assert!(h["simulations"].as_u64().unwrap() > 0);
}

#[test]
fn review_has_one_item_per_decision_of_the_seat() {
    let mut api = Api::new();
    let id = new_game(&mut api, 10);
    play_out(&mut api, &id, "easy");
    let view = ok(&api.view(&id, 0));
    let history = view["history"].as_array().unwrap();
    let seat = 1;
    let mine = history.iter().filter(|t| t["player"] == seat).count();
    assert!(mine > 0);
    let r = ok(&api.review(&id, seat, 1500));
    let items = r["items"].as_array().unwrap();
    assert_eq!(items.len(), mine);
    let grades = ["best", "good", "inaccuracy", "mistake", "blunder"];
    for item in items {
        let turn = item["turn"].as_u64().unwrap() as usize;
        assert_eq!(history[turn]["player"], seat);
        assert_eq!(item["played"], history[turn]["action"]);
        assert!(grades.contains(&item["grade"].as_str().unwrap()), "{item}");
        assert!(item["loss"].as_f64().unwrap() >= 0.0);
        assert!(!item["reasons"].as_array().unwrap().is_empty());
        if item["grade"] == "best"
            && item["loss"] == 0.0
            && item["numbers"].as_object().unwrap().is_empty()
        {
            assert_eq!(item["reasons"][0]["key"], "reason.forced");
        }
    }
    // A review before the deal ends is allowed too, and covers the decisions so far.
    let id2 = new_game(&mut api, 11);
    let a = ok(&api.bot_action(&id2, "easy", 10));
    let r = ok(&api.apply(&id2, &a.to_string()));
    let actor = r["events"][0]["player"].as_u64().unwrap() as usize;
    let early = ok(&api.review(&id2, actor, 300));
    assert_eq!(early["items"].as_array().unwrap().len(), 1);
    assert_eq!(
        ok(&api.review(&id2, (actor + 1) % 4, 10))["items"],
        json!([])
    );
}

#[test]
fn grade_scores_an_action_against_the_best() {
    let mut api = Api::new();
    let id = new_game(&mut api, 12);
    let snap = ok(&api.snapshot(&id)).to_string();
    let legal = ok(&api.legal_actions(&id));
    let actions = legal["actions"].as_array().unwrap();
    // The weakest legal opening play (a single 3d) is often not the best.
    let g = ok(&api.grade(&snap, &actions[0].to_string(), 300));
    assert!(["best", "good", "inaccuracy", "mistake", "blunder"]
        .contains(&g["grade"].as_str().unwrap()));
    assert!(g["best"]["type"].as_str().unwrap().starts_with("action."));
    assert!(!g["reasons"].as_array().unwrap().is_empty());
    assert_eq!(
        error_code(&api.grade(&snap, r#"{"type":"action.pass"}"#, 50)),
        "error.illegal_action"
    );
    // Grading a position does not register a game.
    assert_eq!(api.len(), 1);
}

#[test]
fn variants_round_trip_through_config_and_snapshot() {
    let mut api = Api::new();
    let cfg = json!({
        "seats": 4, "seed": 13,
        "variants": { "straights": "no_twos", "pass_rule": "locked_out", "scoring": "win_loss", "last_card_duty": "off" }
    });
    let id = ok(&api.new_game(&cfg.to_string()))["game_id"]
        .as_str()
        .unwrap()
        .to_string();
    let view = ok(&api.view(&id, 0));
    assert_eq!(view["rules"]["straights"], "no_twos");
    assert_eq!(view["rules"]["pass_rule"], "locked_out");
    assert_eq!(view["rules"]["last_card_duty"], "off");
    assert_eq!(view["rules"]["flush_order"], "rank_first");
    let bad = json!({ "seats": 4, "seed": 1, "variants": { "straights": "wheel" } });
    assert_eq!(
        error_code(&api.new_game(&bad.to_string())),
        "error.unknown_variant"
    );
    let bad = json!({ "seats": 4, "seed": 1, "variants": { "jokers": "two" } });
    assert_eq!(
        error_code(&api.new_game(&bad.to_string())),
        "error.unknown_variant"
    );
    assert_eq!(
        error_code(&api.new_game(r#"{"seats": 3, "seed": 1}"#)),
        "error.bad_config"
    );
}

/// Timings for the README: `cargo test -p big2-app-api --release -- --ignored --nocapture timing`.
#[test]
#[ignore]
fn timing_of_hint_at_5000_and_20000_iterations() {
    let mut api = Api::new();
    let id = new_game(&mut api, 100);
    let opener = ok(&api.view(&id, 0))["current_player"].as_u64().unwrap() as usize;
    for iterations in [5_000u64, 20_000] {
        let h = ok(&api.hint_with_iterations(&id, opener, 0, iterations));
        println!(
            "hint {iterations} iterations: {} ms ({} simulations)",
            h["budget_used_ms"], h["simulations"]
        );
    }
    let a = ok(&api.bot_action(&id, "expert", 0));
    println!("expert bot (10,000 iterations): {} ms", a["budget_used_ms"]);
    play_out(&mut api, &id, "easy");
    let r = ok(&api.review(&id, 0, 0));
    println!(
        "review of seat 0, {} items, 300 deals each: {} ms",
        r["items"].as_array().unwrap().len(),
        r["budget_used_ms"]
    );
}
