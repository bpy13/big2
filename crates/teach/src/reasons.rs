//! Reason keys from the move table (strategy proposal §2.5, hard-problems §8).
//!
//! [`reasons_for`] explains one action of the player to act. Each key is either a
//! fact about the position (the hand's plan, the unseen cards) or a difference
//! between two rows of the [`MoveTable`] (the chosen action and the best other
//! one); nothing is guessed, so every key is true by construction. Keys and
//! arguments only, never display text (hk-games D42): the app's dictionaries
//! turn `reason.plan.breaks {combo, cost}` into words.
//!
//! These keys are additions to the app's existing ones (`reason.keeps_control`
//! and the others in `crates/app-api/src/reasons.rs`), which stay as they are.
//!
//! Notation in the rules below: *chosen* is the action being explained, *other*
//! is the best row of the table that is not *chosen* ([`MoveTable::best_other`]),
//! *unseen* is every card neither played nor in the actor's hand. Card arguments
//! are card ids (`"9s"`); combination arguments are
//! `{"cards": ["9d", "9s"], "combo": "combo.pair"}`. Points are rounded to 0.01,
//! shares to 0.001. Seats are absolute seat indices.
//!
//! | Key | Arguments | Fires when |
//! |---|---|---|
//! | `reason.plan.keeps` | `combo` | chosen keeps a fewest-plays plan, and a lower-rated play breaks a group of the best plan that chosen leaves whole; `combo` is that group, for the best-rated such play |
//! | `reason.plan.breaks` | `combo`, `cost` | chosen breaks a group of the best plan; `cost` = extra plays to go out it causes (≥ 1) |
//! | `reason.control.pass_to_keep` | `card` | chosen is a pass rated above every play; `card` = the highest card of the best-rated play |
//! | `reason.value.score_vs_win` | `score_delta`, `win_delta` | another action has a higher win rate but a lower mean score than chosen |
//! | `reason.tempo.plays_to_out` | `mine`, `theirs` | chosen leaves the fewest plays to go out of all legal plays, and some play leaves more; `theirs` = expected plays per opponent, seat order after the actor |
//! | `reason.control.safe_single` | `card` | chosen is a single, cards stay in hand, and no unseen card beats it |
//! | `reason.control.highest_out` | `card` | the actor holds the highest card not yet played, chosen keeps it, and some legal play spends it |
//! | `reason.value.gap` | `points` | two or more legal actions; `points` = chosen − other (positive: margin over the next best; negative: loss to the best) |
//! | `reason.counterfactual.outcomes` | `best`, `played`, `deals` | chosen is not the best row; `best` and `played` are `{"mean", "win"}` of the two rows on the same `deals` |
//! | `reason.helps_seat` | `seat`, `delta` | per-seat values exist and chosen gives an opponent at least [`HELPS_SEAT_MIN`] points more than other does; the opponent who gains most. In a paired table the gain's 95 % interval must also exclude 0 |
//! | `reason.counterfactual.forced` | `result`, `winner`, `plays` | the exact endgame proved the best row's winner in every deal and chosen is the best row; `result` is `"win"` or `"lose"` |

use crate::table::MoveTable;
use big2_big2::card::{card_at, FULL_DECK};
use big2_big2::plan::{candidates, decompositions, min_plays, Candidate};
use big2_big2::{Action, Combo, State, NUM_PLAYERS};
use big2_cards::mask::highest;
use serde::Serialize;
use serde_json::{json, Map, Value};

/// Smallest gain (points) to an opponent that `reason.helps_seat` reports.
pub const HELPS_SEAT_MIN: f64 = 1.0;

/// One reason: a key and its arguments, e.g.
/// `{"key": "reason.plan.breaks", "args": {"combo": {...}, "cost": 1}}`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reason {
    pub key: &'static str,
    pub args: Value,
}

impl Reason {
    fn new(key: &'static str) -> Reason {
        Reason {
            key,
            args: Value::Object(Map::new()),
        }
    }

    fn arg(mut self, name: &str, value: impl Into<Value>) -> Reason {
        if let Value::Object(m) = &mut self.args {
            m.insert(name.to_string(), value.into());
        }
        self
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn card_id(index: u8) -> String {
    card_at(index).to_string()
}

fn cards_json(mask: u64) -> Value {
    Value::from(
        big2_cards::mask::bits(mask)
            .map(card_id)
            .collect::<Vec<String>>(),
    )
}

fn combo_json(c: &Combo) -> Value {
    json!({ "cards": cards_json(c.cards), "combo": format!("combo.{}", c.kind.name()) })
}

/// [`reasons_for`] with the candidates computed here (every fewest-plays plan of
/// the actor's hand).
pub fn reasons(state: &State, table: &MoveTable, chosen: Action) -> Vec<Reason> {
    let hand = state.hand(table.actor);
    let decomps = decompositions(hand, state.rules(), usize::MAX);
    let cands = candidates(state, &decomps);
    reasons_for(state, table, &cands, chosen)
}

/// The reasons for `chosen`, an action of the player to act in `state`.
///
/// `table` values every legal action of that player; `candidates` labels them
/// against the hand's plans ([`big2_big2::plan::candidates`]). The rules are in
/// the module docs. Returns nothing when `chosen` is not in the table.
pub fn reasons_for(
    state: &State,
    table: &MoveTable,
    candidates: &[Candidate],
    chosen: Action,
) -> Vec<Reason> {
    let mut out = Vec::new();
    let Some(c) = table.row(chosen) else {
        return out;
    };
    let me = table.actor;
    let hand = state.hand(me);
    let rules = state.rules();
    let cand_of = |a: Action| candidates.iter().find(|x| x.action == a);
    let other = table.best_other(chosen);

    // reason.plan.keeps / reason.plan.breaks
    if let Some(cc) = cand_of(chosen) {
        if cc.keeps.is_some() {
            // Only a group that chosen leaves whole: "keeps" must be literally true.
            let broken = table.rows.iter().filter(|r| r.mean < c.mean).find_map(|r| {
                cand_of(r.action)
                    .and_then(|x| x.breaks)
                    .filter(|g| g.cards & chosen.cards() == 0)
            });
            if let Some(g) = broken {
                out.push(Reason::new("reason.plan.keeps").arg("combo", combo_json(&g)));
            }
        }
        if let Some(g) = cc.breaks {
            let cost = (1 + min_plays(hand & !chosen.cards(), rules))
                .saturating_sub(min_plays(hand, rules));
            out.push(
                Reason::new("reason.plan.breaks")
                    .arg("combo", combo_json(&g))
                    .arg("cost", cost),
            );
        }
    }

    // reason.control.pass_to_keep
    if chosen == Action::Pass {
        if let Some(best_play) = table.rows.iter().find(|r| r.action != Action::Pass) {
            if c.mean > best_play.mean {
                if let Some(top) = highest(best_play.action.cards()) {
                    out.push(Reason::new("reason.control.pass_to_keep").arg("card", card_id(top)));
                }
            }
        }
    }

    // reason.value.score_vs_win
    let top_win = table
        .rows
        .iter()
        .fold(None::<&crate::MoveRow>, |acc, r| match acc {
            Some(a) if a.win >= r.win => Some(a),
            _ => Some(r),
        });
    if let Some(w) = top_win {
        if w.action != chosen && c.mean > w.mean && w.win > c.win {
            out.push(
                Reason::new("reason.value.score_vs_win")
                    .arg("score_delta", round2(c.mean - w.mean))
                    .arg("win_delta", round3(c.win - w.win)),
            );
        }
    }

    // reason.tempo.plays_to_out
    if chosen != Action::Pass {
        let after = |a: Action| min_plays(hand & !a.cards(), rules);
        let plays: Vec<u32> = table
            .rows
            .iter()
            .filter(|r| r.action != Action::Pass)
            .map(|r| after(r.action))
            .collect();
        let mine = after(chosen);
        if plays.iter().all(|&p| p >= mine) && plays.iter().any(|&p| p > mine) {
            let theirs: Vec<f64> = (1..NUM_PLAYERS)
                .map(|k| (table.plays_to_out[(me + k) % NUM_PLAYERS] * 10.0).round() / 10.0)
                .collect();
            out.push(
                Reason::new("reason.tempo.plays_to_out")
                    .arg("mine", mine)
                    .arg("theirs", theirs),
            );
        }
    }

    // reason.control.safe_single
    let unseen = FULL_DECK & !state.played() & !hand;
    if let Action::Play(combo) = chosen {
        let left = hand & !combo.cards;
        if combo.size() == 1 && left != 0 {
            let card = combo.cards.trailing_zeros() as u8;
            if highest(unseen).is_none_or(|u| u < card) {
                out.push(Reason::new("reason.control.safe_single").arg("card", card_id(card)));
            }
        }
    }

    // reason.control.highest_out
    if let Some(top) = highest(FULL_DECK & !state.played()) {
        let bit = 1u64 << top;
        if hand & bit != 0
            && chosen.cards() & bit == 0
            && table.rows.iter().any(|r| r.action.cards() & bit != 0)
        {
            out.push(Reason::new("reason.control.highest_out").arg("card", card_id(top)));
        }
    }

    // reason.value.gap
    if let Some(o) = other {
        out.push(Reason::new("reason.value.gap").arg("points", round2(c.mean - o.mean)));
    }

    // reason.counterfactual.outcomes
    let best = table.best();
    if best.action != chosen {
        out.push(
            Reason::new("reason.counterfactual.outcomes")
                .arg(
                    "best",
                    json!({ "mean": round2(best.mean), "win": round3(best.win) }),
                )
                .arg(
                    "played",
                    json!({ "mean": round2(c.mean), "win": round3(c.win) }),
                )
                .arg("deals", table.deals),
        );
    }

    // reason.helps_seat
    if let (Some(o), Some(cs)) = (other, c.seat_means) {
        if let Some(os) = o.seat_means {
            let gain = (1..NUM_PLAYERS)
                .map(|k| (me + k) % NUM_PLAYERS)
                .map(|s| (s, cs[s] - os[s]))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((seat, delta)) = gain {
                // Paired table: the gain must stand out of the sampling noise.
                let clear = match (
                    &table.per_deal,
                    table.index_of(chosen),
                    table.index_of(o.action),
                ) {
                    (Some(pd), Some(i), Some(j)) => {
                        let diffs: Vec<f64> = pd[i]
                            .iter()
                            .zip(&pd[j])
                            .map(|(a, b)| a[seat] - b[seat])
                            .collect();
                        delta - crate::grade::ci95(&diffs) > 0.0
                    }
                    _ => true,
                };
                if delta >= HELPS_SEAT_MIN && clear {
                    out.push(
                        Reason::new("reason.helps_seat")
                            .arg("seat", seat)
                            .arg("delta", round2(delta)),
                    );
                }
            }
        }
    }

    // reason.counterfactual.forced
    if let Some(f) = table.forced {
        if best.action == chosen {
            let result = if f.winner == me { "win" } else { "lose" };
            out.push(
                Reason::new("reason.counterfactual.forced")
                    .arg("result", result)
                    .arg("winner", f.winner)
                    .arg("plays", f.plays),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::MoveRow;
    use big2_big2::card::parse_mask;
    use big2_big2::RuleSet;
    use big2_games::Game;
    use big2_search::Forced;

    fn h(s: &str) -> u64 {
        parse_mask(s).unwrap()
    }

    fn state(hands: [&str; 4], current: usize, play: Option<(usize, &str)>) -> State {
        State::from_position(
            RuleSet::default(),
            hands.map(h),
            current,
            play.map(|(p, c)| (p, h(c))),
        )
        .unwrap()
    }

    /// The legal action that plays exactly `cards` ("" for the pass).
    fn act(s: &State, cards: &str) -> Action {
        let m = if cards.is_empty() { 0 } else { h(cards) };
        s.legal_actions_vec()
            .into_iter()
            .find(|a| a.cards() == m)
            .unwrap_or_else(|| panic!("{cards} is not legal"))
    }

    /// A table over every legal action: the listed ones get the given mean and
    /// win rate, the rest −10 and 0, sorted best first.
    fn table(s: &State, vals: &[(&str, f64, f64)]) -> MoveTable {
        let actor = s.current_player().unwrap();
        let mut rows: Vec<MoveRow> = s
            .legal_actions_vec()
            .into_iter()
            .map(|a| {
                let (mean, win) = vals
                    .iter()
                    .find(|(c, _, _)| act(s, c) == a)
                    .map_or((-10.0, 0.0), |&(_, m, w)| (m, w));
                MoveRow {
                    action: a,
                    mean,
                    ci95: 0.5,
                    win,
                    seat_means: None,
                }
            })
            .collect();
        rows.sort_by(|a, b| b.mean.total_cmp(&a.mean));
        MoveTable {
            actor,
            rows,
            deals: 300,
            plays_to_out: [2.0, 3.0, 4.0, 5.0],
            forced: None,
            per_deal: None,
        }
    }

    fn keys(r: &[Reason]) -> Vec<&'static str> {
        r.iter().map(|x| x.key).collect()
    }

    fn find<'a>(r: &'a [Reason], key: &str) -> Option<&'a Reason> {
        r.iter().find(|x| x.key == key)
    }

    // Seat 0 leads 3c 9d 9s Kd: the best plan is the pair of 9s, the K and the 3.
    fn leading() -> State {
        state(["3c 9d 9s Kd", "4d 4c 4h", "5d 5c 5h", "6d 6c 6h"], 0, None)
    }

    #[test]
    fn plan_keeps_fires_when_a_lower_rated_play_breaks_a_group() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("9s", 1.0, 0.2)]);
        let r = reasons(&s, &t, act(&s, "Kd"));
        let k = find(&r, "reason.plan.keeps").expect("keeps fires");
        assert_eq!(k.args["combo"]["cards"], json!(["9d", "9s"]));
        assert_eq!(k.args["combo"]["combo"], "combo.pair");
    }

    #[test]
    fn plan_keeps_does_not_fire_for_a_breaking_play_or_without_a_lower_breaker() {
        let s = leading();
        // The breaking play itself.
        let t = table(&s, &[("Kd", 2.0, 0.3), ("9s", 1.0, 0.2)]);
        assert!(find(&reasons(&s, &t, act(&s, "9s")), "reason.plan.keeps").is_none());
        // Every breaking play rated above the chosen keeper.
        let t = table(&s, &[("9s", 3.0, 0.3), ("9d", 2.5, 0.3), ("Kd", 2.0, 0.3)]);
        assert!(find(&reasons(&s, &t, act(&s, "Kd")), "reason.plan.keeps").is_none());
        // The lower-rated play breaks the very group chosen plays: not "kept".
        let s = state(["9d 9s 9h Kd", "4d 4c 4h", "5d 5c 5h", "6d 6c 6h"], 0, None);
        let t = table(&s, &[("9d 9s 9h", 2.0, 0.3), ("9d 9s", 1.0, 0.3)]);
        assert!(find(&reasons(&s, &t, act(&s, "9d 9s 9h")), "reason.plan.keeps").is_none());
    }

    #[test]
    fn plan_breaks_fires_with_its_cost() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("9s", 1.0, 0.2)]);
        let r = reasons(&s, &t, act(&s, "9s"));
        let b = find(&r, "reason.plan.breaks").expect("breaks fires");
        assert_eq!(b.args["combo"]["cards"], json!(["9d", "9s"]));
        // 3 plays before (99, K, 3); after 9s: 9d, K, 3 = 3 more, so one extra play.
        assert_eq!(b.args["cost"], 1);
    }

    #[test]
    fn plan_breaks_does_not_fire_for_a_kept_group() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3)]);
        for c in ["Kd", "3c", "9d 9s"] {
            assert!(
                find(&reasons(&s, &t, act(&s, c)), "reason.plan.breaks").is_none(),
                "{c}"
            );
        }
    }

    // Seat 0 follows a single 8d with 3c 9d 9s Kd 2s.
    fn following() -> State {
        state(
            ["3c 9d 9s Kd 2s", "4d 4c 4h", "5d 5c 5h", "6d 6c 6h"],
            0,
            Some((3, "8d")),
        )
    }

    #[test]
    fn pass_to_keep_fires_when_the_pass_beats_every_play() {
        let s = following();
        let t = table(&s, &[("", 1.0, 0.3), ("2s", 0.5, 0.3), ("Kd", 0.0, 0.2)]);
        let r = reasons(&s, &t, Action::Pass);
        let p = find(&r, "reason.control.pass_to_keep").expect("fires");
        assert_eq!(p.args["card"], "2s");
    }

    #[test]
    fn pass_to_keep_does_not_fire_when_a_play_is_rated_higher() {
        let s = following();
        let t = table(&s, &[("Kd", 1.5, 0.3), ("", 1.0, 0.3)]);
        assert!(find(
            &reasons(&s, &t, Action::Pass),
            "reason.control.pass_to_keep"
        )
        .is_none());
        // Nor for a play.
        let t = table(&s, &[("", 1.0, 0.3), ("2s", 0.5, 0.3)]);
        assert!(find(
            &reasons(&s, &t, act(&s, "2s")),
            "reason.control.pass_to_keep"
        )
        .is_none());
    }

    #[test]
    fn score_vs_win_fires_when_another_move_wins_more_but_scores_less() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.2), ("3c", 1.0, 0.4)]);
        let r = reasons(&s, &t, act(&s, "Kd"));
        let x = find(&r, "reason.value.score_vs_win").expect("fires");
        assert_eq!(x.args["score_delta"], 1.0);
        assert_eq!(x.args["win_delta"], -0.2);
    }

    #[test]
    fn score_vs_win_does_not_fire_for_the_top_win_rate() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.2), ("3c", 1.0, 0.4)]);
        assert!(find(&reasons(&s, &t, act(&s, "3c")), "reason.value.score_vs_win").is_none());
        // Nor when the best scorer also wins most.
        let t = table(&s, &[("Kd", 2.0, 0.5), ("3c", 1.0, 0.4)]);
        assert!(find(&reasons(&s, &t, act(&s, "Kd")), "reason.value.score_vs_win").is_none());
    }

    #[test]
    fn plays_to_out_fires_for_the_fewest_plays_left() {
        let s = leading();
        let t = table(&s, &[("9d 9s", 1.0, 0.3)]);
        let r = reasons(&s, &t, act(&s, "9d 9s"));
        let x = find(&r, "reason.tempo.plays_to_out").expect("fires");
        assert_eq!(x.args["mine"], 2);
        // Opponents in seat order after seat 0: seats 1, 2, 3.
        assert_eq!(x.args["theirs"], json!([3.0, 4.0, 5.0]));
    }

    #[test]
    fn plays_to_out_does_not_fire_for_a_play_that_leaves_more() {
        let s = leading();
        let t = table(&s, &[("9s", 1.0, 0.3)]);
        assert!(find(&reasons(&s, &t, act(&s, "9s")), "reason.tempo.plays_to_out").is_none());
    }

    // Everything above the K is played except the 2s (seat 0) and the Ad.
    fn late() -> State {
        state(["3c Kd 2s", "4d 4c", "5d 5c", "Ad 6d"], 0, None)
    }

    #[test]
    fn safe_single_fires_when_no_unseen_card_beats_it() {
        let s = late();
        let t = table(&s, &[("2s", 1.0, 0.5)]);
        let r = reasons(&s, &t, act(&s, "2s"));
        assert_eq!(
            find(&r, "reason.control.safe_single").unwrap().args["card"],
            "2s"
        );
    }

    #[test]
    fn safe_single_does_not_fire_when_an_unseen_card_beats_it() {
        let s = late();
        let t = table(&s, &[("Kd", 1.0, 0.5)]);
        assert!(find(
            &reasons(&s, &t, act(&s, "Kd")),
            "reason.control.safe_single"
        )
        .is_none());
    }

    #[test]
    fn highest_out_fires_when_keeping_the_top_card() {
        let s = late();
        let t = table(&s, &[("Kd", 1.0, 0.5)]);
        let r = reasons(&s, &t, act(&s, "Kd"));
        assert_eq!(
            find(&r, "reason.control.highest_out").unwrap().args["card"],
            "2s"
        );
    }

    #[test]
    fn highest_out_does_not_fire_when_spending_it_or_not_holding_it() {
        let s = late();
        let t = table(&s, &[("2s", 1.0, 0.5)]);
        assert!(find(
            &reasons(&s, &t, act(&s, "2s")),
            "reason.control.highest_out"
        )
        .is_none());
        // Seat 3 holds the 2s, the highest card not yet played.
        let s = state(["3c 9d Kd", "4d 4c", "5d 5c", "2s 6d"], 0, None);
        let t = table(&s, &[("3c", 1.0, 0.5)]);
        assert!(find(
            &reasons(&s, &t, act(&s, "3c")),
            "reason.control.highest_out"
        )
        .is_none());
    }

    #[test]
    fn value_gap_fires_with_the_signed_gap() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("3c", 1.25, 0.3)]);
        assert_eq!(
            find(&reasons(&s, &t, act(&s, "Kd")), "reason.value.gap")
                .unwrap()
                .args["points"],
            0.75
        );
        assert_eq!(
            find(&reasons(&s, &t, act(&s, "3c")), "reason.value.gap")
                .unwrap()
                .args["points"],
            -0.75
        );
    }

    #[test]
    fn value_gap_does_not_fire_with_one_legal_action() {
        // Seat 0 cannot beat the 2s: the pass is the only action.
        let s = state(["3c 4d", "5d 5c", "6d 6c", "7d 7c"], 0, Some((3, "2s")));
        let t = table(&s, &[("", 0.0, 0.0)]);
        assert_eq!(t.rows.len(), 1);
        assert!(find(&reasons(&s, &t, Action::Pass), "reason.value.gap").is_none());
    }

    #[test]
    fn counterfactual_outcomes_fires_for_a_move_that_is_not_best() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("3c", 1.0, 0.25)]);
        let r = reasons(&s, &t, act(&s, "3c"));
        let x = find(&r, "reason.counterfactual.outcomes").expect("fires");
        assert_eq!(x.args["best"], json!({"mean": 2.0, "win": 0.3}));
        assert_eq!(x.args["played"], json!({"mean": 1.0, "win": 0.25}));
        assert_eq!(x.args["deals"], 300);
    }

    #[test]
    fn counterfactual_outcomes_does_not_fire_for_the_best_move() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("3c", 1.0, 0.25)]);
        assert!(find(
            &reasons(&s, &t, act(&s, "Kd")),
            "reason.counterfactual.outcomes"
        )
        .is_none());
    }

    fn with_seats(mut t: MoveTable, seats: &[(&str, [f64; 4])], s: &State) -> MoveTable {
        for r in &mut t.rows {
            r.seat_means = Some(
                seats
                    .iter()
                    .find(|(c, _)| act(s, c) == r.action)
                    .map_or([r.mean, 0.0, 0.0, 0.0], |x| x.1),
            );
        }
        t
    }

    #[test]
    fn helps_seat_fires_for_the_opponent_who_gains_most() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("3c", 1.0, 0.25)]);
        let t = with_seats(
            t,
            &[
                ("Kd", [2.0, -1.0, 0.0, -1.0]),
                ("3c", [1.0, 2.0, -2.0, -1.0]),
            ],
            &s,
        );
        let r = reasons(&s, &t, act(&s, "3c"));
        let x = find(&r, "reason.helps_seat").expect("fires");
        assert_eq!(x.args["seat"], 1);
        assert_eq!(x.args["delta"], 3.0);
    }

    #[test]
    fn helps_seat_does_not_fire_for_small_gains_or_without_seat_values() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("3c", 1.0, 0.25)]);
        assert!(find(&reasons(&s, &t, act(&s, "3c")), "reason.helps_seat").is_none());
        let t = with_seats(
            t,
            &[
                ("Kd", [2.0, -1.0, 0.0, -1.0]),
                ("3c", [1.0, -0.5, 0.5, -1.0]),
            ],
            &s,
        );
        assert!(find(&reasons(&s, &t, act(&s, "3c")), "reason.helps_seat").is_none());
    }

    #[test]
    fn helps_seat_does_not_fire_when_the_paired_gain_is_noise() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("3c", 1.0, 0.25)]);
        let mut t = with_seats(
            t,
            &[
                ("Kd", [2.0, -1.0, 0.0, -1.0]),
                ("3c", [1.0, 2.0, -2.0, -1.0]),
            ],
            &s,
        );
        // Seat 1 gains 3 on average, but +13 / −7 deal by deal over four deals.
        let (kd, c3) = (
            t.index_of(act(&s, "Kd")).unwrap(),
            t.index_of(act(&s, "3c")).unwrap(),
        );
        let mut pd = vec![vec![[0.0; 4]; 4]; t.rows.len()];
        for (d, v) in pd[c3].iter_mut().enumerate() {
            v[1] = if d % 2 == 0 { 12.0 } else { -8.0 };
        }
        for v in &mut pd[kd] {
            v[1] = -1.0;
        }
        t.per_deal = Some(pd.clone());
        assert!(find(&reasons(&s, &t, act(&s, "3c")), "reason.helps_seat").is_none());
        // The same mean with no spread fires.
        for v in &mut pd[c3] {
            v[1] = 2.0;
        }
        t.per_deal = Some(pd);
        assert!(find(&reasons(&s, &t, act(&s, "3c")), "reason.helps_seat").is_some());
    }

    #[test]
    fn forced_fires_for_the_best_move_of_a_proven_endgame() {
        let s = late();
        let mut t = table(&s, &[("2s", 1.0, 1.0)]);
        t.forced = Some(Forced {
            winner: 0,
            plays: 3,
        });
        let r = reasons(&s, &t, act(&s, "2s"));
        let x = find(&r, "reason.counterfactual.forced").expect("fires");
        assert_eq!(x.args, json!({"result": "win", "winner": 0, "plays": 3}));
    }

    #[test]
    fn forced_does_not_fire_without_proof_or_for_another_move() {
        let s = late();
        let mut t = table(&s, &[("2s", 1.0, 1.0)]);
        assert!(find(
            &reasons(&s, &t, act(&s, "2s")),
            "reason.counterfactual.forced"
        )
        .is_none());
        t.forced = Some(Forced {
            winner: 0,
            plays: 3,
        });
        assert!(find(
            &reasons(&s, &t, act(&s, "Kd")),
            "reason.counterfactual.forced"
        )
        .is_none());
    }

    #[test]
    fn every_key_serialises_as_key_and_args() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3), ("9s", 1.0, 0.2)]);
        let r = reasons(&s, &t, act(&s, "9s"));
        assert!(keys(&r).contains(&"reason.plan.breaks"));
        let v = serde_json::to_value(&r[0]).unwrap();
        assert!(v["key"].as_str().unwrap().starts_with("reason."));
        assert!(v["args"].is_object());
    }

    #[test]
    fn a_move_outside_the_table_gets_no_reasons() {
        let s = leading();
        let t = table(&s, &[("Kd", 2.0, 0.3)]);
        let other = act(&following(), "2s");
        assert!(reasons(&s, &t, other).is_empty());
    }
}
