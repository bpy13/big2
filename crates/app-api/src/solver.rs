//! Thin wrappers over the engine's solvers: ISMCTS for hints, bots and drill
//! grading; PIMC for the post-game review. Every search runs with `threads = 1`
//! (the app runs in WebAssembly, where no thread pool exists) and a wall-clock
//! budget.
//!
//! Grade thresholds live here too (`grade_for`), so hint, review and grade
//! agree on what a "mistake" is.

use big2_big2::bots::Greedy;
use big2_big2::{Action, Scoring, State};
use big2_games::Game;
use big2_search::{ismcts, pimc, Analysis, IsmctsConfig, MoveStats, Objective, PimcConfig};
use std::time::Duration;
use web_time::Instant;

/// Exploration noise of the greedy rollout policy inside the solvers, as in the
/// existing CLI and server (`rollout_epsilon=0.1`).
pub const ROLLOUT_EPSILON: f64 = 0.1;

/// ISMCTS iterations of a hint when the budget allows (the web UI's study mode
/// uses the same number).
pub const HINT_ITERATIONS: u64 = 20_000;

/// Sampled deals per reviewed decision when the budget allows (as the web UI).
pub const REVIEW_DEALS: u64 = 300;

/// Deals run per PIMC call inside the review; the review checks the clock
/// between chunks.
const REVIEW_CHUNK: u64 = 20;

/// ISMCTS iterations per bot level.
pub fn bot_iterations(level: &str) -> Option<u64> {
    match level {
        "medium" => Some(300),
        "hard" => Some(2_000),
        "expert" => Some(10_000),
        _ => None,
    }
}

fn rollout() -> Greedy {
    Greedy {
        epsilon: ROLLOUT_EPSILON,
    }
}

/// ISMCTS for the player to move, at most `iterations` simulations and at most
/// `budget` of wall-clock time (checked every 64 simulations by the engine).
pub fn ismcts_analysis(
    state: &State,
    iterations: u64,
    budget: Option<Duration>,
    seed: u64,
) -> Analysis<Action> {
    let seat = state.current_player().expect("the deal is not over");
    let cfg = IsmctsConfig {
        iterations,
        time_limit: budget,
        threads: 1,
        seed,
        objective: Objective::Score,
        ..IsmctsConfig::default()
    };
    ismcts(state, seat, &cfg, &rollout())
}

/// PIMC for the player to move: up to `deals` sampled deals, stopping early
/// when the next chunk would run past `deadline`. Every legal action gets the
/// same number of rollouts, so the played move and the best move are compared
/// fairly. At least one chunk always runs.
pub fn pimc_budgeted(
    state: &State,
    deals: u64,
    deadline: Option<Instant>,
    seed: u64,
) -> Analysis<Action> {
    let seat = state.current_player().expect("the deal is not over");
    let start = Instant::now();
    let mut merged: Option<Analysis<Action>> = None;
    let mut done = 0u64;
    let mut chunk_index = 0u64;
    while done < deals {
        let n = REVIEW_CHUNK.min(deals - done);
        let cfg = PimcConfig {
            determinizations: n,
            threads: 1,
            seed: seed.wrapping_add(chunk_index.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            objective: Objective::Score,
            ..PimcConfig::default()
        };
        let part = pimc(state, seat, &cfg, &rollout());
        done += n;
        chunk_index += 1;
        merged = Some(match merged {
            None => part,
            Some(acc) => merge(acc, &part),
        });
        if let Some(d) = deadline {
            let now = Instant::now();
            // Would another chunk of the same size overrun the deadline?
            let per_deal = now.duration_since(start) / done as u32;
            if now + per_deal * REVIEW_CHUNK as u32 >= d {
                break;
            }
        }
    }
    let mut out = merged.expect("at least one chunk ran");
    out.elapsed = start.elapsed();
    out
}

/// Adds the samples of `part` to `acc` (same position, so the same actions).
/// Means and confidence intervals combine exactly from the per-move counts.
fn merge(mut acc: Analysis<Action>, part: &Analysis<Action>) -> Analysis<Action> {
    for m in &mut acc.moves {
        if let Some(p) = part.moves.iter().find(|p| p.action == m.action) {
            *m = merge_stats(m, p);
        }
    }
    acc.simulations += part.simulations;
    acc.moves
        .sort_by(|a, b| b.mean_score.total_cmp(&a.mean_score));
    acc
}

fn merge_stats(a: &MoveStats<Action>, b: &MoveStats<Action>) -> MoveStats<Action> {
    let (na, nb) = (a.visits as f64, b.visits as f64);
    let n = na + nb;
    if n == 0.0 {
        return a.clone();
    }
    // `score_ci95 = 1.96·√(var/n)`, so `var = (ci/1.96)²·n` and `E[x²] = var + mean²`.
    let second_moment = |m: &MoveStats<Action>, k: f64| {
        let var = (m.score_ci95 / 1.96).powi(2) * k;
        var + m.mean_score * m.mean_score
    };
    let mean = (na * a.mean_score + nb * b.mean_score) / n;
    let ex2 = (na * second_moment(a, na) + nb * second_moment(b, nb)) / n;
    let var = (ex2 - mean * mean).max(0.0);
    let p = (na * a.win_rate + nb * b.win_rate) / n;
    MoveStats {
        action: a.action,
        visits: a.visits + b.visits,
        mean_score: mean,
        score_ci95: 1.96 * (var / n).sqrt(),
        win_rate: p,
        win_ci95: 1.96 * (p * (1.0 - p) / n).sqrt(),
    }
}

/// Score points that one card left at the end is worth under `scoring`: the
/// unit the grade thresholds are expressed in.
///
/// With pairwise settlement (`tiered`, `original`) a player's score is
/// `total − 4·penalty`, so one more penalty point costs 3 points. Under
/// `win_loss` scores run from −1 to 3 and the unit is 1 point.
pub fn score_unit(scoring: Scoring) -> f64 {
    match scoring {
        Scoring::WinLoss => 1.0,
        Scoring::Tiered | Scoring::Original => 3.0,
    }
}

/// Grade buckets, in units of [`score_unit`] (one card):
/// `good` under ⅓ of a card, `inaccuracy` under 1 card, `mistake` under 3
/// cards, `blunder` from 3 cards. `best` is the solver's own first choice.
pub const GRADE_THRESHOLDS: [(&str, f64); 3] =
    [("good", 1.0 / 3.0), ("inaccuracy", 1.0), ("mistake", 3.0)];

/// The grade of a move that loses `loss` score points against the best move.
pub fn grade_for(is_best: bool, loss: f64, scoring: Scoring) -> &'static str {
    if is_best || loss <= 0.0 {
        return "best";
    }
    let cards = loss / score_unit(scoring);
    for (grade, limit) in GRADE_THRESHOLDS {
        if cards < limit {
            return grade;
        }
    }
    "blunder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grades_follow_the_thresholds() {
        let t = Scoring::Tiered;
        assert_eq!(grade_for(true, 5.0, t), "best");
        assert_eq!(grade_for(false, 0.0, t), "best");
        assert_eq!(grade_for(false, 0.5, t), "good");
        assert_eq!(grade_for(false, 2.0, t), "inaccuracy");
        assert_eq!(grade_for(false, 5.0, t), "mistake");
        assert_eq!(grade_for(false, 9.0, t), "blunder");
        assert_eq!(grade_for(false, 0.5, Scoring::WinLoss), "inaccuracy");
    }

    #[test]
    fn merging_two_halves_matches_one_run_of_sums() {
        let a = MoveStats {
            action: Action::Pass,
            visits: 2,
            mean_score: 1.0,
            score_ci95: 0.0, // both samples equal 1
            win_rate: 1.0,
            win_ci95: 0.0,
        };
        let b = MoveStats {
            action: Action::Pass,
            visits: 2,
            mean_score: 3.0,
            score_ci95: 0.0, // both samples equal 3
            win_rate: 0.0,
            win_ci95: 0.0,
        };
        let m = merge_stats(&a, &b);
        assert_eq!(m.visits, 4);
        assert!((m.mean_score - 2.0).abs() < 1e-9);
        // Samples 1,1,3,3: variance 1, ci = 1.96·√(1/4).
        assert!((m.score_ci95 - 0.98).abs() < 1e-9);
        assert!((m.win_rate - 0.5).abs() < 1e-9);
    }
}
