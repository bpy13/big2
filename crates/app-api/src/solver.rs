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
use big2_search::{ismcts, pimc, Analysis, IsmctsConfig, Objective, PimcConfig};
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

/// ISMCTS iterations per search bot level (easy and medium are not searches).
pub fn bot_iterations(level: &str) -> Option<u64> {
    match level {
        "medium" => Some(300),
        // 500, not 2,000, so `expert` is clearly stronger (+1.02 ± 0.40
        // points per deal at 500; −0.03 at 2,000). Brian, 2026-11-02.
        "hard" => Some(500),
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
/// at `deadline`. The engine checks the clock between deals and gives every
/// legal action the same number of deals, so the played move and the best move
/// are compared fairly. One thread: the same code path on native and in
/// WebAssembly, and bit-identical to a run without a time limit.
pub fn pimc_budgeted(
    state: &State,
    deals: u64,
    deadline: Option<Instant>,
    seed: u64,
) -> Analysis<Action> {
    let seat = state.current_player().expect("the deal is not over");
    let cfg = PimcConfig {
        determinizations: deals,
        time_limit: deadline.map(|d| d.saturating_duration_since(Instant::now())),
        threads: 1,
        seed,
        objective: Objective::Score,
        ..PimcConfig::default()
    };
    pimc(state, seat, &cfg, &rollout())
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

/// Grade buckets, in score points under `tiered`/`original`: `good` under
/// 0.69, `inaccuracy` under 2.75, `mistake` under 6.86, `blunder` from 6.86.
/// `best` is the solver's own first choice. The lines are 0.025σ, 0.1σ and
/// 0.25σ of one deal's score (σ = 27.45 in expert self-play, `big2-teach`'s
/// `Thresholds::from_sigma`); Brian chose them over ⅓ / 1 / 3 cards on
/// 2026-11-02. Under `win_loss` they shrink by the same 3 : 1 as
/// [`score_unit`].
pub const GRADE_THRESHOLDS: [(&str, f64); 3] =
    [("good", 0.69), ("inaccuracy", 2.75), ("mistake", 6.86)];

/// The grade of a move that loses `loss` score points against the best move.
pub fn grade_for(is_best: bool, loss: f64, scoring: Scoring) -> &'static str {
    if is_best || loss <= 0.0 {
        return "best";
    }
    // Express the loss in tiered points, the unit of the thresholds.
    let points = loss * score_unit(Scoring::Tiered) / score_unit(scoring);
    for (grade, limit) in GRADE_THRESHOLDS {
        if points < limit {
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
    fn grades_change_at_the_statistics_based_lines() {
        let t = Scoring::Tiered;
        assert_eq!(grade_for(false, 0.68, t), "good");
        assert_eq!(grade_for(false, 0.69, t), "inaccuracy");
        assert_eq!(grade_for(false, 2.74, t), "inaccuracy");
        assert_eq!(grade_for(false, 2.75, t), "mistake");
        assert_eq!(grade_for(false, 6.85, t), "mistake");
        assert_eq!(grade_for(false, 6.86, t), "blunder");
        // win_loss: a third of the tiered lines.
        assert_eq!(grade_for(false, 0.2, Scoring::WinLoss), "good");
        assert_eq!(grade_for(false, 2.3, Scoring::WinLoss), "blunder");
    }

    #[test]
    fn hard_is_ismcts_500() {
        assert_eq!(bot_iterations("hard"), Some(500));
    }
}
