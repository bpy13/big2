//! Quick strength check: one searching seat vs three greedy bots, duplicate deals.
//!
//! `cargo run --release -p poker-search --example arena -- <deals> <iterations> <exploration> [endgame_threshold] [endgame_max_nodes]`
//!
//! Every deal is played four times with the searcher rotated through all seats, so
//! card luck cancels out. If the searcher were as strong as greedy, its mean score
//! would be 0.

use poker_big2::bots::Greedy;
use poker_big2::{RuleSet, State};
use poker_games::{Game, Policy};
use poker_search::{ismcts, IsmctsConfig};
use rand::{rngs::SmallRng, SeedableRng};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let deals: u64 = args.get(1).map_or(50, |s| s.parse().unwrap());
    let iterations: u64 = args.get(2).map_or(1000, |s| s.parse().unwrap());
    let exploration: f64 = args.get(3).map_or(0.7, |s| s.parse().unwrap());
    let endgame_threshold: usize = args.get(4).map_or(0, |s| s.parse().unwrap());
    let endgame_max_nodes: u64 = args.get(5).map_or(20_000, |s| s.parse().unwrap());
    let rollout = Greedy { epsilon: 0.1 };
    let greedy = Greedy::default();
    let mut rng = SmallRng::seed_from_u64(2026);
    let (mut sum, mut sum_sq, mut wins, mut n) = (0.0, 0.0, 0.0, 0.0);
    let start = Instant::now();
    for d in 0..deals {
        let deal = State::new_deal(RuleSet::default(), &mut rng);
        for seat in 0..4 {
            let mut s = deal.clone();
            while let Some(p) = s.current_player() {
                let a = if p == seat {
                    let cfg = IsmctsConfig {
                        iterations,
                        exploration,
                        seed: d * 4 + seat as u64,
                        endgame_threshold,
                        endgame_max_nodes,
                        ..Default::default()
                    };
                    ismcts(&s, p, &cfg, &rollout).best()
                } else {
                    greedy.choose(&s, &s.legal_actions_vec(), &mut rng)
                };
                s.apply(a);
            }
            let o = s.outcome().unwrap();
            sum += o.scores[seat];
            sum_sq += o.scores[seat] * o.scores[seat];
            wins += (o.winner == Some(seat)) as u8 as f64;
            n += 1.0;
        }
    }
    let mean = sum / n;
    let ci = 1.96 * ((sum_sq / n - mean * mean) / n).sqrt();
    println!(
        "iters={iterations} c={exploration} endgame={endgame_threshold}/{endgame_max_nodes}: games={n} mean_score={mean:+.2} ±{ci:.2}  win_rate={:.3} (greedy-only baseline 0.250)  {:.1}s",
        wins / n,
        start.elapsed().as_secs_f64()
    );
}
