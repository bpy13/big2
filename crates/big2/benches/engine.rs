//! Engine throughput: `cargo bench -p big2-big2`.

use big2_big2::{bots::Greedy, plan::min_plays, RuleSet, State};
use big2_games::{play_out, Game, UniformRandom};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::SmallRng, SeedableRng};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let deals: Vec<State> = (0..256)
        .map(|_| State::new_deal(RuleSet::default(), &mut rng))
        .collect();

    // Average moves per random game, to report moves/second.
    let moves: usize = deals
        .iter()
        .map(|d| {
            let mut s = d.clone();
            play_out(&mut s, &UniformRandom, &mut rng);
            s.history().len()
        })
        .sum::<usize>()
        / deals.len();

    let mut g = c.benchmark_group("big2");
    g.throughput(Throughput::Elements(moves as u64));
    let mut i = 0;
    g.bench_function("random_playout_from_deal", |b| {
        b.iter(|| {
            let mut s = deals[i % deals.len()].clone();
            i += 1;
            black_box(play_out(&mut s, &UniformRandom, &mut rng))
        })
    });
    g.bench_function("greedy_playout_from_deal", |b| {
        b.iter(|| {
            let mut s = deals[i % deals.len()].clone();
            i += 1;
            black_box(play_out(&mut s, &Greedy { epsilon: 0.1 }, &mut rng))
        })
    });
    g.throughput(Throughput::Elements(1));
    let mut out = Vec::new();
    g.bench_function("legal_actions_leading_13_cards", |b| {
        b.iter(|| {
            deals[i % deals.len()].legal_actions(&mut out);
            i += 1;
            black_box(out.len())
        })
    });
    let rules = RuleSet::default();
    g.bench_function("min_plays_13_cards", |b| {
        b.iter(|| {
            let hand = deals[i % deals.len()].hand(0);
            i += 1;
            black_box(min_plays(black_box(hand), &rules))
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
