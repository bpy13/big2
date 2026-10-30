//! `big2-teach`: runs the teaching layer on self-play positions and writes JSON.
//!
//! Native only (the app calls the library, not this program). Used by
//! `experiments/build_phase4.py`, so the experiments need no rebuild of the
//! Python extension.
//!
//! ```text
//! big2-teach audit   --out FILE [--positions 50] [--deals 300] [--seed 1]
//! big2-teach regrade --out FILE [--reviewed 50] [--deals 300] [--ref-mult 10]
//!                    [--ref-every 5] [--iterations 300] [--seed-a 1] [--seed-b 2]
//!                    [--threads 8] [--max-deals 0]
//! big2-teach ladder  --out FILE [--deals 500] [--threads 8] [--qnet models/big2-qnet-duty.pknn]
//!                    [--hard-iterations 2000] [--expert-iterations 10000] [--all-pairs 1]
//! big2-teach mine    --out DIR [--games 200] [--deals 600] [--threads 8] [--chunk 32]
//!                    [--qnet ...] [--hard-iterations 2000] [--expert-iterations 10000]
//!                    [--samples 16] [--min-gap 2]
//! ```
//!
//! * `audit` — positions from noisy greedy self-play; for each, the paired
//!   table's best move (the hint) and its reason keys, for Brian's audit (B11).
//! * `regrade` — self-play deals by ISMCTS; every decision graded twice with
//!   different seeds, a sample against a grader with `ref-mult` times the deals,
//!   and the luck ledger of every seat (Task 27, A21).
//! * `ladder` — ladder v2 (Task 29, A23): each pair of adjacent levels (or every
//!   pair) on the same duplicate deals, lineup A B A B, each deal 4× with the
//!   hands rotated; the upper level's mean score per game with a 95 % interval.
//! * `mine` — drill candidates and graded mistakes (Task 28) from self-play by
//!   the ladder levels; writes `DIR/raw.jsonl` and `DIR/progress.json`, resumes
//!   from `progress.json`. `python/big2/teach/drills.py` turns the raw lines into
//!   drills.

use big2_big2::bots::Greedy;
use big2_big2::card::card_at;
use big2_big2::qnet::QNet;
use big2_big2::{Action, RuleSet, State};
use big2_games::{Game, Policy};
use big2_search::{ismcts, InfosetConfig, IsmctsConfig, Objective};
use big2_teach::grade::ROLLOUT_EPSILON;
use big2_teach::ladder::{LadderBot, EASY_RATES, LEVELS, MEDIUM_RATES, MEDIUM_TEMPERATURE};
use big2_teach::ledger::DecisionTable;
use big2_teach::{
    deal_strength, evaluate_paired, grade_from_table, grade_paired_table, reasons, Graded, Ledger,
    PairedConfig,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        eprintln!("usage: big2-teach <audit|regrade|ladder|mine> --out FILE [options]");
        std::process::exit(2);
    };
    let opts = Opts::parse(&args[1..]);
    let result = match cmd.as_str() {
        "audit" => audit(&opts),
        "regrade" => regrade(&opts),
        "ladder" => ladder(&opts),
        "mine" => mine(&opts),
        other => Err(format!("unknown command {other}")),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `--name value` pairs.
struct Opts(BTreeMap<String, String>);

impl Opts {
    fn parse(args: &[String]) -> Opts {
        let mut m = BTreeMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                m.insert(name.to_string(), it.next().cloned().unwrap_or_default());
            }
        }
        Opts(m)
    }

    fn num(&self, name: &str, default: u64) -> u64 {
        self.0
            .get(name)
            .map(|v| {
                v.parse()
                    .unwrap_or_else(|_| panic!("--{name} must be a number"))
            })
            .unwrap_or(default)
    }

    fn qnet_path(&self) -> String {
        self.0
            .get("qnet")
            .cloned()
            .unwrap_or_else(|| "models/big2-qnet-duty.pknn".to_string())
    }

    fn out(&self) -> Result<String, String> {
        self.0
            .get("out")
            .cloned()
            .ok_or_else(|| "--out FILE is required".to_string())
    }
}

fn write_json(path: &str, v: &Value) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{path}: {e}"))
}

fn cards(mask: u64) -> Value {
    Value::from(
        big2_cards::mask::bits(mask)
            .map(|i| card_at(i).to_string())
            .collect::<Vec<_>>(),
    )
}

fn action_json(a: Action) -> Value {
    match a {
        Action::Pass => json!("pass"),
        Action::Play(c) => {
            json!({ "cards": cards(c.cards), "combo": format!("combo.{}", c.kind.name()) })
        }
    }
}

fn position_json(s: &State) -> Value {
    let me = s.current_player().expect("game not over");
    json!({
        "seat": me,
        "hand": cards(s.hand(me)),
        "hand_sizes": s.hand_sizes(),
        "current_play": s.current_play().map(|(p, c)| json!({"seat": p, "cards": cards(c.cards)})),
        "played": cards(s.played()),
        "history_len": s.history().len(),
    })
}

fn rollout() -> Greedy {
    Greedy {
        epsilon: ROLLOUT_EPSILON,
    }
}

fn round(x: f64, digits: i32) -> f64 {
    let f = 10f64.powi(digits);
    (x * f).round() / f
}

/// Positions with a real choice from greedy self-play (ε = 0.2), one per deal,
/// at a random point of the deal.
fn audit(o: &Opts) -> Result<(), String> {
    let out = o.out()?;
    let n = o.num("positions", 50) as usize;
    let deals = o.num("deals", 300);
    let seed = o.num("seed", 1);
    let policy = Greedy { epsilon: 0.2 };
    let mut entries = Vec::new();
    let mut key_counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut deal = 0u64;
    let start = Instant::now();
    while entries.len() < n {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(deal));
        deal += 1;
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        // Play the whole deal once to know its length, then stop at a random decision.
        let mut probe = s.clone();
        let mut actions = Vec::new();
        while !probe.is_terminal() {
            let legal = probe.legal_actions_vec();
            let a = policy.choose(&probe, &legal, &mut rng);
            probe.apply(a);
            actions.push(a);
        }
        let stop = rng.gen_range(0..actions.len());
        for &a in &actions[..stop] {
            s.apply(a);
        }
        if s.legal_actions_vec().len() < 2 {
            continue;
        }
        let cfg = PairedConfig {
            deals,
            seed: deal,
            endgame: Some(InfosetConfig::default()),
            ..PairedConfig::default()
        };
        let table = evaluate_paired(&s, &cfg);
        let hint = table.best().action;
        let rs = reasons(&s, &table, hint);
        for r in &rs {
            *key_counts.entry(r.key).or_default() += 1;
        }
        entries.push(json!({
            "id": format!("big2-audit-{:02}", entries.len() + 1),
            "position": position_json(&s),
            "hint": action_json(hint),
            "reasons": rs,
            "table": table.rows.iter().take(5).map(|r| json!({
                "action": action_json(r.action),
                "mean": round(r.mean, 2),
                "ci95": round(r.ci95, 2),
                "win": round(r.win, 3),
            })).collect::<Vec<_>>(),
            "deals": table.deals,
            "seed": deal,
            "rating": Value::Null,
        }));
    }
    let v = json!({
        "about": "Hints with reason keys for Brian's audit (B11): rate each entry's reasons right / irrelevant / wrong in `rating`.",
        "method": format!("paired table: every legal move on the same {deals} sampled deals, greedy rollouts (epsilon {ROLLOUT_EPSILON}); exact endgame for forced results; positions from greedy self-play (epsilon 0.2), one per deal at a random decision"),
        "key_counts": key_counts,
        "seconds": round(start.elapsed().as_secs_f64(), 1),
        "entries": entries,
    });
    write_json(&out, &v)?;
    println!(
        "wrote {n} entries to {out} in {:.1} s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Grades ordered from best to worst; `best` sits below `good`.
fn grade_rank(g: &str) -> i32 {
    match g {
        "best" => 0,
        "good" => 1,
        "inaccuracy" => 2,
        "mistake" => 3,
        "blunder" => 4,
        _ => -1,
    }
}

/// The span of grades a result allows: one grade, or an `unclear` range.
fn span(g: &Graded) -> (i32, i32) {
    match g.range {
        Some((lo, hi)) => (grade_rank(lo), grade_rank(hi)),
        None => (grade_rank(g.grade), grade_rank(g.grade)),
    }
}

/// The same grade, or overlapping spans when either is `unclear`.
fn compatible(a: &Graded, b: &Graded) -> bool {
    let (a0, a1) = span(a);
    let (b0, b1) = span(b);
    a0 <= b1 && b0 <= a1
}

/// The same grade once `best` and `good` count as one.
fn merged_equal(a: &Graded, b: &Graded) -> bool {
    let m = |g: &'static str| if g == "best" { "good" } else { g };
    m(a.grade) == m(b.grade)
}

struct Decision {
    deal: u64,
    turn: usize,
    a: Graded,
    b: Graded,
    reference: Option<Graded>,
}

type DealResult = (Vec<Decision>, Vec<Ledger>, u64);

fn regrade(o: &Opts) -> Result<(), String> {
    let out = o.out()?;
    let reviewed = o.num("reviewed", 50);
    let deals = o.num("deals", 300);
    let ref_mult = o.num("ref-mult", 10);
    let ref_every = o.num("ref-every", 5).max(1) as usize;
    let iterations = o.num("iterations", 300);
    let seed_a = o.num("seed-a", 1);
    let seed_b = o.num("seed-b", 2);
    let threads = o.num("threads", 8) as usize;
    let max_deals = o.num("max-deals", 0);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    let thresholds = PairedConfig::default().thresholds;

    let per_deal: Vec<DealResult> = (0..reviewed)
        .into_par_iter()
        .map(|d| {
            let mut rng = ChaCha8Rng::seed_from_u64(0xB16_2000 + d);
            let start_state = State::new_deal(RuleSet::default(), &mut rng);
            let mut s = start_state.clone();
            let mut decisions = Vec::new();
            let mut tables: Vec<DecisionTable> = Vec::new();
            let mut forced = 0u64;
            let mut turn = 0usize;
            while !s.is_terminal() {
                let actor = s.current_player().unwrap();
                let legal = s.legal_actions_vec();
                if legal.len() == 1 {
                    forced += 1;
                    tables.push((None, actor, legal[0]));
                    s.apply(legal[0]);
                    turn += 1;
                    continue;
                }
                let search = IsmctsConfig {
                    iterations,
                    threads: 1,
                    seed: d * 1000 + turn as u64,
                    objective: Objective::Score,
                    ..IsmctsConfig::default()
                };
                let played = ismcts(&s, actor, &search, &rollout()).best();
                let mix = d.wrapping_mul(10_007).wrapping_add(turn as u64);
                let cfg = |seed: u64, n: u64| PairedConfig {
                    deals: n,
                    seed: seed.wrapping_mul(0x1_0000_0001).wrapping_add(mix),
                    max_deals,
                    ..PairedConfig::default()
                };
                let (ga, ta) = grade_paired_table(&s, played, &cfg(seed_a, deals));
                let (gb, _) = grade_paired_table(&s, played, &cfg(seed_b, deals));
                let ta = ta.expect("a decision with a choice has a table");
                let reference = (decisions.len() % ref_every == 0).then(|| {
                    let cr = PairedConfig {
                        max_deals: 0,
                        ..cfg(seed_a ^ 0xFFFF, deals * ref_mult)
                    };
                    let tr = evaluate_paired(&s, &cr);
                    grade_from_table(&tr, played, &thresholds, cr.seed)
                });
                decisions.push(Decision {
                    deal: d,
                    turn,
                    a: ga,
                    b: gb,
                    reference,
                });
                tables.push((Some(ta), actor, played));
                s.apply(played);
                turn += 1;
            }
            let outcome = s.outcome().unwrap();
            let ledgers = (0..4)
                .map(|seat| {
                    let strength = deal_strength(&start_state, seat, deals, seed_a + d, &rollout());
                    Ledger::from_tables(seat, outcome.scores[seat], strength, &tables)
                })
                .collect();
            eprintln!("deal {d} done ({} decisions)", decisions.len());
            (decisions, ledgers, forced)
        })
        .collect();

    let all: Vec<&Decision> = per_deal.iter().flat_map(|x| &x.0).collect();
    let ledgers: Vec<&Ledger> = per_deal.iter().flat_map(|x| &x.1).collect();
    let forced: u64 = per_deal.iter().map(|x| x.2).sum();
    let n = all.len() as f64;
    let share = |f: &dyn Fn(&Decision) -> bool| all.iter().filter(|d| f(d)).count() as f64 / n;
    let strict = share(&|d| d.a.grade == d.b.grade);
    let compat = share(&|d| compatible(&d.a, &d.b));
    let merged = share(&|d| merged_equal(&d.a, &d.b));
    let unclear = share(&|d| d.a.grade == "unclear");
    let with_ref: Vec<&&Decision> = all.iter().filter(|d| d.reference.is_some()).collect();
    let nr = with_ref.len() as f64;
    let ref_share = |f: &dyn Fn(&Graded, &Graded) -> bool| {
        with_ref
            .iter()
            .filter(|d| f(&d.a, d.reference.as_ref().unwrap()))
            .count() as f64
            / nr
    };
    let ref_strict = ref_share(&|a, r| a.grade == r.grade);
    let ref_compat = ref_share(&compatible);
    let ref_merged = ref_share(&merged_equal);
    let mut grade_counts: BTreeMap<&str, u64> = BTreeMap::new();
    for d in &all {
        *grade_counts.entry(d.a.grade).or_default() += 1;
    }
    let max_sum_err = ledgers
        .iter()
        .map(|l| (l.total() - l.score).abs())
        .fold(0.0, f64::max);
    let mean_abs = |f: &dyn Fn(&Ledger) -> f64| {
        ledgers.iter().map(|l| f(l).abs()).sum::<f64>() / ledgers.len() as f64
    };
    let v = json!({
        "method": {
            "reviewed_deals": reviewed,
            "players": format!("ISMCTS-{iterations} (greedy rollout, epsilon {ROLLOUT_EPSILON}) in every seat"),
            "grader": format!("paired, {deals} deals (doubled while unclear up to {max_deals}), greedy rollouts epsilon {ROLLOUT_EPSILON}"),
            "max_deals": max_deals,
            "reference": format!("paired, {} deals, every {ref_every}th graded decision of a deal", deals * ref_mult),
            "thresholds_points": [thresholds.good, thresholds.inaccuracy, thresholds.mistake],
            "seeds": [seed_a, seed_b],
            "threads": threads,
        },
        "decisions_graded": all.len(),
        "forced_skipped": forced,
        "agreement": {"strict": strict, "compatible": compat, "best_good_merged": merged},
        "unclear_share": unclear,
        "grade_counts_seed_a": grade_counts,
        "phone_vs_reference": {"decisions": with_ref.len(), "strict": ref_strict, "compatible": ref_compat, "best_good_merged": ref_merged},
        "ledger": {
            "seats": ledgers.len(),
            "max_abs_sum_error": max_sum_err,
            "mean_abs": {
                "score": mean_abs(&|l| l.score),
                "deal_strength": mean_abs(&|l| l.deal_strength),
                "opponents_errors": mean_abs(&|l| l.opponents_errors),
                "own_errors": mean_abs(&|l| l.own_errors),
                "residual": mean_abs(&|l| l.residual),
            },
        },
        "seconds": round(start.elapsed().as_secs_f64(), 1),
        "disagreements": all.iter().filter(|d| d.a.grade != d.b.grade).take(200).map(|d| json!({
            "deal": d.deal, "turn": d.turn,
            "a": {"grade": d.a.grade, "loss": round(d.a.loss, 3), "interval": round(d.a.interval, 3)},
            "b": {"grade": d.b.grade, "loss": round(d.b.loss, 3), "interval": round(d.b.interval, 3)},
        })).collect::<Vec<_>>(),
    });
    write_json(&out, &v)?;
    println!(
        "{} decisions: strict {:.3}, compatible {:.3}, merged {:.3}, unclear {:.3}; vs reference ({}): strict {:.3}, compatible {:.3}, merged {:.3}; ledger max error {:.2e}; {:.0} s",
        all.len(),
        strict,
        compat,
        merged,
        unclear,
        with_ref.len(),
        ref_strict,
        ref_compat,
        ref_merged,
        max_sum_err,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// One pair's duplicate match: the upper level in seats 0 and 2, the lower in
/// 1 and 3; returns per deal the upper level's mean score per game, its wins
/// per game, and the decisions and mistakes per level.
fn ladder_deal(bots: [&LadderBot; 2], seed: u64) -> (f64, f64, [[u64; 2]; 2]) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let base = State::new_deal(RuleSet::default(), &mut rng).hands();
    let (mut score, mut wins) = (0.0, 0.0);
    // counts[level][0] decisions with a choice, [1] mistakes fired.
    let mut counts = [[0u64; 2]; 2];
    for r in 0..4 {
        let hands: [u64; 4] = std::array::from_fn(|i| base[(i + r) % 4]);
        let mut s = State::from_hands(RuleSet::default(), hands).expect("rotated deal");
        let mut game_rng = ChaCha8Rng::seed_from_u64(seed ^ ((r as u64 + 1) * 0x9E37_79B9));
        while !s.is_terminal() {
            let p = s.current_player().unwrap();
            let level = p % 2; // 0: the upper level (seats 0, 2)
            let legal = s.legal_actions_vec();
            let (a, why) = bots[level].choose_explained(&s, &legal, &mut game_rng);
            if legal.len() > 1 {
                counts[level][0] += 1;
                counts[level][1] += u64::from(why.is_some());
            }
            s.apply(a);
        }
        let o = s.outcome().unwrap();
        score += (o.scores[0] + o.scores[2]) / 2.0;
        wins += f64::from(u8::from(matches!(o.winner, Some(0) | Some(2))));
    }
    (score / 4.0, wins / 4.0, counts)
}

fn mean_ci(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    if xs.len() < 2 {
        return (m, 0.0);
    }
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
    (m, 1.96 * (var / n).sqrt())
}

fn ladder(o: &Opts) -> Result<(), String> {
    let out = o.out()?;
    let deals = o.num("deals", 500);
    let threads = o.num("threads", 8) as usize;
    let seed = o.num("seed", 0);
    let all_pairs = o.num("all-pairs", 0) == 1;
    let qnet_path = o.qnet_path();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| e.to_string())?;
    let bots = ladder_bots(o, load_qnet(&qnet_path)?)?;
    let iterations: Vec<u64> = bots
        .iter()
        .map(|b| match b {
            LadderBot::Search(s) => s.iterations,
            _ => 0,
        })
        .collect();
    let pairs: Vec<(usize, usize)> = if all_pairs {
        (0..5)
            .flat_map(|lo| (lo + 1..5).map(move |hi| (lo, hi)))
            .collect()
    } else {
        (0..4).map(|lo| (lo, lo + 1)).collect()
    };
    let start = Instant::now();
    let mut results = Vec::new();
    let mut mistakes = [[0u64; 2]; 5];
    for &(lo, hi) in &pairs {
        let t = Instant::now();
        let per: Vec<(f64, f64, [[u64; 2]; 2])> = (0..deals)
            .into_par_iter()
            .map(|d| ladder_deal([&bots[hi], &bots[lo]], seed.wrapping_mul(1_000_003) + d))
            .collect();
        let scores: Vec<f64> = per.iter().map(|x| x.0).collect();
        let (m, ci) = mean_ci(&scores);
        let win = per.iter().map(|x| x.1).sum::<f64>() / per.len() as f64;
        for x in &per {
            for (k, (u, l)) in x.2[0].iter().zip(&x.2[1]).enumerate() {
                mistakes[hi][k] += u;
                mistakes[lo][k] += l;
            }
        }
        eprintln!(
            "{} vs {}: {m:+.2} ± {ci:.2} ({:.0} s)",
            LEVELS[hi],
            LEVELS[lo],
            t.elapsed().as_secs_f64()
        );
        results.push(json!({
            "upper": LEVELS[hi], "lower": LEVELS[lo], "deals": deals,
            "mean": m, "ci95": ci, "upper_win_rate": win,
            "seconds": round(t.elapsed().as_secs_f64(), 1),
        }));
    }
    let adjacent: Vec<&Value> = results
        .iter()
        .filter(|r| {
            let lo = LEVELS.iter().position(|l| *l == r["lower"]).unwrap();
            let hi = LEVELS.iter().position(|l| *l == r["upper"]).unwrap();
            hi == lo + 1
        })
        .collect();
    let strict = adjacent
        .iter()
        .all(|r| r["mean"].as_f64().unwrap() - r["ci95"].as_f64().unwrap() > 0.0);
    let gaps_ok = adjacent.iter().all(|r| r["mean"].as_f64().unwrap() >= 1.5);
    let v = json!({
        "method": {
            "lineup": "A B A B (upper level in seats 0 and 2), each deal 4x with the hands rotated",
            "deals": deals, "seed": seed, "qnet": qnet_path, "threads": threads,
            "levels": LEVELS, "search_iterations": iterations,
            "easy_rates": EASY_RATES, "medium_rates": MEDIUM_RATES,
            "medium_temperature": MEDIUM_TEMPERATURE,
        },
        "pairs": results,
        "mistake_share": LEVELS.iter().enumerate().map(|(i, l)| json!({
            "level": l, "decisions": mistakes[i][0],
            "share": if mistakes[i][0] > 0 { mistakes[i][1] as f64 / mistakes[i][0] as f64 } else { 0.0 },
        })).collect::<Vec<_>>(),
        "strict_order": strict,
        "gaps_at_least_1_5": gaps_ok,
        "seconds": round(start.elapsed().as_secs_f64(), 1),
    });
    write_json(&out, &v)?;
    println!(
        "ladder: strict order {strict}, every adjacent gap >= 1.5 {gaps_ok}; {:.0} s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn load_qnet(path: &str) -> Result<std::sync::Arc<QNet>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(std::sync::Arc::new(QNet::from_bytes(&bytes)?))
}

/// The five levels with the search budgets from the options.
fn ladder_bots(o: &Opts, net: std::sync::Arc<QNet>) -> Result<Vec<LadderBot>, String> {
    LEVELS
        .iter()
        .map(|name| {
            let mut b = LadderBot::by_name(name, Some(net.clone()))?;
            if let LadderBot::Search(s) = &mut b {
                s.iterations = o.num(&format!("{name}-iterations"), s.iterations);
            }
            Ok(b)
        })
        .collect()
}

/// A position in the app's drill shape (`engine-interface.md` §6): every hand as
/// it is now, the player to act and the play to beat; no history.
fn drill_position(s: &State, seed: u64) -> Value {
    json!({
        "game": "big2",
        "seed": seed,
        "start": {
            "hands": (0..4).map(|p| cards(s.hand(p))).collect::<Vec<_>>(),
            "current_player": s.current_player(),
            "current_play": s.current_play().map(|(p, c)| json!({
                "player": p,
                "cards": cards(c.cards),
                "combo": format!("combo.{}", c.kind.name()),
            })),
        },
        "history": [],
    })
}

/// Share of `n` samples of `bot` that do not play `best`.
fn wrong_share(
    bot: &LadderBot,
    s: &State,
    legal: &[Action],
    best: Action,
    n: u64,
    seed: u64,
) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let wrong = (0..n)
        .filter(|_| bot.choose(s, legal, &mut rng) != best)
        .count();
    wrong as f64 / n as f64
}

/// One self-play deal: seats played by easy, medium, hard, hard (rotated by the
/// deal index). Every decision with a choice gets a paired table; a decision
/// where a level erred is a graded mistake; a decision with a clear best move
/// (paired gap ≥ `min_gap` with its interval above 0) that easy and medium get
/// wrong at least half the time and expert gets right is a drill candidate.
fn mine_deal(o: &Opts, bots: &[LadderBot], d: u64) -> Vec<Value> {
    let deals = o.num("deals", 600);
    let samples = o.num("samples", 16);
    let min_gap = o.num("min-gap", 2) as f64;
    let thresholds = PairedConfig::default().thresholds;
    let seat_level = |p: usize| [1usize, 2, 3, 3][(p + d as usize) % 4];
    let mut rng = ChaCha8Rng::seed_from_u64(0xD811_0000 + d);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    let mut out = Vec::new();
    let mut turn = 0usize;
    while !s.is_terminal() {
        let p = s.current_player().unwrap();
        let legal = s.legal_actions_vec();
        let (played, why) = bots[seat_level(p)].choose_explained(&s, &legal, &mut rng);
        if legal.len() > 1 {
            let cfg = PairedConfig {
                deals,
                seed: d.wrapping_mul(10_007).wrapping_add(turn as u64),
                ..PairedConfig::default()
            };
            let table = evaluate_paired(&s, &cfg);
            let best = table.best().action;
            let best_reasons = reasons(&s, &table, best);
            let keys: Vec<&str> = best_reasons.iter().map(|r| r.key).collect();
            let g = grade_from_table(&table, played, &thresholds, cfg.seed);
            if g.grade != "best" {
                out.push(json!({
                    "type": "mistake", "deal": d, "turn": turn,
                    "level": LEVELS[seat_level(p)], "grade": g.grade, "loss": round(g.loss, 3),
                    "mistake": why.map(|m| m.key()), "best_keys": keys,
                }));
            }
            let second = &table.rows[1];
            let gap = table.best().mean - second.mean;
            let gap_ci = grade_from_table(&table, second.action, &thresholds, cfg.seed).interval;
            if gap >= min_gap && gap - gap_ci > 0.0 {
                let seed = cfg.seed ^ 0xA5A5;
                let easy = wrong_share(&bots[1], &s, &legal, best, samples, seed);
                let medium = wrong_share(&bots[2], &s, &legal, best, samples, seed + 1);
                if easy >= 0.5 && medium >= 0.5 {
                    let expert = wrong_share(&bots[4], &s, &legal, best, 1, seed + 2);
                    if expert == 0.0 {
                        let hard = wrong_share(&bots[3], &s, &legal, best, 1, seed + 3);
                        let random = 1.0 - 1.0 / legal.len() as f64;
                        let answers: Vec<Value> = table
                            .rows
                            .iter()
                            .take(4)
                            .map(|r| {
                                let g = grade_from_table(&table, r.action, &thresholds, cfg.seed);
                                json!({
                                    "action": action_json(r.action),
                                    "grade": g.grade,
                                    "loss": round(g.loss, 2),
                                    "reasons": reasons(&s, &table, r.action),
                                })
                            })
                            .collect();
                        out.push(json!({
                            "type": "candidate", "deal": d, "turn": turn, "seat": p,
                            "position": drill_position(&s, 0xD811_0000 + d),
                            "following": s.current_play().is_some(),
                            "best_keys": keys,
                            "answers": answers,
                            "gap": round(gap, 2), "gap_ci": round(gap_ci, 2),
                            "samples": deals,
                            "wrong": {"random": random, "easy": easy, "medium": medium, "hard": hard, "expert": expert},
                        }));
                    }
                }
            }
        }
        s.apply(played);
        turn += 1;
    }
    out
}

fn mine(o: &Opts) -> Result<(), String> {
    let dir = std::path::PathBuf::from(o.out()?);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let games = o.num("games", 200);
    let chunk = o.num("chunk", 32).max(1);
    let threads = o.num("threads", 8) as usize;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| e.to_string())?;
    let bots = ladder_bots(o, load_qnet(&o.qnet_path())?)?;
    let progress_path = dir.join("progress.json");
    let raw_path = dir.join("raw.jsonl");
    // Resume: deals below `next` are already in raw.jsonl.
    let mut next = std::fs::read_to_string(&progress_path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v["next_deal"].as_u64())
        .unwrap_or(0);
    if next == 0 {
        std::fs::write(&raw_path, "").map_err(|e| e.to_string())?;
    }
    let start = Instant::now();
    while next < games {
        let end = (next + chunk).min(games);
        let lines: Vec<Vec<Value>> = (next..end)
            .into_par_iter()
            .map(|d| mine_deal(o, &bots, d))
            .collect();
        let mut text = String::new();
        let (mut cands, mut mistakes) = (0, 0);
        for v in lines.iter().flatten() {
            if v["type"] == "candidate" {
                cands += 1;
            } else {
                mistakes += 1;
            }
            text.push_str(&v.to_string());
            text.push('\n');
        }
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&raw_path)
            .and_then(|mut f| f.write_all(text.as_bytes()))
            .map_err(|e| e.to_string())?;
        next = end;
        write_json(
            progress_path.to_str().unwrap(),
            &json!({
                "next_deal": next, "games": games,
                "last_chunk": {"candidates": cands, "mistakes": mistakes},
                "seconds_this_run": round(start.elapsed().as_secs_f64(), 1),
            }),
        )?;
        eprintln!("deals {next}/{games}: +{cands} candidates, +{mistakes} mistakes");
    }
    println!(
        "mined {games} deals into {} ({:.0} s)",
        raw_path.display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
