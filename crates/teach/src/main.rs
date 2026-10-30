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
//! ```
//!
//! * `audit` — positions from noisy greedy self-play; for each, the paired
//!   table's best move (the hint) and its reason keys, for Brian's audit (B11).
//! * `regrade` — self-play deals by ISMCTS; every decision graded twice with
//!   different seeds, a sample against a grader with `ref-mult` times the deals,
//!   and the luck ledger of every seat (Task 27, A21).

use big2_big2::bots::Greedy;
use big2_big2::card::card_at;
use big2_big2::{Action, RuleSet, State};
use big2_games::{Game, Policy};
use big2_search::{ismcts, InfosetConfig, IsmctsConfig, Objective};
use big2_teach::grade::ROLLOUT_EPSILON;
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
        eprintln!("usage: big2-teach <audit|regrade> --out FILE [options]");
        std::process::exit(2);
    };
    let opts = Opts::parse(&args[1..]);
    let result = match cmd.as_str() {
        "audit" => audit(&opts),
        "regrade" => regrade(&opts),
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
