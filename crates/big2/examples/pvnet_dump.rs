//! Evaluates positions from seeded games with the Rust [`PvNet`], for comparing it
//! with PyTorch (`python/tests/test_export.py`, the `v2` tests). Like `qnet_dump`,
//! it lives here so the comparison needs no change to the Python extension.
//!
//! ```text
//! cargo run --release -p big2-big2 --example pvnet_dump -- NET.pknn OUT.bin POSITIONS SEED
//! ```
//!
//! Games use the default rules; every seat plays the move with the highest prior,
//! or a random legal move 25 % of the time for variety. Only decisions with at
//! least two legal moves are kept. Output, little-endian:
//!
//! ```text
//! b"PDMP" u32 state_len u32 action_len u32 positions
//! per position: u32 n_moves, u8 state[state_len], u8 actions[n_moves x action_len],
//!               f32 prior[n_moves] (probabilities),
//!               f32 dist[n_moves x 4 x 14] (probabilities, seat-major)
//! ```

use big2_big2::features::{encode_action, encode_state, ACTION_LEN, STATE_LEN};
use big2_big2::pvnet::PvNet;
use big2_big2::{RuleSet, State};
use big2_games::Game;
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [net, out, positions, seed] = args.as_slice() else {
        return Err("usage: pvnet_dump NET.pknn OUT.bin POSITIONS SEED".into());
    };
    let bytes = std::fs::read(net).map_err(|e| format!("{net}: {e}"))?;
    let net = PvNet::from_bytes(&bytes)?;
    let positions: usize = positions
        .parse()
        .map_err(|_| "POSITIONS must be a number")?;
    let seed: u64 = seed.parse().map_err(|_| "SEED must be a number")?;

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut buf = b"PDMP".to_vec();
    for n in [STATE_LEN, ACTION_LEN, positions] {
        buf.extend((n as u32).to_le_bytes());
    }
    let (mut kept, mut legal) = (0, Vec::new());
    while kept < positions {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        while !s.is_terminal() && kept < positions {
            s.legal_actions(&mut legal);
            let mut best = 0;
            if legal.len() >= 2 {
                let out = net.evaluate(&s, &legal);
                buf.extend((legal.len() as u32).to_le_bytes());
                encode_state(&s, &mut buf);
                for a in &legal {
                    encode_action(a, &mut buf);
                }
                buf.extend(out.prior.iter().flat_map(|p| p.to_le_bytes()));
                for d in &out.cand_dist {
                    buf.extend(d.iter().flatten().flat_map(|p| p.to_le_bytes()));
                }
                best = (0..legal.len())
                    .max_by(|&a, &b| out.prior[a].total_cmp(&out.prior[b]))
                    .unwrap();
                kept += 1;
            }
            let a = if rng.gen::<f64>() < 0.25 {
                legal[rng.gen_range(0..legal.len())]
            } else {
                legal[best]
            };
            s.apply(a);
        }
    }
    std::fs::write(out, buf).map_err(|e| format!("{out}: {e}"))
}
