//! Scores positions from seeded games with the Rust [`QNet`], for comparing it
//! with PyTorch (`python/tests/test_export.py`). It lives here, not in the Python
//! module, so the comparison needs no change to the Python extension.
//!
//! ```text
//! cargo run --release -p big2-big2 --example qnet_dump -- NET.pknn OUT.bin POSITIONS SEED
//! ```
//!
//! Games use the default rules; every seat plays the network's best move, or a
//! random legal move 25 % of the time for variety. Only decisions with at least
//! two legal moves are kept. Output, little-endian:
//!
//! ```text
//! b"QDMP" u32 state_len u32 action_len u32 positions
//! per position: u32 n_moves, u8 state[state_len], u8 actions[n_moves × action_len],
//!               f32 score[n_moves] (points), f32 win_logit[n_moves]
//! ```

use std::sync::Arc;

use big2_big2::features::{encode_action, encode_state, ACTION_LEN, STATE_LEN};
use big2_big2::qnet::{QNet, QNetPolicy};
use big2_big2::{RuleSet, State};
use big2_games::{Game, Policy};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [net, out, positions, seed] = args.as_slice() else {
        return Err("usage: qnet_dump NET.pknn OUT.bin POSITIONS SEED".into());
    };
    let bytes = std::fs::read(net).map_err(|e| format!("{net}: {e}"))?;
    let net = Arc::new(QNet::from_bytes(&bytes)?);
    let positions: usize = positions
        .parse()
        .map_err(|_| "POSITIONS must be a number")?;
    let seed: u64 = seed.parse().map_err(|_| "SEED must be a number")?;
    let policy = QNetPolicy {
        net: net.clone(),
        temperature: 0.0,
    };

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut buf = b"QDMP".to_vec();
    for n in [STATE_LEN, ACTION_LEN, positions] {
        buf.extend((n as u32).to_le_bytes());
    }
    let (mut kept, mut legal) = (0, Vec::new());
    while kept < positions {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        while !s.is_terminal() && kept < positions {
            s.legal_actions(&mut legal);
            if legal.len() >= 2 {
                let scores = net.score_moves(&s, &legal);
                buf.extend((legal.len() as u32).to_le_bytes());
                encode_state(&s, &mut buf);
                for a in &legal {
                    encode_action(a, &mut buf);
                }
                buf.extend(scores.iter().flat_map(|x| x.0.to_le_bytes()));
                buf.extend(scores.iter().flat_map(|x| x.1.to_le_bytes()));
                kept += 1;
            }
            let a = if rng.gen::<f64>() < 0.25 {
                legal[rng.gen_range(0..legal.len())]
            } else {
                policy.choose(&s, &legal, &mut rng)
            };
            s.apply(a);
        }
    }
    std::fs::write(out, buf).map_err(|e| format!("{out}: {e}"))
}
