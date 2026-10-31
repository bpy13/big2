//! The native side of `experiments/wasm_timing.mjs`: the same steps as the WebAssembly
//! build, on the calling thread. One JSON line per step on stdout.
//!
//! `cargo run --release -p big2-search-wasm --example puct_timing -- <net.pknn> <steps file>`
//!
//! The steps file holds one step per line ([`big2_search_wasm::Harness::run`]).

use big2_search_wasm::Harness;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let [_, net, steps] = args.as_slice() else {
        return Err("usage: puct_timing <net.pknn> <steps file>".into());
    };
    let bytes = std::fs::read(net).map_err(|e| format!("{net}: {e}"))?;
    let harness = Harness::new(&bytes)?;
    let text = std::fs::read_to_string(steps).map_err(|e| format!("{steps}: {e}"))?;
    for step in text.lines().filter(|l| !l.trim().is_empty()) {
        println!("{}", harness.run(step)?);
    }
    Ok(())
}
