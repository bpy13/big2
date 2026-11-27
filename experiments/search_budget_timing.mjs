// Node WebAssembly side of plan Task 39's timing (experiments/expert_search_budget.py timing
// --where node): runs each step of a steps file in the search-wasm package and prints one
// JSON line of rows per step, as the native example `puct_timing` does.
//
//   node experiments/search_budget_timing.mjs <pkg dir> <net.pknn> <steps file>
//
//   wasm-pack build crates/search-wasm --target nodejs --release --out-dir ../../runs/p6_budget/pkg
import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const [pkg, net, steps] = process.argv.slice(2)
if (!steps) {
  console.error('usage: search_budget_timing.mjs <pkg dir> <net.pknn> <steps file>')
  process.exit(2)
}
const { Harness } = createRequire(import.meta.url)(join(resolve(pkg), 'big2_search_wasm.js'))
const h = new Harness(readFileSync(net))
for (const step of readFileSync(steps, 'utf8').split(/\r?\n/)) {
  if (step.trim()) console.log(h.run(step))
}
