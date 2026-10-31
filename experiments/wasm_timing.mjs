// Times the search in Node WebAssembly and natively, with the same code and the same
// positions (plan Tasks 16 and 20, bars A11 and A15). One thread per position, as the app.
//
//   wasm-pack build crates/search-wasm --target nodejs --release --out-dir ../../runs/wasm_timing/pkg
//   cargo build --release -p big2-search-wasm --example puct_timing
//   python experiments/wasm_positions.py
//   node experiments/wasm_timing.mjs puct    --net runs/net_r3/pvnet-r3.pknn --out runs/wasm_timing/puct.json
//   node experiments/wasm_timing.mjs endgame --net runs/net_r3/pvnet-r3.pknn --out runs/wasm_timing/endgame.json
//
// `puct` runs each step in Node, then in the native example. `endgame` runs the late
// positions of Task 20 in Node on --workers threads, cheapest first, and saves after every
// position, so a rerun resumes. Options: --pkg (another build), --label, --workers,
// --max-cards (positions above this many cards in all hands are left out), --max-nodes.

import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { cpus } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const NATIVE = join(ROOT, 'target', 'release', 'examples', 'puct_timing.exe')
const POSITIONS = join(ROOT, 'runs', 'p3_matches', 'b_endgame', 'positions.txt')

function loadHarness(pkg, net) {
  const { Harness } = createRequire(import.meta.url)(join(pkg, 'big2_search_wasm.js'))
  return new Harness(readFileSync(net))
}

if (!isMainThread) {
  // An endgame worker: solves the positions the main thread hands it.
  const harness = loadHarness(workerData.pkg, workerData.net)
  parentPort.on('message', ({ id, step }) => {
    parentPort.postMessage({ id, row: JSON.parse(harness.run(step))[0] })
  })
} else {
  await main()
}

function option(name, fallback) {
  const i = process.argv.indexOf(`--${name}`)
  return i < 0 ? fallback : process.argv[i + 1]
}

function saveJson(path, data) {
  writeFileSync(`${path}.tmp`, JSON.stringify(data, null, 1))
  renameSync(`${path}.tmp`, path)
}

/** Mean, 95th percentile and largest of numbers. */
function spread(values) {
  const v = [...values].sort((a, b) => a - b)
  const mean = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
  return { n: v.length, mean, p95: v[Math.min(v.length - 1, Math.floor(0.95 * v.length))], max: v.at(-1) }
}

function summarize(rows) {
  const out = { ms: spread(rows.map((r) => r.ms)) }
  if (rows[0]?.changed !== undefined) {
    const reached = rows.filter((r) => r.reached_first_ms)
    out.positions = rows.length
    out.reached_first_ms = reached.length
    out.changed = rows.filter((r) => r.changed).length
    out.changed_share = out.changed / rows.length
    out.changed_share_of_reached = out.changed / Math.max(1, reached.length)
    out.first_simulations = spread(reached.map((r) => r.first_simulations))
  }
  return out
}

async function puct(pkg, net, out) {
  // The first calls run before the JIT has settled: reported apart, then 50 deals each.
  const steps = [
    ['first calls, PUCT-2000 λ 0.5', 'opening puct 2000 0.5 3 1'],
    ['opening PUCT-2000 λ 0.5', 'opening puct 2000 0.5 50 2026'],
    ['opening PUCT-2000 λ 1', 'opening puct 2000 1 50 2026'],
    ['opening PUCT-300 λ 0.5', 'opening puct 300 0.5 50 2026'],
    ['opening PUCT-300 λ 1', 'opening puct 300 1 50 2026'],
    ['opening ISMCTS-1000', 'opening ismcts 1000 50 2026'],
    ['opening ISMCTS-5000', 'opening ismcts 5000 50 2026'],
    ['opening ISMCTS-10000', 'opening ismcts 10000 50 2026'],
    ['midgame ISMCTS-1000', 'midgame ismcts 1000 300 2026'],
    ['midgame ISMCTS-5000', 'midgame ismcts 5000 300 2026'],
    ['midgame ISMCTS-10000', 'midgame ismcts 10000 300 2026'],
    ['anytime PUCT-2000 λ 0.5, 100 ms', 'anytime 2000 0.5 100 300 2026'],
    ['anytime PUCT-2000 λ 1, 100 ms', 'anytime 2000 1 100 300 2026'],
    ['anytime PUCT-300 λ 0.5, 100 ms', 'anytime 300 0.5 100 300 2026'],
  ]
  const harness = loadHarness(pkg, net)
  const results = { label: option('label', ''), cpus: cpus().length, node: process.version, steps: [] }
  for (const [name, step] of steps) {
    const t = performance.now()
    const rows = JSON.parse(harness.run(step))
    const row = { name, step, wasm: summarize(rows), wasm_rows: rows }
    console.log(`${name}: wasm mean ${row.wasm.ms.mean.toFixed(1)} ms, p95 ${row.wasm.ms.p95.toFixed(1)} ms, ` +
      `${((performance.now() - t) / 1000).toFixed(0)} s`, row.wasm.changed_share ?? '')
    results.steps.push(row)
    saveJson(out, results)
  }
  if (option('native', 'yes') === 'yes' && existsSync(NATIVE)) {
    const file = `${out}.steps.txt`
    writeFileSync(file, steps.map(([, s]) => s).join('\n') + '\n')
    const r = spawnSync(NATIVE, [net, file], { encoding: 'utf8', maxBuffer: 1 << 28 })
    if (r.status !== 0) throw new Error(`native example failed: ${r.stderr}`)
    r.stdout.trim().split('\n').forEach((line, i) => {
      const rows = JSON.parse(line)
      results.steps[i].native = summarize(rows)
      results.steps[i].native_rows = rows
      console.log(`${steps[i][0]}: native mean ${results.steps[i].native.ms.mean.toFixed(1)} ms`)
    })
    saveJson(out, results)
  }
}

async function endgame(pkg, net, out) {
  const workers = Number(option('workers', 1))
  const maxCards = Number(option('max-cards', 14))
  const maxNodes = option('max-nodes', '20000000')
  const lines = readFileSync(POSITIONS, 'utf8').trim().split('\n')
  const positions = lines.map((line, id) => {
    const [total, hidden, ...rest] = line.trim().split(/\s+/)
    return { id, total: Number(total), hidden: Number(hidden), step: `endgame 12 52 ${maxNodes} ${rest.join(' ')}` }
  }).filter((p) => p.total <= maxCards).sort((a, b) => a.total - b.total || a.hidden - b.hidden)

  const results = existsSync(out) ? JSON.parse(readFileSync(out, 'utf8'))
    : { label: option('label', ''), workers, max_nodes: Number(maxNodes), rows: {} }
  const todo = positions.filter((p) => results.rows[p.id] === undefined)
  console.log(`${todo.length} of ${positions.length} positions left, ${workers} workers`)
  await new Promise((done, fail) => {
    let next = 0
    let running = 0
    const pool = []
    const give = (w) => {
      if (next >= todo.length) {
        void w.terminate()
        if (running === 0) done()
        return
      }
      running += 1
      const p = todo[next++]
      w.postMessage({ id: p.id, step: p.step })
    }
    for (let i = 0; i < workers; i++) {
      const w = new Worker(fileURLToPath(import.meta.url), { workerData: { pkg, net } })
      w.on('message', ({ id, row }) => {
        running -= 1
        const p = positions.find((q) => q.id === id)
        results.rows[id] = { ...row, total: p.total, hidden: p.hidden }
        saveJson(out, results)
        if (Object.keys(results.rows).length % 25 === 0) {
          console.log(`${Object.keys(results.rows).length} done, last ${p.total} cards ${row.ms.toFixed(0)} ms`)
        }
        give(w)
      })
      w.on('error', fail)
      pool.push(w)
    }
    if (todo.length === 0) done()
    pool.forEach(give)
  })

  const rows = Object.values(results.rows)
  const table = (key, limits) => limits.map((limit) => {
    const sel = rows.filter((r) => r[key] <= limit)
    const ms = spread(sel.map((r) => r.ms))
    return { limit, positions: sel.length, unsolved: sel.filter((r) => !r.exact).length, ...ms }
  })
  results.by_hidden = table('hidden', [8, 10, 12])
  results.by_cards = table('total', [8, 10, 12, 14])
  saveJson(out, results)
  console.table(results.by_hidden)
  console.table(results.by_cards)
}

async function main() {
  const command = process.argv[2]
  const pkg = resolve(ROOT, option('pkg', 'runs/wasm_timing/pkg'))
  const net = resolve(ROOT, option('net', 'runs/net_r3/pvnet-r3.pknn'))
  const out = resolve(ROOT, option('out', `runs/wasm_timing/${command}.json`))
  if (command === 'puct') await puct(pkg, net, out)
  else if (command === 'endgame') await endgame(pkg, net, out)
  else throw new Error('usage: wasm_timing.mjs puct|endgame [options]')
}
