import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
const req = createRequire(import.meta.url)
const { Harness } = req(process.argv[2] + '/big2_search_wasm.js')
const h = new Harness(readFileSync(process.argv[3]))
const rows = JSON.parse(h.run('forward 1500 7'))
const q = (v, p) => v.slice().sort((a, b) => a - b)[Math.min(v.length - 1, Math.floor(p * v.length))]
const warm = rows.slice(100).map(r => r.ms)
console.log(JSON.stringify({n: warm.length, median: q(warm, .5), p95: q(warm, .95), first100_median: q(rows.slice(0,100).map(r=>r.ms), .5)}))
