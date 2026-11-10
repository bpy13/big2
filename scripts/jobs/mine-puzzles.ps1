# Queue job (cpu lane): mine big2 puzzles (hk-games puzzles plan, Task 5).
# Resumes from runs\big2-puzzles\progress.json; adding the job again continues a stopped run.
# 553 puzzles came from 700 deals before; 3,000 deals aim at 2,000 or more.
param([int]$Games = 3000, [int]$DeepMult = 4)
$ErrorActionPreference = 'Stop'
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0   # CPUs 4-19 only
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$threads = if ($env:HKQ_THREADS) { [int]$env:HKQ_THREADS } else { 8 }
$run = Join-Path $root 'runs\big2-puzzles'
$env:CARGO_BUILD_JOBS = '8'
Push-Location $root
try {
  cargo build --release -p big2-teach
  if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
  & (Join-Path $root 'target\release\big2-teach.exe') mine --out $run --games $Games --threads $threads --deep-mult $DeepMult
  if ($LASTEXITCODE -ne 0) { throw 'mine failed' }
  # Turn raw.jsonl into the puzzle file (themes, ratings, brilliant answers) and a summary.
  $py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
  $script = Join-Path $root 'scripts\jobs\write_puzzles.py'
  & $py -I $script (Join-Path $run 'raw.jsonl') (Join-Path $root 'drills\big2-puzzles.json') (Join-Path $run 'summary.json')
  if ($LASTEXITCODE -ne 0) { throw 'write_puzzles failed' }
} finally { Pop-Location }
