# Queue job (cpu lane): top up the big2 endgame_counting theme (hk-games puzzles plan, Task 5b).
# Only exact-endgame candidates are kept (about 1 in 160 deals, so 8,000 deals aim at 30 or more).
# Resumes from runs\big2-puzzles-endgame\progress.json; adding the job again continues a stopped run.
# Afterwards: python -I scripts\jobs\merge_endgame.py (adds the rows to drills\big2-puzzles.json).
param([int]$Games = 8000, [int]$DeepMult = 4)
$ErrorActionPreference = 'Stop'
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF0   # CPUs 4-19 only
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$threads = if ($env:HKQ_THREADS) { [int]$env:HKQ_THREADS } else { 8 }
$run = Join-Path $root 'runs\big2-puzzles-endgame'
$env:CARGO_BUILD_JOBS = '8'
$env:PATH += ";$env:USERPROFILE\.cargo\bin"
Push-Location $root
try {
  cargo build --release -p big2-teach
  if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
  # Deal numbers start at 200000 so no id repeats the main run's.
  & (Join-Path $root 'target\release\big2-teach.exe') mine --out $run --games $Games --threads $threads --deep-mult $DeepMult --top-up 1 --deal-base 200000
  if ($LASTEXITCODE -ne 0) { throw 'mine failed' }
} finally { Pop-Location }
