# Task 32 queued job: rollout best response against 3 x expert (plan Task 32), the full run.
# Runs from the big2-workers worktree. Check: Get-Content runs\rollout_br_full\progress.json ; results: runs\rollout_br_full\results.json
# Resume: rerun this script (finished deals are in deals.jsonl and are skipped).
# Pilot (job 47, iterations 100, 40 deals) wrote to runs\rollout_br in the big2-2 worktree.
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\big2-workers'
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo 'python'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$w = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
& $py experiments/rollout_br.py --deals 60 --rollouts 8 --rollout-iterations 10000 --workers $w --out rollout_br_full
exit $LASTEXITCODE
