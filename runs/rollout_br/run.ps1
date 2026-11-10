# Task 32 queued job: rollout best response against 3 x expert (plan Task 32).
# Runs from the slot-2 worktree. Check: Get-Content <worktree>\runs\rollout_br\progress.json ; results: runs\rollout_br\results.json
# Resume: rerun this script (finished deals are in runs\rollout_br\deals.jsonl and are skipped).
# Raise --deals to add deals later (same seeds, same file).
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\.worktrees\big2-2'
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
Set-Location $repo
$env:PYTHONPATH = Join-Path $repo 'python'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$w = if ($env:HKQ_THREADS) { $env:HKQ_THREADS } else { '8' }
& $py experiments/rollout_br.py --deals 40 --rollouts 8 --rollout-iterations 100 --workers $w --out rollout_br
exit $LASTEXITCODE
