# Queue job for plan Task 21 (build-phase3.md §5): `expert` vs Charlesworth's released network,
# his rules (four-card plays) and scoring (sum of the others' cards), 1,000 deals, both lineups.
#
#   powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\tools\queue\hkq.ps1 add -Lane cpu -Script C:\Users\brian\OneDrive\Documents\workspace\repo\big2\experiments\b2_job.ps1
#
# Needs runs\external\charlesworth\ (his two files, local only, gitignored). The queue sets
# HKQ_THREADS and the CPU rules. Matches resume from saved blocks (runs\b2_charlesworth\);
# rerun the job after a restart. Check: Get-Content ...\big2\runs\b2_charlesworth\progress.json
# Do not rebuild the big2 Python extension while it runs.

$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\big2'
$threads = if ($env:HKQ_THREADS) { [int]$env:HKQ_THREADS } else { 8 }
Set-Location $repo
& 'C:\Users\brian\miniconda3\envs\big2\python.exe' "$repo\experiments\build_phase3.py" b2 `
    --deals 1000 --block 100 --workers $threads --out b2_charlesworth
exit $LASTEXITCODE
