# Queue job for plan Task 40 (ladder rungs below the expert and their Elo).
#   -Stage span   (cpu lane): step 1, random / easy / medium / net t=1 / net / expert against
#                 each other, 500 duplicate deals a pairing (the four expert pairings are the
#                 cost: about 0.26 s an expert move, ~12 thread-hours)
#   -Stage rungs  (cpu lane): step 2, temperatures at equal Elo steps by bisection (cheap)
#   -Stage fit    (cpu lane): step 3, every rung vs the one and two above, pairings grown
#                 until each adjacent gap's 95 % range is within +/-25 Elo or its cap
#   -Stage report (cpu lane): runs\p6_ladder\ladder.json and ladder.md
#   -Stage all    : the four in order (the default)
#   -Threads <T>: worker processes (default HKQ_THREADS, else 8)
# Add with: powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane cpu -Script <this file> -ScriptArgs "-Stage all" [-After '<ids>']
# Every stage resumes when added again (played blocks of 50 deals are cached on disk, so
# progress is saved every few minutes). It imports big2 from THIS checkout (PYTHONPATH):
# do not rebuild python\big2\_core.pyd here while it runs. Needs runs\p6_net\selfplay-h24.pknn.
param(
    [ValidateSet('span', 'rungs', 'fit', 'report', 'all')][string]$Stage = 'all',
    [int]$Threads = 0
)
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\.worktrees\big2-3'
Set-Location $repo
$env:PYTHONPATH = "$repo\python"
if (-not $env:HKQ_THREADS) { $env:HKQ_THREADS = '8' }
if ($Threads -gt 0) { $env:HKQ_THREADS = "$Threads" }
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
& $py "$repo\experiments\elo_ladder.py" $Stage
exit $LASTEXITCODE
