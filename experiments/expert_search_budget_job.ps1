# Queue job for plan Task 39 (the expert solver's depth and budget study): one stage per job.
#   -Stage screen [-Settings a,b]  (cpu lane): every setting (or the named ones) against the raw
#                                  network, 300 deals per lineup; refreshes runs\p6_budget\report.md
#   -Stage timing-native           (quiet lane): native one-thread ms per move, every setting
#   -Stage finalists               (cpu lane, minutes): picks the finalists (screen + native timing)
#   -Stage confirm                 (cpu lane): picks the finalists (screen + native timing), plays
#                                  them at 2,000 deals against the raw network and 1,000 against
#                                  the app's `expert`
#   -Stage timing-node             (quiet lane): the finalists in Node WebAssembly
#   -Stage beyond -Simulations N  (cpu lane, Task 51): PUCT at N simulations vs the raw network,
#                                  2,000 deals per lineup, into runs/p6_budget/beyond
#   -Stage beyond-report           : table and chart of puct200..puct<last> (Task 51)
#   -Stage report                  : table, knee, curve (runs\p6_budget\report.md, curve.svg/png)
# Add with: powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane <lane> -Script <this file> -ScriptArgs "-Stage <stage>" [-After '<ids>']
# Every stage resumes when added again (matches are cached in 100-deal blocks; timed settings
# are kept). The `big2` package comes from the frozen snapshot worktree big2-p6budget (its
# python\big2\_core.pyd has `net_search`; this worktree's must not be rebuilt while Task 36
# jobs run); the scripts, network and outputs live in this worktree.
param(
    [Parameter(Mandatory = $true)][ValidateSet('screen', 'timing-native', 'finalists', 'confirm', 'timing-node', 'report', 'beyond', 'beyond-report')][string]$Stage,
    [int]$Simulations = 0,
    [string[]]$Settings = @('all'),
    [int]$Threads = 0
)
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\big2-workers'
$snapshot = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\big2-p6budget'
Set-Location $repo
$env:PYTHONPATH = "$snapshot\python"
if (-not $env:HKQ_THREADS) { $env:HKQ_THREADS = '8' }
if ($Threads -gt 0) { $env:HKQ_THREADS = "$Threads" }
$env:Path = "C:\Users\brian\miniconda3\envs\big2;C:\Users\brian\miniconda3\envs\big2\Scripts;C:\Users\brian\miniconda3\envs\big2\Library\bin;" + $env:Path
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
$script = "$repo\experiments\expert_search_budget.py"
switch ($Stage) {
    'screen' { & $py $script screen --settings ($Settings -join ',') }
    'timing-native' { & $py $script timing --where native --settings ($Settings -join ',') }
    'finalists' { & $py $script finalists }
    'confirm' { & $py $script finalists; if ($LASTEXITCODE -eq 0) { & $py $script confirm } }
    'timing-node' { & $py $script timing --where node --settings finalists }
    'report' { & $py $script report }
    'beyond' { & $py $script beyond --simulations $Simulations }
    'beyond-report' { & $py $script beyond-report }
}
exit $LASTEXITCODE
