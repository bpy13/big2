# Queue job for plan Task 36 (self-play pilot with the luck taken out): one stage per job.
#   -Stage smoke (gpu lane): three 10-minute runs (pull 0.02 / 0.1 / 0.5), each vs the start;
#                writes runs\selfplay_luck\choice.json (pull and baseline taken)
#   -Stage train -Hours <N> (gpu lane): trains runs\selfplay_luck up to hour N (4-hour jobs);
#                does nothing once runs\selfplay_luck\stop.json says stop
#   -Stage eval -Hours <N> (cpu lane): hour N (and r3) vs the app's `expert` (ISMCTS-10,000),
#                1,000 deals per lineup, seed 7; at hour 8 writes stop.json; refreshes
#                table_expert.md. -Final: the last trained checkpoint up to hour N, 2,000 deals.
#   -Stage h2h -Hours <N> (cpu lane): hour N against hour N-4 and hour 8, networks only,
#                2,000 deals per lineup, seed 7 (Task 36c); refreshes table_h2h.md.
#   -Stage nodup-train (gpu lane): Task 36d, 10-minute arms plain and oracle (no duplicate)
#   -Stage nodup-eval (cpu lane): each arm (and Task 36's smoke network) vs r3 and `expert`
#   -Threads <T>: worker processes (default HKQ_THREADS); 16 only while no gpu job runs.
# Add with: powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane <lane> -Script <this file> -ScriptArgs "-Stage <stage> -Hours <N>" [-After '<ids>']
# Every stage resumes when added again (training from latest.pt, match blocks are cached).
# It imports big2 from THIS worktree (PYTHONPATH). Do not rebuild python\big2\_core.pyd here
# while a stage runs.
param(
    [Parameter(Mandatory = $true)][ValidateSet('smoke', 'train', 'eval', 'h2h', 'nodup-train', 'nodup-eval')][string]$Stage,
    [int]$Hours = 0,
    [switch]$Final,
    [int]$Threads = 0
)
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\big2-workers'
Set-Location $repo
$env:PYTHONPATH = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\big2-p6budget\python'  # has net_search in _core.pyd; this worktree's was not rebuilt after Task 38
if (-not $env:HKQ_THREADS) { $env:HKQ_THREADS = '8' }
if ($Threads -gt 0) { $env:HKQ_THREADS = "$Threads" }
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
$script = "$repo\experiments\selfplay_luck.py"
switch ($Stage) {
    'smoke' { & $py $script smoke }
    'train' { & $py $script train --until-hours $Hours }
    'h2h' { & $py $script h2h --hour $Hours }
    'nodup-train' { & $py $script nodup-train }
    'nodup-eval' { & $py $script nodup-eval }
    'eval' {
        if ($Final) { & $py $script eval --hour $Hours --final }
        else { & $py $script eval --hour $Hours }
    }
}
exit $LASTEXITCODE
