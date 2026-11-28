# Queue job for plan Task 43 (luck-removed rating: luck estimates and the new expert's games).
#   -Stage play   (cpu lane): puct2000 (Task 42's reference) vs net and vs expert, ABAB, 500 deals
#                 each, blocks of 50 under runs\p6_grading\pairs (resumes)
#   -Stage luck   (cpu lane): the hour-24 network plays every ladder, mistakes and new-pair deal in
#                 all four seats, -M sampled games a deal (0 = one top-move game), into
#                 runs\p6_grading\luck-m<M>.jsonl (resumes)
#   -Stage report : runs\p6_grading\report.md and proposal.json (seconds)
#   -Stage all    : play, luck, report (the default)
# Add with: powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane cpu -Script <this file> -ScriptArgs "-Stage all -M 16"
# Imports big2 from THIS checkout (PYTHONPATH): do not rebuild python\big2\_core.pyd while it runs.
# Needs runs\p6_net\selfplay-h24.pknn, runs\p6_mistakes\belief-r1.pknn and runs\p6_ladder\pairs (gitignored).
param(
    [ValidateSet('play', 'luck', 'report', 'all')][string]$Stage = 'all',
    [int]$M = 16,
    [int]$Threads = 0
)
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = "$repo\python"
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:HKQ_THREADS) { $env:HKQ_THREADS = '8' }
if ($Threads -gt 0) { $env:HKQ_THREADS = "$Threads" }
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
& $py "$repo\experiments\grading_rating.py" $Stage --m $M
exit $LASTEXITCODE
