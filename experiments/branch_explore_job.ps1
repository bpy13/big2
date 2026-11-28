# Queue job for plan Task 50 (branch exploration of the qualifying deal-seats).
#   -Stage explore (cpu lane): grow each deal-seat's tree into <Out>\trees.jsonl; resumes
#                  (a deal-seat already written is skipped)
#   -Stage report  : <Out>\report.md, report.json (coverage at caps 20/40/80; seconds)
#   -Pilot <N>: only N deal-seats spread over the bands (0 = all 2,046); -Cap <N> graded
#   decisions a deal-seat (main line included); -Out <dir> (default runs\p6_branches\pilot)
# Pilot: -Stage explore -Pilot 100 -Cap 80 -Out runs\p6_branches\pilot
# Full:  -Stage explore -Cap 40 -Out runs\p6_branches\full
# Add with: powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane cpu -Script <this file> -ScriptArgs "-Stage explore -Pilot 100 -Cap 80"
# Imports big2 from THIS checkout (PYTHONPATH): do not rebuild python\big2\_core.pyd while it runs.
# Needs runs\p6_mistakes\decisions-<band>.jsonl (Task 42), runs\p6_mistakes\belief-r1.pknn and
# runs\p6_net\selfplay-h24.pknn (gitignored).
param(
    [ValidateSet('explore', 'report')][string]$Stage = 'explore',
    [int]$Pilot = 0,
    [int]$Cap = 40,
    [string]$Out = 'runs\p6_branches\pilot',
    [int]$Threads = 0
)
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = "$repo\python"
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:HKQ_THREADS) { $env:HKQ_THREADS = '8' }
if ($Threads -gt 0) { $env:HKQ_THREADS = "$Threads" }
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
& $py "$repo\experiments\branch_explore.py" $Stage --out $Out --pilot $Pilot --cap $Cap
exit $LASTEXITCODE
