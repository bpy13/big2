# Queue job for plan Tasks 42a / 42 (mistake mining).
#   Task 42a (pilot, netsearch reference):
#   -Stage label    (cpu lane): same-rung deals per band, every decision labelled with the
#                   reference (k=8, n=256, full depth) and the knee (k=3, n=64) into
#                   <Out>\decisions.jsonl; resumes (a deal already written is not replayed)
#   -Stage puzzles  : <Out>\candidates.json from decisions.jsonl (seconds)
#   -Stage all      : label, puzzles, summary (the default)
#   Task 42 (PUCT-2000 reference, belief net; decisions 0010, 0011):
#   -Stage label42  (cpu lane): -Deals same-rung deals per band into <Out>\decisions-<band>.jsonl;
#                   resumes, so a later job with a larger -Deals adds the rest
#   -Stage report42 : <Out>\report.md, table.json, hist-<band>.svg/png (seconds)
#   -Stage review36 : the 36 hk-games review puzzles regraded (<Out>\review36-regraded.json)
#   -Deals <N>: deals per band (default 100); -Out <dir> (default runs\p6_mistakes_pilot)
# Add with: powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane cpu -Script <this file> -ScriptArgs "-Stage all"
# Imports big2 from THIS checkout (PYTHONPATH): do not rebuild python\big2\_core.pyd while it runs.
# Task 42 needs runs\p6_net\selfplay-h24.pknn and runs\p6_mistakes\belief-r1.pknn (gitignored).
param(
    [ValidateSet('label', 'puzzles', 'summary', 'all', 'label42', 'report42', 'review36')][string]$Stage = 'all',
    [int]$Deals = 100,
    [string]$Out = 'runs\p6_mistakes_pilot',
    [int]$Threads = 0
)
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = "$repo\python"
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:HKQ_THREADS) { $env:HKQ_THREADS = '8' }
if ($Threads -gt 0) { $env:HKQ_THREADS = "$Threads" }
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
& $py "$repo\experiments\mine_mistakes.py" $Stage --out $Out --deals $Deals
exit $LASTEXITCODE
