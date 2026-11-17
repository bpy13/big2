# Queue job for plan Task 34 (the win-chance bot): one stage per queue job.
#   -Stage label  (cpu lane): win-objective labels for rounds 0-3 (ISMCTS-1000 for round 0,
#                 ISMCTS-5000 for rounds 1-3), runs\round<N>_win\data
#   -Stage train  (gpu lane): networks r0-r3 of the win pipeline, runs\net_r<N>_win, each with
#                 its calibration and .pknn export
#   -Stage match  (cpu lane): experiments\win_chance.py final habits (10,000 duplicate deals
#                 vs `expert`, then the 1,000-deal habits pass), runs\win_chance
# Add with: powershell -NoProfile -File C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add -Lane <lane> -Script <this file> -ScriptArgs "-Stage <stage>" [-After '<ids>']
# Every stage resumes when added again (shards are counted, training continues from latest.pt,
# finished rounds are skipped, match blocks are reused).
# Check: Get-Content runs\round<N>_win\label\progress.json, runs\net_r<N>_win\progress.json,
#        runs\win_chance\progress.json (all in this worktree).
# It imports big2 from THIS worktree (PYTHONPATH): the conda env's big2 points at the
# standalone checkout, which lacks the objective switch. Do not rebuild python\big2\_core.pyd
# here while a stage runs.
param([Parameter(Mandatory = $true)][ValidateSet('label', 'train', 'match')][string]$Stage)
$repo = 'C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\big2-workers'
$threads = if ($env:HKQ_THREADS) { [int]$env:HKQ_THREADS } else { 8 }
Set-Location $repo
$env:PYTHONPATH = "$repo\python"
$py = 'C:\Users\brian\miniconda3\envs\big2\python.exe'
$ei = "$repo\experiments\expert_iteration.py"
switch ($Stage) {
    'label' {
        foreach ($n in 0..3) {
            & $py $ei round $n --objective win --stages label --workers $threads
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
    'train' {
        foreach ($n in 0..3) {
            & $py $ei round $n --objective win --stages train --workers $threads --device cuda
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
    'match' {
        & $py "$repo\experiments\win_chance.py" final habits --workers $threads
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
exit 0
