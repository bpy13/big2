"""Game-agnostic grading tools (plan phase 6): the Elo fit from multi-player games
(:mod:`big2.teach.grading.elo`), loss distributions per band (:mod:`.losses`), grade lines
and the trivial flag (:mod:`.lines`), APL and accuracy (:mod:`.accuracy`) and the
luck-removed rating from points won (:mod:`.rating`). Nothing here imports from the rest of
``big2``: the other games write the same file shapes and reuse this package as it is.
"""
