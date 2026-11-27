# 0007: the hour-24 self-play network is the expert's network

Date: 2026-11-27

The network behind the Big2 expert (search on top of the network, Tasks 38-41, 45) is the self-play
checkpoint `runs/selfplay_luck/hour_24` (exported to .pknn by Task 37). Training plateaued from about
hour 8: no head-to-head between hours 8-24 is clear of luck at 2,000 deals, and hour 24 is the only
one leaning positive against both hour 20 and hour 8. The 2,000-deal final match against the old
expert was dropped (Brian: "we already know it's much stronger than expert").

Brian (2026-11-27), after the vs-expert table (hours 4-20 all about +3 a deal over the old ISMCTS-10000 expert) and the head-to-heads (hour 24 vs hour 20 +0.07/+0.40, vs hour 8 +0.25/+0.38, all ranges include zero): "agree, use hour 24 as the expert"
