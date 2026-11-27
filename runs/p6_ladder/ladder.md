| rung | kind | setting | Elo | 95 % range | deals |
|---|---|---|---|---|---|
| `random` | ladder | `random` | 955 | [949, 960] | 2900 |
| `net-t4.34` | ladder | `netprior:net=<net>,temperature=4.34` | 1096 | [1088, 1102] | 1000 |
| `easy` | old-level | `greedy:structured=1,epsilon=0.1` | 1161 | [1156, 1165] | 3300 |
| `medium` | old-level | `qnet:temperature=1.0` | 1200 | [1200, 1200] | 2400 |
| `hard` | old-level | `ismcts:iterations=500` | 1222 | [1216, 1228] | 800 |
| `net-t1` | ladder | `netprior:net=<net>,temperature=1` | 1230 | [1225, 1234] | 2900 |
| `expert` | old-level | `ismcts:iterations=10000` | 1236 | [1231, 1242] | 2200 |
| `net` | ladder | `netprior:net=<net>,temperature=0` | 1248 | [1243, 1252] | 2500 |

| lower | upper | gap | 95 % range | deals | within ±25 |
|---|---|---|---|---|---|
| `random` | `net-t4.34` | +140 | [133, 148] | 200 | yes |
| `net-t4.34` | `easy` | +65 | [58, 72] | 200 | yes |
| `easy` | `medium` | +39 | [35, 44] | 500 | yes |
| `medium` | `hard` | +22 | [16, 28] | 200 | yes |
| `hard` | `net-t1` | +8 | [2, 15] | 200 | yes |
| `net-t1` | `expert` | +6 | [2, 11] | 500 | yes |
| `expert` | `net` | +11 | [6, 16] | 500 | yes |

| pairing | deals | thread-h |
|---|---|---|
| `easy` vs `expert` | 500 | 0.87 |
| `easy` vs `hard` | 200 | 0.01 |
| `easy` vs `medium` | 500 | 0.01 |
| `easy` vs `net` | 500 | 0.00 |
| `easy` vs `net-t1` | 500 | 0.00 |
| `easy` vs `net-t2.66` | 200 | 0.00 |
| `easy` vs `net-t4.34` | 200 | 0.00 |
| `easy` vs `net-t7.07` | 200 | 0.00 |
| `easy` vs `random` | 500 | 0.00 |
| `expert` vs `hard` | 200 | 0.37 |
| `expert` vs `net` | 500 | 0.87 |
| `expert` vs `net-t1` | 500 | 0.86 |
| `expert` vs `random` | 500 | 1.04 |
| `hard` vs `medium` | 200 | 0.02 |
| `hard` vs `net-t1` | 200 | 0.02 |
| `medium` vs `net` | 500 | 0.01 |
| `medium` vs `net-t1` | 500 | 0.01 |
| `medium` vs `net-t4.34` | 200 | 0.01 |
| `medium` vs `random` | 500 | 0.02 |
| `net-t1` vs `net-t4.34` | 200 | 0.00 |
| `net-t1` vs `random` | 500 | 0.00 |
| `net-t2.66` vs `net-t7.07` | 200 | 0.00 |
| `net-t4.34` vs `net-t7.07` | 200 | 0.00 |
| `net-t4.34` vs `random` | 200 | 0.00 |
| `net-t7.07` vs `random` | 200 | 0.00 |
| `net` vs `net-t1` | 500 | 0.01 |
| `net` vs `random` | 500 | 0.00 |

Total: 4.2 thread-hours, 38400 games on 9600 deals.
