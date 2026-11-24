# CursorMix evaluation results

On the original recorded evaluation, Claude and the local model each routed 0 of 76 classified negative requests to desktop actions. Claude advanced on 0 of 17 unrelated changes in teaching recordings. The local model advanced on 2 of 18. Human cross-session routing error is unmeasured because the personal commands came from one recording session.

The table summarizes a full evaluation on an earlier private revision. This release contains only the public subset of the original test material. These are historical measurements.

| Capability | Claude | Local |
| --- | --- | --- |
| Screen questions | 44/50 correct | 26/50 correct |
| Pointing | 92/100 centers inside target | 79/100 centers inside target |
| Teaching | 6/24 steps verified across 3 workflows | 1/27 steps verified across 3 workflows |
| Full routing set | 99/100 correct | 89/100 correct |
| First visible response | 50/50 valid requests | 49/50 valid requests |
| Quick-answer latency (median / p95) | 423 ms / 1,250 ms | 183 ms / 366 ms |
| Screen-answer latency (median) | 988 ms | 554 ms |

Five Claude pointers missed their target and three returned no center. The teaching gate required ten workflows and at least 80% verified steps. The evaluated build lacked paired live desktop action trials. A locked-desktop check observed zero captures and zero image writes, which did not establish successful capture when summoned on an unlocked Mac.

See [public fixture provenance](catalog/README.md) and [how to run checks](MANUAL.md).
