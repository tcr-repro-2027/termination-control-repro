**One-for-one target replacement. Input, record count and per-record block count are held fixed; only which target blocks are out-of-candidate moves. 24.3% is in Appendix A.**

| condition | replacement_rate | train_seed | records | target_blocks | continuous_pattern_repetition | stable_token_orbit | hit_context_limit | entity_pair_f1 | out_of_candidate_blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OBR 0% s42 | 0 | 42 | 8854 | 5.85e+05 | 0.05515 | 0.00599 | 0.01142 | 0.3389 | 0.02735 |
| OBR 5% s42 | 0.05 | 42 | 8854 | 5.85e+05 | 0.0495 | 0.00599 | 0.007572 | 0.335 | 0.06979 |
| OBR 10% s42 | 0.1 | 42 | 8854 | 5.85e+05 | 0.07369 | 0.01074 | 0.01401 | 0.3258 | 0.101 |
| OBR 15% s42 | 0.15 | 42 | 8854 | 5.85e+05 | 0.09708 | 0.01356 | 0.0243 | 0.3188 | 0.134 |
| OBR 0% s123 | 0 | 123 | 8854 | 5.85e+05 | 0.04396 | 0.005086 | 0.009494 | 0.3389 | 0.02504 |
| OBR 15% s123 | 0.15 | 123 | 8854 | 5.85e+05 | 0.09471 | 0.01311 | 0.01718 | 0.3224 | 0.1404 |
