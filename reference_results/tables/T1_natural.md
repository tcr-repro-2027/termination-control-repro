**Raw versus entity-constraint-cleaned supervision. Rates are over all responses; F1 is pooled micro-F1.**

| scale | condition | train_seed | n_responses | continuous_pattern_repetition | stable_token_orbit | hit_context_limit | gen_tokens_mean | triple_f1 | entity_pair_f1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | no SFT | -- | 8848 | 0.4852 | 0.1759 | 0.4509 | 1.459e+04 | 0.0168 | 0.1045 |
| 1.7B | entity-constraint cleaned | 42 | 8848 | 0.2595 | 0.06499 | 0.1803 | 8685 | 0.0651 | 0.2577 |
| 1.7B | raw | 42 | 8848 | 0.6058 | 0.2288 | 0.5461 | 1.802e+04 | 0.0522 | 0.19 |
| 4B | no SFT | -- | 8848 | 0.01288 | 0.0003391 | 0.0003391 | 3024 | 0.0449 | 0.2323 |
| 4B | entity-constraint cleaned | 42 | 8848 | 0.05515 | 0.00599 | 0.01142 | 4734 | 0.0907 | 0.3389 |
| 4B | raw | 42 | 8848 | 0.323 | 0.07041 | 0.2106 | 1.1e+04 | 0.0822 | 0.2913 |
| 4B | entity-constraint cleaned | 123 | 8848 | 0.04396 | 0.005086 | 0.009494 | 4639 | 0.0924 | 0.3389 |
| 4B | raw | 123 | 8848 | 0.3337 | 0.08476 | 0.2205 | 1.116e+04 | 0.0816 | 0.2918 |
| 8B | no SFT | -- | 8848 | 0.1027 | 0.01017 | 0.07934 | 5514 | 0.0409 | 0.2228 |
| 8B | entity-constraint cleaned | 42 | 8848 | 0.07132 | 0.005086 | 0.01831 | 5166 | 0.0974 | 0.3539 |
| 8B | raw | 42 | 8848 | 0.3178 | 0.06205 | 0.2587 | 1.184e+04 | 0.0866 | 0.2978 |
