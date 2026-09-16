# Eval summary: harness

1 model(s) · 56 runs · 21/56 passed (38%)

Scored by: 56 checker. Judge agreed with the symbolic checker on 56/56 code-scored runs (100%).

Each section is ONE model: success matrix (config × task), then the per-config roll-up sorted cheapest-first. `tok/reason~` is the trace-length ESTIMATE of reasoning tokens (the endpoint reports none); `tok/cache` is summarizer overhead; `fails` is rejected actions per run.

## `qwen/qwen3-32b`

56 runs · 21/56 passed (38%) · 7 configs × 4 tasks

| config           | flag_excavate (R1) | unbox_red (R2) | flag_repair (R3) | pyramid_rebuild (R4) |
| ---------------- | ------------------ | -------------- | ---------------- | -------------------- |
| oneshot_nocot    | 0/2                | 1/2            | 0/2              | 1/2                  |
| oneshot          | 1/2                | 0/2            | 0/2              | 0/2                  |
| act              | 0/2                | 0/2            | 0/2              | 0/2                  |
| react            | 0/2                | 2/2            | 2/2              | 2/2                  |
| react_summary    | 2/2                | 2/2            | 1/2              | 1/2                  |
| on_error_summary | 0/2                | 2/2            | 0/2              | 1/2                  |
| react_raw        | 0/2                | 2/2            | 0/2              | 1/2                  |

| config           | success | rate | tok/tot | tok/reason~ | tok/cache | reason/turns | steps | fails | lat(s) |
| ---------------- | ------- | ---- | ------- | ----------- | --------- | ------------ | ----- | ----- | ------ |
| oneshot_nocot    | 2/8     | 25%  | 3284    | 2265        | 0         | 0.0          | 7.8   | 3.8   | 62.3   |
| oneshot          | 1/8     | 12%  | 3788    | 2778        | 0         | 1.0          | 8.8   | 3.0   | 75.4   |
| react            | 6/8     | 75%  | 38620   | 19764       | 0         | 10.9         | 10.9  | 1.9   | 578.8  |
| act              | 0/8     | 0%   | 39936   | 0           | 0         | 0.0          | 19.9  | 8.9   | 32.9   |
| on_error_summary | 3/8     | 38%  | 52466   | 11139       | 11192     | 5.2          | 14.5  | 4.2   | 347.5  |
| react_summary    | 6/8     | 75%  | 78811   | 15072       | 23430     | 15.8         | 15.8  | 4.2   | 485.9  |
| react_raw        | 3/8     | 38%  | 200516  | 14933       | 0         | 17.9         | 17.9  | 7.0   | 549.5  |
