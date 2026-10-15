# Reproduction margin

Three separate reruns are disclosed here, none of which replaces a published number.

1. `rerun_full_grid.jsonl` — every cell of the grid retrained by the published code path while dumping its predictions (`scripts/dump_all_predictions.py`). Each row carries the published and the reproduced accuracy and F1@50. Training reproduces to about four significant figures; the Viterbi decode does not always return the same path when two paths score within float noise, which is where the remaining differences come from.

2. `discarded_batchsize2_results.jsonl` — a grid run at `--batch-size 2` by mistake. It reproduced only 6 of 39 cells and is archived, not used: accuracy moved 0.436 to 0.478 and one cell swung from 0.893 to 0.036. The published grid is batch-size 1, and that is now the code's default.

3. The fixed-soft arm was run in a later session than the rest. Re-running the rigid arm under identical settings reproduced 32 of 39 cells exactly and moved its F1@50 by 0.007, which bounds the comparability of that row and is stated in the paper.
