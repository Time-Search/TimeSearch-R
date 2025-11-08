#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
submission_path=./ours-ft-gqa-stage4-2epoch-1e-4-64-preds.jsonl
gt_path=./rextime_test.jsonl
save_path=./sample_output.json

PYTHONPATH=$PYTHONPATH:. python rextime_eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}