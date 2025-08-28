#!/usr/bin/env bash
set -euo pipefail

python -m src.mindmodel.train_lora \
	--train_path data/train.jsonl \
	--valid_path data/valid.jsonl \
	--output_dir runs/flan-t5-lora \
	--max_target_len 24 \
	--batch_size 8 \
	--epochs 3
