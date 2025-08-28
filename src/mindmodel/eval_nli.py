from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import tyro
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


@dataclass
class Args:
	pairs_path: str  # JSONL with fields: input, target OR input, prediction
	pred_path: str = ""  # optional predictions JSONL with fields: input, prediction
	text_field_pred: str = "prediction"
	model_name: str = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
	batch_size: int = 16


def load_pairs(args: Args) -> List[dict]:
	pairs = []
	if args.pred_path:
		with open(args.pred_path, "r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				obj = json.loads(line)
				pairs.append({"premise": obj.get("input", obj.get("source", "")), "hypothesis": obj[args.text_field_pred]})
	else:
		with open(args.pairs_path, "r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				obj = json.loads(line)
				pairs.append({"premise": obj["input"], "hypothesis": obj.get("target", obj.get("prediction", ""))})
	return pairs


def main(args: Args) -> None:
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
	model.eval()

	pairs = load_pairs(args)

	labels = []
	scores = []
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	for i in range(0, len(pairs), args.batch_size):
		batch = pairs[i : i + args.batch_size]
		enc = tokenizer(
			[ex["premise"] for ex in batch],
			[ex["hypothesis"] for ex in batch],
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=512,
		)
		enc = {k: v.to(device) for k, v in enc.items()}
		with torch.no_grad():
			out = model(**enc)
			logits = out.logits
			probs = torch.softmax(logits, dim=-1)
			# Label mapping varies; many NLI models use [contradiction, neutral, entailment]
			ent_idx = -1
			scores.extend(probs[:, ent_idx].detach().cpu().tolist())
			labels.extend(torch.argmax(probs, dim=-1).detach().cpu().tolist())

	avg_entail = sum(scores) / max(1, len(scores))
	print(json.dumps({"avg_entailment": avg_entail, "num_pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
	main(tyro.cli(Args))
