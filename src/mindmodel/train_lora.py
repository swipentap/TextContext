from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import tyro
from datasets import load_dataset
from transformers import (
	AutoTokenizer,
	AutoModelForSeq2SeqLM,
	DataCollatorForSeq2Seq,
	Seq2SeqTrainingArguments,
	Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class Args:
	train_path: str
	valid_path: str
	output_dir: str = "runs/flan-t5-lora"
	base_model: str = "google/flan-t5-base"
	max_source_len: int = 512
	max_target_len: int = 24
	batch_size: int = 8
	epochs: int = 3
	lr: float = 2e-4
	warmup_ratio: float = 0.03
	gradient_accumulation_steps: int = 2
	seed: int = 42
	length_tag: Optional[str] = None


def _format_example(example, length_tag: Optional[str]):
	prompt = "You are a faithful conclusion generator.\nRule: Output one short, self-contained conclusion entailed by the input. No new facts.\n\nInput: " + example["input"] + "\nConclusion:"
	if length_tag:
		prompt = f"{length_tag} " + prompt
	return {
		"input_ids": prompt,
		"labels": example.get("target", ""),
	}


def main(args: Args) -> None:
	os.makedirs(args.output_dir, exist_ok=True)

	ds_train = load_dataset("json", data_files=args.train_path, split="train")
	ds_valid = load_dataset("json", data_files=args.valid_path, split="train")

	tokenizer = AutoTokenizer.from_pretrained(args.base_model)
	model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

	peft_config = LoraConfig(
		task_type=TaskType.SEQ_2_SEQ_LM,
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		target_modules=["q", "v"],
	)
	model = get_peft_model(model, peft_config)

	def preprocess(batch):
		inputs = []
		targets = []
		for inp, tgt, tag in zip(batch["input"], batch.get("target", [""] * len(batch["input"])), batch.get("length_tag", [""] * len(batch["input"]))):
			lt = args.length_tag or (tag if tag else None)
			formatted = _format_example({"input": inp, "target": tgt}, lt)
			inputs.append(formatted["input_ids"]) 
			targets.append(formatted["labels"]) 
		model_inputs = tokenizer(
			inputs,
			max_length=args.max_source_len,
			truncation=True,
			padding=False,
		)
		with tokenizer.as_target_tokenizer():
			labels = tokenizer(
				targets,
				max_length=args.max_target_len,
				truncation=True,
				padding=False,
			)
		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	ds_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names)
	ds_valid = ds_valid.map(preprocess, batched=True, remove_columns=ds_valid.column_names)

	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

	training_args = Seq2SeqTrainingArguments(
		output_dir=args.output_dir,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		evaluation_strategy="steps",
		eval_steps=500,
		save_steps=500,
		num_train_epochs=args.epochs,
		learning_rate=args.lr,
		warmup_ratio=args.warmup_ratio,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		predict_with_generate=True,
		logging_steps=50,
		load_best_model_at_end=True,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		seed=args.seed,
		bf16=True,
	)

	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=ds_train,
		eval_dataset=ds_valid,
		data_collator=data_collator,
		tokenizer=tokenizer,
	)

	trainer.train()
	trainer.save_model(args.output_dir)
	print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
	main(tyro.cli(Args))
