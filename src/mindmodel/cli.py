from __future__ import annotations

import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_conclusion(model_name_or_path: str, text: str, max_new_tokens: int = 24, length_tag: str | None = None) -> str:
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
	prompt = "You are a faithful conclusion generator.\nRule: Output one short, self-contained conclusion entailed by the input. No new facts.\n\nInput: " + text + "\nConclusion:"
	if length_tag:
		prompt = f"{length_tag} " + prompt
	inputs = tokenizer(prompt, return_tensors="pt")
	outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
	return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("text", type=str, help="Input text")
	parser.add_argument("--model", type=str, default="google/flan-t5-base")
	parser.add_argument("--max_new_tokens", type=int, default=24)
	parser.add_argument("--length_tag", type=str, default=None)
	args = parser.parse_args()
	print(generate_conclusion(args.model, args.text, args.max_new_tokens, args.length_tag))


if __name__ == "__main__":
	main()
