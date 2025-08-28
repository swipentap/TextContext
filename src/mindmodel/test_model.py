from __future__ import annotations

import json
import argparse
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.mindmodel.data import JsonlDataset


def generate_conclusion(model, tokenizer, text: str, max_new_tokens: int = 24) -> str:
	prompt = (
		"You are a faithful conclusion generator.\n"
		"Rule: Output one short, self-contained conclusion entailed by the input. No new facts.\n\n"
		f"Input: {text}\nConclusion:"
	)
	inputs = tokenizer(prompt, return_tensors="pt")
	outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
	return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_model_on_data(model_path: str, data_path: str, max_samples: int = 100) -> List[Tuple[str, str, str, bool]]:
	"""Test model on data and return (input, target, prediction, is_correct) tuples."""
	
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
	
	dataset = JsonlDataset(data_path)
	results = []
	
	for i, example in enumerate(dataset):
		if i >= max_samples:
			break
			
		prediction = generate_conclusion(model, tokenizer, example.input)
		# Simple exact match for now - could be enhanced with NLI scoring
		is_correct = prediction.strip().lower() == example.target.strip().lower()
		
		results.append((example.input, example.target, prediction, is_correct))
		
		if i < 10:  # Print first 10 examples
			print(f"\nExample {i+1}:")
			print(f"Input: {example.input}")
			print(f"Target: {example.target}")
			print(f"Prediction: {prediction}")
			print(f"Correct: {is_correct}")
	
	return results


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="runs/t5-small-synth")
	parser.add_argument("--data", type=str, default="data/synth/test.jsonl")
	parser.add_argument("--max_samples", type=int, default=100)
	args = parser.parse_args()
	
	results = test_model_on_data(args.model, args.data, args.max_samples)
	
	correct = sum(1 for _, _, _, is_correct in results if is_correct)
	total = len(results)
	accuracy = correct / total if total > 0 else 0
	
	print(f"\n{'='*50}")
	print(f"Results: {correct}/{total} correct ({accuracy:.1%})")
	print(f"{'='*50}")


if __name__ == "__main__":
	main()
