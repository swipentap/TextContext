from __future__ import annotations

import json
import random
from typing import Dict, List

# Rephrasing templates for different phenomena
REPHRASE_TEMPLATES = {
	"numbers": [
		"{company} announced {amount} million in revenue for {quarter} {year}.",
		"{company} posted {amount} million in {quarter} {year} revenue.",
		"Revenue at {company} reached {amount} million during {quarter} {year}.",
		"{company} disclosed {amount} million in {quarter} {year} earnings.",
	],
	"negation": [
		"The {company} board rejected {what} at Monday's meeting.",
		"{company}'s board turned down {what} during the Monday session.",
		"During Monday's meeting, {company} board members declined {what}.",
		"The {company} board voted against {what} on Monday.",
	],
	"modality": [
		"Early findings indicate {drug} might {verb} {what} {cond}.",
		"Initial research shows {drug} could {verb} {what} {cond}.",
		"Study results suggest {drug} potentially {verb}s {what} {cond}.",
		"Research indicates {drug} may {verb} {what} {cond}.",
	],
	"conditional": [
		"Should weather conditions remain favorable, the match will start at {time}.",
		"Assuming good weather, the match will begin at {time}.",
		"Provided the weather stays clear, the match will commence at {time}.",
		"With favorable weather, the match will start at {time}.",
	],
	"dates": [
		"The launch has been set for {utc} on {date}.",
		"Launch time is scheduled for {utc} on {date}.",
		"The rocket will launch at {utc} on {date}.",
		"Launch operations are planned for {utc} on {date}.",
	],
}

def extract_vars(original: str, phenomena: List[str]) -> Dict:
	"""Extract variables from original sentence for rephrasing."""
	vars = {}
	
	if "numbers" in phenomena:
		# Extract company, amount, quarter, year
		parts = original.split()
		for i, part in enumerate(parts):
			if part.endswith("reported"):
				vars["company"] = parts[i-1] if i > 0 else "Acme"
			elif part.startswith("$") and "million" in original:
				vars["amount"] = part[1:]
			elif part.startswith("Q") and part[1].isdigit():
				vars["quarter"] = part
			elif part.isdigit() and len(part) == 4:
				vars["year"] = part
	
	elif "negation" in phenomena:
		# Extract company and what
		parts = original.split()
		for i, part in enumerate(parts):
			if part == "at":
				vars["company"] = parts[i-1] if i > 0 else "Acme"
			elif part in ["proposal", "budget", "plan"]:
				vars["what"] = part
	
	elif "modality" in phenomena:
		# Extract drug, verb, what, cond
		parts = original.split()
		for i, part in enumerate(parts):
			if part in ["the", "drug", "therapy", "vaccine"]:
				vars["drug"] = "the " + parts[i+1] if i+1 < len(parts) else "the drug"
			elif part in ["reduce", "improve", "increase"]:
				vars["verb"] = part
			elif part in ["blood", "memory", "endurance"]:
				vars["what"] = part
			elif part in ["adults", "teens", "patients"]:
				vars["cond"] = "in " + part
	
	elif "conditional" in phenomena:
		# Extract time
		parts = original.split()
		for i, part in enumerate(parts):
			if "pm" in part or "am" in part:
				time_parts = parts[i:i+4]  # "3 pm on 20 Nov 2020"
				vars["time"] = " ".join(time_parts)
	
	elif "dates" in phenomena:
		# Extract UTC and date
		parts = original.split()
		for i, part in enumerate(parts):
			if "UTC" in part:
				vars["utc"] = parts[i-1] + " UTC"
				date_parts = parts[i+2:i+5]  # "24 Apr 2027"
				vars["date"] = " ".join(date_parts)
	
	return vars

def rephrase_sentence(original: str, target: str, phenomena: List[str]) -> str:
	"""Rephrase the input sentence while keeping the same meaning."""
	
	if not phenomena:
		return original
	
	# Get variables from original
	vars = extract_vars(original, phenomena)
	
	# Choose appropriate template
	phenomenon = phenomena[0] if phenomena else "numbers"
	templates = REPHRASE_TEMPLATES.get(phenomenon, [original])
	
	# Fill template with extracted variables
	try:
		rephrased = random.choice(templates).format(**vars)
		return rephrased
	except KeyError:
		# Fallback to original if template filling fails
		return original

def main():
	# Read original data
	with open("data/synth/test_rephrased.jsonl", "r") as f:
		lines = f.readlines()
	
	# Rephrase each line
	rephrased_lines = []
	for line in lines:
		data = json.loads(line)
		original_input = data["input"]
		target = data["target"]
		phenomena = data.get("phenomena_tags", [])
		
		# Rephrase input
		rephrased_input = rephrase_sentence(original_input, target, phenomena)
		
		# Create new data point
		new_data = {
			"input": rephrased_input,
			"target": target,  # Keep same target
			"length_tag": data.get("length_tag", "<LEN_18>"),
			"phenomena_tags": phenomena
		}
		
		rephrased_lines.append(json.dumps(new_data, ensure_ascii=False))
	
	# Write rephrased data
	with open("data/synth/test_rephrased.jsonl", "w") as f:
		for line in rephrased_lines:
			f.write(line + "\n")
	
	print(f"Rephrased {len(rephrased_lines)} examples")

if __name__ == "__main__":
	main()
