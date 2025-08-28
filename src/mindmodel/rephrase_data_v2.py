from __future__ import annotations

import json
import random
import re
from typing import Dict, List

random.seed(42)

# Improved rephrasing templates with better variable handling
REPHRASE_TEMPLATES = {
	"numbers": [
		"{company} announced {amount} million in revenue for {quarter} {year}.",
		"{company} posted {amount} million in {quarter} {year} revenue.",
		"Revenue at {company} reached {amount} million during {quarter} {year}.",
		"{company} disclosed {amount} million in {quarter} {year} earnings.",
		"{company} reported earnings of {amount} million for {quarter} {year}.",
	],
	"negation": [
		"The {company} board rejected {what} at Monday's meeting.",
		"{company}'s board turned down {what} during the Monday session.",
		"During Monday's meeting, {company} board members declined {what}.",
		"The {company} board voted against {what} on Monday.",
		"{company} board decided not to approve {what} on Monday.",
	],
	"modality": [
		"Early findings indicate {drug} might {verb} {what} {cond}.",
		"Initial research shows {drug} could {verb} {what} {cond}.",
		"Study results suggest {drug} potentially {verb}s {what} {cond}.",
		"Research indicates {drug} may {verb} {what} {cond}.",
		"Clinical trials suggest {drug} might {verb} {what} {cond}.",
	],
	"conditional": [
		"Should weather conditions remain favorable, the match will start at {time}.",
		"Assuming good weather, the match will begin at {time}.",
		"Provided the weather stays clear, the match will commence at {time}.",
		"With favorable weather, the match will start at {time}.",
		"If weather permits, the match will begin at {time}.",
	],
	"dates": [
		"The launch has been set for {utc} on {date}.",
		"Launch time is scheduled for {utc} on {date}.",
		"The rocket will launch at {utc} on {date}.",
		"Launch operations are planned for {utc} on {date}.",
		"The mission will launch at {utc} on {date}.",
	],
}

def extract_vars_improved(original: str, phenomena: List[str]) -> Dict:
	"""Improved variable extraction with regex patterns."""
	vars = {}
	
	if "numbers" in phenomena:
		# Extract company, amount, quarter, year using regex
		company_match = re.search(r'(\w+) reported', original)
		if company_match:
			vars["company"] = company_match.group(1)
		
		amount_match = re.search(r'\$([\d.]+) million', original)
		if amount_match:
			vars["amount"] = amount_match.group(1)
		
		quarter_match = re.search(r'(Q[1-4])', original)
		if quarter_match:
			vars["quarter"] = quarter_match.group(1)
		
		year_match = re.search(r'(\d{4})', original)
		if year_match:
			vars["year"] = year_match.group(1)
	
	elif "negation" in phenomena:
		# Extract company and what
		company_match = re.search(r'board at (\w+)', original)
		if company_match:
			vars["company"] = company_match.group(1)
		
		what_match = re.search(r'(proposal|budget|plan)', original)
		if what_match:
			vars["what"] = what_match.group(1)
	
	elif "modality" in phenomena:
		# Extract drug, verb, what, cond
		drug_match = re.search(r'(the (?:drug|therapy|vaccine))', original)
		if drug_match:
			vars["drug"] = drug_match.group(1)
		
		verb_match = re.search(r'(reduce|improve|increase)', original)
		if verb_match:
			vars["verb"] = verb_match.group(1)
		
		what_match = re.search(r'(blood pressure|memory|endurance)', original)
		if what_match:
			vars["what"] = what_match.group(1)
		
		cond_match = re.search(r'(in (?:adults|teens|patients with mild symptoms))', original)
		if cond_match:
			vars["cond"] = cond_match.group(1)
	
	elif "conditional" in phenomena:
		# Extract time
		time_match = re.search(r'(\d{1,2} (?:am|pm) on \d{1,2} \w{3} \d{4})', original)
		if time_match:
			vars["time"] = time_match.group(1)
	
	elif "dates" in phenomena:
		# Extract UTC and date
		utc_match = re.search(r'(\d{2}:\d{2}) UTC', original)
		if utc_match:
			vars["utc"] = utc_match.group(1) + " UTC"
		
		date_match = re.search(r'(\d{1,2} \w{3} \d{4})', original)
		if date_match:
			vars["date"] = date_match.group(1)
	
	return vars

def rephrase_sentence_improved(original: str, target: str, phenomena: List[str]) -> str:
	"""Rephrase the input sentence with improved variable extraction."""
	
	if not phenomena:
		return original
	
	# Get variables from original
	vars = extract_vars_improved(original, phenomena)
	
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
	with open("data/synth/test.jsonl", "r") as f:
		lines = f.readlines()
	
	# Rephrase each line
	rephrased_lines = []
	for line in lines:
		data = json.loads(line)
		original_input = data["input"]
		target = data["target"]
		phenomena = data.get("phenomena_tags", [])
		
		# Rephrase input
		rephrased_input = rephrase_sentence_improved(original_input, target, phenomena)
		
		# Create new data point
		new_data = {
			"input": rephrased_input,
			"target": target,  # Keep same target
			"length_tag": data.get("length_tag", "<LEN_18>"),
			"phenomena_tags": phenomena
		}
		
		rephrased_lines.append(json.dumps(new_data, ensure_ascii=False))
	
	# Write rephrased data
	with open("data/synth/test_rephrased_v2.jsonl", "w") as f:
		for line in rephrased_lines:
			f.write(line + "\n")
	
	print(f"Rephrased {len(rephrased_lines)} examples with improved templates")

if __name__ == "__main__":
	main()
