from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import List

import tyro


random.seed(13)


NAMES = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry"]
COMPANIES = ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Hooli"]
MONTHS = [
	("Jan", 31), ("Feb", 28), ("Mar", 31), ("Apr", 30), ("May", 31), ("Jun", 30),
	("Jul", 31), ("Aug", 31), ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31),
]


def rand_date(year_lo=2020, year_hi=2030):
	m, days = random.choice(MONTHS)
	d = random.randint(1, days)
	y = random.randint(year_lo, year_hi)
	return f"{d} {m} {y}"


def make_numbers_case():
	company = random.choice(COMPANIES)
	amount = round(random.uniform(0.5, 9.9), 1)
	quarter = random.choice(["Q1", "Q2", "Q3", "Q4"])
	year = random.randint(2020, 2030)
	inp = f"{company} reported revenue of ${amount} million in {quarter} {year}."
	tgt = f"{company} reported ${amount} million in {quarter} {year}."
	return inp, tgt, ["numbers"]


def make_negation_case():
	company = random.choice(COMPANIES)
	what = random.choice(["the proposal", "the budget", "the plan"])
	inp = f"The board at {company} did not approve {what} during Monday's meeting."
	tgt = f"The board at {company} did not approve {what}."
	return inp, tgt, ["negation"]


def make_modality_case():
	drug = random.choice(["the drug", "the therapy", "the vaccine"])
	cond = random.choice(["in adults", "in teens", "in patients with mild symptoms"])
	verb = random.choice(["reduce", "improve", "increase"])
	what = random.choice(["blood pressure", "memory", "endurance"])
	inp = f"Preliminary data suggests {drug} may {verb} {what} {cond}."
	tgt = f"{drug.capitalize()} may {verb} {what} {cond}."
	return inp, tgt, ["modality"]


def make_conditional_case():
	time = random.choice(["3 pm", "10:30 am", "7:45 pm"]) + " on " + rand_date()
	inp = f"If the weather holds, the match will start at {time}."
	tgt = f"The match will start at {time} if weather holds."
	return inp, tgt, ["conditional", "time"]


def make_dates_case():
	launch = rand_date(2025, 2030)
	utc = random.choice(["07:30", "18:05", "12:00"]) + " UTC"
	inp = f"The launch is scheduled for {utc} on {launch}."
	tgt = f"The launch is scheduled for {utc} on {launch}."
	return inp, tgt, ["dates", "time"]


GENERATORS = [
	make_numbers_case,
	make_negation_case,
	make_modality_case,
	make_conditional_case,
	make_dates_case,
]


@dataclass
class Args:
	output: str
	n_examples: int = 10000
	length_tag: str = "<LEN_18>"
	seed: int = 13


def main(args: Args) -> None:
	random.seed(args.seed)
	with open(args.output, "w", encoding="utf-8") as f:
		for _ in range(args.n_examples):
			inp, tgt, tags = random.choice(GENERATORS)()
			obj = {
				"input": inp,
				"target": tgt,
				"length_tag": args.length_tag,
				"phenomena_tags": tags,
			}
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
	main(tyro.cli(Args))
