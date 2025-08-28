from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict


@dataclass
class Example:
	input: str
	target: str
	length_tag: Optional[str] = None
	phenomena_tags: Optional[List[str]] = None


class JsonlDataset:
	"""Simple JSONL dataset loader for conclusion generation.

	Each line must include at least: input, target. Optional: length_tag, phenomena_tags.
	"""

	def __init__(self, path: str) -> None:
		self.path = path
		self._data: List[Example] = []
		self._load()

	def _load(self) -> None:
		with open(self.path, "r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				obj: Dict = json.loads(line)
				self._data.append(
					Example(
						input=obj["input"],
						target=obj.get("target", ""),
						length_tag=obj.get("length_tag"),
						phenomena_tags=obj.get("phenomena_tags"),
					)
				)

	def __len__(self) -> int:
		return len(self._data)

	def __iter__(self) -> Iterator[Example]:
		return iter(self._data)

	def to_hf_dataset(self):
		from datasets import Dataset

		records = []
		for ex in self._data:
			records.append(
				{
					"input": ex.input,
					"target": ex.target,
					"length_tag": ex.length_tag or "",
					"phenomena_tags": ex.phenomena_tags or [],
				}
			)
		return Dataset.from_list(records)
