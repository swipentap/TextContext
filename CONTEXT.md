## Project Context: Minimal, Faithful Conclusions

### Problem Statement
Transform an input sentence or short passage into one short, self-contained conclusion that is strictly entailed by the input, concise, and precise (no hallucinations, no added facts).

- **Input**: sentence or short passage (≤ a few sentences)
- **Output**: single declarative sentence (typically ≤ 12–24 tokens) capturing the main implication
- **Constraints**: faithful to input, preserve numbers/entities/modalities, avoid lists or explanations

### Non-Goals
- Long-form summaries, topic overviews, or multi-bullet outputs
- Paraphrase-only rewriting without enforcing faithfulness
- Open-ended Q&A or knowledge augmentation beyond the input

## Scope and Requirements

### Functional Requirements
- Generate exactly one concise conclusion
- Maintain faithfulness: content must be entailed by the input
- Preserve critical facts: numbers, units, named entities, dates
- Respect uncertainty and modality present in the input (e.g., "may", "likely")
- Allow configurable brevity (e.g., target length tags)

### Quality Targets (initial)
- NLI-based entailment ≥ 0.80 on held-out set
- Compression ratio ≤ 0.25 on average
- Numeric/entity fidelity ≥ 0.98 exact-match rate

## Data Plan

### Sources
- Curated subsets of news summarization datasets (e.g., XSum, CNN/DM) filtered to single-sentence inputs
- NLI datasets (SNLI/MNLI) to create hard negatives (concise but not entailed)
- Synthetic pretraining via rule-based simplification (dependency parses) with human cleanup

### Annotation Guidelines (brief)
- Include only propositions strictly entailed by the input
- Prefer active voice; remove adjunct/circumstantial detail unless essential
- Resolve coreference when unambiguous
- Keep numbers, units, dates; avoid qualitative adjectives unless definitional

### Dataset Format
- JSONL fields: `input`, `target`, `length_tag`, `phenomena_tags` (negation, conditional, coordination, etc.)
- Split: train/validation/test with phenomenon stratification

## Modeling Approach

### Baseline
- Fine-tune a compact seq2seq model (e.g., `flan-t5-base`) with LoRA/PEFT
- Decoding controls: max tokens cap, low temperature, nucleus sampling (p≈0.8)

### Faithfulness Enhancements
- Auxiliary NLI head (frozen small DeBERTa/RoBERTa) to score entailment(input → output); integrate as regularizer during training
- Numeric/entity audits: regex/NER checks with loss for mismatches or omissions
- Hard negatives in training to reduce near-entailed hallucinations

### Brevity Control
- Length-control tokens (`<LEN_12>`, `<LEN_18>`, etc.)
- Length penalty during decoding

### Optional Two-Stage Variant
- Stage A: extract proposition candidates from dependencies/information extraction
- Stage B: rewrite best candidate to minimal conclusion

## Training

- Supervised fine-tuning with cross-entropy; enforce max target length
- Curriculum:
  1) Extractive main-clause selection
  2) Controlled rewrite (tense/voice normalization, adjunct removal)
  3) Abstractive minimality with length tags and hard negatives
- Optional RL fine-tuning (PPO) with reward: α·Entailment + β·Conciseness − γ·NumericMismatch − δ·Redundancy

## Evaluation

### Automatic
- NLI entailment P(entail|input, output)
- Compression ratio and length distribution
- Numeric/entity preservation audits (exact match)
- Clause count (ensure single-sentence outputs)

### Human
- Faithfulness (0–2), Brevity (0–2), Usefulness (0–2)
- Phenomenon-specific slices: negation, conditionals, coordination, apposition

### Benchmarks and Gates
- M1: Baseline T5 SFT reaches ≥0.80 entailment, ≤0.25 compression
- M2: Add length tags + hard negatives, maintain entailment, shorten outputs ≥15%
- M3: RL reward improves human faithfulness on tricky sets by ≥0.2

## Inference and UX

### Prompting Template (for instruct variants)
```
You are a faithful conclusion generator.
Rule: Output one short, self-contained conclusion entailed by the input. No new facts.

Input: {text}
Conclusion:
```

### Decoding Defaults
- Max tokens: 12–24
- Temperature: 0.2–0.5; Top-p: 0.8; repetition penalty enabled
- Output must be one sentence; strip list markers and preambles (e.g., "In conclusion,")

### Guardrails
- Preserve modality: if input is uncertain, conclusion must reflect uncertainty
- Safety-sensitive domains (medical/legal): avoid overstatement; only repeat explicit claims
- Optionally return evidence spans (character offsets) used to justify the conclusion

## Interfaces (future API sketch)

### REST
- `POST /v1/conclude` → `{ input: string, target_length?: number }`
- Response: `{ conclusion: string, length: number, evidence_spans?: [start, end] }`

### CLI
- `conclude "<text>" --len 18`

## Risks and Mitigations
- Hallucinations → NLI regularizer, hard negatives, numeric/entity audits
- Over-compression → length tags, minimum informative content check
- Modality loss → rule-based modality preservation checks during eval

## Glossary
- **Entailed**: Output is logically supported by the input; not contradicted and not adding new facts
- **Hard negative**: A plausible, concise output that is not actually entailed
- **Compression ratio**: `len(output) / len(input)`

## Milestones (condensed)
- M1: Baseline SFT + evaluator
- M2: Length control + hard negatives
- M3: RL reward tuning
- M4: Guardrails + evidence spans + API

## Model Availability and Scaffold

- Current: No trained checkpoint in this repo yet. We will scaffold a minimal training/eval setup and small CLI/API to enable quick iteration.
- Scaffold contents (to be added):
  - Dataset schema and loader (`src/mindmodel/data.py`)
  - LoRA fine-tuning for `flan-t5-base` (`src/mindmodel/train_lora.py`)
  - NLI-based entailment evaluator (`src/mindmodel/eval_nli.py`)
  - CLI for inference (`src/mindmodel/cli.py`)
  - FastAPI service at `POST /v1/conclude` (`api/main.py`)
  - Scripts for train/eval/serve (`scripts/*.sh`)

## Quickstart (after scaffold)

1) Install
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Prepare data (JSONL with fields: input, target, length_tag)
```
data/
  train.jsonl
  valid.jsonl
  test.jsonl
```

3) Train LoRA on flan-t5-base
```
python -m src.mindmodel.train_lora \
  --train_path data/train.jsonl \
  --valid_path data/valid.jsonl \
  --output_dir runs/flan-t5-lora \
  --max_target_len 24 \
  --batch_size 8 \
  --epochs 3
```

4) Evaluate entailment
```
python -m src.mindmodel.eval_nli \
  --pairs_path data/test.jsonl \
  --pred_path runs/flan-t5-lora/preds.jsonl
```

5) CLI inference
```
python -m src.mindmodel.cli --model runs/flan-t5-lora "The trial starts on May 5, 2026."
```

6) Serve API
```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
