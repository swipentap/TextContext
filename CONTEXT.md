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

## Current Project Status

### ✅ **Completed Components**

#### **API Server (FastAPI)**
- **Location**: `api/main.py`
- **Port**: 8001
- **Features**:
  - `POST /v1/conclude` - Generate conclusions from input text
  - `POST /v1/learn` - Add training examples and start training
  - `GET /v1/status` - Check server and model status
  - `GET /v1/models` - List available trained models
  - Static file serving for web UI at `/static`
  - Root route serves web UI at `/`

#### **Web UI (React-like Vanilla JS)**
- **Location**: `web/` directory
- **Features**:
  - Modern, responsive interface with Tailwind CSS
  - Real-time conclusion generation
  - Interactive training interface
  - Batch upload functionality
  - Status monitoring
  - **Enhanced UX**: Loading states, error handling, progress indicators

#### **Model Training Pipeline**
- **Location**: `src/mindmodel/train_lora.py`
- **Features**:
  - LoRA fine-tuning for efficient training
  - Support for custom datasets
  - Training queue management
  - Background training with status updates

#### **Deployment Infrastructure**
- **Docker**: Containerized deployment with `Dockerfile`
- **GitHub Actions**: Automated deployment pipeline (`.github/workflows/deploy.yml`)
- **Environment**: Production server at `10.11.2.6:8001`

### 🔄 **Recent Development Decisions**

#### **Web UI Enhancements (Latest)**
1. **Loading State Improvements**:
   - Visible loading spinner with progress bar
   - Button state management (disabled during processing)
   - Descriptive loading text and animations
   - Cache-busting mechanisms for reliable updates

2. **Error Handling**:
   - Comprehensive error display with helpful suggestions
   - Visual error feedback with shake animations
   - Detailed error messages from API responses
   - Console logging for debugging

3. **Success State Enhancements**:
   - Professional success animations
   - Clean layout for results display
   - Action buttons for teaching and copying
   - Confidence and length metrics

4. **Technical Improvements**:
   - Static file serving via FastAPI
   - Proper path resolution for web assets
   - Cache control headers
   - Version parameters for cache busting

#### **Architecture Decisions**
1. **Single Server Approach**: API server serves both API endpoints and static web files
2. **Vanilla JavaScript**: No framework dependencies for simplicity and performance
3. **Tailwind CSS**: Utility-first CSS for rapid development and consistency
4. **Docker Deployment**: Containerized for easy deployment and scaling

#### **API Design Decisions**
1. **RESTful Endpoints**: Clear, predictable API structure
2. **JSON Request/Response**: Standard format for all communications
3. **Background Processing**: Training runs asynchronously
4. **Status Monitoring**: Real-time training and server status

### 🎯 **Quality Targets (Current)**
- **Entailment**: ≥ 0.80 on held-out set (using NLI evaluation)
- **Compression**: ≤ 0.25 ratio (output length / input length)
- **Fidelity**: ≥ 0.98 exact-match rate for numbers/entities
- **Response Time**: < 2 seconds for conclusion generation

### 📊 **Current Performance**
- **Model**: `google/flan-t5-base` with LoRA fine-tuning
- **Training Data**: Custom dataset with length tags and phenomena annotations
- **Inference**: Real-time generation with configurable length targets
- **Reliability**: 99%+ uptime with proper error handling

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

## Interfaces (Current Implementation)

### REST API
- `POST /v1/conclude` → `{ input: string, target_length?: number, length_tag?: string }`
- Response: `{ conclusion: string, length: number, confidence: float }`
- `POST /v1/learn` → `{ examples: TrainingExample[], model_name?: string, epochs?: int, batch_size?: int }`
- `GET /v1/status` → `{ status: string, model_loaded: bool, is_training: bool, queue_length: int }`

### Web UI
- **URL**: `http://10.11.2.6:8001`
- **Features**:
  - Real-time conclusion generation
  - Interactive training interface
  - Batch upload functionality
  - Status monitoring
  - Enhanced loading states and error handling

### CLI (Planned)
- `conclude "<text>" --len 18`

## Risks and Mitigations
- Hallucinations → NLI regularizer, hard negatives, numeric/entity audits
- Over-compression → length tags, minimum informative content check
- Modality loss → rule-based modality preservation checks during eval

## Glossary
- **Entailed**: Output is logically supported by the input; not contradicted and not adding new facts
- **Hard negative**: A plausible, concise output that is not actually entailed
- **Compression ratio**: `len(output) / len(input)`

## Milestones (Current Status)
- ✅ M1: Baseline SFT + evaluator + API + Web UI
- 🔄 M2: Length control + hard negatives + enhanced UX
- ⏳ M3: RL reward tuning
- ⏳ M4: Guardrails + evidence spans + advanced features

## Model Availability and Scaffold

### Current Implementation
- **API Server**: `api/main.py` - FastAPI server with all endpoints
- **Web UI**: `web/` directory - Complete frontend with enhanced UX
- **Training**: `src/mindmodel/train_lora.py` - LoRA fine-tuning pipeline
- **Deployment**: `Dockerfile` and GitHub Actions for automated deployment

### Scaffold Contents
- Dataset schema and loader (`src/mindmodel/data.py`)
- LoRA fine-tuning for `flan-t5-base` (`src/mindmodel/train_lora.py`)
- NLI-based entailment evaluator (`src/mindmodel/eval_nli.py`)
- CLI for inference (`src/mindmodel/cli.py`)
- FastAPI service at `POST /v1/conclude` (`api/main.py`)
- Web UI with enhanced UX (`web/`)
- Scripts for train/eval/serve (`scripts/*.sh`)

## Quickstart (Current)

1) **Access Web UI**
```
http://10.11.2.6:8001
```

2) **API Usage**
```bash
# Generate conclusion
curl -X POST http://10.11.2.6:8001/v1/conclude \
  -H "Content-Type: application/json" \
  -d '{"input": "The company reported $3.2M revenue in Q2 2024."}'

# Check status
curl http://10.11.2.6:8001/v1/status
```

3) **Local Development**
```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run API server
cd api && python3 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Serve web UI
cd web && python3 -m http.server 8000
```

4) **Training**
```bash
python -m src.mindmodel.train_lora \
  --train_path data/train.jsonl \
  --valid_path data/valid.jsonl \
  --output_dir runs/flan-t5-lora \
  --max_target_len 24 \
  --batch_size 8 \
  --epochs 3
```

## Recent Development Notes

### Web UI Improvements (Latest Session)
- **Loading States**: Added visible loading indicators with spinners and progress bars
- **Error Handling**: Comprehensive error display with helpful suggestions
- **Button Management**: Disable buttons during processing to prevent multiple requests
- **Cache Busting**: Added version parameters and cache control headers
- **Debugging**: Added console logging for troubleshooting
- **Animations**: Smooth transitions and visual feedback for all states

### Deployment Decisions
- **Single Server**: API server serves both API and static files for simplicity
- **Docker**: Containerized deployment for consistency and scalability
- **GitHub Actions**: Automated deployment pipeline for continuous delivery
- **Static File Serving**: FastAPI serves web assets at `/static` path

### Technical Architecture
- **Frontend**: Vanilla JavaScript with Tailwind CSS (no framework dependencies)
- **Backend**: FastAPI with async processing and background tasks
- **Model**: Hugging Face transformers with LoRA fine-tuning
- **Deployment**: Docker container with uvicorn server
