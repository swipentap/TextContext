from __future__ import annotations

import os
import json
import asyncio
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="MindModel Learning API")

# Global model state
MODEL_PATH = os.environ.get("MINDMODEL_MODEL", "google/flan-t5-base")
_tokenizer = None
_model = None
_is_training = False
_training_queue = []

# Data storage
DATA_DIR = Path("/app/data")
MODELS_DIR = Path("/app/models")
RUNS_DIR = Path("/app/runs")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

# Request/Response models
class ConcludeRequest(BaseModel):
	input: str
	target_length: Optional[int] = 24
	length_tag: Optional[str] = None

class ConcludeResponse(BaseModel):
	conclusion: str
	length: int
	confidence: float

class TrainingExample(BaseModel):
	input: str
	target: str
	length_tag: Optional[str] = None
	phenomena_tags: Optional[List[str]] = None

class TrainingRequest(BaseModel):
	examples: List[TrainingExample]
	model_name: Optional[str] = "google/flan-t5-base"
	epochs: Optional[int] = 1
	batch_size: Optional[int] = 8

class TrainingResponse(BaseModel):
	status: str
	message: str
	training_id: Optional[str] = None

class StatusResponse(BaseModel):
	status: str
	model_loaded: bool
	is_training: bool
	queue_length: int
	last_training: Optional[str] = None

def load_model():
	"""Load the model and tokenizer."""
	global _tokenizer, _model
	try:
		_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
		_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
		return True
	except Exception as e:
		print(f"Error loading model: {e}")
		return False

def save_training_data(examples: List[TrainingExample], filename: str):
	"""Save training examples to JSONL file."""
	filepath = DATA_DIR / filename
	with open(filepath, "w", encoding="utf-8") as f:
		for example in examples:
			obj = {
				"input": example.input,
				"target": example.target,
				"length_tag": example.length_tag or "",
				"phenomena_tags": example.phenomena_tags or [],
			}
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")

async def train_model_background(training_id: str, examples: List[TrainingExample], 
                                model_name: str, epochs: int, batch_size: int):
	"""Background task for model training."""
	global _is_training, _model, _tokenizer
	
	_is_training = True
	
	try:
		# Save training data
		train_file = f"train_{training_id}.jsonl"
		valid_file = f"valid_{training_id}.jsonl"
		
		# Split examples 80/20
		split_idx = int(len(examples) * 0.8)
		train_examples = examples[:split_idx]
		valid_examples = examples[split_idx:]
		
		save_training_data(train_examples, train_file)
		save_training_data(valid_examples, valid_file)
		
		# Run training
		output_dir = RUNS_DIR / f"training_{training_id}"
		
		# Import here to avoid circular imports
		from src.mindmodel.train_lora import main as train_main
		import tyro
		from src.mindmodel.train_lora import Args
		
		# Create training args
		args = Args(
			train_path=str(DATA_DIR / train_file),
			valid_path=str(DATA_DIR / valid_file),
			output_dir=str(output_dir),
			base_model=model_name,
			epochs=epochs,
			batch_size=batch_size,
			max_target_len=24,
			lr=2e-4,
		)
		
		# Run training
		train_main(args)
		
		# Load the new model
		global MODEL_PATH
		MODEL_PATH = str(output_dir)
		load_model()
		
		# Update training status
		with open(RUNS_DIR / f"training_{training_id}_status.json", "w") as f:
			json.dump({
				"status": "completed",
				"completed_at": datetime.now().isoformat(),
				"examples_processed": len(examples),
			}, f)
			
	except Exception as e:
		# Update training status with error
		with open(RUNS_DIR / f"training_{training_id}_status.json", "w") as f:
			json.dump({
				"status": "failed",
				"error": str(e),
				"failed_at": datetime.now().isoformat(),
			}, f)
		raise
	finally:
		_is_training = False

@app.on_event("startup")
async def startup_event():
	"""Initialize the model on startup."""
	load_model()

@app.get("/health")
async def health_check():
	"""Health check endpoint."""
	return {"status": "healthy", "model_loaded": _model is not None}

@app.get("/status", response_model=StatusResponse)
async def get_status():
	"""Get current system status."""
	last_training = None
	if RUNS_DIR.exists():
		training_files = list(RUNS_DIR.glob("training_*_status.json"))
		if training_files:
			latest = max(training_files, key=lambda x: x.stat().st_mtime)
			with open(latest, "r") as f:
				status_data = json.load(f)
				last_training = status_data.get("completed_at")
	
	return StatusResponse(
		status="running",
		model_loaded=_model is not None,
		is_training=_is_training,
		queue_length=len(_training_queue),
		last_training=last_training,
	)

@app.post("/v1/conclude", response_model=ConcludeResponse)
async def conclude(request: ConcludeRequest):
	"""Generate a conclusion from input text."""
	if _model is None or _tokenizer is None:
		raise HTTPException(status_code=503, detail="Model not loaded")
	
	prompt = (
		"You are a faithful conclusion generator.\n"
		"Rule: Output one short, self-contained conclusion entailed by the input. No new facts.\n\n"
		f"Input: {request.input}\nConclusion:"
	)
	
	if request.length_tag:
		prompt = f"{request.length_tag} " + prompt
	
	inputs = _tokenizer(prompt, return_tensors="pt")
	outputs = _model.generate(
		**inputs, 
		max_new_tokens=request.target_length or 24, 
		do_sample=False
	)
	text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
	
	return ConcludeResponse(
		conclusion=text,
		length=len(text.split()),
		confidence=0.95,  # Placeholder confidence score
	)

@app.post("/v1/learn", response_model=TrainingResponse)
async def learn(request: TrainingRequest, background_tasks: BackgroundTasks):
	"""Add training examples and start training."""
	if not request.examples:
		raise HTTPException(status_code=400, detail="No training examples provided")
	
	# Generate training ID
	training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Add to training queue
	_training_queue.append({
		"id": training_id,
		"examples": request.examples,
		"model_name": request.model_name,
		"epochs": request.epochs,
		"batch_size": request.batch_size,
	})
	
	# Start training if not already training
	if not _is_training:
		background_tasks.add_task(
			train_model_background,
			training_id,
			request.examples,
			request.model_name,
			request.epochs,
			request.batch_size,
		)
	
	return TrainingResponse(
		status="queued",
		message=f"Training queued with {len(request.examples)} examples",
		training_id=training_id,
	)

@app.get("/v1/training/{training_id}")
async def get_training_status(training_id: str):
	"""Get status of a specific training job."""
	status_file = RUNS_DIR / f"training_{training_id}_status.json"
	
	if not status_file.exists():
		raise HTTPException(status_code=404, detail="Training job not found")
	
	with open(status_file, "r") as f:
		status_data = json.load(f)
	
	return status_data

@app.get("/v1/models")
async def list_models():
	"""List available trained models."""
	models = []
	if RUNS_DIR.exists():
		for model_dir in RUNS_DIR.iterdir():
			if model_dir.is_dir() and model_dir.name.startswith("training_"):
				models.append({
					"id": model_dir.name,
					"path": str(model_dir),
					"created": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
				})
	
	return {"models": models}

@app.post("/v1/models/{model_id}/load")
async def load_specific_model(model_id: str):
	"""Load a specific trained model."""
	global MODEL_PATH, _model, _tokenizer
	
	model_path = RUNS_DIR / model_id
	if not model_path.exists():
		raise HTTPException(status_code=404, detail="Model not found")
	
	try:
		MODEL_PATH = str(model_path)
		load_model()
		return {"status": "loaded", "model_path": MODEL_PATH}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Mount static files for web UI
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/")
async def serve_web_ui():
    """Serve the web UI"""
    return FileResponse("web/index.html")
