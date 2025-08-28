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
_model_version = "base"  # Track model version

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
	model_version: str

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
	global _tokenizer, _model, _model_version
	try:
		_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
		_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
		_model_version = MODEL_PATH.split("/")[-1] if "/" in MODEL_PATH else "custom"
		print(f"Model loaded successfully from {MODEL_PATH} (version: {_model_version})")
		return True
	except Exception as e:
		print(f"Error loading model from {MODEL_PATH}: {e}")
		# Try to load the base model as fallback
		try:
			print("Loading base model as fallback...")
			_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
			_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
			_model_version = "base-fallback"
			print("Base model loaded successfully")
			return True
		except Exception as e2:
			print(f"Error loading base model: {e2}")
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
		output_dir.mkdir(exist_ok=True)
		
		# Try to import training module, fallback to simple training if not available
		try:
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
			
		except ImportError as e:
			print(f"Training module not available, using simple fine-tuning: {e}")
			# Fallback to simple fine-tuning
			await simple_fine_tune(training_id, examples, model_name, epochs, batch_size, output_dir)
		
		# Load the new model
		global MODEL_PATH, _model_version
		MODEL_PATH = str(output_dir)
		_model_version = f"trained-{training_id}"
		load_model()
		
		# Update training status
		with open(RUNS_DIR / f"training_{training_id}_status.json", "w") as f:
			json.dump({
				"status": "completed",
				"completed_at": datetime.now().isoformat(),
				"examples_processed": len(examples),
			}, f)
			
	except Exception as e:
		print(f"Training failed: {e}")
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

async def simple_fine_tune(training_id: str, examples: List[TrainingExample], 
                          model_name: str, epochs: int, batch_size: int, output_dir: Path):
	"""Simple fine-tuning fallback when training module is not available."""
	global _model, _tokenizer
	
	try:
		# Load model and tokenizer
		if _model is None or _tokenizer is None:
			_tokenizer = AutoTokenizer.from_pretrained(model_name)
			_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
		
		# Prepare training data
		train_data = []
		for example in examples:
			prompt = (
				"You are a faithful conclusion generator.\n"
				"Rule: Output one short, self-contained conclusion entailed by the input. No new facts.\n\n"
				f"Input: {example.input}\nConclusion:"
			)
			if example.length_tag:
				prompt = f"{example.length_tag} {prompt}"
			
			train_data.append({
				"input": prompt,
				"target": example.target
			})
		
		print(f"Simple fine-tuning with {len(examples)} examples")
		
		# Set up training
		import torch
		from torch.utils.data import Dataset, DataLoader
		
		class SimpleDataset(Dataset):
			def __init__(self, data, tokenizer, max_length=512):
				self.data = data
				self.tokenizer = tokenizer
				self.max_length = max_length
			
			def __len__(self):
				return len(self.data)
			
			def __getitem__(self, idx):
				item = self.data[idx]
				inputs = self.tokenizer(
					item["input"], 
					max_length=self.max_length, 
					truncation=True, 
					padding="max_length", 
					return_tensors="pt"
				)
				with self.tokenizer.as_target_tokenizer():
					labels = self.tokenizer(
						item["target"], 
						max_length=64, 
						truncation=True, 
						padding="max_length", 
						return_tensors="pt"
					)
				return {
					"input_ids": inputs["input_ids"].squeeze(),
					"attention_mask": inputs["attention_mask"].squeeze(),
					"labels": labels["input_ids"].squeeze()
				}
		
		# Create dataset and dataloader
		dataset = SimpleDataset(train_data, _tokenizer)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
		
		# Set up optimizer and training
		optimizer = torch.optim.AdamW(_model.parameters(), lr=1e-4)
		_model.train()
		
		# Training loop
		for epoch in range(epochs):
			total_loss = 0
			for batch in dataloader:
				optimizer.zero_grad()
				
				# Move to device if available
				if torch.cuda.is_available():
					batch = {k: v.cuda() for k, v in batch.items()}
					_model = _model.cuda()
				
				outputs = _model(**batch)
				loss = outputs.loss
				loss.backward()
				optimizer.step()
				
				total_loss += loss.item()
			
			print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
		
		# Save the trained model
		_model.save_pretrained(output_dir)
		_tokenizer.save_pretrained(output_dir)
		
		print(f"Model trained and saved to {output_dir}")
		
	except Exception as e:
		print(f"Simple fine-tuning failed: {e}")
		raise

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
			# Get the most recent training file
			latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
			with open(latest_file, "r") as f:
				status_data = json.load(f)
				last_training = status_data.get("completed_at") or status_data.get("failed_at")
	
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
		prompt = f"{request.length_tag} {prompt}"
	
	inputs = _tokenizer(prompt, return_tensors="pt")
	outputs = _model.generate(
		**inputs, 
		max_new_tokens=request.target_length or 24, 
		do_sample=False
	)
	full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
	
	# Extract only the conclusion part (after "Conclusion:")
	if "Conclusion:" in full_text:
		conclusion = full_text.split("Conclusion:")[-1].strip()
	else:
		conclusion = full_text.strip()
	
	# Remove any remaining prompt parts
	if "Input:" in conclusion:
		conclusion = conclusion.split("Input:")[0].strip()
	
	# If the model just repeated the input, try to generate a simple conclusion
	if conclusion == request.input or len(conclusion) < 5:
		# Simple fallback: create a basic conclusion
		words = request.input.split()
		if len(words) > 5:
			# Take the main part of the sentence
			conclusion = " ".join(words[:min(8, len(words))])
		else:
			conclusion = request.input
	
	print(f"DEBUG - Full generated text: '{full_text}'")
	print(f"DEBUG - Extracted conclusion: '{conclusion}'")
	
	return ConcludeResponse(
		conclusion=conclusion,
		length=len(conclusion.split()),
		confidence=0.95,  # Placeholder confidence score
		model_version=_model_version,
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
