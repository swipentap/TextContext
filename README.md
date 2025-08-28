# MindModel - Always Learning Conclusion Generator

A Docker-based service that continuously learns to generate faithful, concise conclusions from text input.

## Features

- **Always Learning**: Continuously train on new examples
- **Single User**: Designed for personal use with persistent learning
- **Docker Deployment**: Easy setup and deployment
- **REST API**: Simple HTTP interface for inference and training

## Quick Start

### 1. Build and Run

```bash
# Build the Docker image
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 2. Check Status

```bash
curl http://localhost:8000/status
```

### 3. Generate Conclusions

```bash
curl -X POST http://localhost:8000/v1/conclude \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The company reported revenue of $3.2 million in Q2 2024.",
    "target_length": 24
  }'
```

### 4. Teach the Model

```bash
curl -X POST http://localhost:8000/v1/learn \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {
        "input": "The company reported revenue of $3.2 million in Q2 2024.",
        "target": "The company reported $3.2 million in Q2 2024.",
        "phenomena_tags": ["numbers"]
      },
      {
        "input": "If the weather holds, the match will start at 3 pm on Saturday.",
        "target": "The match will start at 3 pm on Saturday if weather holds.",
        "phenomena_tags": ["conditional", "time"]
      }
    ],
    "epochs": 1,
    "batch_size": 8
  }'
```

## API Endpoints

### Processing API

- `POST /v1/conclude` - Generate conclusion from input text
- `GET /health` - Health check
- `GET /status` - System status

### Learning API

- `POST /v1/learn` - Add training examples and start training
- `GET /v1/training/{training_id}` - Get training job status
- `GET /v1/models` - List available trained models
- `POST /v1/models/{model_id}/load` - Load specific model

## Data Persistence

The service uses Docker volumes to persist:

- `/app/data` - Training data and examples
- `/app/models` - Model checkpoints
- `/app/runs` - Training runs and logs

## Usage Examples

### Python Client

```python
import requests

# Generate conclusion
response = requests.post("http://localhost:8000/v1/conclude", json={
    "input": "Shares rose 4% after the CEO said profits would grow next year.",
    "target_length": 20
})
print(response.json()["conclusion"])

# Teach with new examples
training_data = {
    "examples": [
        {
            "input": "The board did not approve the budget proposal.",
            "target": "The board did not approve the budget.",
            "phenomena_tags": ["negation"]
        }
    ],
    "epochs": 1
}
response = requests.post("http://localhost:8000/v1/learn", json=training_data)
print(f"Training ID: {response.json()['training_id']}")
```

### Continuous Learning Workflow

1. **Start with base model**: Service loads `google/flan-t5-base` by default
2. **Add examples**: Use `/v1/learn` to add training examples
3. **Monitor training**: Check `/status` for training progress
4. **Use improved model**: Model automatically updates after training
5. **Repeat**: Continue adding examples for continuous improvement

## Configuration

### Environment Variables

- `MINDMODEL_MODEL`: Base model to use (default: `google/flan-t5-base`)

### Model Options

- `google/flan-t5-base` (250M params) - Fast, good for testing
- `google/flan-t5-large` (780M params) - Better quality, slower
- `t5-small` (60M params) - Very fast, limited quality

## Monitoring

### Check Training Status

```bash
# Get overall status
curl http://localhost:8000/status

# Check specific training job
curl http://localhost:8000/v1/training/20241201_143022
```

### View Logs

```bash
docker-compose logs -f mindmodel
```

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Add New Features

1. Modify `api/main.py` for new endpoints
2. Update `src/mindmodel/` for model improvements
3. Rebuild Docker image: `docker-compose up --build`

## Troubleshooting

### Model Not Loading

```bash
# Check model path
curl http://localhost:8000/status

# Restart service
docker-compose restart mindmodel
```

### Training Fails

```bash
# Check training logs
docker-compose logs mindmodel

# Verify training data format
curl http://localhost:8000/v1/training/{training_id}
```

### Memory Issues

- Use smaller model: `MINDMODEL_MODEL=t5-small`
- Reduce batch size in training requests
- Increase Docker memory limits
