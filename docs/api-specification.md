# MindModel API Specification

## Overview

The MindModel API provides endpoints for generating conclusions from text input and training the model with new examples. The service operates in an "always-learning" mode, allowing continuous improvement through user feedback.

**Base URL**: `http://10.11.2.6:8000`

## Authentication

Currently, the API does not require authentication as it's designed for single-user operation.

## Data Models

### TrainingExample
```json
{
  "input": "string",
  "target": "string",
  "length_tag": "string (optional)",
  "phenomena_tags": ["string"] (optional)
}
```

### ConclusionRequest
```json
{
  "input": "string",
  "target_length": "integer (optional, default: 24)",
  "length_tag": "string (optional)"
}
```

### ConclusionResponse
```json
{
  "conclusion": "string",
  "length": "integer",
  "confidence": "float (0.0-1.0)"
}
```

### TrainingRequest
```json
{
  "examples": [TrainingExample],
  "model_name": "string (optional, default: google/flan-t5-base)",
  "epochs": "integer (optional, default: 1)",
  "batch_size": "integer (optional, default: 8)"
}
```

### TrainingResponse
```json
{
  "training_id": "string",
  "status": "string",
  "message": "string"
}
```

### ModelInfo
```json
{
  "id": "string",
  "name": "string",
  "created": "string (ISO 8601)",
  "size": "string (optional)",
  "performance": "object (optional)"
}
```

### SystemStatus
```json
{
  "model_loaded": "boolean",
  "is_training": "boolean",
  "queue_length": "integer",
  "last_training": "string (ISO 8601, optional)",
  "current_model": "string (optional)"
}
```

## Endpoints

### 1. Generate Conclusion

**POST** `/v1/conclude`

Generate a conclusion from input text.

#### Request Body
```json
{
  "input": "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
  "target_length": 24,
  "length_tag": "<LEN_24>"
}
```

#### Response
```json
{
  "conclusion": "Company revenue increased 15% to $3.2M in Q2.",
  "length": 24,
  "confidence": 0.92
}
```

#### Example Usage
```bash
curl -X POST http://10.11.2.6:8000/v1/conclude \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
    "target_length": 24
  }'
```

### 2. Train Model

**POST** `/v1/learn`

Train the model with new examples.

#### Request Body
```json
{
  "examples": [
    {
      "input": "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
      "target": "Company revenue increased 15% to $3.2M in Q2.",
      "length_tag": "<LEN_24>",
      "phenomena_tags": ["numbers", "percentage"]
    }
  ],
  "epochs": 1,
  "batch_size": 8
}
```

#### Response
```json
{
  "training_id": "train_1234567890",
  "status": "started",
  "message": "Training started successfully"
}
```

#### Example Usage
```bash
curl -X POST http://10.11.2.6:8000/v1/learn \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {
        "input": "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
        "target": "Company revenue increased 15% to $3.2M in Q2.",
        "length_tag": "<LEN_24>",
        "phenomena_tags": ["numbers", "percentage"]
      }
    ],
    "epochs": 1,
    "batch_size": 8
  }'
```

### 3. Get Training Status

**GET** `/v1/training/{training_id}`

Check the status of a specific training job.

#### Response
```json
{
  "training_id": "train_1234567890",
  "status": "completed",
  "progress": 100,
  "message": "Training completed successfully",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z"
}
```

#### Example Usage
```bash
curl http://10.11.2.6:8000/v1/training/train_1234567890
```

### 4. List Available Models

**GET** `/v1/models`

Get a list of all available trained models.

#### Response
```json
{
  "models": [
    {
      "id": "model_20240115_103500",
      "name": "google/flan-t5-base",
      "created": "2024-01-15T10:35:00Z",
      "size": "1.1GB",
      "performance": {
        "accuracy": 0.85,
        "training_examples": 1000
      }
    }
  ]
}
```

#### Example Usage
```bash
curl http://10.11.2.6:8000/v1/models
```

### 5. Load Model

**POST** `/v1/models/{model_id}/load`

Load a specific trained model.

#### Response
```json
{
  "model_id": "model_20240115_103500",
  "status": "loaded",
  "message": "Model loaded successfully"
}
```

#### Example Usage
```bash
curl -X POST http://10.11.2.6:8000/v1/models/model_20240115_103500/load
```

### 6. Get System Status

**GET** `/status`

Get the current system status.

#### Response
```json
{
  "model_loaded": true,
  "is_training": false,
  "queue_length": 0,
  "last_training": "2024-01-15T10:35:00Z",
  "current_model": "model_20240115_103500"
}
```

#### Example Usage
```bash
curl http://10.11.2.6:8000/status
```

### 7. Health Check

**GET** `/health`

Check if the service is healthy and responding.

#### Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Example Usage
```bash
curl http://10.11.2.6:8000/health
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid request format or missing required fields"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error occurred"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Service temporarily unavailable"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. However, it's recommended to:
- Limit requests to reasonable intervals (e.g., 1 request per second)
- Batch training examples when possible
- Monitor system resources during heavy usage

## Best Practices

### 1. Training Examples
- Provide diverse examples covering different topics and styles
- Use consistent length tags for similar types of conclusions
- Include phenomena tags to help the model understand specific patterns
- Aim for high-quality, accurate target conclusions

### 2. Input Text
- Provide clear, well-structured input text
- Include relevant context and details
- Avoid overly complex or ambiguous sentences
- Consider the target audience for the conclusion

### 3. Model Management
- Monitor training progress for large datasets
- Load the most recent model for best performance
- Keep track of training IDs for status monitoring
- Consider model performance metrics when available

## SDK Examples

### Python
```python
import requests

class MindModelClient:
    def __init__(self, base_url="http://10.11.2.6:8000"):
        self.base_url = base_url
    
    def generate_conclusion(self, input_text, target_length=24, length_tag=None):
        response = requests.post(f"{self.base_url}/v1/conclude", json={
            "input": input_text,
            "target_length": target_length,
            "length_tag": length_tag
        })
        response.raise_for_status()
        return response.json()
    
    def train_model(self, examples, epochs=1, batch_size=8):
        response = requests.post(f"{self.base_url}/v1/learn", json={
            "examples": examples,
            "epochs": epochs,
            "batch_size": batch_size
        })
        response.raise_for_status()
        return response.json()
    
    def get_status(self):
        response = requests.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()

# Usage
client = MindModelClient()
result = client.generate_conclusion("The company reported $3.2M revenue in Q2 2024.")
print(result["conclusion"])
```

### JavaScript/Node.js
```javascript
class MindModelClient {
    constructor(baseUrl = 'http://10.11.2.6:8000') {
        this.baseUrl = baseUrl;
    }
    
    async generateConclusion(inputText, targetLength = 24, lengthTag = null) {
        const response = await fetch(`${this.baseUrl}/v1/conclude`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                input: inputText,
                target_length: targetLength,
                length_tag: lengthTag
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async trainModel(examples, epochs = 1, batchSize = 8) {
        const response = await fetch(`${this.baseUrl}/v1/learn`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                examples: examples,
                epochs: epochs,
                batch_size: batchSize
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response.json();
    }
    
    async getStatus() {
        const response = await fetch(`${this.baseUrl}/status`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    }
}

// Usage
const client = new MindModelClient();
client.generateConclusion("The company reported $3.2M revenue in Q2 2024.")
    .then(result => console.log(result.conclusion))
    .catch(error => console.error('Error:', error));
```

## WebSocket Support

For real-time updates (future enhancement), the API may support WebSocket connections for:
- Training progress updates
- Real-time conclusion generation
- System status monitoring

## Versioning

The API uses URL versioning (v1). Future versions will be available at `/v2/`, `/v3/`, etc.

## Changelog

### v1.0.0 (Current)
- Initial API release
- Conclusion generation
- Model training
- System status monitoring
- Model management
