# MindModel SDK

Official SDK libraries for the MindModel API, providing easy integration with your applications.

## Available SDKs

- **Python SDK** (`mindmodel_client.py`) - For Python applications
- **JavaScript/Node.js SDK** (`mindmodel-client.js`) - For JavaScript/Node.js applications

## Python SDK

### Installation

Copy the `mindmodel_client.py` file to your project or install it as a local package.

### Basic Usage

```python
from mindmodel_client import MindModelClient, quick_conclude

# Create a client
client = MindModelClient("http://10.11.2.6:8000")

# Generate a conclusion
result = client.generate_conclusion(
    "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1."
)
print(f"Conclusion: {result['conclusion']}")

# Quick function for simple use cases
conclusion = quick_conclude("The company reported $3.2M revenue in Q2 2024.")
print(conclusion)
```

### Training Examples

```python
# Train the model with examples
examples = [
    {
        "input": "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
        "target": "Company revenue increased 15% to $3.2M in Q2.",
        "length_tag": "<LEN_24>",
        "phenomena_tags": ["numbers", "percentage"]
    }
]

# Start training
result = client.train_model(examples, epochs=1, batch_size=8)
print(f"Training started: {result['training_id']}")

# Wait for completion
final_status = client.wait_for_training_completion(result['training_id'])
print(f"Training completed: {final_status['status']}")
```

### System Management

```python
# Check system status
status = client.get_status()
print(f"Model loaded: {status['model_loaded']}")
print(f"Currently training: {status['is_training']}")

# List available models
models = client.list_models()
for model in models['models']:
    print(f"Model: {model['id']} - Created: {model['created']}")

# Load a specific model
client.load_model("model_20240115_103500")
```

### Error Handling

```python
from mindmodel_client import MindModelError

try:
    result = client.generate_conclusion("")
except MindModelError as e:
    print(f"API Error: {e}")
except ValueError as e:
    print(f"Validation Error: {e}")
```

## JavaScript/Node.js SDK

### Installation

Copy the `mindmodel-client.js` file to your project or install it as a local package.

### Basic Usage

```javascript
const { MindModelClient, quickConclude } = require('./mindmodel-client.js');

// Create a client
const client = new MindModelClient('http://10.11.2.6:8000');

// Generate a conclusion
const result = await client.generateConclusion(
    "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1."
);
console.log(`Conclusion: ${result.conclusion}`);

// Quick function for simple use cases
const conclusion = await quickConclude("The company reported $3.2M revenue in Q2 2024.");
console.log(conclusion);
```

### Browser Usage

```html
<script src="mindmodel-client.js"></script>
<script>
    const client = new MindModelClient('http://10.11.2.6:8000');
    
    client.generateConclusion("The company reported $3.2M revenue in Q2 2024.")
        .then(result => {
            console.log(`Conclusion: ${result.conclusion}`);
        })
        .catch(error => {
            console.error(`Error: ${error.message}`);
        });
</script>
```

### Training Examples

```javascript
// Train the model with examples
const examples = [
    {
        input: "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
        target: "Company revenue increased 15% to $3.2M in Q2.",
        length_tag: "<LEN_24>",
        phenomena_tags: ["numbers", "percentage"]
    }
];

// Start training
const result = await client.trainModel(examples, 1, 8);
console.log(`Training started: ${result.training_id}`);

// Wait for completion
const finalStatus = await client.waitForTrainingCompletion(result.training_id);
console.log(`Training completed: ${finalStatus.status}`);
```

### System Management

```javascript
// Check system status
const status = await client.getStatus();
console.log(`Model loaded: ${status.model_loaded}`);
console.log(`Currently training: ${status.is_training}`);

// List available models
const models = await client.listModels();
models.models.forEach(model => {
    console.log(`Model: ${model.id} - Created: ${model.created}`);
});

// Load a specific model
await client.loadModel("model_20240115_103500");
```

### Error Handling

```javascript
const { MindModelError } = require('./mindmodel-client.js');

try {
    const result = await client.generateConclusion("");
} catch (error) {
    if (error instanceof MindModelError) {
        console.error(`API Error: ${error.message}`);
    } else {
        console.error(`Validation Error: ${error.message}`);
    }
}
```

## API Reference

### MindModelClient

#### Constructor
- `baseUrl` (string): The base URL of the MindModel API (default: "http://10.11.2.6:8000")
- `timeout` (number): Request timeout in seconds/milliseconds

#### Methods

##### generateConclusion(inputText, targetLength, lengthTag)
Generate a conclusion from input text.

- `inputText` (string): The text to generate a conclusion from
- `targetLength` (number): Target length for the conclusion (8-50, default: 24)
- `lengthTag` (string, optional): Length tag for the conclusion

Returns: Object with `conclusion`, `length`, and `confidence`

##### trainModel(examples, epochs, batchSize, modelName)
Train the model with new examples.

- `examples` (array): List of training examples
- `epochs` (number): Number of training epochs (1-10, default: 1)
- `batchSize` (number): Training batch size (1-32, default: 8)
- `modelName` (string, optional): Base model name

Returns: Object with `training_id`, `status`, and `message`

##### getTrainingStatus(trainingId)
Get the status of a training job.

- `trainingId` (string): The ID of the training job

Returns: Object with training status information

##### waitForTrainingCompletion(trainingId, pollInterval, timeout)
Wait for a training job to complete.

- `trainingId` (string): The ID of the training job
- `pollInterval` (number): How often to check status
- `timeout` (number, optional): Maximum time to wait

Returns: Final training status

##### listModels()
Get a list of available trained models.

Returns: Object with list of models

##### loadModel(modelId)
Load a specific trained model.

- `modelId` (string): The ID of the model to load

Returns: Object with load status information

##### getStatus()
Get the current system status.

Returns: Object with system status information

##### healthCheck()
Check if the service is healthy.

Returns: Object with health status

##### isHealthy()
Check if the service is healthy (returns boolean).

Returns: Boolean indicating health status

### Convenience Functions

#### quickConclude(inputText, baseUrl)
Quick function to generate a conclusion from text.

- `inputText` (string): The text to generate a conclusion from
- `baseUrl` (string): The base URL of the MindModel API

Returns: The generated conclusion string

#### batchTrain(examples, baseUrl, waitForCompletion)
Train the model with examples and optionally wait for completion.

- `examples` (array): List of training examples
- `baseUrl` (string): The base URL of the MindModel API
- `waitForCompletion` (boolean): Whether to wait for training to complete

Returns: Training status information

## Error Handling

Both SDKs provide comprehensive error handling:

- **MindModelError**: Raised for API-related errors
- **ValueError** (Python) / **Error** (JavaScript): Raised for validation errors
- **Network errors**: Handled and wrapped in MindModelError

## Examples

See the example code at the bottom of each SDK file for complete usage examples.

## Support

For issues and questions:
- Check the API documentation in `/docs/api-specification.md`
- Review the OpenAPI specification in `/docs/openapi.yaml`
- Open an issue on the GitHub repository
