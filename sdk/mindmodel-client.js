/**
 * MindModel JavaScript/Node.js SDK Client
 * 
 * A JavaScript client library for interacting with the MindModel API.
 */

class MindModelClient {
    /**
     * Create a new MindModel client.
     * 
     * @param {string} baseUrl - The base URL of the MindModel API
     * @param {number} timeout - Request timeout in milliseconds
     */
    constructor(baseUrl = 'http://10.11.2.6:8000', timeout = 30000) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = timeout;
    }

    /**
     * Make an HTTP request to the API.
     * 
     * @param {string} method - HTTP method
     * @param {string} endpoint - API endpoint path
     * @param {Object} options - Request options
     * @returns {Promise<Object>} The response data
     * @throws {MindModelError} If the request fails
     */
    async _makeRequest(method, endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            timeout: this.timeout,
            ...options
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new MindModelError(`HTTP ${response.status}: ${errorText}`);
            }
            
            return await response.json();
        } catch (error) {
            if (error instanceof MindModelError) {
                throw error;
            }
            throw new MindModelError(`API request failed: ${error.message}`);
        }
    }

    /**
     * Generate a conclusion from input text.
     * 
     * @param {string} inputText - The text to generate a conclusion from
     * @param {number} targetLength - Target length for the conclusion (8-50)
     * @param {string} lengthTag - Optional length tag for the conclusion
     * @returns {Promise<Object>} Object containing conclusion, length, and confidence
     * @throws {MindModelError} If the request fails
     * @throws {Error} If input parameters are invalid
     */
    async generateConclusion(inputText, targetLength = 24, lengthTag = null) {
        if (!inputText || !inputText.trim()) {
            throw new Error('Input text cannot be empty');
        }

        if (targetLength < 8 || targetLength > 50) {
            throw new Error('Target length must be between 8 and 50');
        }

        const payload = {
            input: inputText,
            target_length: targetLength
        };

        if (lengthTag) {
            payload.length_tag = lengthTag;
        }

        return await this._makeRequest('POST', '/v1/conclude', {
            body: JSON.stringify(payload)
        });
    }

    /**
     * Train the model with new examples.
     * 
     * @param {Array<Object>} examples - List of training examples
     * @param {number} epochs - Number of training epochs (1-10)
     * @param {number} batchSize - Training batch size (1-32)
     * @param {string} modelName - Optional base model name
     * @returns {Promise<Object>} Object containing training ID, status, and message
     * @throws {MindModelError} If the request fails
     * @throws {Error} If input parameters are invalid
     */
    async trainModel(examples, epochs = 1, batchSize = 8, modelName = null) {
        if (!examples || examples.length === 0) {
            throw new Error('At least one training example is required');
        }

        if (epochs < 1 || epochs > 10) {
            throw new Error('Epochs must be between 1 and 10');
        }

        if (batchSize < 1 || batchSize > 32) {
            throw new Error('Batch size must be between 1 and 32');
        }

        // Validate examples
        for (let i = 0; i < examples.length; i++) {
            const example = examples[i];
            if (typeof example !== 'object' || example === null) {
                throw new Error(`Example ${i} must be an object`);
            }

            if (!example.input || !example.target) {
                throw new Error(`Example ${i} must contain 'input' and 'target' fields`);
            }

            if (!example.input.trim() || !example.target.trim()) {
                throw new Error(`Example ${i} input and target cannot be empty`);
            }
        }

        const payload = {
            examples,
            epochs,
            batch_size: batchSize
        };

        if (modelName) {
            payload.model_name = modelName;
        }

        return await this._makeRequest('POST', '/v1/learn', {
            body: JSON.stringify(payload)
        });
    }

    /**
     * Get the status of a training job.
     * 
     * @param {string} trainingId - The ID of the training job
     * @returns {Promise<Object>} Object containing training status information
     * @throws {MindModelError} If the request fails
     */
    async getTrainingStatus(trainingId) {
        return await this._makeRequest('GET', `/v1/training/${trainingId}`);
    }

    /**
     * Wait for a training job to complete.
     * 
     * @param {string} trainingId - The ID of the training job
     * @param {number} pollInterval - How often to check status (milliseconds)
     * @param {number} timeout - Maximum time to wait (milliseconds, null for no timeout)
     * @returns {Promise<Object>} Final training status
     * @throws {MindModelError} If the request fails or timeout occurs
     */
    async waitForTrainingCompletion(trainingId, pollInterval = 5000, timeout = null) {
        const startTime = Date.now();

        while (true) {
            const status = await this.getTrainingStatus(trainingId);

            if (status.status === 'completed' || status.status === 'failed') {
                return status;
            }

            if (timeout && (Date.now() - startTime) > timeout) {
                throw new MindModelError(`Training timeout after ${timeout} milliseconds`);
            }

            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
    }

    /**
     * Get a list of available trained models.
     * 
     * @returns {Promise<Object>} Object containing list of models
     * @throws {MindModelError} If the request fails
     */
    async listModels() {
        return await this._makeRequest('GET', '/v1/models');
    }

    /**
     * Load a specific trained model.
     * 
     * @param {string} modelId - The ID of the model to load
     * @returns {Promise<Object>} Object containing load status information
     * @throws {MindModelError} If the request fails
     */
    async loadModel(modelId) {
        return await this._makeRequest('POST', `/v1/models/${modelId}/load`);
    }

    /**
     * Get the current system status.
     * 
     * @returns {Promise<Object>} Object containing system status information
     * @throws {MindModelError} If the request fails
     */
    async getStatus() {
        return await this._makeRequest('GET', '/status');
    }

    /**
     * Check if the service is healthy.
     * 
     * @returns {Promise<Object>} Object containing health status
     * @throws {MindModelError} If the service is unhealthy
     */
    async healthCheck() {
        return await this._makeRequest('GET', '/health');
    }

    /**
     * Check if the service is healthy (returns boolean).
     * 
     * @returns {Promise<boolean>} True if healthy, False otherwise
     */
    async isHealthy() {
        try {
            await this.healthCheck();
            return true;
        } catch (error) {
            return false;
        }
    }
}

/**
 * Custom error class for MindModel API errors.
 */
class MindModelError extends Error {
    constructor(message) {
        super(message);
        this.name = 'MindModelError';
    }
}

/**
 * Quick function to generate a conclusion from text.
 * 
 * @param {string} inputText - The text to generate a conclusion from
 * @param {string} baseUrl - The base URL of the MindModel API
 * @returns {Promise<string>} The generated conclusion
 * @throws {MindModelError} If the request fails
 */
async function quickConclude(inputText, baseUrl = 'http://10.11.2.6:8000') {
    const client = new MindModelClient(baseUrl);
    const result = await client.generateConclusion(inputText);
    return result.conclusion;
}

/**
 * Train the model with examples and optionally wait for completion.
 * 
 * @param {Array<Object>} examples - List of training examples
 * @param {string} baseUrl - The base URL of the MindModel API
 * @param {boolean} waitForCompletion - Whether to wait for training to complete
 * @returns {Promise<Object>} Training status information
 * @throws {MindModelError} If the request fails
 */
async function batchTrain(examples, baseUrl = 'http://10.11.2.6:8000', waitForCompletion = true) {
    const client = new MindModelClient(baseUrl);
    const result = await client.trainModel(examples);
    
    if (waitForCompletion) {
        return await client.waitForTrainingCompletion(result.training_id);
    }
    
    return result;
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // Node.js
    module.exports = {
        MindModelClient,
        MindModelError,
        quickConclude,
        batchTrain
    };
} else if (typeof window !== 'undefined') {
    // Browser
    window.MindModelClient = MindModelClient;
    window.MindModelError = MindModelError;
    window.quickConclude = quickConclude;
    window.batchTrain = batchTrain;
}

// Example usage (when run directly)
if (typeof require !== 'undefined' && require.main === module) {
    async function example() {
        const client = new MindModelClient();
        
        try {
            // Generate a conclusion
            const result = await client.generateConclusion(
                "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1."
            );
            console.log(`Conclusion: ${result.conclusion}`);
            console.log(`Length: ${result.length}`);
            console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
            
            // Train the model
            const examples = [
                {
                    input: "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
                    target: "Company revenue increased 15% to $3.2M in Q2.",
                    length_tag: "<LEN_24>",
                    phenomena_tags: ["numbers", "percentage"]
                }
            ];
            
            const trainingResult = await client.trainModel(examples);
            console.log(`Training started: ${trainingResult.training_id}`);
            
            // Wait for completion
            const finalStatus = await client.waitForTrainingCompletion(trainingResult.training_id);
            console.log(`Training completed: ${finalStatus.status}`);
            
            // Check system status
            const status = await client.getStatus();
            console.log(`Model loaded: ${status.model_loaded}`);
            console.log(`Currently training: ${status.is_training}`);
            
        } catch (error) {
            console.error(`Error: ${error.message}`);
        }
    }
    
    example();
}
