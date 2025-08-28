// MindModel Web UI Application
class MindModelUI {
    constructor() {
        this.apiBase = 'http://10.11.2.6:8001';
        this.trainingQueue = [];
        this.currentFile = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadStatus();
        this.setupTabNavigation();
        this.setupFileUpload();
    }

    setupEventListeners() {
        // Test tab
        document.getElementById('generate-btn').addEventListener('click', () => this.generateConclusion());
        document.getElementById('teach-from-result').addEventListener('click', () => this.teachFromResult());
        document.getElementById('copy-result').addEventListener('click', () => this.copyResult());

        // Teach tab
        document.getElementById('teach-single-btn').addEventListener('click', () => this.addTrainingExample());
        document.getElementById('train-btn').addEventListener('click', () => this.startTraining());
        document.getElementById('clear-queue-btn').addEventListener('click', () => this.clearQueue());

        // Batch tab
        document.getElementById('train-batch-btn').addEventListener('click', () => this.trainBatch());
        document.getElementById('clear-file-btn').addEventListener('click', () => this.clearFile());

        // Status tab
        document.getElementById('refresh-status-btn').addEventListener('click', () => this.loadStatus());
    }

    setupTabNavigation() {
        const tabs = ['test', 'teach', 'batch', 'status'];
        tabs.forEach(tab => {
            document.getElementById(`tab-${tab}`).addEventListener('click', () => this.switchTab(tab));
        });
    }

    setupFileUpload() {
        const uploadBtn = document.getElementById('upload-btn');
        const fileInput = document.getElementById('jsonl-file');

        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
    }

    switchTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });

        // Remove active class from all tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active', 'border-blue-500', 'text-blue-600');
            btn.classList.add('border-transparent', 'text-gray-500');
        });

        // Show selected tab content
        document.getElementById(`${tabName}-tab`).classList.remove('hidden');

        // Add active class to selected tab button
        const activeBtn = document.getElementById(`tab-${tabName}`);
        activeBtn.classList.add('active', 'border-blue-500', 'text-blue-600');
        activeBtn.classList.remove('border-transparent', 'text-gray-500');

        // Load specific data for tabs
        if (tabName === 'status') {
            this.loadStatus();
        }
    }

    async generateConclusion() {
        const inputText = document.getElementById('input-text').value.trim();
        const targetLength = document.getElementById('target-length').value;
        const lengthTag = document.getElementById('length-tag').value.trim();

        if (!inputText) {
            this.showToast('Please enter some text to analyze', 'error');
            return;
        }

        this.showLoading();
        
        try {
            const response = await fetch(`${this.apiBase}/v1/conclude`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input: inputText,
                    target_length: parseInt(targetLength),
                    length_tag: lengthTag || undefined
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.showResult(result);
            this.showToast('Conclusion generated successfully!', 'success');
        } catch (error) {
            this.showError(error.message);
            this.showToast('Failed to generate conclusion', 'error');
        }
    }

    showLoading() {
        document.getElementById('loading-output').classList.remove('hidden');
        document.getElementById('result-output').classList.add('hidden');
        document.getElementById('error-output').classList.add('hidden');
    }

    showResult(result) {
        document.getElementById('loading-output').classList.add('hidden');
        document.getElementById('result-output').classList.remove('hidden');
        document.getElementById('error-output').classList.add('hidden');

        document.getElementById('conclusion-text').textContent = result.conclusion;
        document.getElementById('conclusion-length').textContent = result.length;
        document.getElementById('conclusion-confidence').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    }

    showError(message) {
        document.getElementById('loading-output').classList.add('hidden');
        document.getElementById('result-output').classList.add('hidden');
        document.getElementById('error-output').classList.remove('hidden');
        document.getElementById('error-text').textContent = message;
    }

    teachFromResult() {
        const inputText = document.getElementById('input-text').value.trim();
        const conclusion = document.getElementById('conclusion-text').textContent;

        if (!inputText || !conclusion) {
            this.showToast('No result to teach from', 'error');
            return;
        }

        // Switch to teach tab and populate fields
        this.switchTab('teach');
        document.getElementById('teach-input').value = inputText;
        document.getElementById('teach-target').value = conclusion;
        this.showToast('Result copied to teaching form', 'success');
    }

    copyResult() {
        const conclusion = document.getElementById('conclusion-text').textContent;
        navigator.clipboard.writeText(conclusion).then(() => {
            this.showToast('Conclusion copied to clipboard', 'success');
        });
    }

    addTrainingExample() {
        const input = document.getElementById('teach-input').value.trim();
        const target = document.getElementById('teach-target').value.trim();
        const lengthTag = document.getElementById('teach-length-tag').value.trim();
        const phenomena = document.getElementById('teach-phenomena').value.trim();

        if (!input || !target) {
            this.showToast('Please provide both input and target', 'error');
            return;
        }

        const example = {
            input: input,
            target: target,
            length_tag: lengthTag || undefined,
            phenomena_tags: phenomena ? phenomena.split(',').map(tag => tag.trim()) : undefined
        };

        this.trainingQueue.push(example);
        this.updateTrainingQueue();
        this.clearTeachingForm();
        this.showToast('Example added to training queue', 'success');
    }

    updateTrainingQueue() {
        const queueContainer = document.getElementById('training-queue');
        
        if (this.trainingQueue.length === 0) {
            queueContainer.innerHTML = '<div class="text-gray-500 text-center py-4">No examples in queue</div>';
            return;
        }

        queueContainer.innerHTML = this.trainingQueue.map((example, index) => `
            <div class="bg-gray-50 p-3 rounded-md">
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <div class="text-sm font-medium text-gray-700">Input:</div>
                        <div class="text-sm text-gray-600 mb-2">${example.input.substring(0, 100)}${example.input.length > 100 ? '...' : ''}</div>
                        <div class="text-sm font-medium text-gray-700">Target:</div>
                        <div class="text-sm text-gray-600">${example.target}</div>
                    </div>
                    <button onclick="mindModelUI.removeFromQueue(${index})" class="text-red-500 hover:text-red-700 ml-2">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }

    removeFromQueue(index) {
        this.trainingQueue.splice(index, 1);
        this.updateTrainingQueue();
        this.showToast('Example removed from queue', 'info');
    }

    clearTeachingForm() {
        document.getElementById('teach-input').value = '';
        document.getElementById('teach-target').value = '';
        document.getElementById('teach-length-tag').value = '';
        document.getElementById('teach-phenomena').value = '';
    }

    async startTraining() {
        if (this.trainingQueue.length === 0) {
            this.showToast('No examples in training queue', 'error');
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/v1/learn`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    examples: this.trainingQueue,
                    epochs: 1,
                    batch_size: 8
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.showToast(`Training started! ID: ${result.training_id}`, 'success');
            this.trainingQueue = [];
            this.updateTrainingQueue();
        } catch (error) {
            this.showToast(`Training failed: ${error.message}`, 'error');
        }
    }

    clearQueue() {
        this.trainingQueue = [];
        this.updateTrainingQueue();
        this.showToast('Training queue cleared', 'info');
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.name.endsWith('.jsonl')) {
            this.showToast('Please select a JSONL file', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const content = e.target.result;
                const lines = content.split('\n').filter(line => line.trim());
                const examples = lines.map(line => JSON.parse(line));
                
                this.currentFile = {
                    name: file.name,
                    examples: examples
                };
                
                this.showFilePreview();
                this.showToast(`Loaded ${examples.length} examples from ${file.name}`, 'success');
            } catch (error) {
                this.showToast('Invalid JSONL file format', 'error');
            }
        };
        reader.readAsText(file);
    }

    showFilePreview() {
        const previewDiv = document.getElementById('file-preview');
        const contentDiv = document.getElementById('preview-content');
        
        previewDiv.classList.remove('hidden');
        
        contentDiv.innerHTML = this.currentFile.examples.slice(0, 5).map(example => `
            <div class="mb-3 p-3 bg-white rounded border">
                <div class="text-sm font-medium text-gray-700">Input:</div>
                <div class="text-sm text-gray-600 mb-2">${example.input}</div>
                <div class="text-sm font-medium text-gray-700">Target:</div>
                <div class="text-sm text-gray-600">${example.target}</div>
            </div>
        `).join('') + 
        (this.currentFile.examples.length > 5 ? 
            `<div class="text-center text-gray-500">... and ${this.currentFile.examples.length - 5} more examples</div>` : '');
    }

    async trainBatch() {
        if (!this.currentFile) {
            this.showToast('No file loaded', 'error');
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/v1/learn`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    examples: this.currentFile.examples,
                    epochs: 1,
                    batch_size: 8
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.showToast(`Batch training started! ID: ${result.training_id}`, 'success');
            this.clearFile();
        } catch (error) {
            this.showToast(`Batch training failed: ${error.message}`, 'error');
        }
    }

    clearFile() {
        this.currentFile = null;
        document.getElementById('file-preview').classList.add('hidden');
        document.getElementById('jsonl-file').value = '';
        this.showToast('File cleared', 'info');
    }

    async loadStatus() {
        try {
            const response = await fetch(`${this.apiBase}/status`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const status = await response.json();
            
            // Update status indicators
            document.getElementById('model-loaded').textContent = status.model_loaded ? 'Yes' : 'No';
            document.getElementById('training-status').textContent = status.is_training ? 'Training' : 'Idle';
            document.getElementById('queue-length').textContent = status.queue_length;
            document.getElementById('last-training').textContent = status.last_training || 'Never';

            // Update connection status
            const statusDot = document.getElementById('status-dot');
            const statusText = document.getElementById('status-text');
            
            if (status.model_loaded) {
                statusDot.className = 'w-3 h-3 bg-green-400 rounded-full';
                statusText.textContent = 'Connected';
            } else {
                statusDot.className = 'w-3 h-3 bg-red-400 rounded-full';
                statusText.textContent = 'Disconnected';
            }

            // Load models list
            this.loadModelsList();
        } catch (error) {
            console.error('Failed to load status:', error);
            document.getElementById('status-dot').className = 'w-3 h-3 bg-red-400 rounded-full';
            document.getElementById('status-text').textContent = 'Error';
        }
    }

    async loadModelsList() {
        try {
            const response = await fetch(`${this.apiBase}/v1/models`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            const modelsList = document.getElementById('models-list');
            
            if (data.models.length === 0) {
                modelsList.innerHTML = '<div class="text-gray-500 text-center py-4">No trained models available</div>';
                return;
            }

            modelsList.innerHTML = data.models.map(model => `
                <div class="bg-gray-50 p-3 rounded-md">
                    <div class="text-sm font-medium text-gray-700">${model.id}</div>
                    <div class="text-xs text-gray-500">Created: ${new Date(model.created).toLocaleString()}</div>
                    <button onclick="mindModelUI.loadModel('${model.id}')" class="mt-2 text-xs bg-blue-600 text-white py-1 px-2 rounded hover:bg-blue-700">
                        Load Model
                    </button>
                </div>
            `).join('');
        } catch (error) {
            console.error('Failed to load models:', error);
            document.getElementById('models-list').innerHTML = '<div class="text-red-500 text-center py-4">Failed to load models</div>';
        }
    }

    async loadModel(modelId) {
        try {
            const response = await fetch(`${this.apiBase}/v1/models/${modelId}/load`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            this.showToast(`Model ${modelId} loaded successfully`, 'success');
            this.loadStatus(); // Refresh status
        } catch (error) {
            this.showToast(`Failed to load model: ${error.message}`, 'error');
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `px-4 py-2 rounded-md text-white shadow-lg transform transition-all duration-300 translate-x-full`;
        
        switch (type) {
            case 'success':
                toast.className += ' bg-green-500';
                break;
            case 'error':
                toast.className += ' bg-red-500';
                break;
            case 'warning':
                toast.className += ' bg-yellow-500';
                break;
            default:
                toast.className += ' bg-blue-500';
        }
        
        toast.textContent = message;
        
        const container = document.getElementById('toast-container');
        container.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.classList.remove('translate-x-full');
        }, 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.add('translate-x-full');
            setTimeout(() => {
                container.removeChild(toast);
            }, 300);
        }, 3000);
    }
}

// Initialize the application
const mindModelUI = new MindModelUI();
