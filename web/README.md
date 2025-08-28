# MindModel Web UI

A modern, interactive web interface for the MindModel conclusion generation service.

## Features

### 🧪 Test Tab
- **Interactive Testing**: Generate conclusions from any input text
- **Length Control**: Set target length and length tags
- **Quick Actions**: Copy results or add them to training queue
- **Real-time Feedback**: See generation progress and results

### 🎓 Teach Tab
- **Single Examples**: Add training examples one by one
- **Training Queue**: Build up a collection of examples before training
- **Metadata Support**: Add length tags and phenomena tags
- **Queue Management**: Remove examples or clear the entire queue

### 📁 Batch Upload Tab
- **JSONL Support**: Upload files with multiple training examples
- **File Preview**: See the first 5 examples before training
- **Batch Training**: Train on large datasets efficiently
- **File Validation**: Automatic format checking

### 📊 Status Tab
- **System Monitoring**: Check model loading and training status
- **Model Management**: View and load different trained models
- **Real-time Updates**: Refresh status and see current state
- **Connection Status**: Visual indicator of API connectivity

## Usage

1. **Access the UI**: Navigate to `http://10.11.2.6:8000`
2. **Test the Model**: Use the Test tab to generate conclusions
3. **Teach the Model**: Use the Teach tab to add examples
4. **Upload Data**: Use the Batch Upload tab for large datasets
5. **Monitor Status**: Use the Status tab to check system health

## API Integration

The web UI communicates with the MindModel API endpoints:
- `POST /v1/conclude` - Generate conclusions
- `POST /v1/learn` - Train the model
- `GET /v1/models` - List available models
- `GET /status` - Get system status

## Technical Details

- **Framework**: Vanilla JavaScript with Tailwind CSS
- **Responsive**: Works on desktop and mobile devices
- **Real-time**: Live status updates and notifications
- **Error Handling**: Comprehensive error messages and recovery
- **Accessibility**: Keyboard navigation and screen reader support
