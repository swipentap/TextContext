"""
MindModel Python SDK Client

A Python client library for interacting with the MindModel API.
"""

import requests
import json
from typing import List, Dict, Optional, Union
from datetime import datetime
import time


class MindModelClient:
    """
    Client for interacting with the MindModel API.
    
    This client provides methods for generating conclusions, training the model,
    and managing the system status.
    """
    
    def __init__(self, base_url: str = "http://10.11.2.6:8000", timeout: int = 30):
        """
        Initialize the MindModel client.
        
        Args:
            base_url: The base URL of the MindModel API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response: The HTTP response
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise MindModelError(f"API request failed: {e}")
    
    def generate_conclusion(
        self, 
        input_text: str, 
        target_length: int = 24, 
        length_tag: Optional[str] = None
    ) -> Dict[str, Union[str, int, float]]:
        """
        Generate a conclusion from input text.
        
        Args:
            input_text: The text to generate a conclusion from
            target_length: Target length for the conclusion (8-50)
            length_tag: Optional length tag for the conclusion
            
        Returns:
            Dict containing the conclusion, length, and confidence
            
        Raises:
            MindModelError: If the request fails
            ValueError: If input parameters are invalid
        """
        if not input_text.strip():
            raise ValueError("Input text cannot be empty")
        
        if not (8 <= target_length <= 50):
            raise ValueError("Target length must be between 8 and 50")
        
        payload = {
            "input": input_text,
            "target_length": target_length
        }
        
        if length_tag:
            payload["length_tag"] = length_tag
        
        response = self._make_request(
            "POST", 
            "/v1/conclude", 
            json=payload
        )
        
        return response.json()
    
    def train_model(
        self, 
        examples: List[Dict], 
        epochs: int = 1, 
        batch_size: int = 8,
        model_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Train the model with new examples.
        
        Args:
            examples: List of training examples
            epochs: Number of training epochs (1-10)
            batch_size: Training batch size (1-32)
            model_name: Optional base model name
            
        Returns:
            Dict containing training ID, status, and message
            
        Raises:
            MindModelError: If the request fails
            ValueError: If input parameters are invalid
        """
        if not examples:
            raise ValueError("At least one training example is required")
        
        if not (1 <= epochs <= 10):
            raise ValueError("Epochs must be between 1 and 10")
        
        if not (1 <= batch_size <= 32):
            raise ValueError("Batch size must be between 1 and 32")
        
        # Validate examples
        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                raise ValueError(f"Example {i} must be a dictionary")
            
            if 'input' not in example or 'target' not in example:
                raise ValueError(f"Example {i} must contain 'input' and 'target' fields")
            
            if not example['input'].strip() or not example['target'].strip():
                raise ValueError(f"Example {i} input and target cannot be empty")
        
        payload = {
            "examples": examples,
            "epochs": epochs,
            "batch_size": batch_size
        }
        
        if model_name:
            payload["model_name"] = model_name
        
        response = self._make_request(
            "POST", 
            "/v1/learn", 
            json=payload
        )
        
        return response.json()
    
    def get_training_status(self, training_id: str) -> Dict[str, Union[str, int, datetime]]:
        """
        Get the status of a training job.
        
        Args:
            training_id: The ID of the training job
            
        Returns:
            Dict containing training status information
            
        Raises:
            MindModelError: If the request fails
        """
        response = self._make_request("GET", f"/v1/training/{training_id}")
        return response.json()
    
    def wait_for_training_completion(
        self, 
        training_id: str, 
        poll_interval: int = 5, 
        timeout: Optional[int] = None
    ) -> Dict[str, Union[str, int, datetime]]:
        """
        Wait for a training job to complete.
        
        Args:
            training_id: The ID of the training job
            poll_interval: How often to check status (seconds)
            timeout: Maximum time to wait (seconds, None for no timeout)
            
        Returns:
            Final training status
            
        Raises:
            MindModelError: If the request fails or timeout occurs
        """
        start_time = time.time()
        
        while True:
            status = self.get_training_status(training_id)
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            if timeout and (time.time() - start_time) > timeout:
                raise MindModelError(f"Training timeout after {timeout} seconds")
            
            time.sleep(poll_interval)
    
    def list_models(self) -> Dict[str, List[Dict]]:
        """
        Get a list of available trained models.
        
        Returns:
            Dict containing list of models
            
        Raises:
            MindModelError: If the request fails
        """
        response = self._make_request("GET", "/v1/models")
        return response.json()
    
    def load_model(self, model_id: str) -> Dict[str, str]:
        """
        Load a specific trained model.
        
        Args:
            model_id: The ID of the model to load
            
        Returns:
            Dict containing load status information
            
        Raises:
            MindModelError: If the request fails
        """
        response = self._make_request("POST", f"/v1/models/{model_id}/load")
        return response.json()
    
    def get_status(self) -> Dict[str, Union[bool, int, str]]:
        """
        Get the current system status.
        
        Returns:
            Dict containing system status information
            
        Raises:
            MindModelError: If the request fails
        """
        response = self._make_request("GET", "/status")
        return response.json()
    
    def health_check(self) -> Dict[str, str]:
        """
        Check if the service is healthy.
        
        Returns:
            Dict containing health status
            
        Raises:
            MindModelError: If the service is unhealthy
        """
        response = self._make_request("GET", "/health")
        return response.json()
    
    def is_healthy(self) -> bool:
        """
        Check if the service is healthy (returns boolean).
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            self.health_check()
            return True
        except MindModelError:
            return False
    
    def close(self):
        """Close the client session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MindModelError(Exception):
    """Exception raised for MindModel API errors."""
    pass


# Convenience functions for common operations

def quick_conclude(input_text: str, base_url: str = "http://10.11.2.6:8000") -> str:
    """
    Quick function to generate a conclusion from text.
    
    Args:
        input_text: The text to generate a conclusion from
        base_url: The base URL of the MindModel API
        
    Returns:
        The generated conclusion
        
    Raises:
        MindModelError: If the request fails
    """
    with MindModelClient(base_url) as client:
        result = client.generate_conclusion(input_text)
        return result['conclusion']


def batch_train(
    examples: List[Dict], 
    base_url: str = "http://10.11.2.6:8000",
    wait_for_completion: bool = True
) -> Dict[str, Union[str, int, datetime]]:
    """
    Train the model with examples and optionally wait for completion.
    
    Args:
        examples: List of training examples
        base_url: The base URL of the MindModel API
        wait_for_completion: Whether to wait for training to complete
        
    Returns:
        Training status information
        
    Raises:
        MindModelError: If the request fails
    """
    with MindModelClient(base_url) as client:
        result = client.train_model(examples)
        
        if wait_for_completion:
            return client.wait_for_training_completion(result['training_id'])
        
        return result


# Example usage and documentation

if __name__ == "__main__":
    # Example usage
    client = MindModelClient()
    
    try:
        # Generate a conclusion
        result = client.generate_conclusion(
            "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1."
        )
        print(f"Conclusion: {result['conclusion']}")
        print(f"Length: {result['length']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        # Train the model
        examples = [
            {
                "input": "The company reported $3.2M revenue in Q2 2024, representing a 15% increase over Q1.",
                "target": "Company revenue increased 15% to $3.2M in Q2.",
                "length_tag": "<LEN_24>",
                "phenomena_tags": ["numbers", "percentage"]
            }
        ]
        
        training_result = client.train_model(examples)
        print(f"Training started: {training_result['training_id']}")
        
        # Wait for completion
        final_status = client.wait_for_training_completion(training_result['training_id'])
        print(f"Training completed: {final_status['status']}")
        
        # Check system status
        status = client.get_status()
        print(f"Model loaded: {status['model_loaded']}")
        print(f"Currently training: {status['is_training']}")
        
    except MindModelError as e:
        print(f"Error: {e}")
    finally:
        client.close()
