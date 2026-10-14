"""
Cognitive State Predictor for Quantum-Gestalt Integration
==========================================================
Loads and runs the AI model to predict cognitive states from images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class HolisticCognitiveModel(nn.Module):
    """CNN for holistic cognitive state prediction."""
    
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CognitivePredictor:
    """Predict cognitive states from images."""
    
    COGNITIVE_STATES = {
        0: 'Delta (Relaxed)',
        1: 'Theta (Meditative)',
        2: 'Alpha (Calm)',
        3: 'Beta (Active)',
        4: 'Gamma (Hyper-focus)'
    }
    
    def __init__(self, model_path=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model file (default: quantum_gestalt_model.pt in same dir)
        """
        if model_path is None:
            # Default to model in same directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'quantum_gestalt_model.pt')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = HolisticCognitiveModel(num_classes=5)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model_loaded = True
                print(f"✓ Cognitive model loaded from {model_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load model from {model_path}: {e}")
                print("  Using untrained model (random predictions)")
                self.model_loaded = False
        else:
            print(f"⚠ Warning: Model file not found at {model_path}")
            print("  Using untrained model (random predictions)")
            self.model_loaded = False
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Cognitive Predictor initialized on {self.device}")
    
    def predict(self, image):
        """
        Predict cognitive state from image.
        
        Args:
            image: PIL Image or numpy array or path to image
            
        Returns:
            dict: {
                'predicted_class': int (0-4),
                'state_name': str,
                'probabilities': numpy array of 5 probabilities,
                'confidence': float (0-1)
            }
        """
        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
        
        return {
            'predicted_class': predicted_class,
            'state_name': self.COGNITIVE_STATES[predicted_class],
            'probabilities': probabilities,
            'confidence': confidence
        }
    
    def predict_probabilities(self, image):
        """
        Get just the probability distribution.
        
        Args:
            image: PIL Image, numpy array, or path
            
        Returns:
            numpy array: 5 probabilities for each cognitive state
        """
        result = self.predict(image)
        return result['probabilities']


def test_predictor():
    """Test the cognitive predictor."""
    print("Testing Cognitive Predictor...")
    
    # Create test image
    test_img = Image.new('RGB', (256, 256), color=(128, 128, 200))
    
    predictor = CognitivePredictor()
    result = predictor.predict(test_img)
    
    print(f"\nPrediction:")
    print(f"  State: {result['state_name']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Probabilities:")
    for i, prob in enumerate(result['probabilities']):
        print(f"    {predictor.COGNITIVE_STATES[i]}: {prob:.1%}")


if __name__ == "__main__":
    test_predictor()


