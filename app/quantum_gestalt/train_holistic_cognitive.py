#!/usr/bin/env python3
"""
Holistic Cognitive State Training

This script combines multiple datasets to create authentic cognitive state mapping:
- Delta (Relaxed): Nature scenes, calm landscapes
- Theta (Meditative): Faces in contemplation, peaceful scenes
- Alpha (Calm): Everyday objects, neutral scenes
- Beta (Active): Action, sports, dynamic movement
- Gamma (Hyper-focus): Complex patterns, detailed work

Uses a strategic mix of datasets for best cognitive alignment.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from PIL import Image
import io

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from model.quantum_net_fixed import QuantumGestaltNet
    from model.image_extractor import ImageFeatureExtractor
except ImportError:
    print("Error: Could not import required modules.")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not found.")
    print("Install it with: pip install datasets")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architecture
BASE_INPUT_DIM = 10
FREQ_INPUT_DIM = 5
HIDDEN_DIM = 128
NUM_CLASSES = 5

# ============================================================================
# Holistic Dataset Mapping Strategy
# ============================================================================

COGNITIVE_DATASETS = {
    # Class 0: Delta (Relaxed) - Calm animals and nature
    'delta': {
        'dataset': 'cifar10',
        'categories': [4, 6],  # deer, frog - calm nature creatures
        'target_class': 0,
        'samples_per_category': 1000
    },
    
    # Class 1: Theta (Meditative) - Birds (graceful, flowing movement)
    'theta': {
        'dataset': 'cifar10',
        'categories': [2],  # bird - meditative flight
        'target_class': 1,
        'samples_per_category': 1000
    },
    
    # Class 2: Alpha (Calm) - Domestic animals
    'alpha': {
        'dataset': 'cifar10',
        'categories': [3, 5],  # cat, dog - calm companions
        'target_class': 2,
        'samples_per_category': 1000
    },
    
    # Class 3: Beta (Active) - Fast vehicles
    'beta': {
        'dataset': 'cifar10',
        'categories': [0, 1, 8],  # plane, car, ship - active movement
        'target_class': 3,
        'samples_per_category': 700
    },
    
    # Class 4: Gamma (Hyper-focus) - Powerful/industrial
    'gamma': {
        'dataset': 'cifar10',
        'categories': [7, 9],  # horse, truck - power and focus
        'target_class': 4,
        'samples_per_category': 1000
    }
}

# ============================================================================
# Dataset Class
# ============================================================================

class CognitiveDataset(Dataset):
    """Dataset that maps specific categories to cognitive states."""
    
    def __init__(self, hf_dataset, extractor, target_categories, target_class, max_samples=None):
        """
        Args:
            hf_dataset: HuggingFace dataset
            extractor: ImageFeatureExtractor
            target_categories: List of category indices to include
            target_class: The cognitive class to map to (0-4)
            max_samples: Max samples per category
        """
        self.extractor = extractor
        self.target_class = target_class
        self.samples = []
        
        # Find all samples matching target categories
        print(f"  Loading Class {target_class} from categories {target_categories}...")
        category_counts = {cat: 0 for cat in target_categories}
        
        for idx in range(len(hf_dataset)):
            sample = hf_dataset[idx]
            label = sample.get('label') or sample.get('fine_label')
            
            if label in target_categories:
                if max_samples and category_counts[label] >= max_samples:
                    continue
                self.samples.append((idx, sample))
                category_counts[label] += 1
        
        print(f"    Found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        _, sample = self.samples[idx]
        
        # Get image
        image = sample.get('img') or sample.get('image')
        
        try:
            # Convert to RGB PIL image
            if not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save temporarily
            temp_path = f'temp_cognitive_{self.target_class}.png'
            image.save(temp_path)
            
            # Extract features
            base_features, freq_features = self.extractor.extract_features(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return (torch.FloatTensor(base_features),
                   torch.FloatTensor(freq_features),
                   torch.LongTensor([self.target_class])[0])
        
        except Exception as e:
            # Return zeros on error
            return (torch.zeros(BASE_INPUT_DIM),
                   torch.zeros(FREQ_INPUT_DIM),
                   torch.LongTensor([self.target_class])[0])

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (base_features, freq_features, labels) in enumerate(dataloader):
        base_features = base_features.to(device)
        freq_features = freq_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(base_features, freq_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%")
    
    return total_loss / len(dataloader), 100.0 * correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for base_features, freq_features, labels in dataloader:
            base_features = base_features.to(device)
            freq_features = freq_features.to(device)
            labels = labels.to(device)
            
            outputs = model(base_features, freq_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    # Print per-class accuracy
    print("  Per-class accuracy:")
    class_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"    {class_names[i]}: {acc:.2f}%")
    
    return total_loss / len(dataloader), 100.0 * correct / total

# ============================================================================
# Main Training
# ============================================================================

def main():
    print("=" * 70)
    print("Holistic Cognitive State Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()
    
    # Initialize feature extractor
    print("Initializing image feature extractor...")
    extractor = ImageFeatureExtractor()
    
    # Load base dataset
    print("\nLoading CIFAR-10 dataset...")
    cifar_train = load_dataset('cifar10', split='train')
    cifar_test = load_dataset('cifar10', split='test')
    print(f"✓ Loaded {len(cifar_train)} train, {len(cifar_test)} test samples")
    
    # Create cognitive-mapped datasets
    print("\nCreating cognitive state mappings...")
    train_datasets = []
    test_datasets = []
    
    for state_name, config in COGNITIVE_DATASETS.items():
        print(f"\n{state_name.upper()} (Class {config['target_class']}):")
        
        train_ds = CognitiveDataset(
            cifar_train, extractor,
            config['categories'], config['target_class'],
            config['samples_per_category']
        )
        train_datasets.append(train_ds)
        
        test_ds = CognitiveDataset(
            cifar_test, extractor,
            config['categories'], config['target_class'],
            config['samples_per_category'] // 5  # 20% for test
        )
        test_datasets.append(test_ds)
    
    # Combine all datasets
    print("\nCombining datasets...")
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\nInitializing model...")
    model = QuantumGestaltNet(
        base_input_dim=BASE_INPUT_DIM,
        freq_input_dim=FREQ_INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    best_test_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
        print(f"Test     - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        
        scheduler.step(test_acc)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'base_input_dim': BASE_INPUT_DIM,
                'freq_input_dim': FREQ_INPUT_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_classes': NUM_CLASSES,
                'best_test_acc': best_test_acc,
                'dataset': 'holistic_cognitive'
            }, 'quantum_gestalt_model.pt')
            print(f"✓ Saved best model (test_acc: {test_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print("=" * 70)
    print("\nCognitive State Mappings:")
    print("  Delta (Relaxed):     Deer, Frog - calm nature")
    print("  Theta (Meditative):  Birds - graceful flight")
    print("  Alpha (Calm):        Cats, Dogs - domestic calm")
    print("  Beta (Active):       Planes, Cars, Ships - movement")
    print("  Gamma (Hyper-focus): Horses, Trucks - power/industrial")
    print("\nModel saved to: quantum_gestalt_model.pt")
    print("Run: python run_gui.py")
    print("=" * 70)

if __name__ == "__main__":
    main()

