# 🧬 Protein Binding Site Prediction with CNNs

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage Examples](#usage-examples)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a deep learning approach for predicting protein-ligand binding sites using Convolutional Neural Networks (CNNs). The model takes protein sequences as input and predicts per-residue binding probabilities.

### Key Features
- **Multi-scale dilated convolutions** for capturing patterns at different scales
- **Residual blocks** for deep feature learning
- **Self-attention mechanisms** for long-range dependencies  
- **Train/test split** at PDB structure level to prevent data leakage
- **Comprehensive evaluation** with visualization tools

## Requirements

### Dependencies
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import string
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
```

### Hardware Requirements
- **Minimum**: 8GB RAM, GPU with 4GB VRAM
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM
- **Training time**: 2-8 hours depending on dataset size and hardware

## Data Preparation

### Dataset Format
Your CSV file should contain:
- `pdb_id`: Protein structure identifier
- `ligand_name`: Ligand identifier
- `chain_X_sequence`: Amino acid sequence for chain X
- `chain_X_binding_array`: Binary array indicating binding sites

### Dataset Split Strategy
```python
# Automatic 90/10 train/test split at PDB level
train_dataset = BioLiPDataset(csv_file, split='train')  # 90% of PDB structures
test_dataset = BioLiPDataset(csv_file, split='test')   # 10% of PDB structures
```

### Data Processing
- Sequences filtered by length (20-1000 residues)
- 22 amino acid vocabulary (20 standard + PAD + UNK)
- Binding arrays parsed from string representations
- Automatic padding and masking for variable lengths

## Model Architecture

### FastCNNBindingPredictor
```python
FastCNNBindingPredictor(
    vocab_size=22,      # 20 AA + PAD + UNK
    embed_dim=64,       # Embedding dimension
    hidden_dim=128,     # Hidden layer size
    dropout=0.1         # Dropout rate
)
```

#### Architecture Components:
1. **Embedding Layer**: Converts amino acids to dense vectors
2. **Multi-scale Convolutions**: 
   - Kernel sizes: 3, 5 with dilations 1, 2, 4
   - Captures local and distant patterns
3. **Residual Blocks**: Deep feature learning with skip connections
4. **Self-Attention**: Global context modeling
5. **Classification Head**: Per-residue binding prediction

### Enhanced Model
For higher accuracy, use `EnhancedCNNBindingPredictor`:
- 4x more parameters (512 hidden dim vs 128)
- 8 multi-scale convolution layers
- 8 residual blocks with different kernel sizes
- Multiple attention heads
- Channel attention and squeeze-excitation

## Training

### Basic Training Setup
```python
def main():
    # Configuration
    CSV_FILE = 'merged_protein_dataset.csv'
    BATCH_SIZE = 32
    MAX_LENGTH = 300
    MIN_LENGTH = 20
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-4
    
    # Load training data (90% of PDB structures)
    dataset = BioLiPDataset(CSV_FILE, max_length=MAX_LENGTH, 
                           min_length=MIN_LENGTH, split='train')
    
    # Create train/validation split
    train_dataset, val_dataset = create_train_val_split(dataset)
    
    # Initialize model
    model = FastCNNBindingPredictor(embed_dim=64, hidden_dim=128)
    
    # Train
    train_losses, val_f1_scores = train_model(
        train_loader, val_loader, model, device, 
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE
    )
```

### Training Features
- **Class imbalance handling**: Positive weight calculation
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: Reduces LR on plateau
- **Gradient clipping**: Stable training
- **Model checkpointing**: Saves best F1 score model

### Loss Function
```python
# Handles class imbalance automatically
pos_weight = compute_pos_weight(train_loader)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
```

## Evaluation

### Test Set Evaluation
```python
# Evaluate on held-out test set (10% of PDB structures)
evaluator, metrics = run_comprehensive_evaluation(
    model_path='best_binding_model.pth',
    csv_file='merged_protein_dataset.csv',
    subset_size=2000
)
```

### Evaluation Metrics
- **Overall Performance**: Accuracy, Precision, Recall, F1-Score, AUC
- **Probability Distributions**: Separation between binding/non-binding sites
- **No-binding Analysis**: False positive rates on sequences without binding sites
- **Visualization**: ROC curves, PR curves, probability histograms

### Visual Evaluation
```python
# Create detailed comparison plots
results = run_csv_evaluation(
    model_path='best_binding_model.pth',
    csv_file='merged_protein_dataset.csv',
    num_sequences=8,
    detailed_plots=True
)
```

Generates:
- **Per-sequence plots**: Predicted probabilities vs ground truth
- **Overview plots**: Multiple sequences comparison
- **Performance metrics**: Per-sequence F1, precision, recall

## Usage Examples

### 1. Training a New Model
```python
if __name__ == '__main__':
    main()  # Trains model and saves to 'best_binding_model.pth'
```

### 2. Loading and Testing Model
```python
# Test on first sequence from test set
results = test_model_on_first_sequence(
    model_path='best_binding_model.pth',
    csv_file='merged_protein_dataset.csv'
)

# Visualize results
fig = visualize_first_test_result(results)
```

### 3. Predicting Custom Sequence
```python
# Load model
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
model = load_trained_model('best_binding_model.pth', device)

# Predict
sequence = "MKVLWAALLVTFLAG..."  # Your protein sequence
result = predict_binding_sites(model, sequence, device=device)

# Visualize
visualize_binding_prediction(result)
```

### 4. Comprehensive Evaluation
```python
# Full evaluation on test set
evaluator, metrics = run_comprehensive_evaluation(
    model_path='best_binding_model.pth',
    csv_file='merged_protein_dataset.csv'
)

# Print detailed report
evaluator.print_detailed_report(metrics)

# Generate plots
evaluator.plot_evaluation_results('evaluation_results.png')
```

## File Structure

```
project/
├── merged_protein_dataset.csv          # Main dataset
├── best_binding_model.pth             # Saved model checkpoint
├── training_curves.png                # Training progress plots
├── evaluation_plots.png               # Comprehensive evaluation
├── detailed_comparison_seq_*.png       # Per-sequence comparisons
└── training.ipynb                 # Main training notebook
```

## Troubleshooting

### Common Issues

#### 1. **Memory Errors**
```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Use smaller model
model = FastCNNBindingPredictor(embed_dim=32, hidden_dim=64)
```

#### 2. **Low F1 Scores**
- Check class imbalance: `pos_weight` should be > 1
- Increase model capacity: Use `EnhancedFastCNNBindingPredictor`
- Lower learning rate: Try `1e-4` instead of `2e-4`
- More epochs: Increase to 20-50 epochs

#### 3. **Overfitting**
- Increase dropout: Try 0.2-0.3
- Add weight decay: `weight_decay=1e-3` in optimizer
- Reduce model size: Lower `hidden_dim`

#### 4. **Data Loading Errors**
```python
# Check CSV format
print(dataset.data.head())
print(dataset.data.columns)

# Verify sequences are loaded
print(f"Loaded {len(dataset)} sequences")
print(f"Sample sequence: {dataset[0]['sequence']}")
```

#### 5. **Device Compatibility**
```python
# Auto-detect best device
device = torch.device('mps' if torch.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Performance Tips

1. **For Better Accuracy**:
   - Use `EnhancedFastCNNBindingPredictor`
   - Train for more epochs (50-100)
   - Use ensemble of multiple models

2. **For Faster Training**:
   - Increase batch size (if memory allows)
   - Use smaller model dimensions
   - Sample smaller dataset subset

3. **For Debugging**:
   - Set `verbose=True` in evaluation functions
   - Use `subset_size=100` for quick testing
   - Check individual sequence predictions first

### Expected Performance

| Model | Parameters | F1-Score Range | Training Time |
|-------|------------|----------------|---------------|
| FastCNN | ~2M | 0.15-0.35 | 1-2 hours |
| Enhanced | ~15M | 0.25-0.50 | 4-8 hours |

*Performance varies significantly based on dataset quality and class balance.*

---

## Citation

If you use this code, please cite:
```bibtex
@misc{protein_binding_cnn,
  title={Deep Learning for Protein-Ligand Binding Site Prediction},
  author={Alex Haas, Elias Bruss, David Barth},
  year={2025},
  url={https://github.com/gitexa/snapbind}
}
```
