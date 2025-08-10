# SnapBind 🧬

![SnapBind Demo](videos/snapbind.gif)

**Advanced Protein Binding Site Prediction with Deep Learning**

SnapBind is a sophisticated web application that predicts protein binding sites using a Fast Convolutional Neural Network (FastCNN). The application features an interactive web interface with real-time visualization, 3D protein structure rendering, and comprehensive data export capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

### 🔬 **Advanced Machine Learning**
- **FastCNN Architecture**: Optimized convolutional neural network with residual blocks
- **Self-Attention Mechanism**: Captures long-range dependencies in protein sequences
- **Multi-Head Attention**: Enhanced feature extraction for binding site prediction
- **GPU Acceleration**: Supports CUDA, MPS (Apple Silicon), and CPU inference

### 🎨 **Interactive Web Interface**
- **Real-time Predictions**: Instant binding site analysis as you type
- **Adaptive Heatmaps**: Responsive amino acid visualization that scales with sequence length
- **Interactive Charts**: Chart.js-powered probability plots with hover details
- **3D Protein Visualization**: 3Dmol.js integration for structural context

### 📊 **Comprehensive Data Export**
- **Raw Output Access**: Complete prediction data in JSON format
- **One-Click Copy**: Clipboard integration for easy data sharing
- **File Downloads**: Timestamped JSON exports for analysis
- **Structured Results**: Organized binding site information with confidence scores

### 🧪 **Protein Structure Support**
- **Multiple PDB Files**: Crambin, Ubiquitin, and reference structures
- **Automatic Selection**: Random structure assignment for variety
- **3D Rendering**: Interactive molecular visualization with binding site highlighting
- **Structural Context**: Enhanced interpretation of binding predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- Modern web browser with WebGL support

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gitexa/snapbind.git
   cd snapbind
   ```

2. **Create a conda environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate esm-mps
   ```

   Or install with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download protein structures (optional):**
   ```bash
   # The app will download these automatically if missing
   curl -L "https://files.rcsb.org/download/1CRN.pdb" -o outputs/crambin.pdb
   curl -L "https://files.rcsb.org/download/1UBQ.pdb" -o outputs/ubiquitin.pdb
   ```

### Running the Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:5004`

3. **Enter a protein sequence:**
   ```
   Example: MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGE
   ```

4. **View results:**
   - Interactive heatmap showing binding probabilities
   - 3D protein structure with highlighted binding sites
   - Downloadable raw prediction data

## 📁 Project Structure

```
snapbind/
├── app.py                      # Flask web application
├── backend/
│   └── pred.py                # FastCNN model and prediction logic
├── templates/
│   └── index.html             # Web interface with interactive features
├── outputs/                   # PDB files and prediction outputs
├── checkpoints/               # Trained model weights
├── data/                      # Training datasets
├── training/                  # Model training scripts and notebooks
├── requirements.txt           # Python dependencies
├── environment.yaml           # Conda environment specification
└── README.md                  # This file
```

## 🧠 Model Architecture

### FastCNN Overview

The SnapBind model combines several advanced neural network components:

```python
Input Sequence → Embedding → Conv1D → Residual Blocks → Self-Attention → Classification
     ↓              ↓          ↓            ↓              ↓             ↓
   [MKWVT...]    [64-dim]   [Features]  [Skip Conn.]   [Long-range]  [Probabilities]
```

### Key Components

1. **Residual Convolutional Blocks**
   - Skip connections for gradient flow
   - Batch normalization and dropout
   - Multiple kernel sizes for feature diversity

2. **Self-Attention Layer**
   - Multi-head attention mechanism
   - Captures long-range amino acid interactions
   - Residual connections with layer normalization

3. **Classification Head**
   - Global average pooling
   - Fully connected layers with dropout
   - Sigmoid activation for probability output

## 📚 API Documentation

### Main Endpoints

#### `POST /predict`
Predict binding sites for a protein sequence.

**Request:**
```json
{
  "sequence": "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGE"
}
```

**Response:**
```json
{
  "success": true,
  "chart_data": {
    "sequence": ["M", "K", "W", ...],
    "probabilities": [0.12, 0.45, 0.78, ...],
    "positions": [1, 2, 3, ...]
  },
  "binding_sites": [
    {
      "position": 15,
      "probability": 0.85,
      "amino_acid": "S",
      "confidence": 0.85
    }
  ],
  "raw_output": {...}
}
```

#### `GET /protein_structure/<filename>`
Serve PDB files for 3D visualization.

**Supported files:**
- `crambin.pdb` - Small plant protein (46 residues)
- `ubiquitin.pdb` - Regulatory protein (76 residues)
- `test_protein_2.pdb` - Reference structure

## 🎯 Usage Examples

### Basic Prediction

```python
from backend.pred import get_binding_arrays

# Predict binding sites
probabilities, sequence_array = get_binding_arrays(
    protein_sequence="MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGE",
    model_path="checkpoints/best_binding_model.pth"
)

# Find high-confidence binding sites
binding_sites = [(i, prob) for i, prob in enumerate(probabilities) if prob > 0.5]
print(f"Found {len(binding_sites)} potential binding sites")
```

### Web Interface Features

1. **Sequence Input**: Paste or type amino acid sequences
2. **Real-time Analysis**: Instant prediction as you type
3. **Interactive Heatmap**: Click amino acids to see detailed information
4. **3D Visualization**: Rotate and zoom protein structures
5. **Data Export**: Copy or download complete prediction data

## 🔧 Configuration

### Model Parameters

```python
# Model configuration in backend/pred.py
FastCNNBindingPredictor(
    vocab_size=22,      # Standard amino acids + unknown
    embed_dim=64,       # Embedding dimension
    hidden_dim=128,     # Hidden layer size
    dropout=0.1         # Regularization
)
```

### Server Configuration

```python
# Server settings in app.py
app.run(
    debug=True,         # Enable debug mode
    port=5004,          # Server port
    host='127.0.0.1'    # Localhost only
)
```

## 🧪 Training Your Own Model

See [README_TRAINING.md](README_TRAINING.md) for detailed training instructions.

### Quick Training

```bash
# Install training dependencies
pip install -r requirements_training.txt

# Run training script
python train_esm_binding.py --epochs 10 --batch_size 16

# Test the trained model
python test_training.py
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new functions
- Update documentation for API changes

## 📊 Performance Metrics

The current model achieves:

- **Validation F1 Score**: 0.3561
- **Training Loss**: Converges within 10 epochs
- **Inference Speed**: ~50ms per sequence on GPU
- **Memory Usage**: <2GB GPU memory for typical sequences

## 🐛 Troubleshooting

### Common Issues

1. **Model loading errors**:
   ```bash
   # Check model file exists
   ls -la checkpoints/best_binding_model.pth
   ```

2. **WebGL not supported**:
   - Update your browser
   - Enable hardware acceleration
   - Check browser WebGL support

3. **Out of memory errors**:
   - Reduce batch size for training
   - Use CPU inference for very long sequences

4. **Protein structure not loading**:
   ```bash
   # Check PDB files
   python -c "import requests; print(requests.get('http://localhost:5004/test_protein_files').json())"
   ```

## 📖 Citation

If you use SnapBind in your research, please cite:

```bibtex
@software{snapbind2025,
  title={SnapBind: Advanced Protein Binding Site Prediction with Deep Learning},
  author={SnapBind Team},
  year={2025},
  url={https://github.com/gitexa/snapbind}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ESM-2**: Meta AI's protein language model
- **3Dmol.js**: Molecular visualization library
- **Chart.js**: Interactive charting library
- **PyTorch**: Deep learning framework
- **Flask**: Web application framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/gitexa/snapbind/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gitexa/snapbind/discussions)
- **Documentation**: [Wiki](https://github.com/gitexa/snapbind/wiki)

---

<div align="center">
  <strong>Built with ❤️ for the scientific community</strong>
  <br>
  <sub>Advancing protein research through machine learning</sub>
</div>
