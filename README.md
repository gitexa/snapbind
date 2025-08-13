# SnapBind 🧬

**2nd place Global AI Hackathon (Venture Capital Track) by** [**MIT Sloan AI Club and HackNation (2025)**](https://hack-nation.ai/)

![SnapBind Demo](video/snapbind.gif)

**Advanced Protein Binding Site Prediction with Deep Learning**

SnapBind is a sophisticated web application that predicts protein binding sites using a Fast Convolutional Neural Network (FastCNN). The application features an interactive web interface with real-time visualization, 3D protein structure rendering, and comprehensive data export capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

### 🔬 **Advanced Machine Learning**
- **FastCNN Architecture**: Optimized convolutional neural network with residual blocks
- **GPU Acceleration**: Supports CUDA, MPS (Apple Silicon), and CPU inference

### 🎨 **Interactive Web Interface**
- **Real-time Predictions**: Instant binding site analysis
- **Adaptive Heatmaps**: Responsive amino acid visualization that scales with sequence length
- **Interactive Charts**: Chart.js-powered probability plots

### 📊 **Comprehensive Data Export**
- **Raw Output Access**: Complete prediction data in JSON format
- **One-Click Copy**: Clipboard integration for easy data sharing
- **File Downloads**: Timestamped JSON exports for analysis
- **Structured Results**: Organized binding site information with confidence scores

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

<div align="center">
  <strong>Built with ❤️ for the scientific community</strong>
  <br>
  <sub>Advancing protein research through machine learning</sub>
</div>
