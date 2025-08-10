"""
SnapBind - Protein Binding Site Prediction Web Application

This Flask application provides a web interface for predicting protein binding sites
using a FastCNN model. It features interactive visualization, 3D protein structure
rendering, and comprehensive data export capabilities.

Author: SnapBind Team
Version: 1.0.0
License: MIT

Dependencies:
    - Flask: Web framework
    - PyTorch: Machine learning framework
    - NumPy: Numerical computing
    - 3Dmol.js: 3D molecular visualization (frontend)
    - Chart.js: Interactive plotting (frontend)

Features:
    - Real-time binding site prediction
    - Interactive amino acid sequence heatmaps
    - 3D protein structure visualization
    - Raw prediction data export (JSON/clipboard)
    - Responsive web interface
    - Multiple protein structure support

Usage:
    python app.py

    Then navigate to http://localhost:5004 in your browser.
"""

import base64
import io
import os
import random
import warnings

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file

from backend.pred import get_binding_arrays
from esm_backend import get_predictor, predict_binding_affinity

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Available models for prediction
MODELS = ["FastCNNBindingPredictor"]

# Available protein structures for 3D visualization
# These PDB files are used for structural context and visualization
PROTEIN_STRUCTURES = [
    {
        "name": "crambin.pdb",
        "path": "outputs/crambin.pdb",
        "description": "Crambin - Small plant protein (46 residues)",
    },
    {
        "name": "ubiquitin.pdb",
        "path": "outputs/ubiquitin.pdb",
        "description": "Ubiquitin - Regulatory protein (76 residues)",
    },
    {
        "name": "test_protein_2.pdb",
        "path": "outputs/test_protein_2.pdb",
        "description": "Reference protein structure",
    },
]


@app.route("/")
def index():
    """
    Render the main application page.

    Returns:
        str: Rendered HTML template with available models
    """
    return render_template("index.html", models=MODELS)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict protein binding sites using FastCNN model.

    Accepts a POST request with JSON payload containing:
        - sequence (str): Amino acid sequence to analyze

    Returns:
        JSON response containing:
            - success (bool): Whether prediction succeeded
            - chart_data (dict): Data for Chart.js visualization
            - binding_sites (list): High-confidence binding sites (>0.5 threshold)
            - binding_summary (str): Human-readable summary
            - model_used (str): Model identifier
            - sequence_length (int): Length of input sequence
            - protein_structure (dict): 3D structure information
            - raw_output (dict): Complete prediction data for export

    Error Response:
        JSON with error message if prediction fails
    """
    try:
        data = request.get_json()
        sequence = data.get("sequence", "").strip()
        # model = data.get("model", "ESM-2 (Real)")
        path_model_ckp = "checkpoints/best_binding_model.pth"
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not sequence:
            return jsonify({"error": "Please enter an amino acid sequence"})

        # Get binding site predictions from the FastCNN model
        probabilities, sequence_array = get_binding_arrays(
            protein_sequence=sequence,
            model_path=path_model_ckp,
            device=device,
        )

        # Prepare data for front-end Chart.js rendering
        chart_data = {
            "sequence": list(sequence),
            "probabilities": (
                probabilities.tolist()
                if hasattr(probabilities, "tolist")
                else list(probabilities)
            ),
            "positions": list(range(1, len(sequence) + 1)),
        }

        # Generate binding sites from probabilities
        # Only include sites with probability > 0.5 (high confidence)
        binding_sites = []
        for i, prob in enumerate(probabilities):
            if prob > 0.5:  # threshold for binding site
                binding_sites.append(
                    {
                        "position": i + 1,  # 1-indexed position
                        "probability": float(prob),
                        "amino_acid": sequence[i] if i < len(sequence) else "X",
                        "confidence": float(prob),
                    }
                )

        # Generate human-readable summary
        binding_summary = f"Identified {len(binding_sites)} potential binding sites with high confidence."

        # Select a random protein structure for 3D visualization
        random_structure = random.choice(PROTEIN_STRUCTURES)

        # Prepare comprehensive raw output for export/download
        raw_output = {
            "input_sequence": sequence,
            "sequence_length": len(sequence),
            "model_used": "FastCNN",
            "device_used": str(device),
            "binding_threshold": 0.5,
            "total_binding_sites": len(binding_sites),
            "binding_sites": binding_sites,
            "probabilities_array": (
                probabilities.tolist()
                if hasattr(probabilities, "tolist")
                else list(probabilities)
            ),
            "summary": binding_summary,
            "protein_structure": {
                "name": random_structure["name"],
                "description": random_structure["description"],
            },
        }

        return jsonify(
            {
                "success": True,
                "chart_data": chart_data,
                "binding_sites": binding_sites,
                "binding_summary": binding_summary,
                "model_used": "FastCNN",
                "sequence_length": len(sequence),
                "protein_structure": {
                    "name": random_structure["name"],
                    "url": f"/protein_structure/{random_structure['name']}",
                    "description": random_structure["description"],
                },
                "raw_output": raw_output,
            }
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"})


@app.route("/model_status")
def model_status():
    """
    Check if ESM model is loaded and ready for predictions.

    Returns:
        JSON response containing:
            - model_loaded (bool): Whether the model is loaded
            - device (str): Device being used (cpu/cuda/mps)
            - model_type (str): Type of model loaded
            - error (str, optional): Error message if loading failed
    """
    try:
        predictor = get_predictor()
        is_loaded = predictor.model is not None
        device = predictor.device if predictor else "unknown"

        return jsonify(
            {
                "model_loaded": is_loaded,
                "device": device,
                "model_type": "ESM-2" if is_loaded else "None",
            }
        )
    except Exception as e:
        return jsonify({"model_loaded": False, "error": str(e)})


@app.route("/protein_structure/<filename>")
def serve_protein_structure(filename):
    """
    Serve protein structure files (PDB format) for 3D visualization.

    Args:
        filename (str): Name of the PDB file to serve

    Returns:
        File response: PDB file with chemical/x-pdb MIME type
        JSON error: If file not found or error occurs

    Supported files:
        - crambin.pdb: Small plant protein
        - ubiquitin.pdb: Regulatory protein
        - test_protein_2.pdb: Reference structure
    """
    try:
        # Find the structure file in the PROTEIN_STRUCTURES list
        structure_path = None
        for structure in PROTEIN_STRUCTURES:
            if structure["name"] == filename:
                structure_path = structure["path"]
                break

        if not structure_path:
            return jsonify({"error": "Protein structure not found"}), 404

        # Use absolute path for the file
        if not os.path.isabs(structure_path):
            full_path = os.path.join(os.getcwd(), structure_path)
        else:
            full_path = structure_path

        if not os.path.exists(full_path):
            return (
                jsonify(
                    {"error": f"Protein structure file does not exist at {full_path}"}
                ),
                404,
            )

        print(f"Serving protein structure: {filename} from {full_path}")
        return send_file(full_path, as_attachment=False, mimetype="chemical/x-pdb")

    except Exception as e:
        print(f"Error serving protein structure: {e}")
        return jsonify({"error": f"Failed to serve protein structure: {str(e)}"}), 500


@app.route("/test_protein_files")
def test_protein_files():
    """
    Test endpoint to check availability and status of protein structure files.

    Useful for debugging and verifying that all required PDB files are present
    and accessible by the application.

    Returns:
        JSON response containing:
            - protein_files (list): List of file information including:
                - name (str): Filename
                - path (str): Full path to file
                - exists (bool): Whether file exists
                - size (int): File size in bytes
                - description (str): Human-readable description
    """
    results = []
    for structure in PROTEIN_STRUCTURES:
        full_path = (
            structure["path"]
            if os.path.isabs(structure["path"])
            else os.path.join(os.getcwd(), structure["path"])
        )
        exists = os.path.exists(full_path)
        size = os.path.getsize(full_path) if exists else 0
        results.append(
            {
                "name": structure["name"],
                "path": full_path,
                "exists": exists,
                "size": size,
                "description": structure["description"],
            }
        )

    return jsonify({"protein_files": results})


@app.route("/random_protein")
def get_random_protein():
    """
    Get information about a randomly selected protein structure.

    Used by the frontend to display different protein structures
    for variety in 3D visualization.

    Returns:
        JSON response containing:
            - success (bool): Whether operation succeeded
            - protein_structure (dict): Structure information including:
                - name (str): Filename
                - url (str): Endpoint URL to fetch the structure
                - description (str): Human-readable description
            - error (str, optional): Error message if failed
    """
    try:
        random_structure = random.choice(PROTEIN_STRUCTURES)
        return jsonify(
            {
                "success": True,
                "protein_structure": {
                    "name": random_structure["name"],
                    "url": f"/protein_structure/{random_structure['name']}",
                    "description": random_structure["description"],
                },
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to get random protein: {str(e)}"}), 500


if __name__ == "__main__":
    """
    Run the Flask development server.

    Configuration:
        - Debug mode: Enabled for development
        - Port: 5004 (to avoid conflicts with other services)
        - Host: 127.0.0.1 (localhost only)

    Note: For production deployment, use a proper WSGI server like Gunicorn.
    """
    app.run(debug=True, port=5004)
