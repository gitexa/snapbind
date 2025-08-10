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

# Available models
MODELS = ["FastCNNBindingPredictor"]

# Available protein structures for random selection
PROTEIN_STRUCTURES = [
    {
        "name": "crambin.pdb",
        "path": "/Users/alex-mac/Programming/hackathon/outputs/crambin.pdb",
        "description": "Crambin - Small plant protein (46 residues)",
    },
    {
        "name": "ubiquitin.pdb",
        "path": "/Users/alex-mac/Programming/hackathon/outputs/ubiquitin.pdb",
        "description": "Ubiquitin - Regulatory protein (76 residues)",
    },
    {
        "name": "test_protein_2.pdb",
        "path": "/Users/alex-mac/Programming/hackathon/outputs/test_protein_2.pdb",
        "description": "Reference protein structure",
    },
]


@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sequence = data.get("sequence", "").strip()
        # model = data.get("model", "ESM-2 (Real)")
        path_model_ckp = (
            "/Users/alex-mac/Programming/hackathon/checkpoinits/best_binding_model.pth"
        )
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not sequence:
            return jsonify({"error": "Please enter an amino acid sequence"})

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
        binding_sites = []
        for i, prob in enumerate(probabilities):
            if prob > 0.5:  # threshold for binding site
                binding_sites.append(
                    {
                        "position": i + 1,  # 1-indexed
                        "probability": float(prob),
                        "amino_acid": sequence[i] if i < len(sequence) else "X",
                        "confidence": float(prob),
                    }
                )

        # Generate summary
        binding_summary = f"Identified {len(binding_sites)} potential binding sites with high confidence."

        # Select a random protein structure for visualization
        random_structure = random.choice(PROTEIN_STRUCTURES)

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
            }
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"})


@app.route("/model_status")
def model_status():
    """Check if ESM model is loaded and ready"""
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
    """Serve protein structure files"""
    try:
        # Find the structure file
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
    """Test endpoint to check available protein files"""
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
    """Get a random protein structure info"""
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
    app.run(debug=True, port=5004)
