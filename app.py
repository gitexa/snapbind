import base64
import io
import os
import random
import warnings

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from matplotlib.backends.backend_agg import FigureCanvasAgg

from backend.pred import get_binding_arrays, plot_binding_probabilities
from esm_backend import get_predictor, predict_binding_affinity

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Available models
MODELS = ["FastCNNBindingPredictor"]

# Available protein structures for random selection
PROTEIN_STRUCTURES = [
    {
        "name": "test_protein.pdb",
        "path": "outputs/test_protein.pdb",
        "description": "Test protein structure",
    },
    {
        "name": "seq1.pdb",
        "path": "outputs/seq1.pdb",
        "description": "Sequence 1 protein structure",
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
        print(f"Binding probabilities: {probabilities}")

        # Generate plot
        plot_obj = plot_binding_probabilities(probabilities, sequence)
        img_buffer = io.BytesIO()
        plot_obj.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        cnn_plot_data = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

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
                "cnn_plot": cnn_plot_data,
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

        full_path = os.path.join(os.getcwd(), structure_path)

        if not os.path.exists(full_path):
            return jsonify({"error": "Protein structure file does not exist"}), 404

        return send_file(full_path, as_attachment=False, mimetype="chemical/x-pdb")

    except Exception as e:
        print(f"Error serving protein structure: {e}")
        return jsonify({"error": f"Failed to serve protein structure: {str(e)}"}), 500


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
    app.run(debug=True, port=5003)
