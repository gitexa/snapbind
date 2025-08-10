import base64
import io
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg

from esm_backend import get_predictor, predict_binding_affinity

# TODO: Add your model imports here
# from your_esm_model import predict_with_esm_model
# from your_cnn_model import FastCNNBindingPredictor, load_trained_model

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Available models
MODELS = ["FastCNNBindingPredictor"]


def create_cnn_binding_plot(sequence, binding_sites, compound_name):
    """Create a CNN-style binding site prediction plot"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8), height_ratios=[3, 1, 1])

    positions = np.arange(1, len(sequence) + 1)

    # Create probability array from binding sites
    probabilities = np.zeros(len(sequence))
    binary_predictions = np.zeros(len(sequence))

    for site in binding_sites:
        pos = site.get("position", 0) - 1  # Convert to 0-indexed
        if 0 <= pos < len(sequence):
            prob = site.get("probability", 0.5)
            probabilities[pos] = prob
            binary_predictions[pos] = 1 if prob > 0.5 else 0

    # Top plot: Probability curve with highlights
    ax1.plot(
        positions,
        probabilities,
        linewidth=2,
        color="blue",
        alpha=0.8,
        label="Binding Probability",
    )
    ax1.fill_between(positions, probabilities, alpha=0.3, color="blue")
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Threshold (0.5)")

    # Highlight predicted binding sites
    binding_positions = [
        site["position"] for site in binding_sites if site.get("probability", 0) > 0.5
    ]
    if binding_positions:
        binding_probs = [probabilities[pos - 1] for pos in binding_positions]
        ax1.scatter(
            binding_positions,
            binding_probs,
            color="red",
            s=40,
            alpha=0.8,
            zorder=5,
            label="Predicted Binding Sites",
        )

    ax1.set_ylabel("Binding Probability", fontsize=12)
    ax1.set_title(
        f"CNN Model Predictions - {compound_name}\n"
        f"Length: {len(sequence)} residues, "
        f"Predicted sites: {len(binding_positions)} ({len(binding_positions)/len(sequence)*100:.1f}%)\n"
        f"Max prob: {max(probabilities):.3f}, "
        f"Mean prob: {np.mean(probabilities):.3f}",
        fontsize=14,
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(1, len(sequence))

    # Middle plot: Binary predictions as bars
    ax2.bar(
        positions,
        binary_predictions,
        width=1.0,
        color="red",
        alpha=0.7,
        edgecolor="darkred",
    )
    ax2.set_ylabel("Predicted\nBinding", fontsize=10)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(sequence))

    # Bottom plot: Amino acid sequence with color coding
    ax3.set_xlim(1, len(sequence))
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_xlabel("Residue Position", fontsize=12)
    ax3.set_ylabel("Amino\nAcids", fontsize=10)

    # Color-code amino acids by type
    aa_colors = {
        "hydrophobic": ["A", "V", "I", "L", "M", "F", "Y", "W"],
        "polar": ["S", "T", "N", "Q", "C"],
        "positive": ["K", "R", "H"],
        "negative": ["D", "E"],
        "special": ["G", "P"],
    }

    color_map = {
        "hydrophobic": "#FFE4E1",  # Light pink
        "polar": "#E6F3FF",  # Light blue
        "positive": "#FFE6E6",  # Light red
        "negative": "#E6FFE6",  # Light green
        "special": "#F0F0F0",  # Light gray
    }

    for i, aa in enumerate(sequence):
        aa_type = "special"
        for type_name, aa_list in aa_colors.items():
            if aa.upper() in aa_list:
                aa_type = type_name
                break

        color = color_map[aa_type]
        if binary_predictions[i] == 1:  # Highlight predicted binding sites
            color = "#FF4444"  # Bright red for binding sites

        # Create text box for amino acid
        rect = patches.Rectangle(
            (i + 0.7, -0.3),
            0.6,
            0.6,
            linewidth=0.5,
            edgecolor="black",
            facecolor=color,
            alpha=0.8,
        )
        ax3.add_patch(rect)
        ax3.text(
            i + 1,
            0,
            aa.upper(),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return img_data


def create_plot(
    concentrations, response, sequence, smiles_name, model, ic50, binding_sites=None
):
    """Create affinity curve plot with binding site information"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Dose-response curve
    ax1.semilogx(
        np.array(concentrations) * 1e9,
        response,
        "b-",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.set_xlabel("Concentration (nM)")
    ax1.set_ylabel("% Inhibition")
    ax1.set_title(
        f"SBinding Affinity Prediction\n{smiles_name} vs Protein\nModel: {model}"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Add IC50 annotation
    ax1.axvline(ic50 * 1e9, color="red", linestyle="--", alpha=0.7)
    ax1.text(
        ic50 * 1e9,
        50,
        f"IC50 = {ic50*1e9:.1f} nM",
        rotation=90,
        verticalalignment="center",
        color="red",
    )

    # Right plot: Binding site visualization
    if binding_sites and len(binding_sites) > 0:
        positions = [site["position"] for site in binding_sites]
        probabilities = [site["probability"] for site in binding_sites]

        ax2.scatter(
            positions, probabilities, c=probabilities, cmap="Reds", s=60, alpha=0.8
        )
        ax2.set_xlabel("Amino Acid Position")
        ax2.set_ylabel("Binding Probability")
        ax2.set_title(f"Predicted Binding Sites\n{len(binding_sites)} sites identified")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label("Binding Probability")
    else:
        ax2.text(
            0.5,
            0.5,
            "No binding sites\npredicted",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title("Predicted Binding Sites\nNo sites identified")

    plt.tight_layout()

    # Save plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return img_data


@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sequence = data.get("sequence", "").strip()
        model = data.get("model", "ESM-2 (Real)")

        if not sequence:
            return jsonify({"error": "Please enter an amino acid sequence"})

        print(sequence)

        # REPLACE THIS SECTION WITH YOUR ACTUAL MODEL PREDICTIONS
        if model == "CNN Model":
            # TODO: Add your CNN model prediction here
            # from your_cnn_model import FastCNNBindingPredictor, predict_binding
            # model_instance = load_model("path/to/your/model.pth")
            # binding_sites = predict_binding(model_instance, sequence)
            pass

        # For demo purposes, generate random predictions (REMOVE THIS WHEN ADDING REAL MODELS)
        print(f"Generating demo predictions for model: {model}")

        # Generate random binding sites for visualization
        seq_length = len(sequence)
        np.random.seed(len(sequence) + hash(model) % 1000)

        # Create random binding sites (5-15% of sequence length)
        num_binding_sites = max(1, int(seq_length * np.random.uniform(0.05, 0.15)))
        binding_positions = sorted(
            np.random.choice(seq_length, size=num_binding_sites, replace=False) + 1
        )

        # Generate binding site data with probabilities
        binding_sites = []
        for pos in binding_positions:
            binding_sites.append(
                {
                    "position": int(pos),
                    "probability": float(np.random.uniform(0.6, 0.95)),
                    "amino_acid": sequence[pos - 1] if pos - 1 < len(sequence) else "X",
                    "confidence": float(np.random.uniform(0.7, 0.9)),
                }
            )

        # Generate mock dose-response data
        concentrations = np.logspace(-9, -3, 50)
        ic50 = np.random.uniform(1e-8, 1e-5)
        hill_slope = np.random.uniform(0.5, 2.0)
        baseline = np.random.uniform(0, 10)
        top = np.random.uniform(80, 100)

        response = baseline + (top - baseline) / (
            1 + (concentrations / ic50) ** hill_slope
        )
        response += np.random.normal(0, 2, len(response))

        # Create plots
        plot_data = create_plot(
            concentrations,
            response,
            sequence,
            "Target Compound",
            model,
            ic50,
            binding_sites,
        )

        # Create CNN-style binding site plot
        cnn_plot_data = create_cnn_binding_plot(
            sequence, binding_sites, "Target Compound"
        )

        # Generate summary
        binding_summary = f"Identified {len(binding_sites)} potential binding sites with high confidence. "
        binding_summary += f"Key binding regions include positions {', '.join(map(str, binding_positions[:3]))}."

        return jsonify(
            {
                "success": True,
                "plot": plot_data,
                "cnn_plot": cnn_plot_data,
                "ic50": f"{ic50*1e9:.2f} nM",
                "model_used": model,
                "compound": "Target Compound",
                "binding_sites": binding_sites,
                "binding_summary": binding_summary,
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


if __name__ == "__main__":
    app.run(debug=True, port=5003)
