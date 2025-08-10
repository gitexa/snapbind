"""
ESM Model Backend Service
Integrates the trained ESM binding site prediction model with the Flask frontend
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


class ESMTokenSite(nn.Module):
    """ESM model for binding site prediction"""

    def __init__(self, model_name, freeze_backbone=False, n_unfrozen_layers=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, 1)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if n_unfrozen_layers > 0 and hasattr(self.backbone, "encoder"):
                for layer in self.backbone.encoder.layer[-n_unfrozen_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.classifier(x).squeeze(-1)


class ESMPredictor:
    """Main predictor class that handles model loading and inference"""

    def __init__(self):
        self.device = self._pick_device()
        self.model = None
        self.tokenizer = None
        self.max_len = 1024
        self.model_name = "facebook/esm2_t12_35M_UR50D"
        self._load_model()

    def _pick_device(self):
        """Select the best available device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            print(f"Loading ESM model on device: {self.device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, do_lower_case=False
            )

            # Initialize model
            self.model = ESMTokenSite(
                self.model_name, freeze_backbone=True, n_unfrozen_layers=1
            ).to(self.device)

            # Try to load trained weights
            model_path = "runs_site/best_model.pt"
            if os.path.exists(model_path):
                print(f"Loading trained weights from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(
                    f"Model loaded successfully! Best validation ROC-AUC: {checkpoint.get('val_roc_auc', 'N/A')}"
                )
            else:
                print("No trained model found. Using randomly initialized weights.")
                print("Train the model first using the ESM notebook.")

            self.model.eval()

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None

    def _preprocess_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Preprocess protein sequence for model input"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")

        # Clean sequence - only valid amino acids
        valid = set("ACDEFGHIKLMNPQRSTVWYBXZU")
        seq_clean = "".join([c.upper() for c in sequence if c.upper() in valid])

        if not seq_clean:
            raise ValueError("No valid amino acids found in sequence")

        # Tokenize
        encoding = self.tokenizer(
            seq_clean,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
            "sequence": seq_clean,
        }

    def predict_binding_sites(self, sequence: str) -> Dict[str, Any]:
        """Predict binding sites for a protein sequence"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not properly loaded")

        try:
            # Preprocess sequence
            inputs = self._preprocess_sequence(sequence)

            with torch.no_grad():
                # Get model predictions
                logits = self.model(inputs["input_ids"], inputs["attention_mask"])

                # Convert logits to probabilities
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]

                # Get special token positions to exclude them
                special_tokens = {
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id,
                }

                input_ids = inputs["input_ids"].cpu().numpy()[0]
                attention_mask = inputs["attention_mask"].cpu().numpy()[0]

                # Find amino acid positions (exclude special tokens and padding)
                aa_positions = []
                aa_probs = []

                for i, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
                    if mask == 1 and token_id not in special_tokens:
                        aa_positions.append(i)
                        aa_probs.append(probabilities[i])

                # Map back to original sequence
                sequence_clean = inputs["sequence"]
                final_probs = aa_probs[: len(sequence_clean)]

                # Determine binding sites (threshold = 0.5)
                binding_threshold = 0.5
                binding_sites = []
                for i, prob in enumerate(final_probs):
                    if prob > binding_threshold:
                        binding_sites.append(
                            {
                                "position": i + 1,  # 1-indexed
                                "amino_acid": sequence_clean[i],
                                "probability": float(prob),
                            }
                        )

                return {
                    "sequence": sequence_clean,
                    "predictions": final_probs,
                    "binding_sites": binding_sites,
                    "binding_count": len(binding_sites),
                    "sequence_length": len(sequence_clean),
                }

        except Exception as e:
            print(f"Prediction error: {e}")
            raise ValueError(f"Failed to predict binding sites: {str(e)}")

    def generate_affinity_data(self, sequence: str, compound: str) -> Dict[str, Any]:
        """Generate mock binding affinity data based on binding site predictions"""
        try:
            # Get binding site predictions
            binding_results = self.predict_binding_sites(sequence)

            # Generate concentration range
            concentrations = np.logspace(-9, -3, 50)  # 1nM to 1mM

            # Base affinity calculation on number and strength of binding sites
            binding_sites = binding_results["binding_sites"]

            if binding_sites:
                # Calculate average binding probability
                avg_binding_prob = np.mean(
                    [site["probability"] for site in binding_sites]
                )
                # More binding sites and higher probability = better affinity (lower IC50)
                binding_strength = len(binding_sites) * avg_binding_prob

                # IC50 inversely related to binding strength
                base_ic50 = 1e-6 / (1 + binding_strength * 10)  # Range: ~10nM to 1μM

                # Add some compound-specific variation
                compound_factors = {
                    "Aspirin": 1.2,
                    "Caffeine": 0.8,
                    "Ibuprofen": 1.0,
                    "Paracetamol": 1.5,
                    "Morphine": 0.6,
                }

                ic50 = base_ic50 * compound_factors.get(compound, 1.0)

            else:
                # No binding sites predicted = poor affinity
                ic50 = np.random.uniform(1e-4, 1e-3)  # 100μM - 1mM

            # Generate dose-response curve
            hill_slope = np.random.uniform(0.8, 2.0)
            baseline = np.random.uniform(0, 5)
            top = np.random.uniform(85, 98)

            response = baseline + (top - baseline) / (
                1 + (concentrations / ic50) ** hill_slope
            )
            response += np.random.normal(0, 1.5, len(response))  # Add noise
            response = np.clip(response, 0, 100)  # Ensure 0-100% range

            return {
                "concentrations": concentrations.tolist(),
                "response": response.tolist(),
                "ic50": ic50,
                "binding_sites": binding_sites,
                "binding_summary": f"{len(binding_sites)} binding sites predicted",
            }

        except Exception as e:
            print(f"Affinity generation error: {e}")
            # Fallback to simple mock data
            concentrations = np.logspace(-9, -3, 50)
            ic50 = np.random.uniform(1e-8, 1e-5)
            hill_slope = np.random.uniform(0.5, 2.0)
            baseline = np.random.uniform(0, 10)
            top = np.random.uniform(80, 100)

            response = baseline + (top - baseline) / (
                1 + (concentrations / ic50) ** hill_slope
            )
            response += np.random.normal(0, 2, len(response))

            return {
                "concentrations": concentrations.tolist(),
                "response": response.tolist(),
                "ic50": ic50,
                "binding_sites": [],
                "binding_summary": "Prediction failed - using mock data",
            }


# Global predictor instance
predictor = None


def get_predictor():
    """Get or create the global predictor instance"""
    global predictor
    if predictor is None:
        predictor = ESMPredictor()
    return predictor


def predict_binding_affinity(
    sequence: str, compound: str, model_name: str = "ESM-2"
) -> Dict[str, Any]:
    """Main prediction function for the Flask app"""
    try:
        pred = get_predictor()
        if not pred.model:
            raise ValueError("ESM model not available")

        # Generate affinity data using real binding site predictions
        affinity_data = pred.generate_affinity_data(sequence, compound)

        return {
            "success": True,
            "concentrations": affinity_data["concentrations"],
            "response": affinity_data["response"],
            "ic50": affinity_data["ic50"],
            "binding_sites": affinity_data["binding_sites"],
            "model_used": model_name,
            "compound": compound,
            "binding_summary": affinity_data["binding_summary"],
        }

    except Exception as e:
        print(f"Prediction failed: {e}")
        return {"success": False, "error": str(e)}
