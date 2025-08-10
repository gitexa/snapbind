import ast
import os
import random
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)  # Skip connection!


class SelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, channels, L) -> (B, L, channels)
        x = x.transpose(1, 2)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)  # Residual connection
        return x.transpose(1, 2)  # Back to (B, channels, L)


class FastCNNBindingPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=20)

        # Multi-scale dilated convolutions
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    embed_dim, hidden_dim // 4, kernel_size=3, dilation=1, padding=1
                ),
                nn.Conv1d(
                    embed_dim, hidden_dim // 4, kernel_size=5, dilation=1, padding=2
                ),
                nn.Conv1d(
                    embed_dim, hidden_dim // 4, kernel_size=3, dilation=2, padding=2
                ),
                nn.Conv1d(
                    embed_dim, hidden_dim // 4, kernel_size=3, dilation=4, padding=4
                ),
            ]
        )

        self.res_blocks = nn.Sequential(
            ResidualConvBlock(hidden_dim, dropout),
            ResidualConvBlock(hidden_dim, dropout),
            ResidualConvBlock(hidden_dim, dropout),  # 3 residual blocks
        )

        self.attention = SelfAttention1D(hidden_dim)

        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Binary classification head - per-residue binding prediction
        self.classifier = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim // 2, kernel_size=3, padding=1
            ),  # Not kernel=1
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 4, 1, kernel_size=1),
        )

    def forward(self, x, mask=None):
        # Embedding
        x = self.embedding(x)
        x = x.transpose(1, 2)

        # Multi-scale convolutions (keep your current code)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)

        x = torch.cat(conv_outputs, dim=1)
        x = self.batch_norm(x)
        x = self.dropout(x)

        # NEW: Add residual blocks
        x = self.res_blocks(x)

        # NEW: Add attention
        x = self.attention(x)

        # Classification (with better head)
        x = self.classifier(x).squeeze(1)

        if mask is not None:
            x = x * mask.float()

        return x


def load_trained_model(model_path="best_binding_model2.pth", device="mps"):
    """Load the trained binding site prediction model."""

    # Initialize model with same architecture as training
    model = FastCNNBindingPredictor(
        embed_dim=64,  # Same as training
        hidden_dim=128,  # Same as training
        dropout=0.1,
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model with validation F1: {checkpoint['val_f1']:.4f}")
    return model


def sequence_to_tensor(sequence):
    """Convert amino acid sequence to tensor."""
    AA_TO_IDX = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "<PAD>": 20,
        "<UNK>": 21,
    }

    indices = [AA_TO_IDX.get(aa.upper(), AA_TO_IDX["<UNK>"]) for aa in sequence]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


def get_binding_arrays(
    protein_sequence, model_path="best_binding_model.pth", device=None
):

    # Auto-detect device if not provided
    if device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Load model
    model = load_trained_model(model_path, device)

    # Convert sequence to tensor
    seq_tensor = sequence_to_tensor(protein_sequence).to(device)

    # Create mask (all positions are valid for single sequence)
    mask = torch.ones_like(seq_tensor, dtype=torch.float).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(seq_tensor, mask)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # Convert sequence to character array
    sequence_array = np.array(list(protein_sequence))

    # Ensure both arrays have the same length
    assert len(probabilities) == len(
        sequence_array
    ), f"Length mismatch: probabilities ({len(probabilities)}) vs sequence ({len(sequence_array)})"

    return probabilities, sequence_array


def get_binding_arrays_with_model(protein_sequence, model, device):
    """
    Same as get_binding_arrays but with pre-loaded model (faster for multiple sequences).

    Args:
        protein_sequence (str): Amino acid sequence
        model: Pre-loaded model
        device: torch device

    Returns:
        tuple: (probabilities_array, sequence_array)
    """

    # Convert sequence to tensor
    seq_tensor = sequence_to_tensor(protein_sequence).to(device)

    # Create mask
    mask = torch.ones_like(seq_tensor, dtype=torch.float).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(seq_tensor, mask)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # Convert sequence to character array
    sequence_array = np.array(list(protein_sequence))

    return probabilities, sequence_array


def batch_get_binding_arrays(
    protein_sequences, model_path="best_binding_model.pth", device=None
):
    """
    Process multiple sequences efficiently.

    Args:
        protein_sequences (list): List of amino acid sequences
        model_path (str): Path to saved model checkpoint
        device: torch device (auto-detected if None)

    Returns:
        list: List of (probabilities_array, sequence_array) tuples
    """

    # Auto-detect device if not provided
    if device is None:
        device = torch.device(
            "mps"
            if torch.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Load model once
    model = load_trained_model(model_path, device)

    results = []
    for sequence in protein_sequences:
        probs, seq_array = get_binding_arrays_with_model(sequence, model, device)
        results.append((probs, seq_array))

    return results


# # Convert to pandas DataFrame for analysis
# probabilities, sequence = get_binding_arrays(
#     "RDPTQFEERHLKFLQQLGKGNFGSVEMCRYDPLQDNTGEVVAVKKLQHSTEEHLRDFEREIEILKSLQHDNIVKYKGVCYSAGRRNLKLIMEYLPYGSLRDYLQKHKERIDHIKLLQYTSQICKGMEYLGTKRYIHRDLATRNILVENENRVKIGDFGLTKVLPQDKEKVKEPGESPIFWYAPESLTESKFSVASDVWSFGVVLYELFTYIEKSKSPPAEFMRMIGNDKQGQMIVFHLIELLKNNGRLPRPDGCPDEIYMIMTECWNNNVNQRPSFRDLALRVDQIRDNMA"
# )


# Simple usage
sequence = "RDPTQFEERHLKFLQQLGKGNFGSVEMCRYDPLQDNTGEVVAVKKLQHSTEEHLRDFEREIEILKSLQHDNIVKYKGVCYSAGRRNLKLIMEYLPYGSLRDYLQKHKERIDHIKLLQYTSQICKGMEYLGTKRYIHRDLATRNILVENENRVKIGDFGLTKVLPQDKEKVKEPGESPIFWYAPESLTESKFSVASDVWSFGVVLYELFTYIEKSKSPPAEFMRMIGNDKQGQMIVFHLIELLKNNGRLPRPDGCPDEIYMIMTECWNNNVNQRPSFRDLALRVDQIRDNMA"
path_model_ckp = (
    "/Users/alex-mac/Programming/hackathon/checkpoinits/best_binding_model.pth"
)
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# print(probabilities, sequence)

# df = pd.DataFrame(
#     {
#         "position": range(1, len(sequence) + 1),
#         "amino_acid": sequence,
#         "binding_probability": probabilities,
#         "is_binding": probabilities > 0.5,
#     }
# )
# print(df.head())


def plot_binding_probabilities(probabilities, sequence):

    plt.figure(
        figsize=(18, 12)
    )  # Increased from (10, 20) to (18, 12) for better aspect ratio
    plt.plot(
        probabilities, linewidth=3, color="#2E86C1", alpha=0.8
    )  # Thicker line with better color
    plt.axhline(
        y=0.5,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Binding Threshold (50%)",
        alpha=0.8,
    )
    plt.xlabel("Residue Position", fontsize=14, fontweight="bold")
    plt.ylabel("Binding Probability", fontsize=14, fontweight="bold")
    plt.title(
        "Binding Site Prediction Analysis", fontsize=16, fontweight="bold", pad=20
    )

    # Improve x-axis labels for better readability
    step = max(
        1, len(sequence) // 50
    )  # Show every nth amino acid to avoid overcrowding
    plt.xticks(
        range(0, len(sequence), step),
        [f"{sequence[i]}\n{i+1}" for i in range(0, len(sequence), step)],
        rotation=0,
        fontsize=10,
    )

    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Enhance the plot appearance
    plt.ylim(-0.05, 1.05)  # Add some padding
    plt.tight_layout()

    # Add background color and styling
    plt.gca().set_facecolor("#f8f9fa")
    plt.gcf().patch.set_facecolor("white")

    return plt


probabilities, sequence_array = get_binding_arrays(
    protein_sequence=sequence,
    model_path=path_model_ckp,
    device=device,
)

plot_binding_probabilities(probabilities, sequence_array)
