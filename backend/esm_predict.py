# --- Cell 1: Imports & Config ---
import ast
import json
import math
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# Pick device for Mac M1/M2
def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CFG = dict(
    esm_model="facebook/esm2_t12_35M_UR50D",  # Good balance for M1
    max_len=1024,
    batch_size=8,
    epochs=5,
    lr=2e-4,
    device=pick_device(),
    seed=42,
    num_workers=0,
    out_dir="runs_site",
)

os.makedirs(CFG["out_dir"], exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print(f"Using device: {CFG['device']}")


# --- Cell 2: Dataset utilities ---
CHAIN_ORDER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234")


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")


def pick_chain_cols(df: pd.DataFrame) -> Tuple[str, str]:
    for ch in ["A"] + CHAIN_ORDER:
        s_col = f"chain_{ch}_sequence"
        b_col = f"chain_{ch}_binding_array"
        if s_col in df.columns and b_col in df.columns:
            return s_col, b_col
    raise ValueError("No valid chain_*_sequence + chain_*_binding_array found")


def parse_binding_array(x):
    if isinstance(x, list):
        return x
    try:
        return list(ast.literal_eval(x))
    except Exception:
        return []


class SiteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok, max_len: int):
        self.tok, self.max_len = tok, max_len
        self.rows = []
        s_col, b_col = pick_chain_cols(df)
        for _, r in df.iterrows():
            seq = str(r[s_col]) if pd.notna(r[s_col]) else None
            arr = parse_binding_array(r[b_col]) if pd.notna(r[b_col]) else []
            if not seq:
                continue
            seq = seq[:max_len]
            if len(arr) < len(seq):
                arr = arr + [0] * (len(seq) - len(arr))
            arr = arr[: len(seq)]
            self.rows.append((seq, np.array(arr, dtype=np.int64)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        seq, y = self.rows[i]
        y = list(y)
        valid = set("ACDEFGHIKLMNPQRSTVWYBXZU")
        seq_clean, y_clean = [], []
        for c, lbl in zip(seq, y):
            cu = c.upper()
            if cu in valid:
                seq_clean.append(cu)
                y_clean.append(int(lbl))
        enc = self.tok(
            "".join(seq_clean),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)

        labels = torch.full_like(input_ids, -100)
        special = {self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id}
        aa_positions = [
            i for i, tid in enumerate(input_ids.tolist()) if tid not in special
        ]
        L = min(len(aa_positions), len(y_clean))
        if L > 0:
            labels[aa_positions[:L]] = torch.tensor(y_clean[:L], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    # --- Cell 3: Model ---


class ESMTokenSite(nn.Module):
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

    # --- Cell 4: Training & Evaluation Helpers ---


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def run_epoch(model, loader, criterion, optimizer=None):

    train = optimizer is not None
    model.train(train)
    total_loss, all_logits, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for b in tqdm(loader):
            ids = b["input_ids"].to(CFG["device"])
            msk = b["attention_mask"].to(CFG["device"])
            lbl = b["labels"].to(CFG["device"])
            logits = model(ids, msk)
            mask_valid = lbl != -100
            if mask_valid.sum() == 0:
                continue
            target = lbl.masked_select(mask_valid).float()
            pred = logits.masked_select(mask_valid)
            loss = criterion(pred, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * target.numel()
            all_logits.append(pred.detach().cpu())
            all_labels.append(target.detach().cpu())
    if not all_labels:
        return dict(
            loss=float("nan"),
            roc_auc=float("nan"),
            pr_auc=float("nan"),
            f1=float("nan"),
        )
    y = torch.cat(all_labels).numpy()
    s = torch.cat(all_logits).numpy()
    return dict(
        loss=total_loss / max(1, len(y)),
        roc_auc=roc_auc_score(y, s),
        pr_auc=average_precision_score(y, s),
        f1=f1_score(y, (1 / (1 + np.exp(-s)) >= 0.5).astype(int)),
    )


def plot_curves(y, s, tag):
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"ROC {tag}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()
    plt.figure()
    plt.plot(rec, prec)
    plt.title(f"PR {tag}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


# --- Cell 5: Training ---
def train_binding_site(
    train_csv, valid_csv=None, freeze_backbone=False, n_unfrozen_layers=0
):
    tok = AutoTokenizer.from_pretrained(CFG["esm_model"], do_lower_case=False)
    df_tr = safe_read_csv(train_csv)
    ds_tr_full = SiteDataset(df_tr, tok, CFG["max_len"])

    if valid_csv:
        df_va = safe_read_csv(valid_csv)
        ds_va = SiteDataset(df_va, tok, CFG["max_len"])
        ds_tr = ds_tr_full
    else:
        idx = np.arange(len(ds_tr_full))
        np.random.shuffle(idx)
        cut = int(0.9 * len(idx))
        ds_tr = torch.utils.data.Subset(ds_tr_full, idx[:cut])
        ds_va = torch.utils.data.Subset(ds_tr_full, idx[cut:])

    tr_loader = DataLoader(
        ds_tr, batch_size=CFG["batch_size"], shuffle=True, collate_fn=collate
    )
    va_loader = DataLoader(
        ds_va, batch_size=CFG["batch_size"], shuffle=False, collate_fn=collate
    )

    model = ESMTokenSite(CFG["esm_model"], freeze_backbone, n_unfrozen_layers).to(
        CFG["device"]
    )

    # class imbalance weight
    pos, neg = 1, 1
    for b in tr_loader:
        v = b["labels"]
        v = v[v != -100]
        pos += (v == 1).sum().item()
        neg += (v == 0).sum().item()
        break
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], device=CFG["device"])
    )

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
    
    for e in range(1, CFG["epochs"] + 1):
        tr_m = run_epoch(model, tr_loader, criterion, optimizer=opt)
        va_m = run_epoch(model, va_loader, criterion)

        print(
            f"Epoch {e}: Train ROC {tr_m['roc_auc']:.3f}, Val ROC {va_m['roc_auc']:.3f}"
        )
    return model, tok


# --- Cell 7: Evaluation on test set ---
def evaluate_on_csv(model, tok, test_csv):
    df_te = safe_read_csv(test_csv)
    ds_te = SiteDataset(df_te, tok, CFG["max_len"])
    te_loader = DataLoader(
        ds_te, batch_size=CFG["batch_size"], shuffle=False, collate_fn=collate
    )
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for b in tqdm(te_loader):
            ids = b["input_ids"].to(CFG["device"])
            msk = b["attention_mask"].to(CFG["device"])
            lbl = b["labels"].to(CFG["device"])
            logits = model(ids, msk)
            mask_valid = lbl != -100
            target = lbl.masked_select(mask_valid).float()
            pred = logits.masked_select(mask_valid)
            all_logits.append(pred.detach().cpu())
            all_labels.append(target.detach().cpu())
    y = torch.cat(all_labels).numpy()
    s = torch.cat(all_logits).numpy()
    print("ROC-AUC:", roc_auc_score(y, s), "PR-AUC:", average_precision_score(y, s))
    plot_curves(y, s, "Test")

    # --- Cell 6: Training Visualization ---


def plot_training_history(history_path=None, history_dict=None):
    """Plot training and validation metrics over epochs"""
    if history_path:
        with open(history_path, "r") as f:
            history = json.load(f)
    elif history_dict:
        history = history_dict
    else:
        history_path = os.path.join(CFG["out_dir"], "training_history.json")
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            logger.error(f"No training history found at {history_path}")
            return

    epochs = history["epoch"]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ROC-AUC plot
    ax2.plot(epochs, history["train_roc_auc"], "b-", label="Train ROC-AUC", linewidth=2)
    ax2.plot(epochs, history["val_roc_auc"], "r-", label="Val ROC-AUC", linewidth=2)
    ax2.set_title("ROC-AUC Score", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ROC-AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # PR-AUC plot
    ax3.plot(epochs, history["train_pr_auc"], "b-", label="Train PR-AUC", linewidth=2)
    ax3.plot(epochs, history["val_pr_auc"], "r-", label="Val PR-AUC", linewidth=2)
    ax3.set_title("Precision-Recall AUC", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PR-AUC")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # F1 score plot
    ax4.plot(epochs, history["train_f1"], "b-", label="Train F1", linewidth=2)
    ax4.plot(epochs, history["val_f1"], "r-", label="Val F1", linewidth=2)
    ax4.set_title("F1 Score", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("F1 Score")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(CFG["out_dir"], "training_progress.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    plt.show()


# --- Cell: Prepare Train/Val/Test from threaded_binding_sites_1000.csv ---
from sklearn.model_selection import train_test_split


def prepare_site_splits(csv_path, val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits threaded_binding_sites_1000.csv into train, validation, and test sets.
    Stratifies by overall binding presence to keep class balance.
    """
    df = pd.read_csv(csv_path)
    df = df.sample(n=20000, random_state=random_state).reset_index(drop=True)

    # Pick chain columns automatically
    s_col, b_col = pick_chain_cols(df)

    # Binary label: whether this chain has ANY binding site
    stratify_labels = df[b_col].apply(lambda x: 1 if "1" in str(x) else 0)

    # First split: train+val vs test
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )

    # Update stratify labels for train/val split
    stratify_labels_trainval = df_trainval[b_col].apply(
        lambda x: 1 if "1" in str(x) else 0
    )

    # Second split: train vs val
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=stratify_labels_trainval,
    )

    print(
        f"Train size: {len(df_train)}, Validation size: {len(df_val)}, Test size: {len(df_test)}"
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# --- Cell: Run the split and save ---
train_df, val_df, test_df = prepare_site_splits(
    "/Users/alex-mac/Programming/hackathon/data/threaded_binding_sites_1000.csv",
    val_size=0.1,
    test_size=0.1,
    random_state=CFG["seed"],
)

train_df.to_csv("site_train.csv", index=False)
val_df.to_csv("site_valid.csv", index=False)
test_df.to_csv("site_test.csv", index=False)


# Train model using train & val
model, tok = train_binding_site(
    "site_train.csv", "site_valid.csv", freeze_backbone=True, n_unfrozen_layers=1
)


evaluate_on_csv(model, tok, "site_test.csv")
