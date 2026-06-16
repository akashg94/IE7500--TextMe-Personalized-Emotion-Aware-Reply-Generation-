"""
evaluate.py
-----------
Evaluation script for the TextMe BiLSTM Mood Classifier.

Loads the saved checkpoint, runs inference on the test set,
and reports full evaluation metrics across all 18 mood classes.

Usage:
    python mood_classifier/evaluate.py

Output:
    - Classification report printed to console
    - results/confusion_matrix.png
    - results/evaluation_report.md
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

from mood_classifier.model   import MoodClassifier
from mood_classifier.dataset import create_dataloaders


def load_goemotions_data(data_dir):
    """
    Load GoEmotions TSV files from data directory.

    Args:
        data_dir (str): Path to data/ folder

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    col_names = ["text", "labels", "id"]

    train_df = pd.read_csv(
        os.path.join(data_dir, "train.tsv"),
        sep="\t", header=None, names=col_names
    )
    val_df = pd.read_csv(
        os.path.join(data_dir, "dev.tsv"),
        sep="\t", header=None, names=col_names
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, "test.tsv"),
        sep="\t", header=None, names=col_names
    )

    return train_df, val_df, test_df


def load_checkpoint(checkpoint_path, device):
    """
    Load saved model checkpoint.

    Args:
        checkpoint_path (str)       : Path to .pt checkpoint file
        device          (torch.device): cpu or cuda

    Returns:
        tuple: (model, vocab, epoch, val_f1)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}\n"
            f"Run mood_classifier/train.py first."
        )

    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab        = checkpoint["vocab"]
    epoch        = checkpoint["epoch"]
    val_f1       = checkpoint["val_f1"]
    val_loss     = checkpoint["val_loss"]
    embed_dim    = config.EMBED_DIM

    print(f"Checkpoint from epoch : {epoch}")
    print(f"Val loss at save      : {val_loss:.4f}")
    print(f"Val F1 at save        : {val_f1:.4f}")
    print(f"Vocabulary size       : {len(vocab)}")

    # Rebuild embedding matrix with zeros (weights loaded from checkpoint)
    embed_matrix = np.zeros((len(vocab), embed_dim), dtype="float32")

    model = MoodClassifier(
        vocab_size   = len(vocab),
        embed_matrix = embed_matrix
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Model loaded successfully.\n")
    return model, vocab, epoch, val_f1


def run_inference(model, loader, device):
    """
    Run model inference on a DataLoader and collect predictions.

    Args:
        model  : Trained MoodClassifier
        loader : Test DataLoader
        device : torch device

    Returns:
        tuple: (all_predictions, all_true_labels)
    """
    all_preds  = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            logits    = model(input_ids)
            preds     = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def save_confusion_matrix(all_labels, all_preds, class_names, save_path):
    """
    Generate and save confusion matrix as a heatmap.

    Args:
        all_labels  (list): True label indices
        all_preds   (list): Predicted label indices
        class_names (list): List of mood class name strings
        save_path   (str) : Path to save the PNG
    """
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        xticklabels= class_names,
        yticklabels= class_names,
        linewidths = 0.5
    )
    plt.title("Mood Classifier — Confusion Matrix", fontsize=14, pad=15)
    plt.xlabel("Predicted Mood", fontsize=11)
    plt.ylabel("True Mood",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_evaluation_report(
    all_labels, all_preds, class_names,
    macro_f1, macro_precision, macro_recall,
    checkpoint_epoch, save_path
):
    """
    Write evaluation results to a markdown report file.

    Args:
        all_labels        (list): True labels
        all_preds         (list): Predicted labels
        class_names       (list): Mood class names
        macro_f1          (float): Macro F1 score
        macro_precision   (float): Macro precision
        macro_recall      (float): Macro recall
        checkpoint_epoch  (int)  : Epoch the checkpoint was saved at
        save_path         (str)  : Path to save the .md file
    """
    report = classification_report(
        all_labels, all_preds,
        target_names = class_names,
        zero_division= 0
    )

    per_class_f1 = f1_score(
        all_labels, all_preds,
        average      = None,
        zero_division= 0
    )

    lines = []
    lines.append("# TextMe Mood Classifier — Evaluation Report\n")
    lines.append(f"**Checkpoint epoch:** {checkpoint_epoch}\n")
    lines.append("---\n")
    lines.append("## Overall Metrics\n")
    lines.append(f"| Metric | Score |")
    lines.append(f"|---|---|")
    lines.append(f"| Macro F1 | {macro_f1:.4f} |")
    lines.append(f"| Macro Precision | {macro_precision:.4f} |")
    lines.append(f"| Macro Recall | {macro_recall:.4f} |")
    lines.append(f"| Target F1 | 0.6500 |")
    lines.append(f"| Target met | {'Yes' if macro_f1 >= 0.65 else 'No'} |\n")
    lines.append("---\n")
    lines.append("## Per-Class F1 Scores\n")
    lines.append("| Mood Class | F1 Score |")
    lines.append("|---|---|")
    for name, score in zip(class_names, per_class_f1):
        lines.append(f"| {name} | {score:.4f} |")
    lines.append("\n---\n")
    lines.append("## Full Classification Report\n")
    lines.append("```")
    lines.append(report)
    lines.append("```\n")
    lines.append("---\n")
    lines.append("## Notes\n")

    # Find best and worst classes
    best_idx  = int(np.argmax(per_class_f1))
    worst_idx = int(np.argmin(per_class_f1))
    lines.append(f"- Best performing class  : **{class_names[best_idx]}** (F1: {per_class_f1[best_idx]:.4f})")
    lines.append(f"- Worst performing class : **{class_names[worst_idx]}** (F1: {per_class_f1[worst_idx]:.4f})")
    lines.append(f"- Total test samples     : {len(all_labels)}")
    lines.append(f"- Number of classes      : {len(class_names)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Evaluation report saved to {save_path}")


def evaluate():
    """
    Main evaluation function.
    Loads checkpoint, runs inference on test set, saves all results.
    """
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "mood_classifier_best.pt")
    class_names     = config.MOOD_CLASSES

    print("=" * 60)
    print("TextMe Mood Classifier — Evaluation")
    print("=" * 60 + "\n")

    # ── Load checkpoint ──
    model, vocab, checkpoint_epoch, _ = load_checkpoint(checkpoint_path, device)

    # ── Load test data ──
    print("Loading test data...")
    train_df, val_df, test_df = load_goemotions_data(config.DATA_DIR)

    _, _, test_loader, _ = create_dataloaders(
        train_df, val_df, test_df
    )
    print(f"Test samples : {len(test_df)}\n")

    # ── Run inference ──
    print("Running inference on test set...")
    all_preds, all_labels = run_inference(model, test_loader, device)
    print(f"Inference complete. {len(all_preds)} predictions made.\n")

    # ── Compute metrics ──
    macro_f1        = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall    = recall_score(all_labels, all_preds, average="macro",    zero_division=0)

    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Macro F1        : {macro_f1:.4f}  (target: 0.6500)")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Target met      : {'YES' if macro_f1 >= 0.65 else 'NO'}\n")

    # ── Full classification report ──
    print("PER-CLASS RESULTS")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names = class_names,
        zero_division= 0
    ))

    # ── Save confusion matrix ──
    os.makedirs("results", exist_ok=True)
    save_confusion_matrix(
        all_labels, all_preds,
        class_names,
        save_path="results/confusion_matrix.png"
    )

    # ── Save evaluation report ──
    save_evaluation_report(
        all_labels, all_preds,
        class_names,
        macro_f1, macro_precision, macro_recall,
        checkpoint_epoch,
        save_path="results/evaluation_report.md"
    )

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
