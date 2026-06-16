"""
evaluate.py
-----------
Evaluation script for the TextMe BiLSTM Mood Classifier.
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
    col_names = ["text", "labels", "id"]
    train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t", header=None, names=col_names)
    val_df   = pd.read_csv(os.path.join(data_dir, "dev.tsv"),   sep="\t", header=None, names=col_names)
    test_df  = pd.read_csv(os.path.join(data_dir, "test.tsv"),  sep="\t", header=None, names=col_names)
    return train_df, val_df, test_df


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}\nRun train.py first.")

    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint   = torch.load(checkpoint_path, map_location=device)
    vocab        = checkpoint["vocab"]
    epoch        = checkpoint["epoch"]
    val_f1       = checkpoint["val_f1"]
    val_loss     = checkpoint["val_loss"]

    print(f"Checkpoint from epoch : {epoch}")
    print(f"Val loss at save      : {val_loss:.4f}")
    print(f"Val F1 at save        : {val_f1:.4f}")
    print(f"Vocabulary size       : {len(vocab)}")

    embed_matrix = np.zeros((len(vocab), config.EMBED_DIM), dtype="float32")
    model = MoodClassifier(vocab_size=len(vocab), embed_matrix=embed_matrix).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Model loaded successfully.\n")
    return model, vocab, epoch, val_f1


def run_inference(model, loader, device):
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


def save_confusion_matrix(all_labels, all_preds, class_names, present_labels, save_path):
    present_names = [class_names[i] for i in present_labels]
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=present_names, yticklabels=present_names,
        linewidths=0.5
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


def save_evaluation_report(all_labels, all_preds, class_names, present_labels,
                            macro_f1, macro_precision, macro_recall,
                            checkpoint_epoch, save_path):
    present_names = [class_names[i] for i in present_labels]
    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    )
    per_class_f1 = f1_score(all_labels, all_preds, labels=present_labels,
                             average=None, zero_division=0)

    lines = []
    lines.append("# TextMe Mood Classifier — Evaluation Report\n")
    lines.append(f"**Checkpoint epoch:** {checkpoint_epoch}\n")
    lines.append("---\n")
    lines.append("## Overall Metrics\n")
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    lines.append(f"| Macro F1 | {macro_f1:.4f} |")
    lines.append(f"| Macro Precision | {macro_precision:.4f} |")
    lines.append(f"| Macro Recall | {macro_recall:.4f} |")
    lines.append(f"| Target F1 | 0.6500 |")
    lines.append(f"| Target met | {'Yes' if macro_f1 >= 0.65 else 'No'} |\n")
    lines.append("---\n")
    lines.append("## Per-Class F1 Scores\n")
    lines.append("| Mood Class | F1 Score |")
    lines.append("|---|---|")
    for name, score in zip(present_names, per_class_f1):
        lines.append(f"| {name} | {score:.4f} |")
    lines.append("\n---\n")
    lines.append("## Full Classification Report\n")
    lines.append("```")
    lines.append(report)
    lines.append("```\n")
    lines.append("---\n")
    lines.append("## Notes\n")
    best_idx  = int(np.argmax(per_class_f1))
    worst_idx = int(np.argmin(per_class_f1))
    lines.append(f"- Best performing class  : **{present_names[best_idx]}** (F1: {per_class_f1[best_idx]:.4f})")
    lines.append(f"- Worst performing class : **{present_names[worst_idx]}** (F1: {per_class_f1[worst_idx]:.4f})")
    lines.append(f"- Classes present in test: {len(present_labels)} out of {len(class_names)}")
    lines.append(f"- Total test samples     : {len(all_labels)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Evaluation report saved to {save_path}")


def evaluate():
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "mood_classifier_best.pt")
    class_names     = config.MOOD_CLASSES

    print("=" * 60)
    print("TextMe Mood Classifier — Evaluation")
    print("=" * 60 + "\n")

    model, vocab, checkpoint_epoch, _ = load_checkpoint(checkpoint_path, device)

    print("Loading test data...")
    train_df, val_df, test_df = load_goemotions_data(config.DATA_DIR)
    _, _, test_loader, _ = create_dataloaders(train_df, val_df, test_df)
    print(f"Test samples : {len(test_df)}\n")

    print("Running inference on test set...")
    all_preds, all_labels = run_inference(model, test_loader, device)
    print(f"Inference complete. {len(all_preds)} predictions made.\n")

    # Only use labels that actually appear in predictions or ground truth
    present_labels = sorted(list(set(all_labels) | set(all_preds)))

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

    present_names = [class_names[i] for i in present_labels]
    print("PER-CLASS RESULTS")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    os.makedirs("results", exist_ok=True)
    save_confusion_matrix(all_labels, all_preds, class_names, present_labels,
                          save_path="results/confusion_matrix.png")
    save_evaluation_report(all_labels, all_preds, class_names, present_labels,
                           macro_f1, macro_precision, macro_recall,
                           checkpoint_epoch, save_path="results/evaluation_report.md")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
