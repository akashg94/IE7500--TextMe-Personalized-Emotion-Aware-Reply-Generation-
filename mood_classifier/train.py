"""
train.py
--------
Training loop for the TextMe BiLSTM Mood Classifier.

Loads GoEmotions data, trains the MoodClassifier model,
saves the best checkpoint, and plots the loss curve.

Usage:
    python mood_classifier/train.py

Output:
    results/checkpoints/mood_classifier_best.pt
    results/loss_curves.png
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

from mood_classifier.model   import MoodClassifier, load_glove_matrix
from mood_classifier.dataset import create_dataloaders


def load_goemotions_data(data_dir):
    """
    Load GoEmotions TSV files from data directory.

    Args:
        data_dir (str): Path to data/ folder containing train.tsv, dev.tsv, test.tsv

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_path = os.path.join(data_dir, "train.tsv")
    val_path   = os.path.join(data_dir, "dev.tsv")
    test_path  = os.path.join(data_dir, "test.tsv")

    col_names = ["text", "labels", "id"]

    train_df = pd.read_csv(train_path, sep="\t", header=None, names=col_names)
    val_df   = pd.read_csv(val_path,   sep="\t", header=None, names=col_names)
    test_df  = pd.read_csv(test_path,  sep="\t", header=None, names=col_names)

    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")
    print(f"Test samples  : {len(test_df)}")

    return train_df, val_df, test_df


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full training epoch.

    Args:
        model     : MoodClassifier model
        loader    : Training DataLoader
        optimizer : Adam optimizer
        criterion : CrossEntropyLoss
        device    : torch device (cpu or cuda)

    Returns:
        tuple: (avg_loss, macro_f1)
    """
    model.train()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, macro_f1


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation or test set.

    Args:
        model     : MoodClassifier model
        loader    : Val or test DataLoader
        criterion : CrossEntropyLoss
        device    : torch device

    Returns:
        tuple: (avg_loss, macro_f1)
    """
    model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, macro_f1


def save_loss_curve(train_losses, val_losses, train_f1s, val_f1s, save_path):
    """
    Plot and save training curves for loss and F1 score.

    Args:
        train_losses (list): Training loss per epoch
        val_losses   (list): Validation loss per epoch
        train_f1s    (list): Training F1 per epoch
        val_f1s      (list): Validation F1 per epoch
        save_path    (str) : Where to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="coral")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 curve
    ax2.plot(epochs, train_f1s, label="Train F1", color="steelblue")
    ax2.plot(epochs, val_f1s,   label="Val F1",   color="coral")
    ax2.set_title("Macro F1 per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def train(
    num_epochs    = config.NUM_EPOCHS,
    learning_rate = config.LEARNING_RATE,
    patience      = config.PATIENCE,
    checkpoint_dir= config.CHECKPOINT_DIR,
    glove_path    = config.GLOVE_PATH,
    data_dir      = config.DATA_DIR,
):
    """
    Full training pipeline for the MoodClassifier.

    Args:
        num_epochs     (int)  : Maximum training epochs
        learning_rate  (float): Adam learning rate
        patience       (int)  : Early stopping patience
        checkpoint_dir (str)  : Where to save model checkpoints
        glove_path     (str)  : Path to GloVe embeddings
        data_dir       (str)  : Path to data directory
    """

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Load Data ──
    print("Loading GoEmotions dataset...")
    train_df, val_df, test_df = load_goemotions_data(data_dir)

    # ── Create DataLoaders ──
    print("\nBuilding vocabulary and DataLoaders...")
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_df, val_df, test_df
    )
    print(f"Vocabulary size : {len(vocab)}")
    print(f"Train batches   : {len(train_loader)}")
    print(f"Val batches     : {len(val_loader)}\n")

    # ── Load GloVe ──
    print("Building GloVe embedding matrix...")
    embed_matrix = load_glove_matrix(vocab, glove_path, config.EMBED_DIM)
    print()

    # ── Initialize Model ──
    model = MoodClassifier(
        vocab_size   = len(vocab),
        embed_matrix = embed_matrix
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}\n")

    # ── Loss, Optimizer, Scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Checkpoint directory ──
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "mood_classifier_best.pt")

    # ── Training Loop ──
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    train_losses = []
    val_losses   = []
    train_f1s    = []
    val_f1s      = []

    best_val_loss = float("inf")
    best_val_f1   = 0.0
    no_improve    = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Track history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1   = val_f1
            no_improve    = 0

            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    val_loss,
                "val_f1":      val_f1,
                "vocab":       vocab,
            }, checkpoint_path)

            print(f"  --> Best model saved (val_loss: {val_loss:.4f})")

        else:
            no_improve += 1
            print(f"  --> No improvement ({no_improve}/{patience})")

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # ── Save loss curve ──
    save_loss_curve(
        train_losses, val_losses,
        train_f1s, val_f1s,
        save_path="results/loss_curves.png"
    )

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Best Val F1   : {best_val_f1:.4f}")
    print(f"Checkpoint    : {checkpoint_path}")
    print("=" * 60)

    return model, vocab


if __name__ == "__main__":
    train()
