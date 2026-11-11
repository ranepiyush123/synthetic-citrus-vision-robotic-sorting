
"""Train a PyTorch CNN on synthetic citrus images."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline import CLASSES, render


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "cnn_outputs"


class CitrusCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def generate_dataset(samples_per_class, seed, image_size=64):
    rng = np.random.default_rng(seed)

    labels = np.repeat(CLASSES, samples_per_class)
    seeds = rng.integers(
        0, 2**31 - 1, size=len(labels)
    )

    images = []

    for image_seed, label in zip(seeds, labels):
        image = render(
            int(image_seed),
            str(label),
            size=128,
        )

        image = image.resize((image_size, image_size))

        array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        images.append(array)

    X = np.stack(images)
    X = np.transpose(X, (0, 3, 1, 2))

    label_to_id = {
        name: idx for idx, name in enumerate(CLASSES)
    }

    y = np.array(
        [label_to_id[label] for label in labels],
        dtype=np.int64,
    )

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        labels,
    )


def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Generating synthetic citrus images...")

    X, y, labels = generate_dataset(
        args.samples_per_class,
        args.seed,
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.25,
        random_state=args.seed,
        stratify=labels,
    )

    train_dataset = TensorDataset(
        X[train_idx],
        y[train_idx],
    )

    test_dataset = TensorDataset(
        X[test_idx],
        y[test_idx],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CitrusCNN(len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    history = []

    print("Training CNN...")

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == targets
            ).sum().item()

            total += len(targets)

        epoch_loss = total_loss / total
        epoch_accuracy = correct / total

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Loss: {epoch_loss:.4f} "
            f"| Train accuracy: {epoch_accuracy:.4f}"
        )

    print("Evaluating held-out test images...")

    model.eval()

    true_labels = []
    predicted_labels = []
    confidence_scores = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(
                outputs, dim=1
            )

            confidence, predictions = (
                probabilities.max(dim=1)
            )

            true_labels.extend(targets.tolist())
            predicted_labels.extend(
                predictions.cpu().tolist()
            )
            confidence_scores.extend(
                confidence.cpu().tolist()
            )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    class_ids = list(range(len(CLASSES)))

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_ids,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_ids,
    )

    results = {
        "model": "PyTorch CitrusCNN",
        "dataset": "procedurally generated synthetic citrus",
        "samples_total": len(labels),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "seed": args.seed,
        "epochs": args.epochs,
        "test_accuracy": float(accuracy),
        "classes": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "training_history": history,
        "limitations": [
            "Synthetic-only evaluation",
            "No independent real-fruit dataset",
            "No DEM simulation",
            "No real robot integration",
        ],
    }

    metrics_path = OUTPUT_DIR / "cnn_metrics.json"
    model_path = OUTPUT_DIR / "citrus_cnn.pth"

    metrics_path.write_text(
        json.dumps(results, indent=2)
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "image_size": 64,
    }, model_path)

    print()
    print("CNN test accuracy:", f"{accuracy:.2%}")
    print("Confusion matrix:")
    print(matrix)
    print("Model saved:", model_path)
    print("Metrics saved:", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())
