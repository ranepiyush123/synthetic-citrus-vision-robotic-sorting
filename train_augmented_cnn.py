
"""Train an augmented citrus CNN while preserving the original baseline."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pipeline import CLASSES, render
from train_cnn import CitrusCNN

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "augmented_outputs"


class CitrusDataset(Dataset):
    def __init__(self, seeds, labels, augment=False):
        self.seeds = seeds
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = render(
            int(self.seeds[index]),
            CLASSES[int(self.labels[index])]
        )

        if self.augment:
            # Random illumination and colour changes
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.45, 1.5)
            )
            image = ImageEnhance.Color(image).enhance(
                random.uniform(0.4, 1.5)
            )
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(0.75, 1.3)
            )

            if random.random() < 0.25:
                image = image.filter(
                    ImageFilter.GaussianBlur(
                        radius=random.uniform(0.2, 1.5)
                    )
                )

        image = image.resize((64, 64))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(
            np.transpose(array, (2, 0, 1)).copy()
        )

        return tensor, int(self.labels[index])


def main(args):
    OUT.mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    rng = np.random.default_rng(args.seed)
    labels = np.repeat(
        np.arange(len(CLASSES)),
        args.samples_per_class
    )
    seeds = rng.integers(0, 2**31 - 1, len(labels))

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.25,
        random_state=args.seed,
        stratify=labels
    )

    train_dataset = CitrusDataset(
        seeds[train_idx], labels[train_idx], augment=True
    )
    test_dataset = CitrusDataset(
        seeds[test_idx], labels[test_idx], augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    model = CitrusCNN(len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(1)
            correct += (predictions == targets).sum().item()
            total += len(targets)
            loss_sum += loss.item() * len(targets)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| Loss: {loss_sum / total:.4f} "
            f"| Accuracy: {correct / total:.2%}"
        )

    model.eval()
    actual = []
    predicted = []

    with torch.inference_mode():
        for images, targets in test_loader:
            outputs = model(images.to(device))
            predicted.extend(outputs.argmax(1).cpu().tolist())
            actual.extend(targets.tolist())

    accuracy = accuracy_score(actual, predicted)
    matrix = confusion_matrix(
        actual, predicted, labels=list(range(len(CLASSES)))
    )

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": CLASSES,
            "image_size": 64
        },
        OUT / "augmented_cnn.pth"
    )

    metrics = {
        "model": "Augmented CitrusCNN",
        "samples_total": len(labels),
        "test_samples": len(test_idx),
        "test_accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "seed": args.seed,
        "epochs": args.epochs,
        "limitations": [
            "Synthetic-only test data",
            "No independent real-fruit validation"
        ]
    }

    (OUT / "augmented_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print("\nClean test accuracy:", f"{accuracy:.2%}")
    print("Confusion matrix:")
    print(matrix)
    print("Saved to:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-per-class", type=int, default=500
    )
    parser.add_argument(
        "--epochs", type=int, default=25
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    main(parser.parse_args())
