
"""Evaluate a trained citrus CNN under simulated image disturbances."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, confusion_matrix

from pipeline import CLASSES, render
from train_cnn import CitrusCNN


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "augmented_robustness_outputs"


def disturb(image, condition, seed):
    rng = np.random.default_rng(seed)

    if condition == "clean":
        return image

    if condition == "dark":
        return ImageEnhance.Brightness(image).enhance(0.55)

    if condition == "bright":
        return ImageEnhance.Brightness(image).enhance(1.45)

    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.0))

    if condition == "noise":
        array = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0, 22, array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(array)

    if condition == "colour":
        return ImageEnhance.Color(image).enhance(0.5)

    if condition == "combined":
        image = ImageEnhance.Brightness(image).enhance(0.65)
        image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        array = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0, 15, array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(array)

    raise ValueError(f"Unknown condition: {condition}")


def preprocess(image, size):
    image = image.resize((size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        ROOT / "augmented_outputs" / "augmented_cnn.pth",
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]
    size = checkpoint["image_size"]

    model = CitrusCNN(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    conditions = [
        "clean", "dark", "bright", "blur",
        "noise", "colour", "combined"
    ]

    rng = np.random.default_rng(args.seed)
    labels = np.repeat(CLASSES, args.samples_per_class)
    seeds = rng.integers(0, 2**31 - 1, len(labels))

    rows = []
    results = {}

    with torch.inference_mode():
        for condition in conditions:
            actual = []
            predicted = []

            for image_seed, label in zip(seeds, labels):
                image = render(int(image_seed), str(label))
                image = disturb(
                    image,
                    condition,
                    int(image_seed) + 1,
                )

                tensor = preprocess(image, size).unsqueeze(0)
                logits = model(tensor)
                pred = int(logits.argmax(dim=1).item())

                actual.append(classes.index(str(label)))
                predicted.append(pred)

            accuracy = accuracy_score(actual, predicted)
            matrix = confusion_matrix(
                actual, predicted, labels=list(range(len(classes)))
            )

            results[condition] = {
                "accuracy": float(accuracy),
                "confusion_matrix": matrix.tolist(),
                "samples": len(actual),
            }

            rows.append({
                "condition": condition,
                "accuracy": float(accuracy),
                "samples": len(actual),
            })

            print(f"{condition:10s}: {accuracy:.2%}")

    report = {
        "model": "CitrusCNN",
        "evaluation": "Synthetic image disturbance robustness",
        "classes": classes,
        "seed": args.seed,
        "results": results,
        "limitations": [
            "Synthetic imagery only",
            "Image disturbances do not represent all real-world conditions",
            "No real camera validation",
        ],
    }

    (OUT / "robustness_metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    with (OUT / "robustness_summary.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["condition", "accuracy", "samples"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved results to:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-per-class", type=int, default=100
    )
    parser.add_argument(
        "--seed", type=int, default=2027
    )
    main(parser.parse_args())
