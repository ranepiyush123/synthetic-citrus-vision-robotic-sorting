# Synthetic Citrus Vision & Sorting — Proof of Concept

A reproducible, **software-only** demonstration relevant to the Karsten UK / Harper Adams University KTP project. It creates procedural citrus images with three artificial surface-blemish grades, trains a HOG + Random Forest image classifier, evaluates on held-out synthetic samples, and maps predictions to **simulated** sorting commands.

## Run

```bash
python -m pip install -r requirements.txt
python src/pipeline.py --samples-per-class 90 --seed 42
```

Outputs in `outputs/`:
- `synthetic_samples.png` — illustrative generated fruit images
- `metrics.json` — held-out synthetic evaluation and confusion matrix
- `model.joblib` — trained model
- `robot_commands.csv` — simulated target-bin commands (no hardware control)

## Technical approach

1. Procedural PIL image generator varies fruit position, colour, texture, conveyor-like background, blur and blemish count/size.
2. Extract HOG visual features and train a scikit-learn Random Forest classifier.
3. Stratified 75/25 train/test split with independently generated image seeds.
4. Translate predictions to `BIN_A`, `BIN_B` or `REJECT_BIN` in a CSV simulator.

## Important limitations

**Not** a validated fruit grading system. The synthetic labels are defined by the same simple visual rules used to generate the images. Good performance on this dataset does **not** establish real-world grading accuracy or the KTP target of 95% accuracy using synthetic training data. This prototype does not include DEM, Blender photorealism, camera calibration, live conveyor tracking, Fanuc robot control, PLC connectivity, or safety interlocks. `robot_commands.csv` is a mock interface only.

## Roadmap toward a genuine KTP demonstrator

- Capture and label real citrus images under multiple illumination conditions; calibrate camera intrinsics/extrinsics.
- Implement photorealistic Blender renders and compare domain randomisation settings.
- Introduce DEM-based fruit trajectories and contact dynamics, then map simulated poses to rendering.
- Benchmark YOLO/CNN vision models against a real-image holdout; report class-level precision/recall and throughput.
- Add tracking, robot-coordinate transforms, latency measurement, guarded controller integration, and safety validation on a training rig before production.

## CV description (accurate if you run and understand this work)

**Independent project — Synthetic Citrus Vision & Sorting (prototype):** Built a Python pipeline generating labelled synthetic citrus images, training and evaluating an image-grade classifier, and translating predictions into simulated pick-and-place sorting commands. Documented synthetic-to-real limitations and a validation roadmap for industrial deployment.
