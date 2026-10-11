# Cats vs Dogs — CNN classifier (EFREI project)

A local, single-script version of the Colab notebook
[CNN_chats_chiens.ipynb](https://colab.research.google.com/drive/1elP6TGubv4vr1x5-l5fIHBAcwlXK9D6T).
It implements the steps of the assignment: CNN intro → dataset exploration → model → training → evaluation → improvements → bonus (transfer learning).
The code comments and console output are in **French**, following the original notebook and the course. Keep it that way.

## Quick start

```bash
/Volumes/EXT/TOOLS/ansible/bin/python3 -m venv .venv   # Python 3.13 (only interpreter on this machine)
.venv/bin/pip install -r requirements.txt
.venv/bin/python -u main.py                          # ~40 min on CPU
```

- The dataset (`cats_and_dogs_filtered`, ~65 MB) downloads automatically to `data/` on the first run.
- Figures, a results CSV and the trained MobileNetV2 model go to `outputs/`.
- There is **no GPU**: `tensorflow-metal` has no wheel for Python 3.13, so TensorFlow 2.21 runs on CPU (~25–40 s per epoch for the custom CNN).
- Use `python -u` when redirecting output to a log. Without it, stdout is buffered and you see no progress.

## Files

| Path | What |
|---|---|
| `main.py` | The whole pipeline. `main()` runs every step in order. |
| `requirements.txt` | Pinned deps (tensorflow 2.21, pandas, matplotlib, seaborn, scikit-learn, pillow, scipy). `scipy` is required by `ImageDataGenerator` augmentation. |
| `data/` | Downloaded dataset (git-ignored). |
| `outputs/` | `01`–`12` PNG figures, `comparaison_modeles.csv`, `mobilenetv2_cats_dogs.keras` (git-ignored). |
| `run.log` | Console output of the last full run. |

## Pipeline (`main.py`)

1. **Data:** 2000 train images (1000/1000) and 1000 validation images (500/500). The `train` folder is split 80/20 into **train (1600) / test (400)**, stratified, with `SEED=42`. The `validation` folder is used as the validation set. Data is loaded with `ImageDataGenerator.flow_from_dataframe`, images resized to 150×150 and pixels scaled by `1/255`. Labels: `cat=0`, `dog=1`.
2. **Model (`build_cnn`):** 4× [Conv2D 3×3 (32/64/128/128) → ReLU → MaxPool 2×2] → Flatten → Dense(512) → Dense(1, sigmoid). This has 3,453,121 params; `check_param_count` asserts this matches the hand calculation.
3. **Compile:** Adam (lr 1e-3) with binary cross-entropy and accuracy as the metric.
4. **Train:** batch 100, 15 epochs (the assignment allows 5–15), with `EarlyStopping(val_loss, patience=5, restore_best_weights)` and `ReduceLROnPlateau(factor=0.5, patience=2)`.
5. **Evaluate:** on the test set with threshold 0.5. Reports the confusion matrix, accuracy, precision, recall and F1, with "dog" as the positive class.
6. **Improvements, one technique at a time:** data augmentation, then Dropout(0.5) + BatchNorm, then both combined. Validation and test data are **never** augmented.
7. **Bonus:** frozen MobileNetV2 (ImageNet weights, 160×160, `preprocess_input`) + GlobalAveragePooling + Dropout(0.3) + Dense(1). The head is trained for 8 epochs at lr 1e-3, then the last 30 layers are fine-tuned for 5 epochs at lr 1e-5.

## Results (test set, 400 images, last run)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Basic CNN | 0.712 | 0.754 | 0.630 | 0.687 |
| CNN + Data augmentation | 0.715 | 0.690 | 0.780 | 0.732 |
| CNN + Dropout + BatchNorm | 0.768 | 0.772 | 0.760 | 0.766 |
| CNN + Augmentation + Dropout + BN | 0.755 | 0.758 | 0.750 | 0.754 |
| MobileNetV2 (transfer learning) | **0.970** | 0.965 | 0.975 | 0.970 |

Observations:
- The basic CNN **overfits**: 86% train vs 72% validation. Most of its parameters (>93%) are in Dense(512), and there are only 1600 training images.
- Augmentation closes the train/validation gap but is capped by the 15-epoch limit; it would keep improving with more epochs.
- Transfer learning reaches 97–98% with almost no train/validation gap.

## Changes from the original notebook, and why

These changes were made after a first run gave bad results. Don't revert them without re-checking.

- **`BatchNormalization(momentum=0.9)`** (Keras default is 0.99). There are only 16 batches per epoch, so with 0.99 the moving mean and variance never converge. In the first run, the Dropout + BN model had **50% validation accuracy** (constant prediction) while reaching 91% on training images. With 0.9 it gets 77.5% validation.
- **Milder augmentation:** rotation 15, shifts 0.1, shear 0.1, zoom 0.15, horizontal flip. The notebook's settings (rotation 30, shifts 0.2…) left the model stuck around 50% for the first 5 epochs, and it only reached 65% test accuracy within 15 epochs.
- **Adam learning rate 1e-3** instead of 1e-4, plus EarlyStopping and ReduceLROnPlateau, so the models converge within the 15-epoch cap.
- **MobileNetV2 fine-tuning keeps its BatchNorm layers frozen**, to avoid damaging the pretrained statistics.
- Figures are saved with `matplotlib` backend `Agg` instead of being shown.

## Ideas if you continue

- Keep the assignment constraints: batch size 100, 5–15 epochs, augment only training data, evaluate on the test split.
- Possible next steps: tune the dropout rate, try `GlobalAveragePooling2D` instead of `Flatten` to cut parameters, other backbones (EfficientNet, ResNet50), Grad-CAM visualizations.
- A full run takes ~40 min. To iterate on one step, comment out the other steps in `main()` rather than lowering the epoch counts. The step functions are independent apart from the shared generators.
