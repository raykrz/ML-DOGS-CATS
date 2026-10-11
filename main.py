"""
Teaching Computers to See : classifier des chats et des chiens avec des CNN
Projet EFREI – Convolutional Neural Networks

Version locale du notebook Colab. Étapes :
  2. Exploration du dataset      4. Entraînement     6. Améliorations
  3. Construction du modèle      5. Évaluation       Bonus : Transfer Learning (MobileNetV2)

Les graphiques sont enregistrés dans ./outputs et les résultats sont affichés dans la console.
"""
import os
import glob
import random
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # pas de fenêtre : on enregistre les figures
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

warnings.filterwarnings("ignore")

# Reproductibilité : on fixe les graines aléatoires
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (150, 150)
BATCH_SIZE = 100          # imposé par le sujet
EPOCHS = 15               # le sujet demande 5 à 15 époques
LR = 1e-3                 # learning rate d'Adam


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print("  -> figure enregistrée :", os.path.relpath(path, ROOT))


# ============================================================
# Étape 2 : Exploration du dataset
# ============================================================
def download_dataset():
    # Télécharge et décompresse cats_and_dogs_filtered (une seule fois)
    base_dir = os.path.join(DATA_DIR, "cats_and_dogs_filtered")
    if not os.path.isdir(base_dir):
        url = "https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip"
        keras.utils.get_file("cats_and_dogs_filtered.zip", url, cache_dir=ROOT,
                             cache_subdir="data", extract=True)
        # Selon la version de Keras, l'archive est extraite dans un sous-dossier
        if not os.path.isdir(base_dir):
            found = glob.glob(os.path.join(DATA_DIR, "**", "cats_and_dogs_filtered"), recursive=True)
            base_dir = [d for d in found if os.path.isdir(os.path.join(d, "train"))][0]
    return os.path.join(base_dir, "train"), os.path.join(base_dir, "validation")


def count_images(directory):
    # Retourne le nombre d'images par classe (sous-dossiers 'cats' et 'dogs')
    return {cls: len(os.listdir(os.path.join(directory, cls))) for cls in ["cats", "dogs"]}


def explore(train_dir, validation_dir):
    print("\n=== Étape 2 : Exploration du dataset ===")
    counts = pd.DataFrame({"train": count_images(train_dir), "validation": count_images(validation_dir)})
    counts.loc["total"] = counts.sum()
    print(counts)
    print("Le dataset est équilibré : l'accuracy est une métrique fiable (hasard ≈ 50 %).")

    # Distribution des classes
    counts.drop("total").T.plot(kind="bar", figsize=(6, 4), color=["#4C9BE8", "#F2994A"])
    plt.title("Distribution des classes"); plt.ylabel("Nombre d'images"); plt.xticks(rotation=0)
    save_fig("01_distribution_classes.png")

    # Forme des images : elles n'ont pas toutes la même taille
    sample_files = glob.glob(os.path.join(train_dir, "*", "*.jpg"))
    sizes = [Image.open(f).size for f in random.sample(sample_files, 200)]
    sizes_df = pd.DataFrame(sizes, columns=["largeur", "hauteur"])
    print("\nTaille des images (échantillon de 200) :")
    print(sizes_df.describe().loc[["min", "mean", "max"]])
    print(f"-> toutes les images seront redimensionnées en {IMG_SIZE + (3,)} et normalisées dans [0, 1].")

    # Quelques exemples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for row, cls in enumerate(["cats", "dogs"]):
        files = random.sample(os.listdir(os.path.join(train_dir, cls)), 5)
        for col, f in enumerate(files):
            img = Image.open(os.path.join(train_dir, cls, f))
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"{cls[:-1]} {img.size}")
            axes[row, col].axis("off")
    plt.tight_layout()
    save_fig("02_exemples_images.png")


def make_df(directory):
    # Construit un DataFrame (filepath, label) pour toutes les images d'un dossier
    rows = []
    for cls, label in [("cats", "cat"), ("dogs", "dog")]:
        for f in os.listdir(os.path.join(directory, cls)):
            rows.append((os.path.join(directory, cls, f), label))
    return pd.DataFrame(rows, columns=["filepath", "label"])


def make_generators(train_datagen, eval_datagen, train_df, val_df, test_df, img_size=IMG_SIZE):
    # train_datagen : appliqué au train (éventuellement avec augmentation)
    # eval_datagen  : appliqué à validation ET test (JAMAIS d'augmentation sur le test)
    kw = dict(x_col="filepath", y_col="label", target_size=img_size,
              batch_size=BATCH_SIZE, class_mode="binary", classes=["cat", "dog"])
    train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, seed=SEED, **kw)
    val_gen = eval_datagen.flow_from_dataframe(val_df, shuffle=False, **kw)
    test_gen = eval_datagen.flow_from_dataframe(test_df, shuffle=False, **kw)  # shuffle=False : ordre = étiquettes
    return train_gen, val_gen, test_gen


# ============================================================
# Étape 3 : Construction du modèle CNN
# ============================================================
def build_cnn(dropout=False, batch_norm=False, input_shape=IMG_SIZE + (3,)):
    # 4 blocs Conv(3x3) + ReLU + MaxPool(2x2), puis Flatten -> Dense(512) -> Dense(1, sigmoïde)
    # Les options dropout / batch_norm servent à l'étape 6.
    model = models.Sequential(name="cnn_cats_dogs")
    model.add(layers.Input(shape=input_shape))
    for filters in [32, 64, 128, 128]:
        model.add(layers.Conv2D(filters, (3, 3)))
        if batch_norm:
            model.add(layers.BatchNormalization(momentum=0.9))  # momentum 0.9 : peu de batchs par époque
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    if dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512))
    if batch_norm:
        model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation("relu"))
    if dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))   # sortie : probabilité de 'chien'
    return model


def compile_model(m, lr=LR):
    # binary_crossentropy : classification binaire avec sortie sigmoïde
    # Adam : pas d'apprentissage adaptatif, converge vite avec peu de réglages
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def check_param_count(model):
    # Vérification du calcul manuel du nombre de paramètres
    def conv_params(k, c_in, c_out): return (k * k * c_in + 1) * c_out
    def dense_params(n_in, n_out): return (n_in + 1) * n_out
    # 150 -> conv 148 -> pool 74 -> conv 72 -> pool 36 -> conv 34 -> pool 17 -> conv 15 -> pool 7
    manual = (conv_params(3, 3, 32) + conv_params(3, 32, 64) + conv_params(3, 64, 128)
              + conv_params(3, 128, 128) + dense_params(7 * 7 * 128, 512) + dense_params(512, 1))
    print(f"Total calculé à la main : {manual:,}")
    print(f"Total selon Keras       : {model.count_params():,}")
    assert manual == model.count_params()


# ============================================================
# Étape 4 : Entraînement
# ============================================================
def callbacks():
    # On garde les meilleurs poids (val_loss minimale) et on réduit le LR quand la validation stagne
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


def train(model, train_gen, val_gen, epochs=EPOCHS):
    return model.fit(train_gen, epochs=epochs, validation_data=val_gen,
                     callbacks=callbacks(), verbose=2)


def plot_history(h, title, filename):
    # Trace accuracy et loss (train vs validation) au fil des époques
    acc, val_acc = h.history["accuracy"], h.history["val_accuracy"]
    loss, val_loss = h.history["loss"], h.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(epochs_range, acc, "o-", label="Train"); ax[0].plot(epochs_range, val_acc, "o-", label="Validation")
    ax[0].set_title(f"Accuracy {title}"); ax[0].set_xlabel("Époque"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(epochs_range, loss, "o-", label="Train"); ax[1].plot(epochs_range, val_loss, "o-", label="Validation")
    ax[1].set_title(f"Loss {title}"); ax[1].set_xlabel("Époque"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    save_fig(filename)

    gap = acc[-1] - val_acc[-1]
    print(f"  accuracy finale : train {acc[-1]:.3f} | validation {val_acc[-1]:.3f} | écart {gap:+.3f}")
    if gap > 0.10:
        print("  -> écart important : sur-apprentissage.")
    elif max(val_acc) < 0.65:
        print("  -> accuracy faible partout : sous-apprentissage.")
    else:
        print("  -> écart raisonnable : le modèle généralise correctement.")


# ============================================================
# Étape 5 : Évaluation sur l'ensemble de test
# ============================================================
def evaluate_model(m, gen, name, filename=None):
    # Évalue un modèle sur un générateur (shuffle=False !), matrice de confusion + métriques
    gen.reset()
    probs = m.predict(gen, verbose=0).ravel()
    y_pred = (probs > 0.5).astype(int)         # seuil 0,5 : > 0,5 => chien (classe 1)
    y_true = gen.classes

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "Modèle": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Précision": precision_score(y_true, y_pred),
        "Rappel": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
    }
    print(f"\n--- Test : {name} ---")
    print(classification_report(y_true, y_pred, target_names=["cat", "dog"], digits=3))
    if filename:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
        plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title(f"Matrice de confusion – {name}")
        save_fig(filename)
    return metrics


def main():
    # ---------- Étape 2 ----------
    train_dir, validation_dir = download_dataset()
    explore(train_dir, validation_dir)

    # Découpage du dossier train en train (80 %) / test (20 %), stratifié ; validation = dossier validation
    full_train_df = make_df(train_dir)
    val_df = make_df(validation_dir)
    train_df, test_df = train_test_split(full_train_df, test_size=0.20,
                                         stratify=full_train_df["label"], random_state=SEED)
    print(f"\nTrain : {len(train_df)} | Test : {len(test_df)} | Validation : {len(val_df)}")

    base_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_gen, val_gen, test_gen = make_generators(base_datagen, base_datagen, train_df, val_df, test_df)
    print("Indices des classes :", train_gen.class_indices)   # cat -> 0, dog -> 1

    # ---------- Étape 3 ----------
    print("\n=== Étape 3 : Construction du modèle ===")
    model = compile_model(build_cnn())
    model.summary()
    check_param_count(model)

    # ---------- Étapes 4 et 5 : modèle de base ----------
    print("\n=== Étape 4 : Entraînement du CNN de base ===")
    results = []
    history = train(model, train_gen, val_gen)
    plot_history(history, "(modèle de base)", "03_courbes_base.png")
    print("\n=== Étape 5 : Évaluation ===")
    results.append(evaluate_model(model, test_gen, "CNN de base", "04_confusion_base.png"))

    # ---------- Étape 6 : améliorations (une technique à la fois) ----------
    print("\n=== Étape 6.1 : Data augmentation ===")
    # Augmentation modérée : assez pour régulariser, mais le modèle converge en 15 époques
    aug_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    # Le train est augmenté ; validation et test restent seulement normalisés
    train_gen_aug, _, _ = make_generators(aug_datagen, base_datagen, train_df, val_df, test_df)
    model_aug = compile_model(build_cnn())
    history_aug = train(model_aug, train_gen_aug, val_gen)
    plot_history(history_aug, "(data augmentation)", "05_courbes_augmentation.png")
    results.append(evaluate_model(model_aug, test_gen, "CNN + Data augmentation", "06_confusion_augmentation.png"))

    print("\n=== Étape 6.2 : Dropout + Batch Normalization ===")
    model_reg = compile_model(build_cnn(dropout=True, batch_norm=True))
    history_reg = train(model_reg, train_gen, val_gen)
    plot_history(history_reg, "(Dropout + BatchNorm)", "07_courbes_dropout_bn.png")
    results.append(evaluate_model(model_reg, test_gen, "CNN + Dropout + BatchNorm", "08_confusion_dropout_bn.png"))

    print("\n=== Étape 6.3 : Augmentation + Dropout + BatchNorm ===")
    model_both = compile_model(build_cnn(dropout=True, batch_norm=True))
    history_both = train(model_both, train_gen_aug, val_gen)
    plot_history(history_both, "(Augmentation + Dropout + BN)", "09_courbes_combine.png")
    results.append(evaluate_model(model_both, test_gen, "CNN + Augmentation + Dropout + BN"))

    # ---------- Bonus : Transfer Learning avec MobileNetV2 ----------
    print("\n=== Bonus : Transfer Learning (MobileNetV2 pré-entraîné sur ImageNet) ===")
    tl_size = (160, 160)
    # MobileNetV2 attend des pixels dans [-1, 1] : on utilise SA fonction de prétraitement
    tl_aug = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20,
                                width_shift_range=0.1, height_shift_range=0.1,
                                zoom_range=0.15, horizontal_flip=True)
    tl_eval = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen_tl, val_gen_tl, test_gen_tl = make_generators(tl_aug, tl_eval, train_df, val_df, test_df,
                                                            img_size=tl_size)

    base_model = MobileNetV2(input_shape=tl_size + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False       # phase 1 : base gelée, on n'entraîne que la tête
    model_tl = models.Sequential([
        layers.Input(shape=tl_size + (3,)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="mobilenetv2_transfer")
    compile_model(model_tl, lr=1e-3)
    history_tl = train(model_tl, train_gen_tl, val_gen_tl, epochs=8)

    # Phase 2 : fine-tuning des 30 dernières couches avec un learning rate très faible
    # (les couches BatchNorm restent en mode inférence pour ne pas casser les poids appris)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model_tl, lr=1e-5)
    history_ft = train(model_tl, train_gen_tl, val_gen_tl, epochs=5)

    for k in ["accuracy", "loss", "val_accuracy", "val_loss"]:
        history_tl.history[k] += history_ft.history[k]
    plot_history(history_tl, "(MobileNetV2 : extraction + fine-tuning)", "10_courbes_mobilenetv2.png")
    results.append(evaluate_model(model_tl, test_gen_tl, "MobileNetV2 (transfer learning)",
                                  "11_confusion_mobilenetv2.png"))
    model_tl.save(os.path.join(OUT_DIR, "mobilenetv2_cats_dogs.keras"))

    # ---------- Comparaison finale ----------
    comparison = pd.DataFrame(results).set_index("Modèle").round(3)
    print("\n=== Comparaison des modèles sur l'ensemble de test ===")
    print(comparison.to_string())
    comparison.to_csv(os.path.join(OUT_DIR, "comparaison_modeles.csv"))

    comparison.plot(kind="bar", figsize=(10, 5), ylim=(0.5, 1.0))
    plt.title("Comparaison des modèles sur l'ensemble de test"); plt.xticks(rotation=20, ha="right")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    save_fig("12_comparaison_modeles.png")


if __name__ == "__main__":
    main()
