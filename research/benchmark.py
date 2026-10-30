"""Reproducible frozen prototype benchmark for a selected BANKING77 subset.

Downloads the public PolyAI dataset and Qwen3-Embedding-0.6B into the user
cache. It never trains or fine-tunes the encoder. The official test split is
evaluated once after correction strength is selected on a validation split.
"""
from __future__ import annotations

import json
import os
import platform
import random
from collections import Counter
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score


SEED = 731
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DATASET_ID = "PolyAI/banking77"
DATASET_REPOSITORY = "PolyAI-LDN/task-specific-datasets"
DATASET_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
LOCAL_DATA_DIR = os.environ.get("BANKING77_DATA_DIR")
INTENTS = [
    "cash_withdrawal_not_recognised",
    "wrong_exchange_rate_for_cash_withdrawal",
    "pending_cash_withdrawal",
    "cash_withdrawal_charge",
    "declined_cash_withdrawal",
]
INSTRUCTION = (
    "Instruct: Group short online-banking support messages by their specific "
    "customer intent, distinguishing related cash-withdrawal issues.\nQuery: "
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "dist" / "assets" if (PROJECT_ROOT / "dist").is_dir() else PROJECT_ROOT / "assets"
OUTPUT = OUTPUT_DIR / "benchmark-results.json"


def encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        prompt=INSTRUCTION,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)


def stratified_partition(rows: list[dict], rng: random.Random, proportions: tuple[float, float, float]):
    parts = [[], [], []]
    for label in INTENTS:
        group = [row for row in rows if row["label"] == label]
        rng.shuffle(group)
        first = round(len(group) * proportions[0])
        second = first + round(len(group) * proportions[1])
        parts[0].extend(group[:first])
        parts[1].extend(group[first:second])
        parts[2].extend(group[second:])
    for part in parts:
        rng.shuffle(part)
    return parts


def predictions(scores: np.ndarray) -> np.ndarray:
    return scores.argmax(axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=np.arange(len(INTENTS)))),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    selected = set(INTENTS)
    base_url = (
        f"https://raw.githubusercontent.com/{DATASET_REPOSITORY}/"
        f"{DATASET_REVISION}/banking_data"
    )

    def collect(split: str) -> list[dict]:
        source = Path(LOCAL_DATA_DIR) / f"banking77-{split}.csv" if LOCAL_DATA_DIR else f"{base_url}/{split}.csv"
        frame = pd.read_csv(source)
        frame = frame[frame["category"].isin(selected)]
        return [
            {"text": str(row.text), "label": str(row.category)}
            for row in frame.itertuples(index=False)
        ]

    official_train = collect("train")
    official_test = collect("test")
    if any(not any(r["label"] == label for r in official_train) for label in INTENTS):
        raise RuntimeError("A selected label is missing from the pinned BANKING77 training CSV.")
    rng = random.Random(SEED)
    # The official train split is split three ways: prototype fit, independent
    # background reference, and validation. Official test stays untouched.
    prototype_rows, reference_rows, validation_rows = stratified_partition(
        official_train, rng, (0.62, 0.18, 0.20)
    )
    missing = [label for label in INTENTS if not any(r["label"] == label for r in prototype_rows)]
    if missing:
        raise RuntimeError(f"Selected intent labels absent from dataset: {missing}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batch_size = 32 if device == "mps" else 8
    print(f"Device: {device.upper()} (MPS available: {torch.backends.mps.is_available()})")
    model = SentenceTransformer(MODEL_ID, device=device)
    prototype_vecs = encode(model, [r["text"] for r in prototype_rows], batch_size)
    reference_vecs = encode(model, [r["text"] for r in reference_rows], batch_size)
    val_vecs = encode(model, [r["text"] for r in validation_rows], batch_size)
    test_vecs = encode(model, [r["text"] for r in official_test], batch_size)
    label_index = {label: i for i, label in enumerate(INTENTS)}

    def centroid_matrix(rows: list[dict], vectors: np.ndarray) -> np.ndarray:
        centers = []
        for label in INTENTS:
            indexes = [i for i, row in enumerate(rows) if row["label"] == label]
            center = vectors[indexes].mean(axis=0)
            center /= np.linalg.norm(center)
            centers.append(center)
        return np.stack(centers)

    centers = centroid_matrix(prototype_rows, prototype_vecs)
    ref_sim = reference_vecs @ centers.T
    val_sim = val_vecs @ centers.T
    test_sim = test_vecs @ centers.T
    y_val = np.array([label_index[r["label"]] for r in validation_rows])
    y_test = np.array([label_index[r["label"]] for r in official_test])

    # Proposed mean-background correction; its single global strength is
    # selected strictly on validation macro-F1.
    background = ref_sim.mean(axis=0)
    candidates = []
    for lam in np.round(np.arange(0.0, 1.5001, 0.025), 3):
        score = val_sim - float(lam) * background[None, :]
        metric = metrics(y_val, predictions(score))
        candidates.append((metric["macro_f1"], metric["accuracy"], -float(lam), float(lam)))
    selected_lambda = max(candidates)[3]

    # CSLS local scaling: average each prototype's top-k similarity to the
    # independent reference-query sample. Query-side density is per-query but
    # constant across candidates, so it cannot change that query's argmax.
    k = min(10, len(reference_vecs))
    candidate_density = np.sort(ref_sim, axis=0)[-k:, :].mean(axis=0)

    method_scores = {
        "cosine": test_sim,
        "mean_baseline": test_sim - selected_lambda * background[None, :],
        "csls": 2 * test_sim - candidate_density[None, :],
    }
    prediction_sets = {name: predictions(score) for name, score in method_scores.items()}
    test_metrics = {name: metrics(y_test, pred) for name, pred in prediction_sets.items()}

    # A simple hubness diagnostic: distribution of top-1 predictions. A skewed
    # count can be inspected, but with only five candidates it is not evidence
    # of high-dimensional hubness on its own.
    top1_counts = Counter(INTENTS[int(i)] for i in prediction_sets["cosine"])
    result = {
        "status": "measured",
        "retrieved_on": date.today().isoformat(),
        "task": "Frozen cosine-prototype assignment on five closely related BANKING77 intents",
        "dataset": {
            "id": DATASET_ID,
            "source": f"https://github.com/{DATASET_REPOSITORY}/tree/{DATASET_REVISION}/banking_data",
            "revision": DATASET_REVISION,
            "citation": "Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders (2020), NLP4ConvAI / ACL.",
            "license": "CC BY 4.0",
            "selected_intents": INTENTS,
            "counts": {
                "prototype_fit": len(prototype_rows),
                "background_reference": len(reference_rows),
                "validation": len(validation_rows),
                "official_test": len(official_test),
            },
            "split_note": "Original official test split used only for final evaluation; official train split stratified by seed 731 into fit, reference and validation.",
        },
        "model": {
            "id": MODEL_ID,
            "library": "sentence-transformers",
            "library_version": version("sentence-transformers"),
            "encoder_frozen": True,
            "instruction": INSTRUCTION.strip(),
            "device": device,
            "normalization": "L2 unit vectors; cosine similarity by dot product",
        },
        "methods": {
            "cosine": "Uncorrected cosine similarity to L2-normalized intent centroids.",
            "mean_baseline": "cosine(q,c_k) - lambda * mean_{r in independent reference}(cosine(r,c_k)); lambda selected on validation macro-F1.",
            "csls": f"2*cosine(q,c_k) - r_query(q) - r_candidate(c_k), with top-{k} local neighbourhood averages; query-side term is constant across candidates for each query.",
        },
        "selected_lambda": selected_lambda,
        "metrics": test_metrics,
        "absolute_delta_vs_cosine": {
            method: {
                metric: values[metric] - test_metrics["cosine"][metric]
                for metric in ("accuracy", "macro_f1")
            }
            for method, values in test_metrics.items()
            if method != "cosine"
        },
        "raw_cosine_top1_intent_counts": {label: top1_counts.get(label, 0) for label in INTENTS},
        "interpretation": (
            "Small domain-specific evaluation, not a general benchmark. The encoder was frozen; "
            "a label-centroid prototype was fitted on training examples, so this measures known-intent "
            "assignment, not unsupervised cluster discovery. No correction is presumed to help."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": version("scikit-learn"),
            "random_seed": SEED,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
