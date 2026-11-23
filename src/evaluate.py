"""Métricas y gráficos compartidos por todos los modelos (mismo protocolo de evaluación)."""
import time

import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)

from src.config import LABELS


def compute_metrics(y_true, y_pred):
    """Accuracy, F1 ponderado/macro y recall por clase (la clase 2 es la más crítica)."""
    rec = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_clase0": rec[0], "recall_clase1": rec[1], "recall_clase2": rec[2],
    }


def report_text(y_true, y_pred):
    return classification_report(y_true, y_pred, labels=[0, 1, 2],
                                 target_names=list(LABELS.values()), zero_division=0)


def plot_confusion(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=list(LABELS.values()))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    disp.ax_.set_title(title)
    return cm


def measure_latency(predict_one, texts, n_warmup=5, n_runs=100):
    """Latencia por observación (ms) con batch=1. `predict_one(texto)` debe devolver la clase."""
    for t in texts[:n_warmup]:
        predict_one(t)
    tiempos = []
    for i in range(n_runs):
        t = texts[i % len(texts)]
        t0 = time.perf_counter()
        predict_one(t)
        tiempos.append((time.perf_counter() - t0) * 1000)
    tiempos = np.array(tiempos)
    return {"latencia_media_ms": tiempos.mean(), "latencia_p95_ms": np.percentile(tiempos, 95)}
