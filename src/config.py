"""Configuración central del proyecto (rutas, semilla, etiquetas, splits)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "observaciones_raw.csv"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42

LABELS = {0: "Normal", 1: "Riesgo Leve", 2: "Riesgo Alto"}

# Split estratificado 70 / 15 / 15 (definido en la metodología del Entregable 1)
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"  # BETO
