"""Baselines: clase mayoritaria y TF-IDF + Regresión Logística."""
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.config import SEED


def make_majority():
    return DummyClassifier(strategy="most_frequent")


def make_tfidf_lr(C=1.0, class_weight=None, seed=SEED):
    """TF-IDF (unigramas+bigramas, sin tildes, tf sublineal) + regresión logística."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")),
        ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=seed)),
    ])


def select_C(X_train, y_train, X_val, y_val, grid=(0.1, 1, 10, 100), class_weight=None, seed=SEED):
    """Elige C según F1 ponderado en validación (el test no se toca). Devuelve (mejor_C, tabla)."""
    tabla = {}
    for C in grid:
        m = make_tfidf_lr(C, class_weight, seed).fit(X_train, y_train)
        tabla[C] = f1_score(y_val, m.predict(X_val), average="weighted")
    return max(tabla, key=tabla.get), tabla
