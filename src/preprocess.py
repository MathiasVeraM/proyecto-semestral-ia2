"""Preprocesamiento + split estratificado 70/15/15 para cualquier CSV (texto, etiqueta).

Uso por defecto (dataset del proyecto):
    python -m src.preprocess

Uso con otro dataset:
    python -m src.preprocess --input otro.csv --text-col comentario --label-col clase --output-dir data/otro

Pasos: (1) descartar textos nulos, (2) limpiar texto, (3) descartar vacíos y textos muy cortos,
(4) validar etiquetas, (5) eliminar duplicados (y textos con etiquetas contradictorias),
(6) split estratificado train/val/test. El split se hace DESPUÉS de deduplicar para evitar fuga de datos.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED_DIR, DATA_RAW, SEED, TEST_SIZE, TRAIN_SIZE, VAL_SIZE
from src.text_cleaning import clean_text


def preprocess_dataframe(df, text_col="texto", label_col="etiqueta", min_words=3, valid_labels=(0, 1, 2)):
    """Limpia un DataFrame. Devuelve (df_limpio con columnas texto_original, texto, etiqueta; reporte)."""
    report = {"filas_iniciales": int(len(df))}
    out = pd.DataFrame({"texto_original": df[text_col], "texto": df[text_col].map(clean_text),
                        "etiqueta": df[label_col]})

    n = len(out)
    out = out[out["texto_original"].notna()]
    report["descartadas_nulas"] = n - len(out)

    n = len(out)
    out = out[out["texto"].str.len() > 0]
    report["descartadas_vacias_tras_limpieza"] = n - len(out)

    n = len(out)
    out = out[out["texto"].str.split().str.len() >= min_words]
    report["descartadas_muy_cortas"] = n - len(out)

    n = len(out)
    out = out[out["etiqueta"].isin(valid_labels)].copy()
    out["etiqueta"] = out["etiqueta"].astype(int)
    report["descartadas_etiqueta_invalida"] = n - len(out)

    # Mismo texto con etiquetas distintas -> contradictorio, se elimina todo el grupo
    n = len(out)
    conflictivos = out.groupby("texto")["etiqueta"].nunique()
    conflictivos = set(conflictivos[conflictivos > 1].index)
    out = out[~out["texto"].isin(conflictivos)]
    report["descartadas_etiquetas_contradictorias"] = n - len(out)

    n = len(out)
    out = out.drop_duplicates(subset="texto", keep="first")
    report["descartadas_duplicadas"] = n - len(out)

    out = out.reset_index(drop=True)
    report["filas_finales"] = int(len(out))
    report["distribucion_clases"] = {int(k): int(v) for k, v in out["etiqueta"].value_counts().sort_index().items()}
    return out, report


def split_dataframe(df, seed=SEED, train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE):
    """Split estratificado train/val/test (por defecto 70/15/15)."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9
    train, rest = train_test_split(df, train_size=train_size, stratify=df["etiqueta"], random_state=seed)
    val, test = train_test_split(rest, train_size=val_size / (val_size + test_size),
                                 stratify=rest["etiqueta"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DATA_RAW)
    ap.add_argument("--output-dir", type=Path, default=DATA_PROCESSED_DIR)
    ap.add_argument("--text-col", default="texto")
    ap.add_argument("--label-col", default="etiqueta")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    raw = pd.read_csv(args.input, encoding="utf-8")
    clean, report = preprocess_dataframe(raw, args.text_col, args.label_col, args.min_words)
    train, val, test = split_dataframe(clean, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for nombre, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(args.output_dir / f"{nombre}.csv", index=False, encoding="utf-8")
        report[f"{nombre}_tamano"] = int(len(d))
        report[f"{nombre}_clases"] = {int(k): int(v) for k, v in d["etiqueta"].value_counts().sort_index().items()}
    report["semilla"] = args.seed
    (args.output_dir / "preprocess_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
