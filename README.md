# Clasificación automática de observaciones con Transformers (BETO) para un motor de bienestar animal

Proyecto semestral de Inteligencia Artificial 2 — Universidad de las Américas.
Integrantes: Elizabeth Guerron, Jean Luc Morales y Mathias Vera.

Fine-tuning de BETO (`dccuchile/bert-base-spanish-wwm-cased`) para clasificar observaciones breves de un centro de
rescate animal en tres niveles de gravedad: **0** Normal, **1** Riesgo Leve, **2** Riesgo Alto.
Se compara contra baselines clásicos (clase mayoritaria y TF-IDF + Regresión Logística).

## Estructura

```
data/
  raw/observaciones_raw.csv   # dataset sintético sin limpiar (id, texto, etiqueta)
  processed/                  # splits train/val/test ya preprocesados (se generan)
src/
  config.py                   # rutas, semilla, etiquetas, split 70/15/15
  generate_dataset.py         # generador reproducible del dataset sintético
  text_cleaning.py            # limpieza de una observación (solo librería estándar)
  preprocess.py               # limpieza + deduplicado + split estratificado (CLI, acepta otros CSV)
  baseline.py                 # clase mayoritaria y TF-IDF + Regresión Logística
  train_beto.py               # fine-tuning de BETO (versión inicial)
  evaluate.py                 # métricas, matriz de confusión y latencia compartidas
notebooks/01_baseline_vs_beto.ipynb   # experimento completo (Colab)
results/figures, tables       # gráficos y tablas para el informe
requirements.txt
```

## Instalación y ejecución local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.generate_dataset                       # 1) regenera data/raw/observaciones_raw.csv
python -m src.preprocess                             # 2) limpia y genera data/processed/{train,val,test}.csv
```

Preprocesar otro dataset con la misma estructura (una columna de texto y una de etiqueta 0/1/2):

```bash
python -m src.preprocess --input mi.csv --text-col texto --label-col etiqueta --output-dir data/mi_dataset
```

Después abrir `notebooks/01_baseline_vs_beto.ipynb` (Colab con GPU recomendado para BETO).

## Ejecución en Google Colab

La primera celda de cada notebook clona el repo e instala dependencias:

```python
!git clone https://github.com/<usuario>/proyecto-semestral-ia2.git
%cd proyecto-semestral-ia2
!pip install -q -r requirements.txt
```
Para BETO activar GPU: Entorno de ejecución → Cambiar tipo → T4 GPU.

## Dataset (sintético)

597 filas en `data/raw/observaciones_raw.csv`, generadas con semilla fija (`SEED=42`):

| Clase | Filas aprox. |
|---|---|
| 0 Normal | 284 |
| 1 Riesgo Leve | 185 |
| 2 Riesgo Alto | 128 |

Contenido: ~46 casos escritos a mano (informales, ambiguos, negaciones como "no presenta sangrado") más frases por
plantillas, incluyendo observaciones mixtas ("come bien *pero* cojea", etiqueta = gravedad máxima).
Desbalance intencional (la clase 2 es minoritaria).

**Suciedad deliberada que el preprocesamiento debe resolver:** mayúsculas, tildes faltantes, abreviaturas de chat
(`q`, `x`, `xq`, `c/`, `vet`), alargamientos (`bieeen`), typos, emojis, signos repetidos, espacios y saltos de línea
extra, prefijos de turno/hora (`Obs:`, `(08:30)`), colas administrativas, filas vacías/nulas, textos de una palabra y
duplicados exactos.

> Limitación: es un dataset sintético. Las frases de plantilla se repiten con distinto animal, por lo que las
> métricas preliminares pueden ser optimistas. Hacer el split **después** de eliminar duplicados y reportar esta
> limitación en el informe.
