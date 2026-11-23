"""Fine-tuning de BETO para clasificación de observaciones (versión inicial, Entrega 2).

BETO + cabeza de clasificación (AutoModelForSequenceClassification, num_labels=3). Bucle de entrenamiento
explícito en PyTorch para poder registrar pérdida/F1 por época y elegir la mejor época con VALIDACIÓN.

Configuración deliberadamente simple (sin pérdida ponderada, sin congelar capas, sin búsqueda de
hiperparámetros): las mejoras quedan para la entrega final.
"""
import copy
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import MODEL_NAME, SEED


@dataclass
class BetoConfig:
    model_name: str = MODEL_NAME
    max_len: int = 96
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_ratio: float = 0.1
    seed: int = SEED
    use_class_weights: bool = False  # mejora prevista: Weighted Cross-Entropy (ver Riesgos del Entregable 1)

    def to_dict(self):
        return asdict(self)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding="max_length",
                             return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


@torch.no_grad()
def _evaluate(model, loader, device, loss_fn):
    model.eval()
    total, preds, gold = 0.0, [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        total += loss_fn(logits, b["labels"]).item() * len(b["labels"])
        preds += logits.argmax(-1).tolist()
        gold += b["labels"].tolist()
    return total / len(gold), f1_score(gold, preds, average="weighted"), preds


def train_beto(train_df, val_df, cfg: BetoConfig, verbose=True):
    """Entrena y devuelve (model, tokenizer, history). Restaura los pesos de la mejor época (F1 val)."""
    set_seed(cfg.seed)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=3).to(device)

    g = torch.Generator().manual_seed(cfg.seed)
    tr = DataLoader(TextDataset(train_df["texto"], train_df["etiqueta"], tok, cfg.max_len),
                    batch_size=cfg.batch_size, shuffle=True, generator=g)
    va = DataLoader(TextDataset(val_df["texto"], val_df["etiqueta"], tok, cfg.max_len), batch_size=64)

    if cfg.use_class_weights:
        counts = np.bincount(train_df["etiqueta"], minlength=3)
        w = torch.tensor(len(train_df) / (3 * counts), dtype=torch.float, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    eval_loss_fn = torch.nn.CrossEntropyLoss()  # pérdida de validación sin pesos, comparable entre corridas

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)

    history, best_f1, best_state = [], -1.0, None
    for ep in range(1, cfg.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            loss = loss_fn(logits, b["labels"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(b["labels"])
            n += len(b["labels"])
        val_loss, val_f1, _ = _evaluate(model, va, device, eval_loss_fn)
        history.append({"epoca": ep, "train_loss": run / n, "val_loss": val_loss, "val_f1": val_f1})
        if verbose:
            print(f"época {ep}/{cfg.epochs}  train_loss={run / n:.4f}  val_loss={val_loss:.4f}  val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return model, tok, history


@torch.no_grad()
def predict(model, tok, texts, max_len=96, batch_size=64):
    """Predice clases para una lista de textos. Devuelve (preds, probs)."""
    device = next(model.parameters()).device
    model.eval()
    preds, probs = [], []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, dim=-1)
        preds += p.argmax(-1).tolist()
        probs += p.cpu().tolist()
    return np.array(preds), np.array(probs)


def make_predict_one(model, tok, max_len=96):
    """Función de 1 texto para medir latencia (batch=1, incluye tokenización)."""
    def f(texto):
        return predict(model, tok, [texto], max_len=max_len, batch_size=1)[0][0]
    return f
