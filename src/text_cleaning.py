"""Limpieza de una observación individual (solo librería estándar).

Conserva tildes y puntuación básica (BETO es *cased* y las usa); elimina el ruido
que no aporta gravedad: emojis, prefijos de turno/hora, hashtags, alargamientos,
signos repetidos, abreviaturas de chat y espacios extra.
"""
import re
import unicodedata

# Categorías Unicode a eliminar: símbolos (emojis), modificadores, control/formato, marcas sueltas
_CATEGORIAS_ELIMINAR = {"So", "Sk", "Cf", "Cs", "Co", "Cn", "Mn"}

_PREFIJOS = re.compile(
    r"^\s*(?:"
    r"(?:obs|nota|reporte|observaci[oó]n)\s*[:\-–]\s*"
    r"|turno\s+(?:ma[nñ]ana|tarde|noche)\s*:\s*"
    r"|\(?\d{1,2}:\d{2}\)?\s*"
    r"|#\w+\s*"
    r")+",
    flags=re.IGNORECASE,
)
_HORA = re.compile(r"\(?\b\d{1,2}:\d{2}\b\)?")
_HASHTAG = re.compile(r"#\w+")

# Abreviaturas de chat -> forma estándar (se aplican sobre texto en minúsculas)
ABREVIATURAS = {
    r"\bxq\b": "porque", r"\bpq\b": "porque", r"\bx\b": "por", r"\bq\b": "que",
    r"\bc/": "con ", r"\bpa\b": "para", r"\btmb\b": "también", r"\bmui\b": "muy",
    r"\bta\b": "está", r"\bmñn\b": "mañana", r"\bvet\b": "veterinario",
}
_ABREV_COMPILADAS = [(re.compile(p), r) for p, r in ABREVIATURAS.items()]

_ALARGAMIENTO = re.compile(r"([a-záéíóúüñ])\1{2,}")
_PUNT_REPETIDA = re.compile(r"([!?¡¿.,;:])\1+")
_ESPACIOS = re.compile(r"\s+")


def _quitar_simbolos(s: str) -> str:
    return "".join(c for c in s if unicodedata.category(c) not in _CATEGORIAS_ELIMINAR)


def clean_text(texto) -> str:
    """Devuelve la observación limpia. Texto nulo o no-string -> cadena vacía."""
    if texto is None or not isinstance(texto, str):
        return ""
    s = unicodedata.normalize("NFC", texto)
    s = _quitar_simbolos(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = _PREFIJOS.sub("", s)
    s = _HORA.sub(" ", s)
    s = _HASHTAG.sub(" ", s)
    s = s.lower()
    for patron, reemplazo in _ABREV_COMPILADAS:
        s = patron.sub(reemplazo, s)
    s = _ALARGAMIENTO.sub(r"\1", s)
    s = re.sub(r"yy\b", "y", s)  # "hoyy" -> "hoy"
    s = re.sub(r"\s+([!?])\1+", r"\1", s)  # " ??" -> "?"
    s = _PUNT_REPETIDA.sub(r"\1", s)
    s = _ESPACIOS.sub(" ", s).strip()
    return s
