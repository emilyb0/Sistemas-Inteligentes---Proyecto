"""Monitoreo ligero de cambios en entradas/salidas del clasificador.

Mide drift de distribución sin almacenar comentarios ni información personal.
No demuestra una disminución de accuracy (no hay etiquetas de referencia).
"""
from __future__ import annotations

from collections import Counter
import math

CLASES = ("negativo", "neutral", "positivo")
BORDES_PALABRAS = (0, 10, 25, 50, 100, math.inf)


def _bin_palabras(t: str) -> str:
    n = len(str(t).split())
    if n < 10:
        return "0-9"
    if n < 25:
        return "10-24"
    if n < 50:
        return "25-49"
    if n < 100:
        return "50-99"
    return "100+"


def distribuciones(textos_preparados: list[str], clases_predichas: list[str]) -> dict:
    if len(textos_preparados) != len(clases_predichas):
        raise ValueError("Las listas de textos y clases deben tener la misma longitud")
    if any(c not in CLASES for c in clases_predichas):
        raise ValueError("Clase desconocida en predicciones")
    return {
        "n": len(clases_predichas),
        "conteos_clases": {c: Counter(clases_predichas)[c] for c in CLASES},
        "conteos_longitud": {
            b: Counter(map(_bin_palabras, textos_preparados))[b]
            for b in ("0-9", "10-24", "25-49", "50-99", "100+")
        },
    }


def psi_conteos(referencia: dict, actual: dict, epsilon: float = 1e-4) -> float:
    """PSI usando suavizado para categorías poco frecuentes o ausentes."""
    if set(referencia) != set(actual):
        raise ValueError("Las distribuciones no usan las mismas categorías")
    claves = sorted(referencia)
    s0, s1 = sum(referencia.values()), sum(actual.values())
    if s0 <= 0 or s1 <= 0:
        raise ValueError("Las distribuciones deben contener observaciones")
    n = len(claves)
    valor = 0.0
    for k in claves:
        p = (referencia[k] / s0 + epsilon) / (1 + epsilon * n)
        q = (actual[k] / s1 + epsilon) / (1 + epsilon * n)
        valor += (q - p) * math.log(q / p)
    return float(valor)


def evaluar_drift(referencia: dict, actual: dict, min_muestra: int = 50, umbral: float = .2) -> dict:
    if actual["n"] < min_muestra:
        return {
            "estado": "muestra_insuficiente", "n": actual["n"],
            "min_muestra": min_muestra,
            "nota": "No interpretar PSI con una muestra pequeña.",
        }
    resultados = {
        "psi_clases": psi_conteos(referencia["conteos_clases"], actual["conteos_clases"]),
        "psi_longitud": psi_conteos(referencia["conteos_longitud"], actual["conteos_longitud"]),
    }
    alerta = any(v >= umbral for v in resultados.values())
    return {
        "estado": "alerta_revisar" if alerta else "sin_alerta",
        "n": actual["n"], "umbral_heuristico": umbral,
        **resultados,
        "nota": "PSI no mide exactitud ni sustituye revisión humana con etiquetas reales.",
    }
