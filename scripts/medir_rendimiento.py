#!/usr/bin/env python3
"""Medición reproducible de tiempos (modelo real, después de instalar dependencias).

MODELO_VERSION=v2_embeddings python -m scripts.medir_rendimiento --salida resultados/rendimiento_v2.json
"""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import statistics
import time

import pandas as pd

from src.modelo import obtener_modelo, predecir, predecir_lote

FRASES = [
    "El curso fue muy interesante y aprendí muchísimo.",
    "La evaluación fue poco clara y el docente no respondió preguntas.",
    "Las clases son los jueves y hay tres parciales.",
    "El material era adecuado, aunque faltó retroalimentación.",
    "El profesor explica muy bien, pero el tiempo no alcanza.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--csv", type=Path, help="CSV local opcional con comentarios de evaluación reales")
    ap.add_argument("--columna", default="feedback")
    args = ap.parse_args()
    if args.n < 10:
        ap.error("n debe ser >= 10")
    if args.csv:
        datos = pd.read_csv(args.csv, encoding="utf-8-sig")
        if args.columna not in datos.columns:
            ap.error(f"No existe la columna {args.columna!r}")
        muestras = datos[args.columna].dropna().astype(str)
        muestras = muestras[muestras.str.strip() != ""].head(args.n).tolist()
        if len(muestras) < 10:
            ap.error("El CSV debe contener por lo menos 10 comentarios válidos")
        fuente = "csv_local_no_persistido"
    else:
        muestras = (FRASES * ((args.n + len(FRASES)-1)//len(FRASES)))[:args.n]
        fuente = "5_frases_de_ejemplo_repetidas"
    inicio = time.perf_counter()
    modelo = obtener_modelo()
    carga = time.perf_counter() - inicio
    inicio_primera = time.perf_counter()
    predecir(FRASES[0])  # primera inferencia puede cargar/descargar encoder
    primera_seg = time.perf_counter() - inicio_primera
    tiempos_ms = []
    for texto in muestras:
        t0 = time.perf_counter()
        predecir(texto)
        tiempos_ms.append((time.perf_counter() - t0) * 1000)
    lotes_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        predecir_lote(muestras)
        lotes_ms.append((time.perf_counter() - t0) * 1000)
    lote_ms = statistics.median(lotes_ms)
    p95 = sorted(tiempos_ms)[int(.95 * (len(tiempos_ms) - 1))]
    resultado = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "version_modelo": modelo.version,
        "sha256_modelo": modelo.sha256,
        "python": platform.python_version(), "plataforma": platform.platform(),
        "fuente": fuente, "n": len(muestras),
        "segundos_carga_artefacto": carga,
        "segundos_primera_prediccion_con_carga_encoder": primera_seg,
        "segundos_primera_respuesta_aproximada": carga + primera_seg,
        "latencia_individual_ms_promedio": statistics.mean(tiempos_ms),
        "latencia_individual_ms_mediana": statistics.median(tiempos_ms),
        "latencia_individual_ms_p95": p95,
        "tiempo_lote_ms_mediana_5_repeticiones": lote_ms,
        "tiempo_lote_ms_p95_5_repeticiones": sorted(lotes_ms)[int(.95 * (len(lotes_ms) - 1))],
        "tiempo_lote_ms_por_comentario": lote_ms / len(muestras),
        "nota": "Primera predicción puede cargar/descargar encoder. Estado estable separado de carga; CPU local. No se almacena texto.",
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
