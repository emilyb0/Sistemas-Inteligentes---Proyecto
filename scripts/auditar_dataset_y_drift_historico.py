#!/usr/bin/env python3
"""Auditoría reproducible del corpus y comparación retrospectiva de drift.

No reentrena ni usa el clasificador. El drift compara *etiquetas humanas*
entre semestres (NO predicciones en producción). Solo genera agregados, sin
comentarios, identificadores ni información personal.

Ejemplo:
python -m scripts.auditar_dataset_y_drift_historico \
  --csv datos/comentarios_limpio_ids_ok.csv \
  --salida resultados/auditoria_dataset_y_drift.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from src.embeddings_pipeline import PrepararComentarios
from src.monitoreo import distribuciones, evaluar_drift

CLASES = ("negativo", "neutral", "positivo")
COLUMNAS = {"archivo", "semestre", "id", "id_unico", "feedback", "polaridad"}


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open('rb') as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloque)
    return h.hexdigest()


def preparar_textos(df: pd.DataFrame) -> list[str]:
    return PrepararComentarios(modo="corregido").transform(df["feedback"].tolist())


def resumen_temporal(df: pd.DataFrame, textos: list[str]) -> dict:
    """Referencia 2023-1/2023-2; monitoreo 2024-1 y 2024-2."""
    mascara_ref = df["semestre"].isin(["2023-1", "2023-2"])
    ref = distribuciones([x for x, m in zip(textos, mascara_ref) if m],
                         df.loc[mascara_ref, "polaridad"].tolist())
    if not ref["n"]:
        raise ValueError("No hay observaciones históricas para 2023-1/2023-2")
    series = {}
    for periodo in ["2024-1", "2024-2"]:
        mascara = df["semestre"].eq(periodo)
        actual = distribuciones([x for x, m in zip(textos, mascara) if m],
                                df.loc[mascara, "polaridad"].tolist())
        series[periodo] = {
            "agregados": actual,
            "diagnostico": evaluar_drift(ref, actual),
        }
    return {
        "tipo": "retrospectivo_etiquetas_humanas_no_predicciones",
        "referencia_periodos": ["2023-1", "2023-2"],
        "referencia": ref,
        "periodos_comparados": series,
        "interpretacion": (
            "El PSI de las etiquetas mide cambios entre distribuciones de etiquetas humanas. "
            "No demuestra deriva de las predicciones ni caída de desempeño del modelo. "
            "Para monitoreo de producción, usar drift_referencia.json del Pipeline reentrenado "
            "y comentarios nuevos en scripts.monitorear_drift."
        ),
    }


def medir_preprocesamiento(textos: list[str], repeticiones: int) -> dict:
    preparar = PrepararComentarios(modo="corregido")
    preparar.transform(textos)  # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        preparar.transform(textos)
        tiempos.append((time.perf_counter() - inicio) * 1000)
    tiempos.sort()
    return {
        "tipo": "solo_limpieza_no_inferencia",
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "repeticiones": repeticiones,
        "n_comentarios": len(textos),
        "tiempo_lote_ms_mediana": statistics.median(tiempos),
        "tiempo_lote_ms_p95": tiempos[int(.95 * (len(tiempos) - 1))],
        "ms_por_comentario_mediana": statistics.median(tiempos) / len(textos),
        "nota": "Medición de preprocesamiento únicamente; no incluye embeddings ni LogisticRegression.",
    }


def auditar(ruta: Path, repeticiones: int = 30) -> dict:
    if repeticiones < 5:
        raise ValueError("repeticiones debe ser >= 5")
    df = pd.read_csv(ruta, encoding="utf-8-sig", dtype={"semestre": str})
    faltan = COLUMNAS - set(df.columns)
    if faltan:
        raise ValueError(f"Columnas faltantes: {sorted(faltan)}")
    if df["feedback"].isna().any() or df["feedback"].astype(str).str.strip().eq("").any():
        raise ValueError("Hay textos faltantes o vacíos")
    if df["polaridad"].isna().any() or not set(df["polaridad"]).issubset(CLASES):
        raise ValueError("Las etiquetas son inválidas o faltan")
    if df["feedback"].duplicated().any() or df.duplicated(["archivo", "semestre", "id"]).any():
        raise ValueError("Hay comentarios duplicados o claves compuestas repetidas")
    _, i_test = train_test_split(range(len(df)), test_size=.2, random_state=42,
                                 stratify=df["polaridad"].tolist())
    textos = df["feedback"].tolist()
    preparados = preparar_textos(df)
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "nombre_archivo": ruta.name,
            "sha256_dataset": sha256_archivo(ruta),
            "n": len(df), "columnas": len(df.columns),
            "etiquetas": dict(Counter(df["polaridad"])),
            "periodos": dict(Counter(df["semestre"])),
            "duplicados_feedback": int(df["feedback"].duplicated().sum()),
            "duplicados_id_unico": int(df["id_unico"].duplicated().sum()),
            "duplicados_clave_compuesta": int(df.duplicated(["archivo", "semestre", "id"]).sum()),
            "particion_propuesta": {"semilla":42,"metodo":"estratificado_80_20","n_train":len(df)-len(i_test),"n_test":len(i_test)},
        },
        "drift_historico": resumen_temporal(df, preparados),
        "rendimiento_preprocesamiento": medir_preprocesamiento(textos, repeticiones),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--repeticiones", type=int, default=30)
    args = ap.parse_args()
    resultado = auditar(args.csv, args.repeticiones)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
    print(json.dumps({"salida":str(args.salida),"sha256_dataset":resultado["dataset"]["sha256_dataset"],
        "drift_por_periodo":{k:v["diagnostico"] for k,v in resultado["drift_historico"]["periodos_comparados"].items()},
        "rendimiento_limpieza":resultado["rendimiento_preprocesamiento"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
