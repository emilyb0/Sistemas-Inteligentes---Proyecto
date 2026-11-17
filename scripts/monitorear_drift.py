#!/usr/bin/env python3
"""Monitoreo agregado de distribución: NO persiste comentarios de estudiantes.

MODELO_VERSION=v2_embeddings python -m scripts.monitorear_drift \\
  --csv ejemplos.csv --columna comentario \\
  --referencia modelos/v2_embeddings/drift_referencia.json \\
  --salida resultados/drift_2026-11-09.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.modelo import obtener_modelo, predecir_lote
from src.monitoreo import distribuciones, evaluar_drift


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--columna", default="comentario")
    ap.add_argument("--referencia", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    args = ap.parse_args()
    modelo = obtener_modelo()
    referencia = json.loads(args.referencia.read_text(encoding="utf-8"))
    if referencia.get("sha256_modelo") != modelo.sha256:
        raise ValueError("La referencia de drift corresponde a otro modelo")
    df = pd.read_csv(args.csv, encoding="utf-8-sig", dtype={args.columna: str})
    if args.columna not in df:
        raise ValueError(f"El CSV no contiene la columna {args.columna!r}")
    comentarios = df[args.columna].dropna().astype(str)
    comentarios = comentarios[comentarios.str.strip() != ""].tolist()
    if not comentarios:
        raise ValueError("No hay comentarios válidos para monitorear")
    clase_preds, procesados = [], []
    for i in range(0, len(comentarios), 64):
        lote = comentarios[i:i+64]
        clase_preds.extend(p["clase"] for p in predecir_lote(lote))
        if hasattr(modelo, "preparar_para_aviso"):
            procesados.extend(modelo.preparar_para_aviso(lote))
        else:
            from src.preprocesamiento import preparar
            procesados.extend(map(preparar, lote))
    actual = distribuciones(procesados, clase_preds)
    evaluacion = evaluar_drift(referencia, actual)
    salida = {
        "modelo": modelo.version, "referencia": str(args.referencia),
        "distribuciones_actuales": actual, "diagnostico": evaluacion,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(salida, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(evaluacion, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
