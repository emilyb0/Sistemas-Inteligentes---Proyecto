#!/usr/bin/env python3
"""Crea una copia con identificador compuesto único, sin modificar etiquetas/textos.

python -m scripts.corregir_ids --csv datos/comentarios_limpio.csv \\
  --salida datos/comentarios_limpio_ids_ok.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    args = ap.parse_args()
    if args.csv.resolve() == args.salida.resolve():
        ap.error("La salida debe ser un archivo distinto para conservar el original")
    df = pd.read_csv(args.csv, encoding="utf-8-sig", dtype={"semestre": str})
    requeridas = {"archivo", "semestre", "id"}
    if not requeridas.issubset(df.columns):
        raise ValueError("Faltan columnas para la clave compuesta")
    df["id_unico"] = (
        df["archivo"].astype(str) + "_" + df["semestre"].astype(str)
        + "_" + df["id"].astype(str)
    )
    if df["id_unico"].duplicated().any():
        raise ValueError("Sigue habiendo IDs repetidos; revisar origen")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False, encoding="utf-8-sig")
    print(f"{len(df)} filas; {df['id_unico'].nunique()} identificadores distintos en {args.salida}")


if __name__ == "__main__":
    main()
