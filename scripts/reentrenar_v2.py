#!/usr/bin/env python3
"""Entrena v2 con C=0.3, serializa un Pipeline único y genera trazabilidad.

Ejecutar desde la raíz del repositorio:
 python -m scripts.reentrenar_v2 --csv datos/comentarios_limpio.csv

Modo 'historico' = reproducir artículo original; mantiene el error de
anonimización del notebook. Modo 'corregido' (por defecto) = evitar dicho
error y actualizar las métricas publicadas con las métricas que salgan.

No sobrescribe modelos hasta que termina el entrenamiento y la evaluación.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.embeddings_pipeline import (
    ENCODER_PREDETERMINADO, PrepararComentarios, VectorizadorEmbeddings,
)
from src.monitoreo import distribuciones

CLASES = ("negativo", "neutral", "positivo")
SEMILLA = 42
C_ARTICULO = 0.3


def hash_archivo(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def revision_encoder(nombre: str, revision: str | None) -> str:
    if revision:
        return revision
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(nombre).sha
    except Exception as e:
        raise RuntimeError(
            "No se pudo fijar una revisión de Hugging Face. Indica --revision "
            "con el SHA de commit del encoder o verifica el acceso a Internet."
        ) from e


def leer_datos(ruta: Path) -> pd.DataFrame:
    datos = pd.read_csv(ruta, encoding="utf-8-sig", dtype={"semestre": str})
    necesarias = {"feedback", "polaridad", "archivo", "semestre", "id"}
    if not necesarias.issubset(datos.columns):
        raise ValueError(f"Faltan columnas: {sorted(necesarias - set(datos.columns))}")
    if datos["feedback"].isna().any() or datos["polaridad"].isna().any():
        raise ValueError("Datos con texto o polaridad faltante")
    if datos["feedback"].astype(str).str.strip().eq("").any():
        raise ValueError("Hay comentarios vacíos")
    if not set(datos["polaridad"]).issubset(CLASES):
        raise ValueError("Hay polaridades desconocidas")
    if datos["feedback"].duplicated().any():
        raise ValueError("Hay comentarios exactamente duplicados")
    claves = (datos["archivo"].astype(str) + "|" + datos["semestre"].astype(str)
              + "|" + datos["id"].astype(str))
    if claves.duplicated().any():
        raise ValueError("Colisión de (archivo, semestre, id)")
    return datos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--salida", type=Path, default=Path("modelos/v2_embeddings_candidato"))
    ap.add_argument("--modo", choices=("corregido", "historico"), default="corregido")
    ap.add_argument("--revision", help="SHA de commit de Hugging Face; por defecto, se consulta a HF")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--c", type=float, default=C_ARTICULO)
    args = ap.parse_args()

    if args.c <= 0 or args.batch_size < 1:
        ap.error("C y batch-size deben ser positivos")

    datos = leer_datos(args.csv)
    indices = np.arange(len(datos))
    y = datos["polaridad"].tolist()
    i_train, i_test = train_test_split(
        indices, test_size=.2, random_state=SEMILLA, stratify=y,
    )
    x_train = datos.iloc[i_train]["feedback"].tolist()
    x_test = datos.iloc[i_test]["feedback"].tolist()
    y_train = datos.iloc[i_train]["polaridad"].tolist()
    y_test = datos.iloc[i_test]["polaridad"].tolist()
    revision = revision_encoder(ENCODER_PREDETERMINADO, args.revision)
    limpiar = PrepararComentarios(modo=args.modo)
    embeddings = VectorizadorEmbeddings(
        nombre_modelo=ENCODER_PREDETERMINADO,
        revision=revision,
        batch_size=args.batch_size,
    )

    # El encoder es preentrenado y NO se ajusta a estos comentarios.
    # Por tanto, es seguro calcular embeddings de TRAIN antes de CV.
    e_train = embeddings.transform(limpiar.transform(x_train))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"C": [.01, .1, .3, 1, 3, 10, 30]},
        scoring="f1_macro", cv=cv, n_jobs=1,
    )
    grid.fit(e_train, y_train)
    c_cv = float(grid.best_params_["C"])
    print(f"Selección por CV: C={c_cv} | Macro-F1={grid.best_score_:.4f}")
    if abs(c_cv - args.c) > 1e-9:
        print(
            f"ADVERTENCIA: C fijado por argumento ({args.c}) no coincide con "
            f"el óptimo por CV ({c_cv}). No cambiar C basándose en el test.",
            file=sys.stderr,
        )

    pipe = Pipeline([
        ("limpieza", PrepararComentarios(modo=args.modo)),
        ("embeddings", embeddings),
        ("clf", LogisticRegression(C=args.c, max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(x_train, y_train)

    # Reservar el test hasta haber seleccionado el procedimiento y C.
    pred = pipe.predict(x_test)
    prob = pipe.predict_proba(x_test)
    metrics = {
        "accuracy_test": float(accuracy_score(y_test, pred)),
        "macro_f1_test": float(f1_score(y_test, pred, average="macro", labels=CLASES, zero_division=0)),
        "matriz_confusion": confusion_matrix(y_test, pred, labels=CLASES).tolist(),
        "reporte_test": classification_report(y_test, pred, labels=CLASES, output_dict=True, zero_division=0),
    }
    rng = np.random.default_rng(SEMILLA)
    yt, yp = np.array(y_test), np.asarray(pred)
    bs = [f1_score(yt[j], yp[j], labels=CLASES, average="macro", zero_division=0)
          for j in (rng.integers(len(yt), size=len(yt)) for _ in range(1000))]
    metrics["ic_95_macro_f1_bootstrap"] = [float(v) for v in np.percentile(bs, [2.5, 97.5])]

    # Pipeline liviano: los pesos HF se recuperan por revisión fijada.
    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_modelo = args.salida / "modelo.joblib"
    joblib.dump(pipe, ruta_modelo)
    recargado = joblib.load(ruta_modelo)  # artefacto producido localmente en esta ejecución
    if not np.array_equal(recargado.predict(x_test), pred):
        raise AssertionError("Predicciones cambiaron al guardar/recargar el Pipeline")
    hash_modelo = hash_archivo(ruta_modelo)
    (args.salida / "modelo.sha256").write_text(
        f"{hash_modelo}  modelo.joblib\n", encoding="utf-8",
    )

    import sentence_transformers
    info = {
        "version": "v2_embeddings",
        "modelo": "Pipeline de limpieza + embeddings + regresión logística",
        "C": args.c,
        "C_optimo_por_CV": c_cv,
        "macro_f1_cv_optimo": float(grid.best_score_),
        "macro_f1_cv_C_elegido": float(next(
            score for c, score in zip(grid.cv_results_["param_C"], grid.cv_results_["mean_test_score"])
            if abs(float(c) - args.c) < 1e-9
        )) if any(abs(float(c) - args.c) < 1e-9 for c in grid.cv_results_["param_C"]) else None,
        "modelo_embeddings": ENCODER_PREDETERMINADO,
        "revision_embeddings": revision,
        "modo_preprocesamiento": args.modo,
        "clases": list(pipe.classes_),
        "n_train": len(i_train), "n_test": len(i_test),
        "distribucion_train": dict(Counter(y_train)),
        "distribucion_test": dict(Counter(y_test)),
        "semilla": SEMILLA,
        "csv": args.csv.name,
        "sha256_dataset": hash_archivo(args.csv),
        "sha256_modelo": hash_modelo,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "fecha": datetime.now(timezone.utc).date().isoformat(),
        "advertencia_metodologica": (
            "El test de 123 comentarios ya se exploró en versiones previas: esta evaluación "
            "no representa un holdout completamente virgen."
        ),
        **metrics,
    }
    (args.salida / "modelo_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    # Referencia de drift calculada únicamente con las predicciones del train.
    referencia = distribuciones(
        limpiar.transform(x_train), list(pipe.predict(x_train)),
    )
    referencia.update({
        "version_modelo": f"v2_embeddings-{info['fecha']}-{hash_modelo[:8]}",
        "sha256_modelo": hash_modelo,
        "origen": "predicciones en entrenamiento, referencia exploratoria",
    })
    (args.salida / "drift_referencia.json").write_text(
        json.dumps(referencia, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    # No escribir textos privados entre los artefactos de salida.
    print(json.dumps({
        "c_elegido": args.c,
        "c_cv": c_cv,
        "modo_preprocesamiento": args.modo,
        "accuracy_test": metrics["accuracy_test"],
        "macro_f1_test": metrics["macro_f1_test"],
        "sha256_modelo": hash_modelo,
    }, ensure_ascii=False, indent=2))
    print(f"Archivos guardados en {args.salida}")


if __name__ == "__main__":
    main()
