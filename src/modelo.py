"""Carga segura del Pipeline entrenado y funciones de predicción.

- El modelo se carga una sola vez por proceso (``obtener_modelo`` está en caché).
- Solo se cargan archivos dentro de ``modelos/`` cuyo SHA-256 coincide con el
  archivo ``<nombre>.sha256`` que lo acompaña: ``joblib.load`` puede ejecutar
  código, así que el hash se verifica ANTES de deserializar.
- La ruta se toma de la variable de entorno ``MODELO_PATH`` (por defecto
  ``./modelos/modelo.joblib``) para poder cambiar o revertir de versión.
- Una predicción es de confianza baja si su probabilidad máxima es menor que
  ``UMBRAL_CONFIANZA`` (variable de entorno del mismo nombre, por defecto 0.50) o
  si el modelo reconoce menos de ``MIN_PALABRAS_RECONOCIDAS`` palabras del texto.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from src.preprocesamiento import preparar

log = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "modelos"
RUTA_POR_DEFECTO = "./modelos/modelo.joblib"
CLASES = ("negativo", "neutral", "positivo")
MIN_PALABRAS_RECONOCIDAS = 2

AVISO_POCAS_PALABRAS = (
    "El modelo reconoce muy pocas palabras de este comentario, así que la clasificación no es confiable. "
    "Intenta describir con más detalle qué te gustó o no te gustó del curso."
)
AVISO_PROBABILIDAD_BAJA = "El modelo no está seguro: las probabilidades de las clases son muy parecidas."


def leer_umbral(valor_por_defecto: float = 0.50) -> float:
    """Umbral de confianza desde la variable de entorno UMBRAL_CONFIANZA."""
    crudo = os.environ.get("UMBRAL_CONFIANZA")
    if crudo is None or not crudo.strip():
        return valor_por_defecto
    try:
        umbral = float(crudo.replace(",", "."))
    except ValueError:
        raise ValueError(f"UMBRAL_CONFIANZA debe ser un número entre 0 y 1; se recibió {crudo!r}.") from None
    if not 0.0 <= umbral <= 1.0:
        raise ValueError(f"UMBRAL_CONFIANZA debe estar entre 0 y 1; se recibió {umbral}.")
    return umbral


UMBRAL_CONFIANZA = leer_umbral()


@dataclass(frozen=True)
class ModeloCargado:
    pipeline: Pipeline
    ruta: Path
    sha256: str
    version: str
    info: dict = field(default_factory=dict)


def ruta_configurada() -> Path:
    """Ruta del modelo según MODELO_PATH; las rutas relativas parten de la raíz del repo."""
    ruta = Path(os.environ.get("MODELO_PATH") or RUTA_POR_DEFECTO)
    return (ruta if ruta.is_absolute() else RAIZ / ruta).resolve()


def calcular_sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_modelo(ruta: Path, dir_confianza: Path = DIR_MODELOS) -> ModeloCargado:
    """Verifica origen y hash del archivo y solo entonces lo deserializa."""
    ruta = Path(ruta).resolve()
    dir_confianza = Path(dir_confianza).resolve()

    if not ruta.is_relative_to(dir_confianza):
        raise PermissionError(f"Solo se cargan modelos de {dir_confianza}; se pidió {ruta}.")
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el modelo {ruta}.")

    ruta_hash = ruta.with_suffix(".sha256")
    if not ruta_hash.is_file():
        raise FileNotFoundError(f"Falta el archivo de hash {ruta_hash.name} junto al modelo.")
    esperado = ruta_hash.read_text(encoding="utf-8").split()[0].strip().lower()
    real = calcular_sha256(ruta)
    if real != esperado:
        raise ValueError(
            f"El SHA-256 de {ruta.name} no coincide con {ruta_hash.name} "
            f"(esperado {esperado[:12]}…, obtenido {real[:12]}…). No se carga el modelo."
        )

    info = {}
    ruta_info = ruta.with_name(f"{ruta.stem}_info.json")
    if ruta_info.is_file():
        info = json.loads(ruta_info.read_text(encoding="utf-8"))
        if info.get("sklearn") and info["sklearn"] != sklearn.__version__:
            raise RuntimeError(
                f"El modelo se entrenó con scikit-learn {info['sklearn']} y está instalada la "
                f"{sklearn.__version__}. Instala la versión fijada en requirements.txt."
            )
        if info.get("python") and info["python"].rsplit(".", 1)[0] != platform.python_version().rsplit(".", 1)[0]:
            log.warning("Modelo entrenado con Python %s; se está usando %s.",
                        info["python"], platform.python_version())
    else:
        log.warning("No se encontró %s; no se puede comprobar la versión de entrenamiento.", ruta_info.name)

    pipeline = joblib.load(ruta)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"{ruta.name} no contiene un sklearn Pipeline.")
    if tuple(pipeline.classes_) != CLASES:
        raise ValueError(f"Clases inesperadas en el modelo: {list(pipeline.classes_)}.")

    version = f"{ruta.stem}-{info.get('fecha', 'sin-fecha')}-{real[:8]}"
    log.info("Modelo cargado: %s (%s)", ruta, version)
    return ModeloCargado(pipeline, ruta, real, version, info)


@lru_cache(maxsize=1)
def obtener_modelo() -> ModeloCargado:
    """Modelo único del proceso: se carga la primera vez y se reutiliza."""
    return cargar_modelo(ruta_configurada())


def _contar_palabras(preparado: str) -> tuple[int, int]:
    """(palabras reconocidas por el vocabulario, palabras totales) de un texto ya preparado."""
    vec = obtener_modelo().pipeline.steps[0][1]
    palabras = vec.build_tokenizer()(preparado)
    return sum(p in vec.vocabulary_ for p in palabras), len(palabras)


def _advertencia(prob_max: float, reconocidas: int) -> str | None:
    if reconocidas < MIN_PALABRAS_RECONOCIDAS:
        return AVISO_POCAS_PALABRAS
    if prob_max < UMBRAL_CONFIANZA:
        return AVISO_PROBABILIDAD_BAJA
    return None


def _resultado(probas, reconocidas: int, version: str) -> dict:
    advertencia = _advertencia(float(probas.max()), reconocidas)
    return {
        "clase": CLASES[probas.argmax()],
        "probabilidades": {c: float(p) for c, p in zip(CLASES, probas)},
        "version_modelo": version,
        "confianza_baja": advertencia is not None,
        "advertencia": advertencia,
    }


def predecir_lote(textos: list[str]) -> list[dict]:
    m = obtener_modelo()
    preparados = [preparar("" if t is None else str(t)) for t in textos]
    if not preparados:
        return []
    probas = m.pipeline.predict_proba(preparados)
    return [_resultado(p, _contar_palabras(t)[0], m.version) for p, t in zip(probas, preparados)]


def predecir(texto: str) -> dict:
    """Devuelve {"clase", "probabilidades": {negativo, neutral, positivo}, "version_modelo",
    "confianza_baja", "advertencia"}."""
    return predecir_lote([texto])[0]


def palabras_reconocidas(texto: str) -> int:
    """Cuántas palabras del texto (tras preparar) están en el vocabulario del modelo."""
    return contar_palabras(texto)[0]


def contar_palabras(texto: str) -> tuple[int, int]:
    """(reconocidas, totales) con el mismo tokenizador del vectorizador."""
    return _contar_palabras(preparar("" if texto is None else str(texto)))
