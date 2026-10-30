"""Carga segura de la versión activa del modelo y funciones de predicción.

Versiones (cada una en su carpeta de ``modelos/``):
- ``v1_tfidf``: Pipeline TfidfVectorizer + LogisticRegression (``modelo.joblib``).
- ``v2_embeddings``: embeddings de sentence-transformers + LogisticRegression
  (``clasificador.joblib``). El modelo de embeddings se descarga de Hugging Face
  la primera vez; no está en el repositorio.

- La versión se elige con la variable de entorno ``MODELO_VERSION`` (por defecto
  ``v2_embeddings``) y se carga una sola vez por proceso. ``MODELO_PATH`` permite,
  opcionalmente, apuntar a otro .joblib de esa versión dentro de ``modelos/``.
- ``joblib.load`` puede ejecutar código, así que antes de deserializar se verifica
  que el archivo esté en ``modelos/`` y que su SHA-256 coincida con el
  ``modelo.sha256`` de su carpeta.
- Una predicción es de confianza baja si su probabilidad máxima es menor que
  ``UMBRAL_CONFIANZA`` (variable de entorno, por defecto 0.50). Además, en v1 si
  el modelo reconoce menos de 2 palabras del vocabulario y en v2 si el texto
  tiene menos de 3 palabras.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import sklearn

from src.preprocesamiento import preparar

log = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "modelos"
CLASES = ("negativo", "neutral", "positivo")
ARCHIVOS = {"v1_tfidf": "modelo.joblib", "v2_embeddings": "clasificador.joblib"}
VERSIONES = tuple(ARCHIVOS)
VERSION_POR_DEFECTO = "v2_embeddings"

MIN_PALABRAS_RECONOCIDAS = 2   # v1
MIN_PALABRAS_TEXTO = 3         # v2

AVISO_POCAS_PALABRAS = (
    "El modelo reconoce muy pocas palabras de este comentario, así que la clasificación no es confiable. "
    "Intenta describir con más detalle qué te gustó o no te gustó del curso."
)
AVISO_TEXTO_CORTO = (
    "El comentario es demasiado corto para clasificarlo con confianza. "
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


# ------------------------------------------------------------------ versiones
class ModeloBase(ABC):
    """Interfaz común: la API y la app solo usan estos métodos y atributos."""

    def __init__(self, version_id: str, ruta: Path, sha256: str, info: dict):
        self.version_id = version_id
        self.ruta = ruta
        self.sha256 = sha256
        self.info = info
        self.version = f"{version_id}-{info.get('fecha', 'sin-fecha')}-{sha256[:8]}"

    @abstractmethod
    def predecir_proba(self, textos_preparados: list[str]) -> np.ndarray:
        """Matriz (n, 3) con las probabilidades en el orden de CLASES."""

    @abstractmethod
    def aviso_texto(self, preparado: str) -> str | None:
        """Advertencia que depende solo del texto (antes de mirar la probabilidad)."""

    @abstractmethod
    def nota_confianza(self, texto: str) -> str:
        """Línea corta que la app muestra junto a la advertencia de confianza baja."""


class ModeloTfidf(ModeloBase):
    def __init__(self, pipeline, **kw):
        super().__init__("v1_tfidf", **kw)
        self.pipeline = pipeline
        self._vec = pipeline.steps[0][1]
        self._tokenizar = self._vec.build_tokenizer()

    def predecir_proba(self, textos_preparados):
        return self.pipeline.predict_proba(textos_preparados)

    def contar_palabras(self, preparado: str) -> tuple[int, int]:
        """(palabras reconocidas por el vocabulario, palabras totales)."""
        palabras = self._tokenizar(preparado)
        return sum(p in self._vec.vocabulary_ for p in palabras), len(palabras)

    def aviso_texto(self, preparado):
        return AVISO_POCAS_PALABRAS if self.contar_palabras(preparado)[0] < MIN_PALABRAS_RECONOCIDAS else None

    def nota_confianza(self, texto):
        reconocidas, totales = self.contar_palabras(preparar(str(texto)))
        return f"Palabras reconocidas: {reconocidas} de {totales}"


class ModeloEmbeddings(ModeloBase):
    def __init__(self, clasificador, codificador, **kw):
        super().__init__("v2_embeddings", **kw)
        self.clasificador = clasificador
        self.codificador = codificador

    def predecir_proba(self, textos_preparados):
        # encode() por defecto (sin normalizar), igual que en el entrenamiento
        embeddings = self.codificador.encode(
            textos_preparados, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
        return self.clasificador.predict_proba(embeddings)

    def aviso_texto(self, preparado):
        return AVISO_TEXTO_CORTO if len(re.findall(r"\w+", preparado)) < MIN_PALABRAS_TEXTO else None

    def nota_confianza(self, texto):
        return "Modelo: v2 · embeddings"


# ------------------------------------------------------------------ carga
def version_configurada() -> str:
    version = (os.environ.get("MODELO_VERSION") or VERSION_POR_DEFECTO).strip()
    if version not in ARCHIVOS:
        raise ValueError(f"MODELO_VERSION debe ser uno de {', '.join(VERSIONES)}; se recibió {version!r}.")
    return version


def ruta_configurada(version: str | None = None) -> Path:
    """Ruta del .joblib: MODELO_PATH si existe; si no, el archivo de la versión. Relativa a la raíz del repo."""
    version = version or version_configurada()
    ruta = Path(os.environ.get("MODELO_PATH") or Path("modelos") / version / ARCHIVOS[version])
    return (ruta if ruta.is_absolute() else RAIZ / ruta).resolve()


def calcular_sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _verificar(ruta: Path, dir_confianza: Path) -> tuple[str, dict]:
    """Comprueba origen, hash y versiones; devuelve (sha256, info). No deserializa nada."""
    ruta = Path(ruta).resolve()
    dir_confianza = Path(dir_confianza).resolve()

    if not ruta.is_relative_to(dir_confianza):
        raise PermissionError(f"Solo se cargan modelos de {dir_confianza}; se pidió {ruta}.")
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el modelo {ruta}.")

    ruta_hash = ruta.parent / "modelo.sha256"
    if not ruta_hash.is_file():
        raise FileNotFoundError(f"Falta el archivo de hash modelo.sha256 junto a {ruta.name}.")
    partes = ruta_hash.read_text(encoding="utf-8").split()
    esperado = partes[0].strip().lower()
    if len(partes) > 1 and partes[1].lstrip("*") != ruta.name:
        raise ValueError(f"modelo.sha256 corresponde a {partes[1]}, no a {ruta.name}.")
    real = calcular_sha256(ruta)
    if real != esperado:
        raise ValueError(
            f"El SHA-256 de {ruta.name} no coincide con modelo.sha256 "
            f"(esperado {esperado[:12]}…, obtenido {real[:12]}…). No se carga el modelo."
        )

    info = {}
    ruta_info = ruta.parent / "modelo_info.json"
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
        log.warning("No se encontró modelo_info.json junto a %s; no se pueden comprobar versiones.", ruta.name)
    return real, info


@lru_cache(maxsize=None)
def _codificador(nombre: str):
    """SentenceTransformer cargado una sola vez por proceso (pesos en safetensors, sin código remoto)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nombre, device="cpu")


def cargar_modelo(ruta: Path, version: str = "v1_tfidf", dir_confianza: Path = DIR_MODELOS) -> ModeloBase:
    """Verifica origen, hash y versiones del archivo y solo entonces lo deserializa."""
    if version not in ARCHIVOS:
        raise ValueError(f"Versión desconocida: {version!r}.")
    ruta = Path(ruta).resolve()
    sha, info = _verificar(ruta, dir_confianza)

    if version == "v2_embeddings":
        import sentence_transformers
        esperado = info.get("sentence_transformers")
        if esperado and esperado != sentence_transformers.__version__:
            raise RuntimeError(
                f"v2 se entrenó con sentence-transformers {esperado} y está instalada la "
                f"{sentence_transformers.__version__}. Instala la versión fijada en requirements.txt."
            )
        if not info.get("modelo_embeddings"):
            raise ValueError("modelo_info.json de v2 debe indicar 'modelo_embeddings'.")

    objeto = joblib.load(ruta)
    if tuple(getattr(objeto, "classes_", ())) != CLASES:
        raise ValueError(f"Clases inesperadas en {ruta.name}: {list(getattr(objeto, 'classes_', []))}.")

    if version == "v1_tfidf":
        from sklearn.pipeline import Pipeline
        if not isinstance(objeto, Pipeline):
            raise TypeError(f"{ruta.name} no contiene un sklearn Pipeline.")
        modelo = ModeloTfidf(objeto, ruta=ruta, sha256=sha, info=info)
    else:
        from sklearn.linear_model import LogisticRegression
        if not isinstance(objeto, LogisticRegression):
            raise TypeError(f"{ruta.name} no contiene una LogisticRegression.")
        codificador = _codificador(info["modelo_embeddings"])
        dim = codificador.get_embedding_dimension()
        if dim != objeto.n_features_in_:
            raise ValueError(f"El clasificador espera {objeto.n_features_in_} dimensiones y los embeddings tienen {dim}.")
        modelo = ModeloEmbeddings(objeto, codificador, ruta=ruta, sha256=sha, info=info)

    log.info("Modelo cargado: %s (%s)", ruta, modelo.version)
    return modelo


@lru_cache(maxsize=None)
def _modelo_en_cache(version: str, ruta: Path) -> ModeloBase:
    return cargar_modelo(ruta, version)


def obtener_modelo() -> ModeloBase:
    """Modelo de la versión configurada: se carga la primera vez y se reutiliza."""
    version = version_configurada()
    return _modelo_en_cache(version, ruta_configurada(version))


# ------------------------------------------------------------------ predicción
def _advertencia(modelo: ModeloBase, preparado: str, prob_max: float) -> str | None:
    aviso = modelo.aviso_texto(preparado)
    if aviso:
        return aviso
    if prob_max < UMBRAL_CONFIANZA:
        return AVISO_PROBABILIDAD_BAJA
    return None


def _resultado(modelo: ModeloBase, probas, preparado: str) -> dict:
    advertencia = _advertencia(modelo, preparado, float(probas.max()))
    return {
        "clase": CLASES[probas.argmax()],
        "probabilidades": {c: float(p) for c, p in zip(CLASES, probas)},
        "version_modelo": modelo.version,
        "confianza_baja": advertencia is not None,
        "advertencia": advertencia,
    }


def predecir_lote(textos: list[str]) -> list[dict]:
    m = obtener_modelo()
    preparados = [preparar("" if t is None else str(t)) for t in textos]
    if not preparados:
        return []
    probas = m.predecir_proba(preparados)
    return [_resultado(m, p, t) for p, t in zip(probas, preparados)]


def predecir(texto: str) -> dict:
    """Devuelve {"clase", "probabilidades": {negativo, neutral, positivo}, "version_modelo",
    "confianza_baja", "advertencia"}."""
    return predecir_lote([texto])[0]


def nota_confianza(texto: str) -> str:
    """Detalle que acompaña la advertencia en la app (depende de la versión activa)."""
    return obtener_modelo().nota_confianza(texto)
