"""Transformadores sklearn para v2.

El Pipeline joblib contiene los pasos de limpieza + embeddings + clasificador.
Los pesos del encoder de Hugging Face se descargan/cargan por separado al
iniciar la inferencia; se fija la revisión del repositorio para reproducibilidad.
Por ello el artefacto es UN Pipeline, pero NO un archivo totalmente autónomo.
"""
from __future__ import annotations

from functools import lru_cache
import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin

ENCODER_PREDETERMINADO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def limpiar_corregido(texto: str) -> str:
    """Anonimización aproximada que no elimina verbos/negaciones en minúscula.

    Atención: las reglas por expresión regular no garantizan anonimización total.
    """
    t = str(texto)
    t = re.sub(r"\S+@\S+", "[CORREO]", t)
    t = re.sub(r"\+?\d[\d\s-]{6,}\d", "[TELEFONO]", t)
    # Solo la palabra 'profesor/profe/docente' ignora mayúsculas;
    # el supuesto nombre SÍ debe empezar por mayúscula.
    t = re.sub(
        r"\b(?P<titulo>(?i:profesor|profesora|profe|docente))\s+"
        r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b",
        r"\g<titulo> [NOMBRE]",
        t,
    )
    t = unicodedata.normalize("NFC", t).lower()
    return re.sub(r"\s+", " ", t).strip()


class PrepararComentarios(BaseEstimator, TransformerMixin):
    """'historico' reproduce exactamente el preprocesamiento del notebook.

    'corregido' evita perder verbos y negaciones al anonimizar nombres.
    Cambiar entre ambos modifica las features: exige reentrenar y reevaluar.
    """

    def __init__(self, modo: str = "corregido"):
        self.modo = modo

    def fit(self, X, y=None):
        if self.modo not in ("historico", "corregido"):
            raise ValueError("modo debe ser 'historico' o 'corregido'")
        return self

    def transform(self, X):
        if self.modo == "historico":
            from src.preprocesamiento import preparar  # conservar v1 intacta
            return [preparar(x) for x in X]
        if self.modo == "corregido":
            return [limpiar_corregido(x) for x in X]
        raise ValueError(f"Modo de limpieza no válido: {self.modo}")


@lru_cache(maxsize=4)
def _modelo_encoder(nombre: str, revision: str | None):
    """Se carga una sola vez por proceso y NO se incluye en el joblib."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nombre, revision=revision, device="cpu")


class VectorizadorEmbeddings(BaseEstimator, TransformerMixin):
    """Adaptador de SentenceTransformer compatible con sklearn Pipeline.

    Solo guarda configuración; los pesos externos NO se serializan con joblib.
    """

    def __init__(
        self,
        nombre_modelo: str = ENCODER_PREDETERMINADO,
        revision: str | None = None,
        batch_size: int = 64,
    ):
        self.nombre_modelo = nombre_modelo
        self.revision = revision
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self  # modelo preentrenado congelado, no se ajusta sobre train

    def transform(self, X):
        textos = list(X)
        if not textos:
            import numpy as np
            return np.empty((0, 384), dtype="float32")
        return _modelo_encoder(self.nombre_modelo, self.revision).encode(
            textos,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
