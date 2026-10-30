"""API REST del clasificador de polaridad.

Ejecutar desde la raíz del repositorio:  uvicorn api.main:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.modelo import UMBRAL_CONFIANZA, obtener_modelo, predecir, predecir_lote

MAX_CARACTERES = 5000
MAX_TEXTOS = 1000


def _validar_texto(texto: str) -> str:
    if not texto.strip():
        raise ValueError("El texto no puede estar vacío ni contener solo espacios.")
    if len(texto) > MAX_CARACTERES:
        raise ValueError(f"El texto tiene {len(texto)} caracteres; el máximo permitido es {MAX_CARACTERES}.")
    return texto


class EntradaTexto(BaseModel):
    texto: str = Field(..., description="Comentario del estudiante", examples=["El curso fue muy dinámico y aprendí mucho"])

    @field_validator("texto")
    @classmethod
    def validar(cls, v: str) -> str:
        return _validar_texto(v)


class EntradaLote(BaseModel):
    textos: list[str] = Field(..., description=f"Entre 1 y {MAX_TEXTOS} comentarios")

    @field_validator("textos")
    @classmethod
    def validar(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("La lista de textos no puede estar vacía.")
        if len(v) > MAX_TEXTOS:
            raise ValueError(f"Se recibieron {len(v)} textos; el máximo por solicitud es {MAX_TEXTOS}.")
        for i, t in enumerate(v):
            try:
                _validar_texto(t)
            except ValueError as e:
                raise ValueError(f"Texto en la posición {i}: {e}") from None
        return v


class Probabilidades(BaseModel):
    negativo: float
    neutral: float
    positivo: float


class Prediccion(BaseModel):
    clase: str
    probabilidades: Probabilidades
    version_modelo: str
    confianza_baja: bool = Field(..., description="True si la probabilidad máxima es menor que el umbral "
                                                  "o el modelo reconoce muy pocas palabras del texto")
    advertencia: str | None = Field(None, description="Explicación en español cuando confianza_baja es True")


class PrediccionLote(BaseModel):
    predicciones: list[Prediccion]


@asynccontextmanager
async def lifespan(app: FastAPI):
    obtener_modelo()  # carga única al arrancar; si el hash falla, la API no inicia
    yield


app = FastAPI(
    title="Clasificador de polaridad de comentarios estudiantiles",
    description="Sistemas Inteligentes · Emily Bonilla y Santiago Grande",
    version="1.0.0",
    lifespan=lifespan,
)

_MENSAJES = {
    "missing": "Falta este campo obligatorio.",
    "string_type": "Debe ser una cadena de texto.",
    "list_type": "Debe ser una lista de textos.",
    "json_invalid": "El cuerpo no es un JSON válido.",
    "model_attributes_type": "El cuerpo debe ser un objeto JSON.",
}


@app.exception_handler(RequestValidationError)
async def error_validacion(request: Request, exc: RequestValidationError):
    errores = []
    for e in exc.errors():
        if e["type"] == "value_error":
            mensaje = str(e["ctx"]["error"])
        else:
            mensaje = _MENSAJES.get(e["type"], "Valor no válido.")
        campo = ".".join(str(p) for p in e["loc"] if p != "body") or "cuerpo"
        errores.append({"campo": campo, "mensaje": mensaje})
    return JSONResponse(status_code=422, content={"detail": errores})


@app.get("/health")
def health():
    m = obtener_modelo()
    return {"estado": "ok", "modelo_cargado": True, "version_modelo": m.version,
            "umbral_confianza": UMBRAL_CONFIANZA}


@app.post("/predict", response_model=Prediccion)
def predict(entrada: EntradaTexto):
    return predecir(entrada.texto)


@app.post("/predict_batch", response_model=PrediccionLote)
def predict_batch(entrada: EntradaLote):
    return {"predicciones": predecir_lote(entrada.textos)}
