"""Tabla comparativa v1 contra v2. Solo falla ante errores graves de polaridad en v2."""
import pytest

from src.modelo import obtener_modelo, predecir

FRASES = [
    "no me gustó para nada el curso",
    "odié el curso",
    "el profe explica súper bien, me encantó",
    "el curso tiene tres parciales",
    "las clases son un desorden y nunca responde correos",
]


def _predicciones(version):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODELO_VERSION", version)
        mp.delenv("MODELO_PATH", raising=False)
        obtener_modelo()
        return [predecir(f) for f in FRASES]


def _celda(r):
    marca = " ⚠" if r["confianza_baja"] else ""
    return f"{r['clase']} {max(r['probabilidades'].values()) * 100:4.1f} %{marca}"


def test_tabla_comparativa_v1_v2(capsys):
    v1, v2 = _predicciones("v1_tfidf"), _predicciones("v2_embeddings")

    ancho = max(map(len, FRASES))
    filas = [f"| {'frase':<{ancho}} | {'v1_tfidf':<17} | {'v2_embeddings':<17} |",
             f"|{'-' * (ancho + 2)}|{'-' * 19}|{'-' * 19}|"]
    filas += [f"| {f:<{ancho}} | {_celda(a):<17} | {_celda(b):<17} |" for f, a, b in zip(FRASES, v1, v2)]
    with capsys.disabled():
        print("\n\nComparativa (clase y probabilidad máxima; ⚠ = confianza baja)")
        print("\n".join(filas))

    por_frase = dict(zip(FRASES, v2))
    assert por_frase["no me gustó para nada el curso"]["clase"] != "positivo"
    assert por_frase["el profe explica súper bien, me encantó"]["clase"] != "negativo"
