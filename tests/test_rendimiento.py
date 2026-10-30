import random
import time

from src.modelo import obtener_modelo, predecir, predecir_lote

INICIOS = ["El curso", "La profesora", "El docente", "Las clases", "El material", "La evaluación"]
CENTROS = ["fue muy dinámico", "no tuvo claridad", "estuvo bien en general", "fue bastante desafiante",
           "me ha gustado mucho", "no respondió las preguntas", "fue satisfactorio", "tuvo algunas fallas"]
FINALES = ["y aprendí mucho.", "pero faltó tiempo.", "aunque algunas cosas pudieron mejorar.", "y lo recomiendo.", ""]


def _comentarios(n=100, semilla=0):
    rnd = random.Random(semilla)
    return [f"{rnd.choice(INICIOS)} {rnd.choice(CENTROS)} {rnd.choice(FINALES)}".strip() for _ in range(n)]


def test_tiempo_promedio_inferencia(capsys):
    obtener_modelo()  # la carga no cuenta en la medición
    textos = _comentarios(100)
    predecir(textos[0])  # calentamiento

    t0 = time.perf_counter()
    for t in textos:
        predecir(t)
    individual_ms = (time.perf_counter() - t0) / len(textos) * 1000

    t0 = time.perf_counter()
    predecir_lote(textos)
    lote_ms = (time.perf_counter() - t0) * 1000

    with capsys.disabled():
        print(f"\n[rendimiento] 100 comentarios uno a uno: {individual_ms:.3f} ms promedio por comentario")
        print(f"[rendimiento] 100 comentarios en un lote: {lote_ms:.2f} ms en total "
              f"({lote_ms / len(textos):.3f} ms por comentario)")
    assert individual_ms < 200
