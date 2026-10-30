import json
import os
import random
import subprocess
import sys
import time

from src.modelo import RAIZ, obtener_modelo, predecir, predecir_lote

INICIOS = ["El curso", "La profesora", "El docente", "Las clases", "El material", "La evaluación"]
CENTROS = ["fue muy dinámico", "no tuvo claridad", "estuvo bien en general", "fue bastante desafiante",
           "me ha gustado mucho", "no respondió las preguntas", "fue satisfactorio", "tuvo algunas fallas"]
FINALES = ["y aprendí mucho.", "pero faltó tiempo.", "aunque algunas cosas pudieron mejorar.", "y lo recomiendo.", ""]

ARRANQUE = """
import json, time
t0 = time.perf_counter()
from src.modelo import obtener_modelo, predecir
t1 = time.perf_counter()
obtener_modelo()
t2 = time.perf_counter()
predecir("El curso fue muy dinámico y aprendí mucho")
t3 = time.perf_counter()
print(json.dumps({"importar": t1 - t0, "cargar": t2 - t1, "primera": t3 - t2}))
"""


def _comentarios(n=100, semilla=0):
    rnd = random.Random(semilla)
    return [f"{rnd.choice(INICIOS)} {rnd.choice(CENTROS)} {rnd.choice(FINALES)}".strip() for _ in range(n)]


def test_tiempo_promedio_inferencia(version, capsys):
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
        print(f"\n[rendimiento {version}] 100 comentarios uno a uno: {individual_ms:.2f} ms promedio por comentario")
        print(f"[rendimiento {version}] 100 comentarios en un lote: {lote_ms:.1f} ms en total "
              f"({lote_ms / len(textos):.2f} ms por comentario)")
    assert individual_ms < 1000


def test_tiempo_de_arranque(version, capsys):
    """Proceso nuevo: importar el módulo, cargar el modelo (desde la caché local) y la primera predicción."""
    salida = subprocess.run([sys.executable, "-c", ARRANQUE], cwd=RAIZ, capture_output=True, text=True,
                            env={**os.environ, "MODELO_VERSION": version}, check=True)
    t = json.loads(salida.stdout.strip().splitlines()[-1])
    with capsys.disabled():
        print(f"\n[arranque {version}] importar {t['importar']:.2f} s · cargar modelo {t['cargar']:.2f} s · "
              f"primera predicción {t['primera'] * 1000:.0f} ms · total {sum(t.values()):.2f} s")
    assert t["cargar"] < 300
