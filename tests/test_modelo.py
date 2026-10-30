import math
import os
import subprocess
import sys

import pytest
from sklearn.pipeline import Pipeline

from src import modelo
from src.modelo import (AVISO_POCAS_PALABRAS, AVISO_PROBABILIDAD_BAJA, CLASES, cargar_modelo,
                        contar_palabras, leer_umbral, obtener_modelo, predecir, predecir_lote)

POSITIVO = "El curso fue muy dinámico y excelente, estoy muy satisfecho, he aprendido mucho"
NEGATIVO = "No me gustó el curso, las explicaciones no tuvieron claridad y no hubo tiempo para las preguntas"


def test_modelo_carga():
    m = obtener_modelo()
    assert isinstance(m.pipeline, Pipeline)
    assert tuple(m.pipeline.classes_) == CLASES
    assert m.version.startswith("modelo-")


def test_modelo_se_carga_una_sola_vez():
    assert obtener_modelo() is obtener_modelo()


def test_estructura_de_salida():
    r = predecir(POSITIVO)
    assert set(r) == {"clase", "probabilidades", "version_modelo", "confianza_baja", "advertencia"}
    assert isinstance(r["confianza_baja"], bool)
    assert set(r["probabilidades"]) == set(CLASES)
    assert r["clase"] in CLASES
    assert r["version_modelo"] == obtener_modelo().version


def test_comentario_positivo():
    assert predecir(POSITIVO)["clase"] == "positivo"


def test_comentario_negativo():
    assert predecir(NEGATIVO)["clase"] == "negativo"


@pytest.mark.parametrize("texto", [
    "",
    "     ",
    "😀🔥👏🙌",
    "excelente " * 2000,
    "The course was great and I learned a lot",
    None,
])
def test_entradas_extremas_no_rompen(texto):
    r = predecir(texto)
    assert r["clase"] in CLASES
    assert math.isclose(sum(r["probabilidades"].values()), 1.0, abs_tol=1e-9)


def test_probabilidades_suman_uno_y_estan_en_rango():
    for r in predecir_lote([POSITIVO, NEGATIVO, "El curso es los martes"]):
        assert math.isclose(sum(r["probabilidades"].values()), 1.0, abs_tol=1e-9)
        assert all(0.0 <= p <= 1.0 for p in r["probabilidades"].values())


def test_lote_coincide_con_individual():
    lote = predecir_lote([POSITIVO, NEGATIVO])
    assert lote == [predecir(POSITIVO), predecir(NEGATIVO)]


def test_rechaza_hash_incorrecto(tmp_path):
    falso = tmp_path / "modelo.joblib"
    falso.write_bytes(b"no es un modelo")
    (tmp_path / "modelo.sha256").write_text("0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        cargar_modelo(falso, dir_confianza=tmp_path)


def test_rechaza_modelo_sin_hash(tmp_path):
    falso = tmp_path / "modelo.joblib"
    falso.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        cargar_modelo(falso, dir_confianza=tmp_path)


def test_rechaza_ruta_fuera_de_modelos(tmp_path):
    falso = tmp_path / "modelo.joblib"
    falso.write_bytes(b"x")
    with pytest.raises(PermissionError):
        cargar_modelo(falso)


def test_modelo_path_desde_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("MODELO_PATH", "modelos/otro.joblib")
    assert modelo.ruta_configurada() == (modelo.RAIZ / "modelos" / "otro.joblib").resolve()
    monkeypatch.delenv("MODELO_PATH")
    assert modelo.ruta_configurada() == (modelo.RAIZ / "modelos" / "modelo.joblib").resolve()


# ------------------------------------------------------------ confianza baja
def test_odie_el_curso_es_confianza_baja():
    r = predecir("Odié el curso")
    assert r["confianza_baja"] is True
    assert r["advertencia"] in (AVISO_POCAS_PALABRAS, AVISO_PROBABILIDAD_BAJA)
    assert max(r["probabilidades"].values()) < modelo.UMBRAL_CONFIANZA


def test_comentario_negativo_claro():
    r = predecir("El curso fue desorganizado y confuso, los parciales no tenían relación con las clases")
    assert r["clase"] == "negativo"


def test_comentario_seguro_sin_advertencia():
    r = predecir(POSITIVO)
    assert r["confianza_baja"] is False and r["advertencia"] is None


@pytest.mark.parametrize("texto", ["😀🔥👏", "great course", "excelente"])
def test_pocas_palabras_reconocidas(texto):
    r = predecir(texto)
    assert r["confianza_baja"] is True
    assert r["advertencia"] == AVISO_POCAS_PALABRAS


def test_contar_palabras():
    assert contar_palabras("Odié el curso") == (2, 3)
    assert contar_palabras("😀") == (0, 0)


def test_umbral_por_defecto(monkeypatch):
    monkeypatch.delenv("UMBRAL_CONFIANZA", raising=False)
    assert leer_umbral() == 0.50


@pytest.mark.parametrize("valor, esperado", [("0.7", 0.7), ("0,35", 0.35), ("1", 1.0)])
def test_umbral_desde_variable_de_entorno(monkeypatch, valor, esperado):
    monkeypatch.setenv("UMBRAL_CONFIANZA", valor)
    assert leer_umbral() == esperado


@pytest.mark.parametrize("valor", ["alto", "1.5", "-0.1"])
def test_umbral_invalido(monkeypatch, valor):
    monkeypatch.setenv("UMBRAL_CONFIANZA", valor)
    with pytest.raises(ValueError, match="UMBRAL_CONFIANZA"):
        leer_umbral()


def test_constante_umbral_se_lee_al_importar():
    entorno = {**os.environ, "UMBRAL_CONFIANZA": "0.8"}
    salida = subprocess.run(
        [sys.executable, "-c", "from src import modelo; print(modelo.UMBRAL_CONFIANZA)"],
        cwd=modelo.RAIZ, env=entorno, capture_output=True, text=True, check=True,
    )
    assert salida.stdout.strip() == "0.8"


def test_umbral_cambia_la_decision(monkeypatch):
    monkeypatch.setattr(modelo, "UMBRAL_CONFIANZA", 0.99)
    r = predecir(POSITIVO)
    assert r["confianza_baja"] is True and r["advertencia"] == AVISO_PROBABILIDAD_BAJA
    monkeypatch.setattr(modelo, "UMBRAL_CONFIANZA", 0.0)
    assert predecir(POSITIVO)["confianza_baja"] is False
