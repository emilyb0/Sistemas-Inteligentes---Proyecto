"""Comportamiento de predicción; cada prueba corre con v1_tfidf y con v2_embeddings."""
import math

import pytest

from src import modelo
from src.modelo import (AVISO_POCAS_PALABRAS, AVISO_PROBABILIDAD_BAJA, AVISO_TEXTO_CORTO, CLASES, ModeloBase,
                        nota_confianza, obtener_modelo, predecir, predecir_lote)

pytestmark = pytest.mark.usefixtures("version")

POSITIVO = "El curso fue muy dinámico y excelente, estoy muy satisfecho, he aprendido mucho"
NEGATIVO = "No me gustó el curso, las explicaciones no tuvieron claridad y no hubo tiempo para las preguntas"


def solo(version, esperada):
    if version != esperada:
        pytest.skip(f"solo aplica a {esperada}")


def test_modelo_carga(version):
    m = obtener_modelo()
    assert isinstance(m, ModeloBase)
    assert m.version_id == version
    assert m.version.startswith(f"{version}-")
    assert tuple(m.info["clases"]) == CLASES


def test_modelo_se_carga_una_sola_vez():
    assert obtener_modelo() is obtener_modelo()


def test_estructura_de_salida():
    r = predecir(POSITIVO)
    assert set(r) == {"clase", "probabilidades", "version_modelo", "confianza_baja", "advertencia"}
    assert set(r["probabilidades"]) == set(CLASES)
    assert r["clase"] in CLASES
    assert isinstance(r["confianza_baja"], bool)
    assert r["version_modelo"] == obtener_modelo().version


def test_comentario_positivo():
    assert predecir(POSITIVO)["clase"] == "positivo"


def test_comentario_negativo():
    assert predecir(NEGATIVO)["clase"] == "negativo"


def test_comentario_negativo_claro():
    r = predecir("El curso fue desorganizado y confuso, los parciales no tenían relación con las clases")
    assert r["clase"] == "negativo"


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
    assert math.isclose(sum(r["probabilidades"].values()), 1.0, abs_tol=1e-6)


def test_probabilidades_suman_uno_y_estan_en_rango():
    for r in predecir_lote([POSITIVO, NEGATIVO, "El curso es los martes"]):
        assert math.isclose(sum(r["probabilidades"].values()), 1.0, abs_tol=1e-6)
        assert all(0.0 <= p <= 1.0 for p in r["probabilidades"].values())


def test_lote_coincide_con_individual():
    # en v2 el relleno del lote cambia los embeddings en el orden de 1e-7
    for lote, individual in zip(predecir_lote([POSITIVO, NEGATIVO]), [predecir(POSITIVO), predecir(NEGATIVO)]):
        assert lote["clase"] == individual["clase"]
        for c in CLASES:
            assert lote["probabilidades"][c] == pytest.approx(individual["probabilidades"][c], abs=1e-5)


# ------------------------------------------------------------ confianza baja
def test_comentario_seguro_sin_advertencia():
    r = predecir(POSITIVO)
    assert r["confianza_baja"] is False and r["advertencia"] is None


def test_umbral_cambia_la_decision(monkeypatch):
    monkeypatch.setattr(modelo, "UMBRAL_CONFIANZA", 1.0)
    r = predecir(POSITIVO)
    assert r["confianza_baja"] is (max(r["probabilidades"].values()) < 1.0)
    if r["confianza_baja"]:
        assert r["advertencia"] == AVISO_PROBABILIDAD_BAJA
    monkeypatch.setattr(modelo, "UMBRAL_CONFIANZA", 0.0)
    assert predecir(POSITIVO)["confianza_baja"] is False


def test_odie_el_curso(version):
    r = predecir("Odié el curso")
    if version == "v1_tfidf":   # "odié" no está en el vocabulario de TF-IDF
        assert r["confianza_baja"] is True
        assert max(r["probabilidades"].values()) < modelo.UMBRAL_CONFIANZA
    else:
        assert r["clase"] == "negativo"


@pytest.mark.parametrize("texto", ["😀🔥👏", "great course", "excelente"])
def test_v1_pocas_palabras_reconocidas(version, texto):
    solo(version, "v1_tfidf")
    r = predecir(texto)
    assert r["confianza_baja"] is True and r["advertencia"] == AVISO_POCAS_PALABRAS


def test_v1_nota_palabras_reconocidas(version):
    solo(version, "v1_tfidf")
    assert obtener_modelo().contar_palabras("odié el curso") == (2, 3)
    assert nota_confianza("Odié el curso") == "Palabras reconocidas: 2 de 3"


@pytest.mark.parametrize("texto", ["excelente", "pésimo curso", "😀🔥", ""])
def test_v2_texto_corto(version, texto):
    solo(version, "v2_embeddings")
    r = predecir(texto)
    assert r["confianza_baja"] is True and r["advertencia"] == AVISO_TEXTO_CORTO


def test_v2_tres_palabras_no_es_corto(version):
    solo(version, "v2_embeddings")
    assert predecir("odié el curso")["advertencia"] != AVISO_TEXTO_CORTO


def test_v2_nota_modelo(version):
    solo(version, "v2_embeddings")
    assert nota_confianza("cualquier texto") == "Modelo: v2 · embeddings"
