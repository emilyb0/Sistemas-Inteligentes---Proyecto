import math

import pytest
from sklearn.pipeline import Pipeline

from src import modelo
from src.modelo import CLASES, cargar_modelo, obtener_modelo, predecir, predecir_lote

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
    assert set(r) == {"clase", "probabilidades", "version_modelo"}
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
