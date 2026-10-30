import pytest
from fastapi.testclient import TestClient

from api.main import MAX_CARACTERES, MAX_TEXTOS, app


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:  # ejecuta el lifespan (carga del modelo)
        yield c


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_predict_ok(cliente):
    r = cliente.post("/predict", json={"texto": "Estoy muy contento, el curso fue muy gratificante y excelente"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["clase"] == "positivo"
    assert set(cuerpo["probabilidades"]) == {"negativo", "neutral", "positivo"}
    assert abs(sum(cuerpo["probabilidades"].values()) - 1) < 1e-6


def test_predict_batch_ok(cliente):
    r = cliente.post("/predict_batch", json={"textos": ["Excelente curso", "No me gustó", "😀"]})
    assert r.status_code == 200
    assert len(r.json()["predicciones"]) == 3


def test_predict_batch_en_el_limite(cliente):
    r = cliente.post("/predict_batch", json={"textos": ["buen curso"] * MAX_TEXTOS})
    assert r.status_code == 200


@pytest.mark.parametrize("cuerpo, fragmento", [
    ({"texto": ""}, "vacío"),
    ({"texto": "   \n\t "}, "vacío"),
    ({"texto": "a" * (MAX_CARACTERES + 1)}, "máximo"),
    ({}, "obligatorio"),
    ({"texto": 123}, "cadena de texto"),
])
def test_predict_422(cliente, cuerpo, fragmento):
    r = cliente.post("/predict", json=cuerpo)
    assert r.status_code == 422
    assert fragmento in r.json()["detail"][0]["mensaje"]


@pytest.mark.parametrize("cuerpo, fragmento", [
    ({"textos": []}, "vacía"),
    ({"textos": ["ok"] * (MAX_TEXTOS + 1)}, "máximo"),
    ({"textos": ["bien", "  "]}, "posición 1"),
    ({"textos": "no es lista"}, "lista"),
])
def test_predict_batch_422(cliente, cuerpo, fragmento):
    r = cliente.post("/predict_batch", json=cuerpo)
    assert r.status_code == 422
    assert fragmento in r.json()["detail"][0]["mensaje"]


def test_texto_en_el_limite_de_caracteres(cliente):
    assert cliente.post("/predict", json={"texto": "a" * MAX_CARACTERES}).status_code == 200
