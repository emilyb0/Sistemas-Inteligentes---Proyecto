"""Pruebas unitarias nuevas: no requieren descargar el encoder real."""
import hashlib
import json
from pathlib import Path
import sys
import types

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import embeddings_pipeline as ep
from src.embeddings_pipeline import PrepararComentarios, VectorizadorEmbeddings
from src.monitoreo import distribuciones, evaluar_drift, psi_conteos
from src import modelo


class EncoderFalso:
    def encode(self, textos, **kw):
        return np.asarray([
            [len(t.split()), sum(w in t for w in ["bien", "excelente"]),
             sum(w in t for w in ["mal", "pésimo"])] for t in textos
        ], dtype="float32")


@pytest.fixture
def pipeline_falso(monkeypatch):
    monkeypatch.setattr(ep, "_modelo_encoder", lambda nombre, revision: EncoderFalso())
    pipe = Pipeline([
        ("limpieza", PrepararComentarios(modo="corregido")),
        ("embeddings", VectorizadorEmbeddings(revision="abcdef")),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    X = ["excelente curso", "curso excelente bien", "curso bastante bueno",
         "curso terrible", "pésimo curso", "fue mal", "temario semanal", "horario de lunes", "tres temas"]
    y = ["positivo"] * 3 + ["negativo"] * 3 + ["neutral"] * 3
    pipe.fit(X, y)
    return pipe


def test_limpieza_corrige_perdida_de_negacion():
    frase = "El profesor no explicó bien el curso"
    assert "profesor no explicó" in ep.limpiar_corregido(frase)
    assert "[nombre]" in ep.limpiar_corregido("El profesor Ramírez explicó bien")
    assert "profesor fue" in ep.limpiar_corregido("El profesor fue claro")


def test_limpieza_historica_se_mantiene_para_reproducibilidad():
    frase = "El profesor no explicó bien"
    assert "[nombre]" in PrepararComentarios(modo="historico").transform([frase])[0]
    assert "profesor no" in PrepararComentarios(modo="corregido").transform([frase])[0]


def test_pipeline_joblib_se_recarga_sin_guardar_encoder(tmp_path, pipeline_falso):
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(pipeline_falso, ruta)
    pipe2 = joblib.load(ruta)
    X = ["excelente curso", "pésimo curso", "horario de lunes"]
    assert list(pipe2.predict(X)) == list(pipeline_falso.predict(X))
    assert ruta.stat().st_size < 100_000
    assert set(pipe2.named_steps) == {"limpieza", "embeddings", "clf"}


def test_carga_v2_valida_pipeline_y_sha(tmp_path, pipeline_falso, monkeypatch):
    fake_st = types.SimpleNamespace(__version__="99.0")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(pipeline_falso, ruta)
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
    (tmp_path / "modelo.sha256").write_text(f"{sha}  modelo.joblib\n")
    info = {
        "version": "v2_embeddings", "modelo_embeddings": ep.ENCODER_PREDETERMINADO,
        "revision_embeddings": "abcdef", "modo_preprocesamiento": "corregido",
        "C": 1.0, "clases": list(modelo.CLASES), "sha256_modelo": sha,
    }
    (tmp_path / "modelo_info.json").write_text(json.dumps(info))
    m = modelo.cargar_modelo(ruta, "v2_embeddings", dir_confianza=tmp_path)
    assert m.version_id == "v2_embeddings"
    assert m.predecir_proba(["excelente curso"]).shape == (1, 3)
    assert m.preparar_para_aviso(["El profesor no explicó"])[0] == "el profesor no explicó"


def test_metadata_sha_inconsistente_se_rechaza(tmp_path, pipeline_falso):
    ruta = tmp_path / "modelo.joblib"
    joblib.dump(pipeline_falso, ruta)
    sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
    (tmp_path / "modelo.sha256").write_text(f"{sha}  modelo.joblib")
    (tmp_path / "modelo_info.json").write_text(json.dumps({"sha256_modelo": "0"*64}))
    with pytest.raises(ValueError, match="modelo_info.json"):
        modelo.cargar_modelo(ruta, "v2_embeddings", dir_confianza=tmp_path)


def test_psi_identico_cero():
    assert psi_conteos({"a": 30, "b": 70}, {"a": 30, "b": 70}) == pytest.approx(0.0)


def test_drift_detecta_cambio_grande():
    ref = distribuciones(["muy bien curso"] * 70 + ["no cumple"] * 30,
                         ["positivo"] * 70 + ["negativo"] * 30)
    nuevo = distribuciones(["mal"] * 100, ["negativo"] * 100)
    d = evaluar_drift(ref, nuevo)
    assert d["estado"] == "alerta_revisar" and d["psi_clases"] > .2


def test_drift_muestra_insuficiente():
    ref = distribuciones(["comentario"] * 100, ["positivo"] * 100)
    nuevo = distribuciones(["comentario"] * 10, ["positivo"] * 10)
    assert evaluar_drift(ref, nuevo)["estado"] == "muestra_insuficiente"


def test_artefacto_pipeline_v2_soportado():
    assert modelo.ARCHIVO_PIPELINE_V2 == "modelo.joblib"
    assert modelo.ARCHIVOS["v2_embeddings"] == "clasificador.joblib"  # compatibilidad
