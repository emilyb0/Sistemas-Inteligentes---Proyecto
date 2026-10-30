"""Carga segura, selección de versión y umbral (no dependen de la versión activa)."""
import os
import shutil
import subprocess
import sys

import pytest

from src import modelo
from src.modelo import ARCHIVOS, DIR_MODELOS, VERSIONES, cargar_modelo, leer_umbral


@pytest.mark.parametrize("version", VERSIONES)
def test_archivos_de_cada_version_y_hash_valido(version):
    carpeta = DIR_MODELOS / version
    for nombre in (ARCHIVOS[version], "modelo.sha256", "modelo_info.json"):
        assert (carpeta / nombre).is_file(), nombre
    esperado, archivo = (carpeta / "modelo.sha256").read_text(encoding="utf-8").split()
    assert archivo == ARCHIVOS[version]
    assert modelo.calcular_sha256(carpeta / archivo) == esperado


def test_rechaza_hash_incorrecto(tmp_path):
    falso = tmp_path / "modelo.joblib"
    falso.write_bytes(b"no es un modelo")
    (tmp_path / "modelo.sha256").write_text("0" * 64 + "  modelo.joblib")
    with pytest.raises(ValueError, match="SHA-256"):
        cargar_modelo(falso, dir_confianza=tmp_path)


def test_rechaza_hash_de_otro_archivo(tmp_path):
    falso = tmp_path / "clasificador.joblib"
    falso.write_bytes(b"x")
    (tmp_path / "modelo.sha256").write_text(modelo.calcular_sha256(falso) + "  modelo.joblib")
    with pytest.raises(ValueError, match="corresponde a"):
        cargar_modelo(falso, "v2_embeddings", dir_confianza=tmp_path)


def test_rechaza_modelo_sin_hash(tmp_path):
    falso = tmp_path / "modelo.joblib"
    falso.write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        cargar_modelo(falso, dir_confianza=tmp_path)


def test_rechaza_ruta_fuera_de_modelos(tmp_path):
    for nombre in ("modelo.joblib", "modelo.sha256", "modelo_info.json"):
        shutil.copy(DIR_MODELOS / "v1_tfidf" / nombre, tmp_path / nombre)
    with pytest.raises(PermissionError):
        cargar_modelo(tmp_path / "modelo.joblib")


def test_version_por_defecto_es_v2(monkeypatch):
    monkeypatch.delenv("MODELO_VERSION", raising=False)
    monkeypatch.delenv("MODELO_PATH", raising=False)
    assert modelo.version_configurada() == "v2_embeddings"
    assert modelo.ruta_configurada() == (DIR_MODELOS / "v2_embeddings" / "clasificador.joblib").resolve()


def test_modelo_version_desde_variable_de_entorno(monkeypatch):
    monkeypatch.delenv("MODELO_PATH", raising=False)
    monkeypatch.setenv("MODELO_VERSION", "v1_tfidf")
    assert modelo.ruta_configurada() == (DIR_MODELOS / "v1_tfidf" / "modelo.joblib").resolve()


def test_modelo_version_invalida(monkeypatch):
    monkeypatch.setenv("MODELO_VERSION", "v3_magia")
    with pytest.raises(ValueError, match="MODELO_VERSION"):
        modelo.version_configurada()


def test_modelo_path_sobrescribe_la_ruta(monkeypatch):
    monkeypatch.setenv("MODELO_VERSION", "v1_tfidf")
    monkeypatch.setenv("MODELO_PATH", "modelos/v1_tfidf/otro.joblib")
    assert modelo.ruta_configurada() == (DIR_MODELOS / "v1_tfidf" / "otro.joblib").resolve()


# ------------------------------------------------------------ umbral
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
