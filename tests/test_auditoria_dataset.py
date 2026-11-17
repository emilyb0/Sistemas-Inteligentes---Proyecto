"""La auditoría de datos funciona con un corpus ficticio y no publica textos."""
import hashlib
import json

import pandas as pd
import pytest

from scripts.auditar_dataset_y_drift_historico import auditar, sha256_archivo


def _datos_ejemplo():
    datos = []
    for semestre, cantidad in [("2023-1", 70), ("2023-2", 70), ("2024-1", 70), ("2024-2", 9)]:
        for i in range(cantidad):
            datos.append({
                "archivo": 1, "semestre": semestre, "id": i,
                "id_unico": f"1_{semestre}_{i}",
                "feedback": f"frase sintética {semestre} número {i}",
                "polaridad": ["negativo", "neutral", "positivo"][i % 3],
            })
    return pd.DataFrame(datos)


def test_auditoria_registra_hash_sin_texto(tmp_path):
    ruta = tmp_path / "comentarios_ficticios.csv"
    _datos_ejemplo().to_csv(ruta, index=False, encoding="utf-8-sig")
    res = auditar(ruta, repeticiones=5)
    assert res["dataset"]["sha256_dataset"] == hashlib.sha256(ruta.read_bytes()).hexdigest()
    assert res["dataset"]["n"] == 219
    assert res["dataset"]["duplicados_id_unico"] == 0
    assert res["dataset"]["particion_propuesta"]["n_train"] == 175
    assert res["dataset"]["particion_propuesta"]["n_test"] == 44
    assert res["drift_historico"]["periodos_comparados"]["2024-2"]["diagnostico"]["estado"] == "muestra_insuficiente"
    assert res["drift_historico"]["periodos_comparados"]["2024-1"]["diagnostico"]["estado"] == "sin_alerta"
    assert "frase sintética" not in json.dumps(res, ensure_ascii=False)


def test_datos_duplicados_se_rechazan(tmp_path):
    datos = _datos_ejemplo()
    datos.loc[1, "feedback"] = datos.loc[0, "feedback"]
    ruta = tmp_path / "duplicado.csv"
    datos.to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="duplicados"):
        auditar(ruta, repeticiones=5)


def test_no_se_generan_hashes_ficticios(tmp_path):
    ruta = tmp_path / "prueba.csv"
    ruta.write_bytes(b"abc")
    assert sha256_archivo(ruta) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
