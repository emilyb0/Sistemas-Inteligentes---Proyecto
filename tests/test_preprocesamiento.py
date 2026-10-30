from src.preprocesamiento import anonimizar, normalizar, preparar


def test_anonimiza_correo_telefono_y_nombre():
    t = anonimizar("Escríbele al profesor Ramírez a juan@uni.edu o al 300 123 4567")
    assert "[CORREO]" in t and "[TELEFONO]" in t and "profesor [NOMBRE]" in t
    assert "juan@uni.edu" not in t and "Ramírez" not in t


def test_normaliza_minusculas_y_espacios():
    assert normalizar("  Muy   BUENO\n\tcurso ") == "muy bueno curso"


def test_preparar_aplica_ambos_pasos():
    assert preparar("La DOCENTE Pérez   explica bien") == "la docente [nombre] explica bien"
