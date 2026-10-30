"""Preprocesamiento compartido por la API y la app.

Debe ser EXACTAMENTE el mismo que se aplicó al texto con el que se entrenó el
modelo; si cambia, las predicciones dejan de ser válidas.
"""
import re, unicodedata


def anonimizar(t):
    t = re.sub(r'\S+@\S+', '[CORREO]', t)
    t = re.sub(r'\+?\d[\d\s-]{6,}\d', '[TELEFONO]', t)
    t = re.sub(r'\b(profe(sor|sora)?|docente)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+', r'\1 [NOMBRE]', t, flags=re.I)
    return t


def normalizar(t):
    t = unicodedata.normalize('NFC', str(t)).lower()
    return re.sub(r'\s+', ' ', t).strip()


def preparar(t):
    return normalizar(anonimizar(t))
