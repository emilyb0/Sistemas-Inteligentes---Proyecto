# Polaridad de comentarios estudiantiles

Proyecto de la materia **Sistemas Inteligentes**, por **Emily Bonilla** y **Santiago Grande**.

Este MVP clasifica comentarios de estudiantes sobre sus cursos en **negativo**, **neutral** o **positivo**. Usa un
`Pipeline` de scikit-learn (`TfidfVectorizer` + `LogisticRegression`) que entrenamos en Colab. El proyecto **no
reentrena** el modelo: lo carga una sola vez al iniciar y lo expone de dos formas:

- una **API REST** con FastAPI (`api/main.py`);
- una **app web** con Streamlit (`app/streamlit_app.py`), que analiza un comentario o un archivo CSV completo.

Ambas usan el mismo módulo `src/modelo.py` y el mismo preprocesamiento (`src/preprocesamiento.py`), que es
idéntico al que se aplicó en el entrenamiento.

| Métrica (conjunto de prueba) | Valor |
|---|---|
| F1 macro | 0,610 |
| Ejemplos de entrenamiento / prueba | 490 / 123 |
| Versiones de entrenamiento | Python 3.13 · scikit-learn 1.6.1 |

> **Limitación conocida:** el vocabulario sale de solo 490 comentarios. Las palabras que el modelo nunca vio
> (por ejemplo, "pésimo") o el texto en otro idioma no aportan información. En esos casos el modelo devuelve
> casi la proporción base de las clases, y la app lo advierte.

## Estructura

```
modelos/
  modelo.joblib        Pipeline entrenado (el único artefacto que se carga)
  modelo.sha256        hash SHA-256 esperado del .joblib
  modelo_info.json     métricas, clases y versiones de entrenamiento
src/
  preprocesamiento.py  anonimizar + normalizar (igual que en el entrenamiento)
  modelo.py            carga segura, caché única y funciones predecir / predecir_lote
api/main.py            FastAPI: /health, /predict, /predict_batch
app/
  streamlit_app.py     interfaz "cuaderno de evaluación"
  estilos.css          identidad visual (inyectada en la app)
.streamlit/config.toml tema base, fuentes y paleta
tests/                 pruebas con pytest
```

## Instalación

Se necesita **Python 3.13** (la misma versión menor del entrenamiento).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements-dev.txt     # app + API + pruebas
# o solo lo necesario para ejecutar:  pip install -r requirements.txt
```

`requirements.txt` fija `scikit-learn==1.6.1`. Si se instala otra versión, `src/modelo.py` se niega a cargar el modelo.

## Ejecutar la API

Desde la raíz del repositorio:

```bash
uvicorn api.main:app --reload
```

- Documentación interactiva: <http://localhost:8000/docs>
- `GET /health`: estado y versión del modelo.
- `POST /predict` con `{"texto": "..."}`.
- `POST /predict_batch` con `{"textos": ["...", "..."]}` (entre 1 y 1000 textos).

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"texto": "El curso fue muy dinámico y aprendí mucho"}'
```

```json
{
  "clase": "positivo",
  "probabilidades": {"negativo": 0.16, "neutral": 0.16, "positivo": 0.68},
  "version_modelo": "modelo-2026-10-29-bc27692a",
  "confianza_baja": false,
  "advertencia": null
}
```

`/predict_batch` responde `{"predicciones": [ ... ]}` con un objeto como el anterior por cada texto.

Un texto vacío, con solo espacios o de más de 5000 caracteres (o una lista vacía o de más de 1000 textos) recibe
**422** con un mensaje en español:

```json
{"detail": [{"campo": "texto", "mensaje": "El texto no puede estar vacío ni contener solo espacios."}]}
```

## Confianza baja

TF-IDF solo conoce las palabras que aparecieron en los 490 comentarios de entrenamiento e **ignora todas las
demás**. Con un texto como "Odié el curso", la palabra "odié" no está en el vocabulario y solo quedan "el" y
"curso", que no dicen nada de la polaridad. El modelo reparte entonces la probabilidad casi en partes iguales
(27 % / 37 % / 35 %) y elige una clase casi al azar.

Por eso cada predicción incluye `confianza_baja` (bool) y `advertencia` (texto o `null`). Una predicción se marca
como de confianza baja si se cumple alguna de estas condiciones:

1. el modelo reconoce **menos de 2 palabras** del comentario. En ese caso, la advertencia pide describir con más
   detalle qué gustó o no gustó;
2. la **probabilidad máxima es menor que el umbral** (0,50 por defecto). En ese caso, la advertencia dice que las
   probabilidades de las clases son muy parecidas.

En la app, el sello se ve atenuado con la palabra "INCIERTO" y aparece una nota con las palabras reconocidas.
En el CSV se agrega la columna `confianza_baja`, y el resumen cuenta cuántos comentarios quedaron así.

El umbral se cambia con la variable de entorno **`UMBRAL_CONFIANZA`** (un número entre 0 y 1), que se lee al
iniciar:

```bash
# macOS / Linux
UMBRAL_CONFIANZA=0.6 uvicorn api.main:app
# Windows (PowerShell)
$env:UMBRAL_CONFIANZA = "0.6"; streamlit run app/streamlit_app.py
```

En Streamlit Community Cloud se agrega como secreto: `UMBRAL_CONFIANZA = "0.6"`. Con tres clases, una
probabilidad máxima cercana a 0,33 equivale a no saber nada. Subir el umbral marca más predicciones como dudosas;
`GET /health` muestra el valor que está en uso.

## Ejecutar la app

```bash
streamlit run app/streamlit_app.py
```

La app carga el modelo directamente con `src/modelo.py`, así que **no necesita la API**. Tiene dos pestañas:

- **Un comentario:** escribes el texto y ves el sello con la clase, la probabilidad de cada clase y el texto tal
  como lo leyó el modelo (ya anonimizado y normalizado).
- **Archivo CSV:** subes un CSV, eliges la columna de texto y clasificas todo. Ves el conteo y el porcentaje por
  clase con un gráfico, una vista previa, y puedes descargar el CSV con las columnas `clase`, `prob_negativo`,
  `prob_neutral`, `prob_positivo` y `confianza_baja`. Las filas vacías quedan como `sin_texto`.

## Pruebas

```bash
pytest -v
```

Las pruebas verifican lo siguiente:

- el modelo carga una sola vez;
- un comentario positivo y uno negativo claros dan la clase esperada;
- el texto vacío, solo emojis, muy largo o en inglés no rompe nada;
- "Odié el curso" queda marcado con confianza baja y el umbral se lee de `UMBRAL_CONFIANZA`;
- las probabilidades suman 1;
- el hash incorrecto o una ruta fuera de `modelos/` se rechazan;
- la API responde 200 y 422 donde corresponde.

`tests/test_rendimiento.py` mide e imprime el tiempo promedio de inferencia de 100 comentarios.

## Seguridad del modelo

`joblib.load` deserializa con pickle, y **eso puede ejecutar código**. Por eso `src/modelo.py` hace lo siguiente:

1. Solo acepta rutas dentro de la carpeta `modelos/` del repositorio.
2. Calcula el SHA-256 del archivo y lo compara con `<nombre>.sha256` **antes** de deserializar.
3. Comprueba que la versión de scikit-learn instalada sea la de `<nombre>_info.json`.
4. Verifica que lo cargado sea un `Pipeline` con las clases `negativo`, `neutral` y `positivo`.

Si algo falla, la API no arranca y la app muestra el error. En ningún caso se usa un modelo no verificado.

## Cambiar o revertir la versión del modelo

La ruta del modelo se lee de la variable de entorno **`MODELO_PATH`**, cuyo valor por defecto es
`./modelos/modelo.joblib`. Una ruta relativa se interpreta desde la raíz del repositorio. Cada versión vive en
`modelos/` con tres archivos que comparten el mismo nombre base:

```
modelos/modelo_v2.joblib
modelos/modelo_v2.sha256       # "<hash>  modelo_v2.joblib"
modelos/modelo_v2_info.json    # métricas y versiones (opcional, pero recomendado)
```

1. Copia el nuevo `.joblib` y su `_info.json` a `modelos/`.
2. Genera el hash:
   ```bash
   # macOS / Linux
   sha256sum modelos/modelo_v2.joblib | sed 's#modelos/##' > modelos/modelo_v2.sha256
   # Windows (PowerShell)
   "$((Get-FileHash modelos\modelo_v2.joblib -Algorithm SHA256).Hash.ToLower())  modelo_v2.joblib" | Out-File -Encoding ascii modelos\modelo_v2.sha256
   ```
3. Apunta la variable a la nueva versión y reinicia:
   ```bash
   # macOS / Linux
   MODELO_PATH=modelos/modelo_v2.joblib uvicorn api.main:app
   # Windows (PowerShell)
   $env:MODELO_PATH = "modelos/modelo_v2.joblib"; uvicorn api.main:app
   ```
4. **Para revertir**, vuelve a apuntar `MODELO_PATH` a la versión anterior (o elimina la variable para usar
   `modelo.joblib`) y reinicia. Las versiones anteriores siguen en `modelos/` y en el historial de git.

El campo `version_modelo` de cada respuesta (por ejemplo `modelo-2026-10-29-bc27692a`, es decir
nombre-fecha-hash) indica qué versión respondió.

## Desplegar la app en Streamlit Community Cloud

1. Sube el repositorio a GitHub (si es privado, autoriza a Streamlit para leer repositorios privados).
2. Entra a <https://share.streamlit.io>, inicia sesión con GitHub y pulsa **Create app → Deploy a public app from
   GitHub**.
3. Configura:
   - **Repository:** `emilyb0/Sistemas-Inteligentes---Proyecto`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. En **Advanced settings** elige **Python 3.13**. Streamlit Cloud instala `requirements.txt` desde la raíz.
5. (Opcional) Para usar otra versión del modelo, agrega en **Secrets**:
   ```toml
   MODELO_PATH = "modelos/modelo_v2.joblib"
   ```
   Las claves de nivel raíz de los secretos se exponen como variables de entorno.
6. Pulsa **Deploy**. Streamlit Cloud toma `.streamlit/config.toml` (tema y fuentes) y `app/estilos.css`
   automáticamente.

## Datos

El repositorio **no incluye comentarios de estudiantes**: `.gitignore` excluye `*.csv`, `datos/` y `data/`. Solo se
versiona el modelo entrenado. El preprocesamiento anonimiza correos, teléfonos y nombres de docentes antes de
predecir.
