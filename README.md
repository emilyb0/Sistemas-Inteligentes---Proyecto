# Polaridad de comentarios estudiantiles

Proyecto de la materia **Sistemas Inteligentes**, por **Emily Bonilla**, **Santiago Grande** y **Daniel Posada**

Este proyecto clasifica los comentarios que los estudiantes escriben sobre sus cursos en **negativo**, **neutral**
o **positivo**. El clasificador se ofrece de dos formas:

- **Una app web** (Streamlit): permite analizar un comentario o un archivo CSV completo. Muestra la clase, la
  probabilidad de cada clase y un resumen con gráfico, y permite descargar los resultados.
- **Una API REST** (FastAPI): ofrece los endpoints `/predict` y `/predict_batch` para integrar el clasificador en
  otros sistemas.

Antes de clasificar, el texto se anonimiza: se reemplazan correos, teléfonos y nombres de docentes. Cuando el
modelo no está seguro de su predicción, lo indica con una advertencia.

## Versiones del modelo

| Versión | Modelo | Macro-F1 (prueba) |
|---|---|---|
| `v1_tfidf` | TF-IDF + regresión logística | 0,610 |
| `v2_embeddings` (por defecto) | Embeddings de `paraphrase-multilingual-MiniLM-L12-v2` + regresión logística | 0,745 |

**v1** solo conoce las palabras que aparecieron en el entrenamiento. Por eso falla con negaciones y con palabras
nuevas: "no me gustó para nada el curso" y "odié el curso" salen como neutral.

**v2** representa cada comentario por su significado (embeddings de 384 dimensiones), así que entiende sinónimos,
negaciones y palabras que no vio al entrenar. Además mejora el macro-F1 de 0,610 a 0,745. El modelo de embeddings
se descarga de Hugging Face la primera vez que se usa; no está en el repositorio.

Cada versión vive en su carpeta dentro de `modelos/`, con su `modelo.sha256`. El hash se verifica antes de cargar el
modelo. La versión activa se elige con la variable de entorno `MODELO_VERSION`. Para **revertir a v1**:

```bash
MODELO_VERSION=v1_tfidf streamlit run app/streamlit_app.py
```

En Streamlit Community Cloud se agrega `MODELO_VERSION = "v1_tfidf"` en **Secrets** y se reinicia la app.
