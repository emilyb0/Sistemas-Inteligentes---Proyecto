# Polaridad de comentarios estudiantiles

Proyecto de la materia **Sistemas Inteligentes**, por **Emily Bonilla** y **Santiago Grande**.

Este proyecto clasifica los comentarios que los estudiantes escriben sobre sus cursos en **negativo**, **neutral**
o **positivo**. Usa un modelo de TF-IDF con regresión logística entrenado previamente y lo ofrece de dos formas:

- **Una app web** (Streamlit): permite analizar un comentario o un archivo CSV completo. Muestra la clase, la
  probabilidad de cada clase y un resumen con gráfico, y permite descargar los resultados.
- **Una API REST** (FastAPI): ofrece los endpoints `/predict` y `/predict_batch` para integrar el clasificador en
  otros sistemas.

Antes de clasificar, el texto se anonimiza: se reemplazan correos, teléfonos y nombres de docentes. Cuando el
modelo no está seguro de su predicción, lo indica con una advertencia.
