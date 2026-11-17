# Pendientes de entrega: v2 con C=0.3, trazabilidad, rendimiento y drift

Este paquete es una **integración de código** para el repositorio `emilyb0/Sistemas-Inteligentes---Proyecto`.
**No incluye el dataset ni un modelo reentrenado**; no se generaron métricas reales de v2 ni se inventaron valores de latencia o drift.

## Compatibilidad y alcance

- El v2 histórico (`modelos/v2_embeddings/clasificador.joblib`) sigue cargando como antes.
- Si existe `modelos/v2_embeddings/modelo.joblib`, se da prioridad al Pipeline completo (limpieza + embeddings + clasificador); se valida su SHA-256 y su metadato.
- `MODELO_PATH` permite probar un candidato antes de publicarlo. El encoder es externo a joblib y se fija por **revisión de commit** de Hugging Face, registrada en `modelo_info.json`.
- La app muestra, al clasificar un CSV, los PSI de clases predichas y longitud **solo si existe una referencia válida del mismo modelo**.
- La lógica v1 no se reentrena ni se modifica.

## 1. Preparar dataset original (NO incluirlo en GitHub)

Usar el CSV depurado de comentarios etiquetados. El esquema requerido es: `feedback`, `polaridad`, `archivo`, `semestre`, `id`; las polaridades admitidas son `negativo`, `neutral`, `positivo`. El corpus de trabajo previamente documentado contiene 613 comentarios válidos (490 train / 123 test con 80/20 estratificado, semilla 42). **Validar estos conteos contra el CSV efectivo.**

Si se necesita corregir `id_unico` porque IDs de cursos se repiten por archivo/semestre:

```bash
python -m scripts.corregir_ids --csv datos/comentarios_limpio.csv \
  --salida datos/comentarios_limpio_ids_ok.csv
```

Esta copia modifica el CSV y, en consecuencia, tiene un SHA-256 propio que debe registrar el entrenamiento.

## 2. Reentrenar v2 y generar su hash

Desde la raíz del repositorio, con el entorno de `requirements.txt` instalado:

```bash
python -m scripts.reentrenar_v2 \
  --csv datos/comentarios_limpio_ids_ok.csv \
  --modo corregido --c 0.3 \
  --salida modelos/v2_embeddings_candidato
```

El script hace split estratificado 80/20 con semilla 42, CV estratificada de 5 folds **exclusivamente sobre train** para auditar la selección de `C`, entrena finalmente con `C=0.3`, evalúa en test y recarga el artefacto para comprobar predicciones. Registra matriz de confusión, métricas, versiones y hash de los **bytes exactos** del CSV (`sha256_dataset`). Genera:

- `modelo.joblib` — Pipeline completo con tres pasos (`limpieza`, `embeddings`, `clf`). Los pesos del encoder se cargan por separado.
- `modelo.sha256` — integridad del joblib.
- `modelo_info.json` — incluye `C`, `sha256_dataset`, `sha256_modelo`, `revision_embeddings`, métricas y versiones.
- `drift_referencia.json` — referencia agregada obtenida en train, compatible únicamente con ese SHA-256 del modelo.

El modo `corregido` soluciona el bug de anonimización histórica que interpretaba negaciones como nombres propios. **No reutilizar las métricas históricas** (`macro-F1=0.7451` para C=1; `0.714` del artículo con otra metodología). Reportar lo que produzca este entrenamiento; el conjunto test ya fue observado en experimentos previos, por lo que no es un holdout absolutamente virgen.

Antes de promover la carpeta candidata, comprobar `modelo_info.json`, los hashes y las pruebas. Para evaluar el candidato sin tocar la carpeta vigente:

```bash
MODELO_VERSION=v2_embeddings \
MODELO_PATH=modelos/v2_embeddings_candidato/modelo.joblib \
python -m pytest -q tests/test_pipeline_v2_nuevo.py
```

Luego actualizar **código + modelo + hashes + metadatos + referencia** en el mismo commit de despliegue. Mantener copia de la versión anterior para revertir.

## 3. Medir latencias reales

```bash
MODELO_VERSION=v2_embeddings \
MODELO_PATH=modelos/v2_embeddings_candidato/modelo.joblib \
python -m scripts.medir_rendimiento --n 100 \
  --salida resultados/rendimiento_v2.json
```

Registra plataforma, versión del modelo, tiempo de carga, tiempo individual promedio/mediana/p95 y lote de 100. El calentamiento se ejecuta antes de medir la inferencia; la carga inicial es una métrica aparte. Los tiempos dependen del equipo y la caché de Hugging Face. Incluir el archivo JSON real en la entrega (sin datos personales).

## 4. Monitorear drift sobre datos nuevos

```bash
MODELO_VERSION=v2_embeddings \
MODELO_PATH=modelos/v2_embeddings_candidato/modelo.joblib \
python -m scripts.monitorear_drift \
  --csv datos/comentarios_nuevos.csv --columna feedback \
  --referencia modelos/v2_embeddings_candidato/drift_referencia.json \
  --salida resultados/drift_ultimo.json
```

Calcula **PSI sobre clases predichas y longitud de textos**; PSI ≥ 0.2 genera una alerta heurística. Con menos de 50 comentarios válidos el resultado es `muestra_insuficiente`. No guarda comentarios en el reporte, solo agregados y diagnóstico. PSI no sustituye la medición de desempeño sobre nuevas etiquetas humanas ni detecta toda forma de drift semántico. Evitar usar el mismo CSV de entrenamiento como "datos nuevos" para afirmar que no existe drift.

## 5. Pruebas y entregables verificables

```bash
python -m pytest -q tests/test_pipeline_v2_nuevo.py
python -m pytest tests/ -ra --tb=short
```

Los nueve tests nuevos de Pipeline, hashing, limpieza y PSI se probaron localmente con un codificador simulado; **no** se ejecutó la batería de tests del sistema con el encoder real ni se entrenó el modelo final. Antes de desplegar, ejecutar todos los tests sobre la carpeta candidata y luego sobre la carpeta promovida. Algunas expectativas de clase sobre frases fijas pueden cambiar tras reentrenar; documentar las diferencias con sus métricas antes de ajustar tests.

Para entregar todo hace falta recopilar también los JSON de métricas/tiempos/drift, el artefacto real y sus hashes, y una captura real de Streamlit con el v2 final.

## Limitaciones de seguridad

No agregar comentarios originales, CSV, contraseñas o tokens a GitHub. Los hashes aseguran integridad frente a un valor esperado, no autenticidad si ambos archivos fueran cambiados. La anonimización con expresiones regulares no garantiza eliminación completa de datos personales.


## Verificación del dataset aportado (16 de noviembre de 2026)

Se inspeccionaron los dos CSV subidos: son idénticos. El archivo original tiene **613 filas, 20 columnas, 0 comentarios duplicados y etiquetas completas**: 258 neutros, 204 negativos y 151 positivos. Se hallaron **cinco `id_unico` repetidos**, aunque `(archivo, semestre, id)` no tenía duplicados.

- SHA-256 de `comentarios_limpio.csv` **original**: `435b325da6a6ab34f6d31afb354af0c53698193c640431fe5f8c4df70c0a8a5e`
- Se generó `comentarios_limpio_ids_ok.csv`, con 613 IDs únicos. Se modificó **solo** la columna `id_unico`; ni `feedback` ni `polaridad` cambiaron.
- SHA-256 del **dataset corregido para entrenamiento**: `7ef215fcfcd62f6d8c80155667a6b83874efc7fc7087b1d1456838247588f146`
- La partición propuesta es **490 train / 123 test**, estratificada con semilla 42. La ejecución definitiva debe obtener los mismos conteos.
- Los hash y conteos se registraron en `resultados/trazabilidad_dataset.json` y `resultados/auditoria_dataset_y_drift.json`. El CSV corregido se entrega separadamente por privacidad, no dentro de este ZIP.

Para reproducir la auditoría con el CSV corregido guardado bajo `datos/`:

```bash
python -m scripts.auditar_dataset_y_drift_historico \
  --csv datos/comentarios_limpio_ids_ok.csv \
  --salida resultados/auditoria_dataset_y_drift.json --repeticiones 50
```

### Resultados reales de drift retrospectivo (etiquetas humanas, no modelo)

Comparación de referencia **2023-1 y 2023-2 (n=377)** frente a **2024-1 (n=227)**: `PSI_clases=0.226837`, `PSI_longitud=1.104268`. Los dos superan el umbral orientativo de 0.2: se emite **alerta de revisión**. El periodo 2024-2 tiene solamente 9 ejemplos: **muestra insuficiente**, no se calcula PSI.

Esto mide **cambio de distribución histórica de las etiquetas y longitudes**, no la deriva de la distribución de clases *predichas por v2* en datos nuevos y no evidencia una caída de precisión. Un drift de producción requiere el nuevo modelo, su referencia y observaciones posteriores sin reutilizar el conjunto de entrenamiento.

### Medición ejecutada: SOLO preprocesamiento

Se procesaron 613 textos con `PrepararComentarios(modo="corregido")`, 50 repeticiones, CPU Linux/Python 3.13.5. Mediana del lote: **24.04 ms**; p95: **27.33 ms**, aproximadamente **0.039 ms por comentario** de limpieza. Los números exactos y el entorno están en `resultados/auditoria_dataset_y_drift.json`. **No son tiempos de inferencia ni incluyen `SentenceTransformer` y la regresión logística**.

### Estado de reentrenamiento real

Se comprobó que `scripts.reentrenar_v2.leer_datos()` acepta el CSV corregido, pero no se entrenó un Pipeline final aquí: este entorno no tiene instalados `sentence-transformers` ni `transformers`, no contiene los pesos de `paraphrase-multilingual-MiniLM-L12-v2` y el acceso externo de red está deshabilitado. No existe un `modelo.joblib` nuevo, ni se generaron `macro-F1`, accuracy, tiempos de inferencia o `drift_referencia.json` del candidato. **No reemplazar el v2 vigente ni atribuirle el hash del CSV corregido al artefacto anterior.**

En el equipo con las dependencias y el encoder real disponibles, ejecutar el entrenamiento (comando en §2), verificar los hashes y guardar las métricas/tiempos. El script escribe `sha256_dataset` en el `modelo_info.json` **del modelo entrenado**; no modificar `modelo_info.json` histórico porque no se ha verificado su procedencia exacta.

### Pruebas ejecutadas en entorno aislado

- `python -m pytest -q tests/test_pipeline_v2_nuevo.py tests/test_auditoria_dataset.py`: **12 passed**.
- `python -m pytest -q tests/test_carga.py -k 'not test_archivos_de_cada_version_y_hash_valido and not test_rechaza_ruta_fuera_de_modelos'`: **17 passed, 3 deselected**.
- Las tres pruebas dependientes de los archivos de modelos reales no se pudieron ejecutar localmente: los artefactos originales no están montados en este entorno. **No afirmar que pasó toda la batería del repositorio**.
