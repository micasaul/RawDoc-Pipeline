# RawDoc-Pipeline: Detección de Layout en Documentos Universitarios

## Integrantes y Roles

* **Axel Blanc** — Responsable de Testing (validación manual y evaluación de resultados).
* **Donato De Battista** — Responsable del Código (implementación del pipeline de detección y análisis).
* **Benjamín Ayala** — Responsable de Arquitectura (entorno Docker, preprocesamiento y diseño del pipeline).
* **Ximena Carmona** — Responsable de Documentación (redacción del informe técnico y ground truth).
* **Micaela Saul** — Responsable del Repositorio (gestión de Git, organización de archivos y estructura del proyecto).


## 1. Introducción

Este proyecto aborda un problema concreto que tiene la Universidad Autónoma de Entre Ríos: la información contenida en los actos administrativos del Consejo Superior (ordenanzas, resoluciones) está disponible únicamente como archivos PDF, muchos de ellos escaneados. Esto dificulta la búsqueda, indexación y reutilización de esa información.

El objetivo es construir un pipeline que, dado un documento PDF del Consejo Superior, pueda:
1. Convertirlo en imágenes procesables.
2. Detectar automáticamente la estructura de layout de cada página (identificar dónde hay texto, tablas, encabezados, firmas, etc.).
3. En etapas futuras, aplicar OCR sobre las regiones detectadas para extraer texto estructurado.

La idea no es simplemente pasar un OCR sobre la página entera, sino primero entender la estructura del documento. Saber que un bloque es una tabla permite procesarlo distinto que un párrafo de texto o una firma escaneada. Este enfoque se conoce como *document layout analysis* y es un paso previo fundamental para cualquier sistema serio de extracción de información en documentos.

## 2. Contexto técnico

### 2.1 ¿Qué es Document Layout Analysis?

Document Layout Analysis (DLA) es la tarea de identificar y clasificar las regiones que componen una página de documento. En lugar de tratar la página como una imagen plana, el modelo segmenta las áreas y les asigna un tipo: párrafo, título, tabla, imagen, etc.

Los enfoques modernos tratan este problema como una tarea de detección de objetos, usando modelos como Faster R-CNN, DETR o YOLO, entrenados sobre datasets de documentos anotados.

### 2.2 DocLayNet

DocLayNet es un dataset publicado por IBM Research en 2022 que contiene más de 80.000 páginas de documentos anotadas manualmente con 11 categorías de layout. Es uno de los datasets más grandes y diversos para esta tarea, e incluye documentos financieros, legales, científicos y gubernamentales.

Las categorías que define son: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text y Title.

Elegimos un modelo preentrenado sobre DocLayNet porque las categorías coinciden bien con la estructura de los documentos del Consejo Superior: texto normativo, encabezados de sección (CONSIDERANDO, RESUELVE), listas numeradas (artículos), tablas y firmas/sellos.

### 2.3 ¿Por qué YOLO?

YOLO (You Only Look Once) es una familia de modelos de detección de objetos que se caracterizan por su velocidad. A diferencia de otros enfoques que procesan la imagen en múltiples pasadas, YOLO realiza la detección en una sola pasada, lo que lo hace práctico para procesar documentos de muchas páginas.

En particular usamos **YOLOv10s** (variante small), que ofrece un buen balance entre precisión y velocidad. El modelo preentrenado que utilizamos (`yolov10s-doclaynet.pt`) fue entrenado específicamente sobre DocLayNet, por lo que no requiere fine-tuning para una primera evaluación.

## 3. Arquitectura del pipeline

El pipeline se compone de tres etapas principales, cada una implementada como un módulo independiente:

```
PDFs originales          Imágenes por página         Regiones detectadas
(data/raw/)         →    (data/processed/)       →   (runs/detect/predict/)
                    ↑                             ↑
          preprocesamiento.py              src/YOLO/yolo.py
          (pdf2image + Poppler)            (YOLOv10s-DocLayNet)
```

### Etapa 1 — Preprocesamiento (`src/preprocesamiento.py`)

Convierte cada PDF en un conjunto de imágenes JPG (una por página) usando la librería `pdf2image`, que internamente utiliza Poppler como motor de renderizado.

El script recorre todos los PDFs en `data/raw/`, crea una subcarpeta por cada documento en `data/processed/`, y guarda las páginas como `pagina_1.jpg`, `pagina_2.jpg`, etc. Tiene un mecanismo de skip para no reprocesar documentos que ya fueron convertidos.

### Etapa 2 — Detección de layout (`src/YOLO/yolo.py`)

Carga el modelo YOLOv10s-DocLayNet y lo ejecuta sobre todas las imágenes generadas en la etapa anterior. Para cada imagen, el modelo produce un conjunto de bounding boxes con la clase detectada y su nivel de confianza.

Los resultados se guardan como imágenes anotadas (con los bounding boxes dibujados) en `runs/detect/predict/`, manteniendo la misma estructura de subcarpetas por documento. Se utiliza un umbral de confianza de 0.20, que es deliberadamente bajo para esta primera evaluación: preferimos obtener más detecciones (incluso con algo de ruido) y después ajustar.

### Etapa 3 — Análisis de resultados (`src/analisis_resultados.py`)

Re-ejecuta la inferencia sobre las imágenes para recopilar estadísticas detalladas. Genera un reporte en formato JSON (`runs/analisis/reporte.json`) que incluye:
- Cantidad de detecciones por categoría de layout
- Estadísticas de confianza (promedio, mínimo, máximo) por categoría
- Desglose por documento

## 4. Entorno de ejecución

### 4.1 Contenedor Docker

Para garantizar reproducibilidad, el proyecto incluye un Dockerfile y un docker-compose.yml que configuran el entorno completo:

**Imagen base:** `python:3.10-slim` (Debian)

**Dependencias de sistema:**
- `tesseract-ocr` y `tesseract-ocr-spa`: motor OCR con soporte para español (para etapas futuras).
- `poppler-utils`: renderizado de PDFs a imágenes.
- `libgl1`, `libglib2.0-0`: dependencias de OpenCV que necesita Ultralytics.

**Dependencias Python** (vía `requirements.txt`):
- `pdf2image 1.17.0`: wrapper de Poppler para Python.
- `pillow 12.2.0`: manipulación de imágenes.
- `pytesseract 0.3.13`: wrapper de Tesseract para Python.
- `ultralytics`: framework de YOLO (incluye YOLOv10).

### 4.2 Volúmenes

El docker-compose mapea tres directorios entre el host y el contenedor:

| Host | Contenedor | Contenido |
|---|---|---|
| `./src` | `/app/src` | Código fuente |
| `./data` | `/app/data` | PDFs y imágenes procesadas |
| `./runs` | `/app/runs` | Resultados de inferencia |

Esto permite editar el código y ver los resultados directamente desde el sistema host, sin necesidad de copiar archivos dentro y fuera del contenedor.

### 4.3 Cómo reproducir

```bash
# Levantar el contenedor
docker compose up -d

# Entrar al contenedor
docker exec -it RawDoc-Pipeline bash

# Ejecutar el preprocesamiento (si no se hizo antes)
python src/preprocesamiento.py

# Ejecutar YOLO
python src/YOLO/yolo.py

# Ejecutar el análisis
python src/analisis_resultados.py
```

## 5. Conjunto de referencia

Para la evaluación del pipeline, se configuró un repositorio de archivos de proceso compuesto por **10 documentos reales** de la Universidad Autónoma de Entre Ríos (UADER), divididos equitativamente en 5 ordenanzas y 5 resoluciones. Este corpus suma un total de **148 páginas** digitalizadas en formato PDF, almacenadas de forma local en `data/raw/` (excluidas del control de versiones por su volumen y privacidad).

El detalle cualitativo de la procedencia, la composición del conjunto de prueba y la especificación formal de las categorías esperadas se documenta en [ground_truth.md](file:///c:/Users/benja/Proyectos/Documentos/4°Lic.Sistemas/IA-PRACTICE/Proyecto_ocr_uader/docs/ground_truth.md). Los documentos cubren estructuras sumamente diversas:
- Páginas de texto continuo (decretos, considerandos).
- Artículos estructurados como listas numeradas.
- Tablas presupuestarias y formularios administrativos complejos.
- Firmas manuscritas y sellos institucionales escaneados.
- Páginas vacías correspondientes a reversos de hojas escaneadas.

## 6. Resultados preliminares

### 6.1 Ejecución general

El modelo procesó las 148 páginas sin errores. El tiempo promedio de inferencia fue de aproximadamente 350-400ms por imagen (sin GPU, usando CPU dentro del contenedor Docker).

Se detectaron regiones en la gran mayoría de las páginas. Las únicas páginas sin detecciones corresponden a reversos en blanco de hojas escaneadas y alguna última página de ordenanza que solo contiene espacio vacío, lo cual es un resultado correcto.

### 6.2 Categorías detectadas

De acuerdo con el reporte cuantitativo consolidado en [reporte.json](file:///c:/Users/benja/Proyectos/Documentos/4°Lic.Sistemas/IA-PRACTICE/Proyecto_ocr_uader/runs/analisis/reporte.json), se registraron un total de **1579 detecciones** en las 148 páginas procesadas, con una confianza promedio global de **0.595**. El desglose detallado de las detecciones por clase es el siguiente:

1. **Text (1101 detecciones, confianza prom: 0.642)**: Es la categoría predominante. El modelo agrupa correctamente los párrafos de texto normativo extenso, mostrando un comportamiento muy sólido (con picos de confianza de hasta `0.99`).
2. **Section-header (160 detecciones, confianza prom: 0.452)**: Identificó de manera consistente las divisiones lógicas del documento como "CONSIDERANDO:", "RESUELVE:", "ANEXO I". La confianza promedio moderada se debe a que tipográficamente estos encabezados son similares al texto plano (mismo tamaño, a veces sin negrita) en comparación con el dataset DocLayNet.
3. **List-item (127 detecciones, confianza prom: 0.649)**: Detectó con alta precisión los artículos numerados del articulado de las ordenanzas e ítems de listas.
4. **Picture (79 detecciones, confianza prom: 0.380)**: Agrupa tanto logotipos/escudos institucionales como las firmas manuscritas y sellos. Esta categorización genérica es importante para la exclusión en la etapa de OCR posterior.
5. **Page-footer (71 detecciones, confianza prom: 0.457)**: Detecta los números de página y pies de página.
6. **Table (15 detecciones, confianza prom: 0.574)**: Identifica bloques tabulares completos (por ejemplo, planillas de firmas o anexos de presupuestos).
7. **Page-header (12 detecciones, confianza prom: 0.244)**: Detecta el membrete superior institucional en las hojas membretadas.
8. **Title (10 detecciones, confianza prom: 0.230)**: Detecta títulos de documentos con baja confianza debido a su similitud visual con encabezados comunes en este tipo de actas.
9. **Caption (4 detecciones, confianza prom: 0.283)**: Epígrafes de tablas y cuadros anexos.

### 6.3 Observaciones cualitativas

Al revisar visualmente las imágenes anotadas generadas en `runs/detect/predict/`, se observó lo siguiente:

**Fortalezas del pipeline actual:**
- Los párrafos normativos y bloques de texto largo se segmentan con alta precisión (confianzas superiores a `0.90`).
- Las listas numeradas (artículos) se aíslan correctamente ítem por ítem.
- Las tablas se enmarcan como regiones únicas, ideal para su posterior extracción tabular o exclusión.
- Las páginas en blanco (reversos escaneados) no producen falsas detecciones (se registraron exactamente **6 páginas sin detección** de forma correcta).

**Oportunidades de mejora y limitaciones:**
- El umbral de `0.20` es bajo y genera algunas detecciones ruidosas de baja confianza (rango `0.21 - 0.30`).
- Escudos o logotipos pequeños se confunden a veces con `Text` en lugar de `Picture`.
- Títulos del Consejo Superior como "ORDENANZA CS Nº..." se catalogan como `Text` porque visualmente no tienen la prominencia typográfica (fuente gigante o negrita extrema) que el modelo espera para un `Title`.
- Algunas firmas desvaídas o con trazos finos no son segmentadas, quedando fuera de la región `Picture`.

### 6.4 Calibración del Umbral y F1-Score (Evaluación de Confianza)

Para refinar el comportamiento del modelo, se realizaron experimentos utilizando tres umbrales de confianza (**0.15**, **0.20** y **0.25**), evaluando manualmente la precisión y recall sobre clases clave. Como caso de estudio representativo, se analizó el rendimiento de la detección de la clase `Page-footer` (pie de página / numeración) en el documento `ORD-CS-N°-187` (que cuenta con un Ground Truth real de 23 pies de página esperados):

| Umbral de Confianza | True Positives (TP) | False Positives (FP) | Ground Truth (Real) | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **0.15** | 18 | 1 | 23 | **0.857** |
| **0.20** | 17 | 1 | 23 | **0.829** |
| **0.25** | 15 | 1 | 23 | **0.769** |

*Nota matemática: F1-Score calculado como $\frac{2 \cdot TP}{GroundTruth + TP + FP}$.*

**Conclusiones de la calibración:**
El análisis indica que el umbral de **0.15** ofrece el mayor F1-Score (0.857) para esta clase específica al capturar más pies de página reales (mayor Recall) sin penalizar en gran medida los falsos positivos. No obstante, al generalizar el pipeline a todo el dataset normativo, un umbral general muy bajo (como 0.15) introduce demasiado ruido en clases secundarias. Por ello, se ratificó el umbral de **0.20** como el valor por defecto para esta primera entrega funcional, logrando un balance robusto.

### 6.5 Tiempos de ejecución

El procesamiento se realizó dentro de la imagen de contenedor reproducible del proyecto ejecutada sobre Windows (CPU local host):

| Etapa | Tiempo aproximado |
|---|---|
| Preprocesamiento básico (148 páginas) | ~2 minutos |
| Inferencia YOLOv10s (148 páginas) | ~1 minuto |
| Análisis de resultados estadísticos | ~1 minuto |

Los tiempos son satisfactorios para el volumen de la entrega parcial. La GPU será crítica únicamente al escalar el procesamiento a repositorios masivos.

## 7. Decisiones técnicas

| Decisión | Justificación |
|---|---|
| YOLOv10s (small) | Balance entre velocidad y precisión. No es necesario el modelo large para una primera evaluación. |
| DocLayNet como base | Las categorías del dataset se alinean con la estructura de documentos administrativos universitarios. |
| Umbral de confianza 0.20 | Valor bajo deliberado para la primera ejecución: permite observar todas las detecciones posibles y luego filtrar. |
| pdf2image + Poppler | Es la combinación más robusta para convertir PDFs a imágenes en Python. Funciona con PDFs escaneados y digitales. |
| Docker con docker-compose | Simplifica la instalación de Tesseract, Poppler y las dependencias de sistema. Garantiza que el entorno sea reproducible. |
| Formato JPG (default DPI) | Suficiente para la detección de layout. Para OCR se evaluará usar 300 DPI y otros formatos. |

## 8. Trabajo futuro

Con esta primera etapa completada, los próximos pasos del proyecto son:

1. **Aplicar OCR por región**: usar Tesseract sobre las regiones de texto detectadas por YOLO, en lugar de sobre la página completa. Esto debería mejorar la calidad del texto extraído al evitar que el OCR intente leer firmas o imágenes.

2. **Ajustar el umbral de confianza**: a partir del análisis de la primera ejecución, calibrar el umbral para reducir falsos positivos sin perder detecciones válidas.

3. **Crear anotaciones manuales (ground truth formal)**: anotar un subconjunto de páginas con herramientas como Label Studio o CVAT para poder calcular métricas cuantitativas (mAP, precision, recall).

4. **Evaluar preprocesamiento de imagen**: probar técnicas como binarización, corrección de inclinación (deskew) y aumento de contraste para mejorar tanto la detección de layout como el OCR posterior.

5. **Exportar texto estructurado**: generar salida en formato JSON o XML que respete la estructura del documento (secciones, artículos, tablas como datos tabulares).

6. **Evaluar modelos alternativos**: comparar los resultados de YOLOv10 con otros enfoques de DLA como LayoutLM, DiT o Faster R-CNN entrenados sobre DocLayNet.

## 9. Estructura y Documentación del Repositorio

El repositorio se encuentra completamente configurado, con el código modularizado y funcionando bajo una estructura reproducible. A continuación, se detalla la organización de los componentes y el rol de cada pieza de documentación técnica elaborada hasta este punto:

### Documentación del Proyecto
- [README.md](file:///c:/Users/benja/Proyectos/Documentos/4°Lic.Sistemas/IA-PRACTICE/Proyecto_ocr_uader/README.md): Guía de configuración rápida para desarrolladores, instrucciones de montaje del contenedor Docker y comandos de ejecución paso a paso.
- [docs/ground_truth.md](file:///c:/Users/benja/Proyectos/Documentos/4°Lic.Sistemas/IA-PRACTICE/Proyecto_ocr_uader/docs/ground_truth.md): Especificación detallada del corpus documental de referencia y la categorización de layout del dataset DocLayNet.
- [docs/INFORME.md](file:///c:/Users/benja/Proyectos/Documentos/4°Lic.Sistemas/IA-PRACTICE/Proyecto_ocr_uader/docs/INFORME.md): Este documento, el cual recopila el informe técnico de la arquitectura, decisiones de diseño y análisis de resultados de la primera entrega.

### Árbol de Directorios
```
Proyecto_ocr_uader/
├── data/
│   ├── raw/                  # Repositorio de PDFs originales (10 archivos de proceso)
│   └── processed/            # Imágenes JPG por página generadas en el preprocesamiento
├── docs/
│   ├── ground_truth.md       # Definición de Ground Truth
│   └── INFORME.md            # Informe de avance de la entrega
├── runs/
│   ├── detect/predict/       # Imágenes de debug anotadas con bounding boxes de YOLO
│   └── analisis/             # Directorio de salida del reporte estadístico consolidado
├── src/
│   ├── preprocesamiento.py   # Implementación del preprocesamiento básico (PDF -> JPG)
│   ├── analisis_resultados.py # Análisis preliminar y acumulación de métricas
│   └── YOLO/
│       └── yolo.py           # Script de inferencia funcional con YOLOv10s
├── Dockerfile                # Definición de la imagen del contenedor reproducible
├── docker-compose.yml        # Orquestación de volúmenes y servicios del entorno
├── requirements.txt          # Dependencias de librerías Python instaladas
├── yolov10s-doclaynet.pt     # Pesos descargados localmente del modelo YOLOv10
└── README.md                 # Guía técnica de uso
```

## 10. Referencias

- DocLayNet: A Large-scale Dataset for Document Layout Analysis (Pfitzmann et al., 2022). https://arxiv.org/abs/2206.01062
- Ultralytics YOLO: https://docs.ultralytics.com/
- Poppler: https://poppler.freedesktop.org/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
