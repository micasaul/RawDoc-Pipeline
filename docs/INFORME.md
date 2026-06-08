# 1. Introducción

Este proyecto aborda un problema concreto que tiene la Universidad Autónoma de Entre Ríos: la información contenida en los actos administrativos del Consejo Superior (ordenanzas, resoluciones) está disponible únicamente como archivos PDF, muchos de ellos escaneados. Esto dificulta la búsqueda, indexación y reutilización de esa información.

El objetivo es construir un pipeline que, dado un documento PDF del Consejo Superior, pueda:
1. Convertirlo en imágenes procesables.
2. Detectar automáticamente la estructura de layout de cada página (identificar dónde hay texto, tablas, encabezados, firmas, etc.).
3. En etapas futuras, aplicar OCR sobre las regiones detectadas para extraer texto estructurado.

La idea no es simplemente pasar un OCR sobre la página entera, sino primero entender la estructura del documento. Saber que un bloque es una tabla, permite procesarlo distinto que un párrafo de texto o una firma escaneada. Esto se conoce como **Document Layout Analysis (DLA)**, y consiste en identificar y clasificar las distintas regiones que componen una página de documento. En lugar de tratar la página como una imagen plana, los modelos segmentan las áreas y les asignan un tipo específico, como párrafo, título, tabla o imagen.

# 2. Contexto técnico

### 2.1 ¿Qué es Document Layout Analysis?

Document Layout Analysis (DLA) es la tarea de identificar y clasificar las regiones que componen una página de documento. En lugar de tratar la página como una imagen plana, el modelo segmenta las áreas y les asigna un tipo: párrafo, título, tabla, imagen, etc.

Los enfoques modernos tratan este problema como una tarea de detección de objetos, usando modelos como Faster R-CNN, DETR o YOLO, entrenados sobre datasets de documentos anotados.

## 2.2 DocLayNet
DocLayNet es un dataset publicado por IBM Research en 2022 que contiene más de 80.000 páginas de documentos anotadas manualmente con 11 categorías de layout. Es uno de los datasets más grandes y diversos para esta tarea, e incluye documentos financieros, legales, científicos y gubernamentales.

Las categorías que define son: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text y Title.

Elegimos un modelo preentrenado sobre DocLayNet porque las categorías coinciden bien con la estructura de los documentos del Consejo Superior: texto normativo, encabezados de sección (CONSIDERANDO, RESUELVE), listas numeradas (artículos), tablas y firmas/sellos.

## 2.3 ¿Por qué YOLO?
YOLO (You Only Look Once) es una familia de modelos de detección de objetos que se caracterizan por su velocidad. A diferencia de otros enfoques que procesan la imagen en múltiples pasadas, YOLO realiza la detección en una sola pasada, lo que lo hace práctico para procesar documentos de muchas páginas.

En particular usamos **YOLOv10s** (variante small), que ofrece un buen balance entre precisión y velocidad. El modelo preentrenado que utilizamos (yolov10s-doclaynet.pt) fue entrenado específicamente sobre DocLayNet, por lo que no requiere fine-tuning para una primera evaluación.

# 3. Arquitectura del pipeline

El pipeline se compone de tres etapas principales, cada una implementada como un módulo independiente:

```
PDFs originales          Imágenes por página         Regiones detectadas
(data/raw/)         →    (data/processed/)       →   (runs/detect/predict/)
                    ↑                             ↑
          preprocesamiento.py              src/YOLO/yolo.py
          (pdf2image + Poppler)            (YOLOv10s-DocLayNet)
```

### Etapa 1: Preprocesamiento (src/preprocesamiento.py)
Convierte cada PDF en un conjunto de imágenes JPG (una por página) usando la librería `pdf2image`, que internamente utiliza Poppler como motor de renderizado.

El script recorre todos los PDFs en data/raw/, crea una subcarpeta por cada documento en data/processed/, y guarda las páginas como pagina_1.jpg, pagina_2.jpg, etc. Tiene un mecanismo de skip para no procesar documentos que ya fueron convertidos.

### Etapa 2: Detección de layout (src/YOLO/yolo.py)
Carga el modelo YOLOv10s-DocLayNet y lo ejecuta sobre todas las imágenes generadas en la etapa anterior. Para cada imagen, el modelo produce un conjunto de bounding boxes con la clase detectada y su nivel de confianza.

Los resultados se guardan como imágenes anotadas (con los bounding boxes dibujados) en runs/detect/predict/, manteniendo la misma estructura de subcarpetas por documento. Se utiliza un umbral de confianza de 0.20, que es deliberadamente bajo para esta primera evaluación: preferimos obtener más detecciones (incluso con algo de ruido) y después ajustar.

### Etapa 3: Análisis de resultados (src/analisis_resultados.py)
Re-ejecuta la inferencia sobre las imágenes para recopilar estadísticas detalladas. Genera un reporte en formato JSON (runs/análisis/reporte.json) que incluye:
* Cantidad de detecciones por categoría de layout
* Estadísticas de confianza (promedio, mínimo, máximo) por categoría
* Desglose por documento

# 3. Entorno de ejecución

## 3.1 Contenedor Docker
Para garantizar reproducibilidad, el proyecto incluye un Dockerfile y un docker-compose.yml que configuran el entorno completo:

**Imagen base:** python:3.10-slim (Debian)

**Dependencias de sistema:**
* tesseract-ocr y tesseract-ocr-spa: motor OCR con soporte para español (para etapas futuras).
* poppler-utils: renderizado de PDFs a imágenes.
* libgl1, libglib2.0-0: dependencias de OpenCV que necesitan Ultralytics.

**Dependencias Python** (vía requirements.txt):
* pdf2image 1.17.0: wrapper de Poppler para Python.
* pillow 12.2.0: la manipulación de imágenes.
* pytesseract 0.3.13: wrapper de Tesseract para Python.
* ultralytics: framework de YOLO (incluye YOLOv10).

## 3.2 Volúmenes
El docker-compose mapea tres directorios entre el host y el contenedor:

| Host | Contenedor | Contenido |
| :--- | :--- | :--- |
| ./src | /app/src | Código fuente |
| ./data | /app/data | PDFs e imágenes procesadas |
| ./runs | /app/runs | Resultados de inferencia |

Esto permite editar el código y ver los resultados directamente desde el sistema host, sin necesidad de copiar archivos dentro y fuera del contenedor.

## 3.3 Cómo reproducir
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

# 4. Conjunto de referencia

Para la evaluación del pipeline, se configuró un repositorio de archivos de proceso compuesto por 10 documentos reales de la Universidad Autónoma de Entre Ríos (UADER), divididos equitativamente en 5 ordenanzas y 5 resoluciones emitidas por el Consejo Superior. Los documentos fueron descargados desde el portal institucional oficial en formato PDF y almacenados localmente en data/raw/.

El corpus utilizado suma un total de 148 páginas digitalizadas y representa un caso de uso real orientado a la digitalización y estructuración automática de actos administrativos universitarios. Los documentos seleccionados corresponden a:
* **Ordenanzas del Consejo Superior** (prefijo ORD-CS), vinculadas a normativas de alcance general.
* **Resoluciones del Consejo Superior** (prefijo Res-CS), asociadas a disposiciones específicas y administrativas.

La composición del conjunto de referencia se detalla a continuación:

| Documento | Tipo | Páginas |
| :--- | :--- | :--- |
| ORD-185 | Ordenanza | 8 |
| ORD-CS-186-25 | Ordenanza | 20 |
| ORD-CS-N°-187 | Ordenanza | 24 |
| ORD-CS-N°-188-25 | Ordenanza | 20 |
| ORD-CS-N°-189-comprimido | Ordenanza | 56 |
| Res-CS-054-25-27-03-2025 | Resolución | 2 |
| Res-CS-055-25-27-03-2025 | Resolución | 8 |
| Res-CS-056-25-27-03-2025 | Resolución | 2 |
| Res-CS-073-25-27-03-2025 | Resolución | 4 |
| Res-CS-078-25-27-03-2025 | Resolución | 4 |

**Total: 10 documentos y 148 páginas.**

Los documentos cubren estructuras sumamente diversas, incluyendo:
* Páginas de texto continuo.
* Artículos estructurados como listas numeradas.
* Tablas presupuestarias y formularios administrativos complejos.
* Firmas manuscritas y sellos institucionales escaneados.
* Encabezados y pies de página administrativos.
* Páginas vacías correspondientes a reversos de hojas escaneadas.

El modelo utilizado corresponde a YOLOv10s entrenado sobre el dataset DocLayNet, el cual reconoce distintas categorías de regiones documentales, entre ellas: Section-header, List-item, Text, Table, Picture, Title, Caption, Page-header y Page-footer.

A partir de estas categorías generales, el grupo de trabajo definió además un conjunto de observaciones específicas orientadas a las características propias de los documentos institucionales analizados.

Con el objetivo de complementar las categorías originales del modelo y realizar un análisis más cercano al dominio documental universitario, se elaboró una planilla experimental de observación manual donde se registraron distintos elementos presentes en las páginas procesadas. Entre ellos:
* Firmas manuscritas.
* Sellos institucionales.
* Logos y escudos.
* Perforaciones de hojas escaneadas.
* Manchas o ruido visual.
* Páginas en blanco.
* Bullets y elementos de listas.
* Imágenes y tablas detectadas.

En varios casos, estas observaciones representan subdivisiones o interpretaciones específicas de categorías más amplias definidas por DocLayNet. Por ejemplo:

| Categoría experimental | Categoría equivalente en DocLayNet / YOLO |
| :--- | :--- |
| Firmas | Picture |
| Sellos | Picture |
| Logos / escudos | Picture |
| Imágenes | Picture |
| Bullets / listas | List-item |
| Tablas | Table |
| Encabezados administrativos | Page-header |
| Pies de página | Page-footer |
| Títulos | Title |
| Títulos de secciones | Section-header |
| Páginas en blanco | Ausencia de detecciones |

De esta manera, el análisis experimental permitió complementar la salida estándar del modelo con observaciones cualitativas específicas del contexto institucional analizado.

Se inspeccionaron visualmente las detecciones generadas sobre una muestra representativa de documentos, verificando la coherencia entre las regiones detectadas y el contenido real presente en las páginas.

Asimismo, se realizaron pruebas variando distintos umbrales de confianza del modelo para analizar el comportamiento de las detecciones frente a documentos con diferente complejidad visual. Los resultados obtenidos fueron registrados en una planilla comparativa desarrollada por el equipo.

**Planilla de experimentación y resultados:**  
[Link a la Planilla de Google Drive](https://docs.google.com/spreadsheets/d/1auYnDuoBU9F2DGiNMLovTlKGoXOYFVR43KZiilfujnU/edit?usp=sharing)

Entre las observaciones más relevantes identificadas durante el análisis se destacan:
* Las resoluciones poseen una estructura más uniforme y predecible, generalmente compuesta por texto administrativo y firmas institucionales.
* Algunas páginas corresponden a reversos en blanco de escaneos doble faz, las cuales fueron correctamente interpretadas por el modelo mediante ausencia de detecciones.
* La calidad del escaneo varía significativamente entre documentos, existiendo casos con buena resolución y otros con leve inclinación, compresión o pérdida de nitidez.

# 5. Resultados preliminares

## 5.1 Ejecución general
El modelo procesó las 148 páginas sin errores. El tiempo promedio de inferencia fue de aproximadamente 350-400ms por imagen (sin GPU, usando CPU dentro del contenedor Docker).

## 5.2 Categorías detectadas
De acuerdo con el reporte cuantitativo consolidado en reporte.json, se registraron un total de 1579 detecciones en las 148 páginas procesadas, con una confianza promedio global de 0.595. El desglose detallado de las detecciones por clase es el siguiente:

1. **Text (1101 detecciones, confianza prom: 0.642)**: Es la categoría predominante. El modelo agrupa correctamente los párrafos de texto normativo extenso, mostrando un comportamiento muy sólido (con picos de confianza de hasta 0.99).
2. **Section-header (160 detecciones, confianza prom: 0.452)**: Identificó de manera consistente las divisiones lógicas del documento como "CONSIDERANDO:", "RESUELVE:", "ANEXO I". La confianza promedio moderada se debe a que tipográficamente estos encabezados son similares al texto plano (mismo tamaño, a veces sin negrita) en comparación con el dataset DocLayNet.
3. **List-item (127 detecciones, confianza prom: 0.649)**: Detectó con alta precisión los artículos numerados del articulado de las ordenanzas e ítems de listas.
4. **Picture (79 detecciones, confianza prom: 0.380)**: Agrupa tanto logotipos/escudos institucionales como las firmas manuscritas y sellos. Esta categorización genérica es importante para la exclusión en la etapa de OCR posterior.
5. **Page-footer (71 detecciones, confianza prom: 0.457)**: Detecta los números de página y pies de página.
6. **Table (15 detecciones, confianza prom: 0.574)**: Identifica bloques tabulares completos (por ejemplo, planillas de firmas o anexos de presupuestos).
7. **Page-header (12 detecciones, confianza prom: 0.244)**: Detecta el membrete superior institucional en las hojas membretadas.
8. **Title (10 detecciones, confianza prom: 0.230)**: Detecta títulos de documentos con baja confianza debido a su similitud visual con encabezados comunes en este tipo de actas.
9. **Caption (4 detecciones, confianza prom: 0.283)**: Epígrafes de tablas y cuadros anexos.

## 5.3 Observaciones cualitativas
Al revisar visualmente las imágenes anotadas generadas en runs/detect/predict/, se observó lo siguiente:

**Fortalezas del pipeline actual:**
* Los párrafos normativos y bloques de texto largo se segmentan con alta precisión (confianzas superiores a 0.90).
* Las listas numeradas (artículos) se aíslan correctamente ítem por ítem.
* Las tablas se enmarcan como regiones únicas, ideal para su posterior extracción tabular o exclusión.
* Las páginas en blanco (reversos escaneados) no producen falsas detecciones (se registraron exactamente 6 páginas sin detección de forma correcta).

**Oportunidades de mejora y limitaciones:**
* El umbral de 0.20 es bajo y genera algunas detecciones ruidosas de baja confianza (rango 0.21 - 0.30).
* Escudos o logotipos pequeños se confunden a veces con Text en lugar de Picture.
* Títulos del Consejo Superior como "ORDENANZA CS Nº..." se catalogan como Text porque visualmente no tienen la prominencia typográfica (fuente gigante o negrita extrema) que el modelo espera para un Title.
* Algunas firmas desvaídas o con trazos finos no son segmentadas, quedando fuera de la región Picture.

## 5.4 Calibración del Umbral y F1-Score (Evaluación de Confianza)
Para refinar el comportamiento del modelo, se realizaron experimentos utilizando tres umbrales de confianza (0.15, 0.20 y 0.25), evaluando manualmente la precisión y recall sobre clases clave. Como caso de estudio representativo, se analizó el rendimiento de la detección de la clase Page-footer (pie de página / numeración) en el documento ORD-CS-N°-187 (que cuenta con un Ground Truth real de 23 pies de página esperados):

| Umbral de Confianza | True Positives | False Positives | Ground Truth | F1 - Score |
| :---: | :---: | :---: | :---: | :---: |
| 0.15 | 18 | 1 | 23 | 0.857 |
| 0.20 | 17 | 1 | 23 | 0.829 |
| 0.25 | 15 | 1 | 23 | 0.769 |

**Conclusiones de la calibración:** El análisis indica que el umbral de 0.15 ofrece el mayor F1-Score (0.857) para esta clase específica al capturar más pies de página reales (mayor Recall) sin penalizar en gran medida los falsos positivos. No obstante, al generalizar el pipeline a todo el dataset normativo, un umbral general muy bajo (como 0.15) introduce demasiado ruido en clases secundarias. Por ello, se ratificó el umbral de 0.20 como el valor por defecto para esta primera entrega funcional, logrando un balance robusto.

## 5.5 Tiempos de ejecución
El procesamiento se realizó dentro de la imagen de contenedor reproducible del proyecto ejecutada sobre Windows (CPU local host):

| Etapa | Tiempo aproximado |
| :--- | :--- |
| Preprocesamiento básico (148 páginas) | ~2 minutos |
| Inferencia YOLOv10s (148 páginas) | ~1 minuto |
| Análisis de resultados estadísticos | ~1 minuto |

# 6. Decisiones técnicas

| Decisión | Justificación |
| :--- | :--- |
| YOLOv10s (small) | Balance entre velocidad y precisión. No es necesario el modelo large para una primera evaluación. |
| DocLayNet como base | Las categorías del dataset se alinean con la estructura de documentos administrativos universitarios. |
| Umbral de confianza 0.20 | Valor bajo deliberado para la primera ejecución: permite observar todas las detecciones posibles y luego filtrar. |
| pdf2image + Poppler | Es la combinación más robusta para convertir PDFs a imágenes en Python. Funciona con PDFs escaneados y digitales. |
| Docker con docker-compose | Simplifica la instalación de Tesseract, Poppler y las dependencias de sistema. Garantiza que el entorno sea reproducible. |
| Formato JPG (default DPI) | Suficiente para la detección de layout. |

# 7. Estructura y Documentación del Repositorio

El repositorio se encuentra completamente configurado, con el código modularizado y funcionando bajo una estructura reproducible. A continuación, se detalla la organización de los componentes y el rol de cada pieza de documentación técnica elaborada hasta este punto:

### Documentación del Proyecto
* **README.md**: Guía de configuración rápida para desarrolladores, instrucciones de montaje del contenedor Docker y comandos de ejecución paso a paso.
* **docs/ground_truth.md**: Especificación detallada del corpus documental de referencia y la categorización de layout del dataset DocLayNet.
* **docs/INFORME.md**: Este documento, el cual recopila el informe técnico de la arquitectura, decisiones de diseño y análisis de resultados de la primera entrega.

### Árbol de Directorios
```
RawDoc-Pipeline/
├── data/
│   ├── raw/                  # Repositorio de PDFs originales (10 archivos de proceso)
│   └── processed/            # Imágenes JPG por página generadas en el preprocesamiento
├── docs/
│   ├── ground_truth.md       # Definición de Ground Truth
│   └── INFORME.md            # Informe de avance de la entrega
├── runs/
│   ├── detect/predict/       # Imágenes de debug anotadas con bounding boxes de YOLO
│   └── analisis/             # Directorio de salida del reporte consolidado
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

# 8. Referencias

* DocLayNet: A Large-scale Dataset for Document Layout Analysis (Pfitzmann et al., 2022). https://arxiv.org/abs/2206.01062
* Ultralytics YOLO: https://docs.ultralytics.com/
* Poppler: https://poppler.freedesktop.org/
* Tesseract OCR: https://github.com/tesseract-ocr/tesseract
