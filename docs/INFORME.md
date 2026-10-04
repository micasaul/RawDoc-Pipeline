# Informe Técnico Final: Pipeline Inteligente para Análisis Estructural y OCR Selectivo de Documentos Normativos

**Institución:** Universidad Autónoma de Entre Ríos (UADER)  
**Materia:** Inteligencia Artificial / Práctica Profesional  
**Proyecto:** RawDoc-Pipeline (Análisis Estructural y OCR Selectivo)  
**Fecha:** Octubre 2026  

---

# 1. Introducción y Planteo del Problema

Los actos administrativos emitidos por el Consejo Superior de la Universidad Autónoma de Entre Ríos (ordenanzas, resoluciones) se encuentran preservados y publicados como documentos PDF, en su gran mayoría escaneados o digitalizados en baja resolución. Estos documentos presentan una estructura visual heterogénea y compleja: membretes institucionales, texto normativo en columnas o bloques continuos, articulados numerados, tablas presupuestarias, sellos oficiales y firmas manuscritas de autoridades.

### El problema del OCR Tradicional
La aplicación directa de motores de Reconocimiento Óptico de Caracteres (OCR) tradicionales (como Tesseract aplicado sobre la página entera) presenta limitaciones severas:
1. **Contaminación por ruido estructural:** El motor intenta interpretar trazos manuscritos, sellos circulares o escudos como texto, generando secuencias de caracteres basura (`",\no .\n.\n”\nds\nb\n+\n.ES\n,"`).
2. **Pérdida de jerarquía y orden de lectura:** Al tratar la página como una imagen plana, se destruye la separación entre títulos, considerandos y articulados, produciendo un texto continuo no estructurado que dificulta su posterior indexación o procesamiento con modelos de lenguaje (LLMs).
3. **Procesamiento ciego de páginas no textuales:** Se procesan innecesariamente reversos en blanco y anexos gráficos.

### Propuesta de Solución
Este proyecto implementa y evalúa un **pipeline integral basado en técnicas modernas de Document AI**, que combina:
* Detección de Layout de Documentos (**Document Layout Analysis - DLA**) mediante redes neuronales de detección de objetos (familia YOLO entrenada sobre DocLayNet).
* Reglas heurísticas de post-procesamiento geométrico para la exclusión deliberada de elementos no textuales (`Picture`) y supresión de cajas anidadas redundantes.
* Extracción selectiva de texto mediante OCR guiado por bloques en orden de lectura natural con Tesseract 5 (`spa`), generando salidas estructuradas en **Markdown semántico**, **JSON enriquecido** y **Texto Plano**.

---

# 2. Contexto Técnico y Selección de Modelos

## 2.1 Document Layout Analysis (DLA) y Dataset DocLayNet
Document Layout Analysis consiste en segmentar y clasificar las diferentes regiones funcionales de una página. En este trabajo se adopta como referencia el dataset **DocLayNet** (IBM Research, 2022), compuesto por más de 80.000 páginas anotadas con 11 categorías: *Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text y Title*.

## 2.2 Evolución Experimental: De YOLOv10s a YOLO11m
En la primera fase del proyecto (junio) se utilizó como baseline el modelo `yolov10s-doclaynet.pt`. Si bien demostró viabilidad técnica, el análisis cualitativo reveló que omitía firmas desvaídas y confundía encabezados de sección con párrafos planos.

Para resolver esto, se llevó a cabo un **benchmark experimental comparativo exhaustivo** evaluando 4 variantes sobre el corpus completo de la UADER (10 documentos, 148 páginas) bajo idéntico umbral de confianza (`conf=0.20`):

| Métrica / Categoría | `yolov10s` (Baseline) | `yolov11s` (YOLO11 Small) | 🏆 `yolov11m` (YOLO11 Medium) | `yolov11l` (YOLO11 Large) |
| :--- | :---: | :---: | :---: | :---: |
| **Tamaño de pesos (.pt)** | 16.6 MB | 19.3 MB | **40.6 MB** | 51.3 MB |
| **Detecciones totales** | 1.579 | 1.591 | **1.814** | 1.750 |
| **Confianza promedio global** | 0.595 | 0.587 | **0.600 (Máxima)** | 0.583 |
| **Hojas en blanco (reversos)** | 6 sin detección | 4 (2 falsos positivos) | **6 (100% robusto)** | **6 (100% robusto)** |
| **Velocidad (CPU Docker)** | 389 ms/pág | 417 ms/pág | **1.013 ms/pág (~1.0s)** | 1.391 ms/pág |
| **`Picture` (Firmas/sellos)** | 79 det (conf 0.38) | 122 det (conf 0.36) | **137 det (conf 0.41)** | 119 det (conf 0.42) |
| **`Section-header`** | 160 det (conf 0.45) | 128 det (conf 0.43) | **241 det (conf 0.43)** | 188 det (conf 0.44) |
| **`Table` (Tablas)** | 15 det | 21 det (fragmentadas) | **15 det (Exacto)** | 23 det (fragmentadas) |

### Justificación de la Elección de YOLO11m
1. **Máxima captura de firmas y sellos (+73%):** Pasó de 79 a 137 detecciones de `Picture`, aislando exitosamente trazos finos manuscritos que anteriormente generaban ruido en el OCR.
2. **Jerarquía normativa superior (+50% en Section-header):** Captura consistentemente fórmulas jurídicas como *"CONSIDERANDO:"* y *"RESUELVE:"*.
3. **Integridad tabular:** Detectó exactamente las 15 tablas existentes sin fragmentarlas, a diferencia de las versiones Small y Large que sobre-segmentaron celdas individuales.
4. **Cero falsos positivos en páginas vacías:** 100% de especificidad en las 6 páginas de reversos escaneados.

---

# 3. Arquitectura del Pipeline Implementado

El pipeline consta de 4 etapas modulares y reproducibles:

```
    [PDFs en data/raw/]
            │
            ▼ 1. Preprocesamiento (src/preprocesamiento.py)
    [Imágenes PNG 300 DPI en data/processed/]
            │
            ▼ 2. Layout Detection (src/YOLO/yolo.py)
    [BBoxes ordenados + layout.json + runs/detect/]
            │
            ▼ 3. Post-procesamiento Heurístico (src/ocr/preprocessor.py)
    [Filtrado de Picture + Supresión de Cajas Anidadas (IoU)]
            │
            ▼ 4. OCR Selectivo y Reconstrucción (src/ocr/pipeline.py)
    [Salidas en runs/ocr/: documento.md, documento.txt, ocr_results.json]
```

### Etapa 1: Preprocesamiento Atómico y Streaming O(1)
* **Resolución nativa:** Conversión a 300 DPI en formato sin pérdida PNG (`pdf2image` + Poppler).
* **Consumo de memoria constante O(1):** Utiliza `paths_only=True` para volcar directamente el renderizado a disco, evitando fugas de memoria en documentos extensos (ej. 56 páginas).
* **Nomenclatura con Zero-Padding:** Guarda archivos como `pagina_0001.png`, garantizando consistencia estricta en el orden lexicográfico de lectura.

### Etapa 2: Detección y Clasificación Estructural (DLA)
* Inferencia con **YOLO11m-DocLayNet** (`conf=0.20`, `iou=0.45`).
* Ordenamiento de regiones en secuencia de lectura natural (*top-to-bottom, left-to-right*) mediante clave de ordenamiento $\text{sort\_key} = (y_{\min}, x_{\min})$.
* Generación de metadatos estructurados por página (`layout.json`) y consolidado global (`manifest.json`).

### Etapa 3: Post-procesamiento Heurístico y Filtrado Geométrico
* **Exclusión Selectiva:** Mapeo de taxonomía que declara no elegibles para OCR a las clases no textuales (`Picture` $\to$ firmas/sellos/escudos, `Formula` $\to$ fórmulas).
* **Supresión de Regiones Redundantes:** Algoritmo de intersección geométrica que descarta cajas anidadas si el ratio de contención supera el 85% ($\frac{\text{Área}(A \cap B)}{\text{Área}(B)} \ge 0.85$) o $\text{IoU} \ge 0.70$, evitando la duplicación de líneas de texto.

### Etapa 4: Extracción OCR Selectiva y Reconstrucción Estructurada
* **Aislamiento con Margen de Seguridad (*Padding*):** Recorte dinámico con margen de 8 píxeles (`padding=8`), impidiendo que el contorno de la caja mutile trazos curvos de caracteres alfanuméricos.
* **Configuración Adaptativa de Tesseract (PSM):**
  * `PSM 6` (`--oem 1 -l spa`): Bloque uniforme para párrafos normativos y encabezados.
  * `-c preserve_interword_spaces=1`: Para preservar columnas en tablas detectadas.
* **Reconstrucción Semántica:** Genera `documento.md` formateando títulos con `#`, encabezados con `###`, articulados con listas `-` y bloques tabulares en código.

---

# 4. Entorno de Ejecución Reproducible

Para asegurar total reproducibilidad técnica multiplataforma, el entorno se encapsula mediante Docker:

* **Imagen Base:** `python:3.10-slim` (Debian Bookworm).
* **Dependencias de Sistema:** `tesseract-ocr`, paquete de idioma español `tesseract-ocr-spa`, utilitarios de renderizado `poppler-utils`, y librerías gráficas `libgl1`, `libglib2.0-0`.
* **Dependencias Python:** `ultralytics` (YOLO11), `pytesseract 0.3.13`, `pdf2image 1.17.0`, `pillow 12.2.0`, `opencv-python-headless`.
* **Orquestación:** `docker-compose.yml` mapeando volúmenes independientes para `/app/src`, `/app/data`, `/app/runs`, `/app/docs` y los pesos del modelo `yolov11m-doclaynet.pt`.

---

# 5. Evaluación Experimental y Resultados

## 5.1 Rendimiento Global del Batch OCR (Corpus UADER)
El pipeline procesó la totalidad del corpus institucional (10 documentos, 148 páginas) arrojando las siguientes métricas globales registradas en `runs/ocr/ocr_manifest.json`:

* **Páginas procesadas:** 148 páginas.
* **Palabras totales extraídas:** 40.645 palabras.
* **Caracteres totales extraídos:** 260.821 caracteres.
* **Confianza promedio global de OCR:** **90.52%**.
* **Tiempo total de ejecución en CPU:** 609.2 segundos (~10 minutos para todo el archivo histórico universitario).

| Documento | Tipo | Páginas | Palabras Extraídas | Confianza Promedio |
| :--- | :---: | :---: | :---: | :---: |
| ORD-185 | Ordenanza | 8 | 1.272 | 91.64% |
| ORD-CS-186-25 | Ordenanza | 20 | 6.207 | 89.98% |
| ORD-CS-N°-187 | Ordenanza | 24 | 9.395 | 91.55% |
| ORD-CS-N°-188-25 | Ordenanza | 20 | 6.471 | 90.25% |
| ORD-CS-N°-189-comprimido | Ordenanza | 56 | 12.636 | 89.56% |
| Res-CS-054-25-27-03-2025 | Resolución | 2 | 599 | 90.93% |
| Res-CS-055-25-27-03-2025 | Resolución | 8 | 1.677 | 90.35% |
| Res-CS-056-25-27-03-2025 | Resolución | 2 | 505 | 87.76% |
| Res-CS-073-25-27-03-2025 | Resolución | 4 | 741 | 92.05% |
| Res-CS-078-25-27-03-2025 | Resolución | 4 | 1.142 | 90.40% |
| **Total General** | — | **148** | **40.645** | **90.52%** |

---

## 5.2 Evaluación Experimental Comparativa: OCR Completo vs. OCR Selectivo

En cumplimiento estricto con los objetivos de la materia, se diseñó un experimento cuantitativo formal ejecutado mediante `src/ocr/comparar_ocr.py`, contrastando el OCR tradicional directo (página completa) frente al OCR selectivo por layout sobre páginas con firmas y sellos:

| Caso Evaluado | OCR Tradicional Completo | OCR Selectivo por Layout | Reducción de Ruido / Mejora |
| :--- | :---: | :---: | :---: |
| **Página de firmas institucionales** *(ORD-189, Pág 56)* | 15 líneas de basura (`,\no .\nds\nb\n+`) | **0 líneas basura (Página limpia)** | 🎯 **100.0% reducción de ruido** |
| **Página final con múltiples firmas** *(ORD-185, Pág 8)* | 7 palabras erróneas (conf: 49.1%) | **0 palabras falsas (conf: 100%)** | 🎯 **100.0% reducción de ruido** |
| **Página con firmas y sello al pie** *(Res-CS-054, Pág 2)* | Firmas leídas como caracteres corruptos | Firmas excluidas; solo texto institucional | 🎯 **Aislamiento perfecto de autoridades** |
| **Página inicial de Ordenanza** *(ORD-185, Pág 1)* | Bloque plano indiferenciado | **Markdown semántico (`#`, `###`, `-`)** | 📖 **Jerarquía y orden de lectura preservados** |

### Hallazgo Clave del Experimento
En páginas que contienen únicamente firmas manuscritas y sellos, el OCR convencional produce hasta un 25% de caracteres de ruido alucinados por el motor de reconocimiento. El pipeline propuesto **elimina el 100% de este ruido**, evitando la contaminación de bases de datos o índices de búsqueda.

---

## 5.3 Análisis de Errores y Calibración de Hiperparámetros

Durante la evaluación empírica se detectó un caso de estudio crítico en la primera página de la ordenanza `ORD-185`: el encabezado `"ORDENANZA CS Nº 1 8 5"` era reconocido erróneamente por Tesseract como `"ORDENANZA CS Nº 1 3 5"`.

### Diagnóstico y Resolución mediante Calibración de Padding
Al realizar un análisis visual del recorte a nivel de píxeles, se constató que la caja delimitadora provista por YOLO ajustaba exactamente sobre el glifo del número `8`. Con el padding original de 4 píxeles (`padding=4`), el recorte cercenaba 1 píxel del trazo exterior curvo izquierdo, transformando visualmente el `8` en un `3` abierto ante la red neuronal LSTM de Tesseract.

Se evaluaron experimentalmente distintos márgenes de recorte sobre dicha región:

| Padding de Recorte | Texto Reconocido por Tesseract | Confianza | Diagnóstico Técnico |
| :---: | :---: | :---: | :--- |
| `padding = 0 px` | `ORDENANZA “Cs” N 1 8 5` | 69.0% | Reconoce el 8 pero pierde puntuación. |
| `padding = 2 px` | `ORDENANZA “cs”N» 1 85` | 74.5% | Fusión de caracteres adyacentes. |
| `padding = 4 px` *(antiguo)* | `ORDENANZA “cs”N* 1 3 5` | 78.2% | **Falso negativo:** corte de trazo convierte 8 en 3. |
| **`padding = 8 px` *(adoptado)*** | **`ORDENANZA “cs”No 1 8 5`** | **83.1%** | **Óptimo:** margen perimetral suficiente para el modelo LSTM. |
| `padding = 12 px` | `ORDENANZA “cs”No 1 8 5` | 82.8% | Estable, pero introduce riesgo de capturar líneas adyacentes. |

**Conclusión:** Se fijó `padding = 8` píxeles como estándar en el módulo `src/ocr/pipeline.py`, erradicando el error sin invadir regiones vecinas.

---

# 6. Estructura del Repositorio y Módulos de Código

```
Proyecto_ocr_uader/
├── data/
│   ├── raw/                      # PDFs originales de Consejo Superior (10 archivos)
│   └── processed/                # Imágenes renderizadas en alta resolución (PNG @ 300 DPI)
├── docs/
│   ├── ground_truth.md           # Definición de Ground Truth y dataset de validación
│   └── INFORME.md                # Este informe técnico consolidado
├── runs/
│   ├── detect/
│   │   └── predict_yolov11m-doclaynet/ # BBoxes anotados (.jpg) y layout.json por documento
│   ├── ocr/                      # Salidas finales: documento.md, documento.txt y ocr_results.json
│   └── analisis/                 # Reportes consolidados y JSONs de benchmark experimental
├── src/
│   ├── preprocesamiento.py       # Renderizado O(1) de PDFs a 300 DPI con zero-padding
│   ├── comparar_modelos.py       # Script de benchmarking multi-modelo YOLO
│   ├── YOLO/
│   │   ├── __init__.py           # Exportación de utilitarios de recorte geométrico
│   │   └── yolo.py               # Inferencia YOLO11m, orden natural y exportación de layout
│   └── ocr/
│       ├── __init__.py           # Paquete de OCR
│       ├── config.py             # Configuración dinámica multiplataforma de Tesseract/Tessdata
│       ├── preprocessor.py       # Filtros de imagen y supresión de cajas anidadas (IoU)
│       ├── engine.py             # Envoltorio MotorOCR con PSM adaptativo por bloque
│       ├── pipeline.py           # Pipeline integral batch y reconstrucción Markdown
│       └── comparar_ocr.py       # Experimento comparativo cuantitativo OCR Completo vs Selectivo
├── Dockerfile                    # Entorno reproducible con Debian, Tesseract 5 spa y Poppler
├── docker-compose.yml            # Orquestación de volúmenes y servicios
├── requirements.txt              # Dependencias fijadas de Python
└── README.md                     # Guía de despliegue y manual de ejecución
```

---

# 7. Conclusiones y Trabajo Futuro

### Conclusiones Técnicas
1. **Validación del Enfoque Document AI:** Se demostró cuantitativamente que aplicar OCR de forma ciega sobre documentos normativos genera salidas corruptas. La incorporación de detección de layout previa como filtro estructural permite **eliminar el 100% de los caracteres basura** provenientes de firmas y sellos.
2. **Superioridad de YOLO11m:** La adopción de YOLO11m frente a YOLOv10s incrementó en un **73% la captura de firmas y un 50% la de encabezados normativos**, alcanzando un punto de equilibrio óptimo entre precisión y latencia (~1 seg/pág en CPU estándar).
3. **Calidad de Reconstrucción Semántica:** El pipeline no solo extrae texto con una confianza promedio superior al **90%**, sino que reconstruye documentos legibles en **Markdown jerárquico**, permitiendo la preservación de la lógica jurídica de ordenanzas y resoluciones universitarias.

### Líneas de Trabajo Futuro
* **Post-procesamiento con LLMs ligeros locales:** Integrar modelos de lenguaje pequeños (SLMs vía Ollama) para corregir errores menores de OCR en nombres propios históricos o números de expediente deteriorados.
* **Extracción de Tablas a CSV/DataFrames:** Integrar herramientas especializadas de reconstrucción tabular (como Table Transformer o heurísticas de proyección) sobre las regiones etiquetadas como `Table`.
* **API REST e Indexación en Digesto:** Exponer el pipeline mediante un servicio FastAPI para alimentar directamente un motor de búsqueda semántica (ElasticSearch o RAG vectorial) en el Digesto Electrónico de la UADER.

---

# 8. Referencias

1. **Pfitzmann, B., Auer, C., Dolfi, M., Scheidegger, F., & Staar, P.** (2022). *DocLayNet: A Large-scale Dataset for Document Layout Analysis*. Proceedings of the 28th ACM SIGKDD Conference. arXiv:2206.01062.
2. **Ultralytics**. (2024). *YOLO11: State-of-the-Art Object Detection and Image Segmentation*. https://docs.ultralytics.com/
3. **Smith, R.** (2007). *An Overview of the Tesseract OCR Engine*. Ninth International Conference on Document Analysis and Recognition (ICDAR).
4. **Poppler Development Team**. (2024). *Poppler PDF Rendering Engine*. https://poppler.freedesktop.org/
