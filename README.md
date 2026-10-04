# RawDoc-Pipeline: Análisis Estructural y OCR Selectivo de Documentos Normativos (UADER) 📄🔍

Pipeline integral de **Document AI** diseñado para procesar actos administrativos escaneados (ordenanzas y resoluciones del Consejo Superior de la UADER). Combina detección de layout (**YOLO11m-DocLayNet**), post-procesamiento heurístico y **OCR selectivo** con **Tesseract 5** (`spa`) para eliminar ruido de firmas/sellos y reconstruir el documento en **Markdown semántico** y **JSON estructurado**.

---

## 🏗️ Arquitectura del Pipeline

```
    [PDFs en data/raw/]
            │
            ▼ 1. Preprocesamiento (src/preprocesamiento.py)
    [Imágenes PNG 300 DPI en data/processed/]
            │
            ▼ 2. Detección de Layout (src/YOLO/yolo.py)
    [BBoxes ordenados + layout.json en runs/detect/]
            │
            ▼ 3. Post-procesamiento Heurístico (src/ocr/preprocessor.py)
    [Filtrado de Picture (firmas/sellos) + Supresión de solapamientos (IoU)]
            │
            ▼ 4. OCR Selectivo y Reconstrucción (src/ocr/pipeline.py)
    [Salidas en runs/ocr/: documento.md, documento.txt y ocr_results.json]
```

---

## 📁 Estructura del Repositorio

* `data/raw/`: Documentos PDF originales del Consejo Superior (10 archivos, 148 páginas).
* `data/processed/`: Imágenes renderizadas a 300 DPI nativos con zero-padding (`pagina_0001.png`).
* `docs/`:
  * `INFORME.md`: **Informe Técnico Final** con fundamentación, benchmark de modelos, análisis de errores y conclusiones.
  * `ground_truth.md`: Definición formal del dataset de referencia y validación manual.
* `runs/`:
  * `detect/predict_yolov11m-doclaynet/`: Imágenes anotadas con bounding boxes y metadatos `layout.json`.
  * `ocr/`: Resultados de extracción de texto estructurado (`documento.md`, `ocr_results.json`, `ocr_manifest.json`).
  * `analisis/`: Reportes de benchmark multi-modelo y experimentos comparativos.
* `src/`:
  * `preprocesamiento.py`: Conversión de PDFs a imágenes 300 DPI con streaming O(1) de memoria.
  * `YOLO/yolo.py`: Inferencia de layout con YOLO11m y exportación estructurada.
  * `comparar_modelos.py`: Herramienta de benchmarking multi-modelo (YOLOv10 vs YOLO11s/m/l).
  * `ocr/`: Paquete de OCR selectivo (motor Tesseract, heurísticas geométricas, pipeline y comparador).

---

## 🚀 Guía de Ejecución Rápida con Docker

El proyecto está completamente contenerizado y reproducible.

### 1. Iniciar el Contenedor
```bash
docker compose up -d
```

### 2. Ejecutar el Pipeline Completo Paso a Paso

#### Paso 1: Preprocesamiento de PDFs a 300 DPI
Convierte los PDFs de `data/raw/` en imágenes PNG de alta resolución:
```bash
docker exec RawDoc-Pipeline python3 /app/src/preprocesamiento.py
```

#### Paso 2: Detección de Layout con YOLO11m
Detecta regiones (títulos, párrafos, tablas, encabezados y firmas) y exporta `layout.json`:
```bash
docker exec RawDoc-Pipeline python3 /app/src/YOLO/yolo.py --model /app/yolov11m-doclaynet.pt
```

#### Paso 3: Extracción OCR Selectiva y Reconstrucción en Markdown
Filtra firmas y sellos (`Picture`), limpia solapamientos y extrae el texto en español:
```bash
docker exec RawDoc-Pipeline python3 /app/src/ocr/pipeline.py --lang spa
```
> Las salidas finales se generan en `runs/ocr/<documento>/documento.md` (Markdown estructurado) y `ocr_results.json`.

---

## 🔬 Scripts de Evaluación y Benchmarking

### Comparativa: OCR Completo vs. OCR Selectivo (Demostración de reducción de ruido)
```bash
docker exec RawDoc-Pipeline python3 /app/src/ocr/comparar_ocr.py
```

### Benchmark Multi-Modelo YOLO (DocLayNet)
```bash
docker exec RawDoc-Pipeline python3 /app/src/comparar_modelos.py --models /app/yolov11m-doclaynet.pt
```

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.10** en entorno Linux contenedorizado (Debian Bookworm).
* **Ultralytics YOLO11** (Modelo preentrenado sobre DocLayNet).
* **Tesseract OCR 5** (`tesseract-ocr-spa` + `pytesseract`).
* **Poppler** (`pdf2image`) para renderizado rasterizado de alta precisión.
* **OpenCV & Pillow** para transformaciones espaciales y mejoras de imagen.
