# Proyecto OCR UADER 📄🔍

Este proyecto tiene como objetivo procesar documentos PDF, convertirlos en imágenes y extraer su contenido mediante OCR (Reconocimiento Óptico de Caracteres).

## Estructura del Proyecto
- `src/`: Código fuente.
- `data/raw/`: PDFs originales (no se suben al repo).
- `data/processed/`: Imágenes generadas (no se suben al repo).
- `requirements.txt`: Dependencias del proyecto.
- `Dockerfile`: Configuración para entorno contenedorizado.

## Configuración del Entorno

### 1. Requisitos Locales (Windows)
Para correr el proyecto fuera de Docker en Windows, necesitas:
- **Python 3.10+**
- **Poppler:** Descargar y extraer en `C:\poppler`. El script está configurado para buscar en `C:\poppler\Library\bin`.
- **Tesseract OCR:** Instalar en el sistema para realizar el reconocimiento de texto.

### 2. Instalación de Dependencias
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Uso de Docker
Para no tener que escribir comandos largos ni reconstruir la imagen constantemente, usamos `docker-compose`. Este método ya monta automáticamente las carpetas de código y datos.

```bash
# 1. Levantar el contenedor en segundo plano (solo la primera vez tardará en construir)
docker compose up -d

# 2. Entrar a la terminal del contenedor
docker exec -it ocr_uader_dev bash

# 3. Cuando termines de trabajar en el día, para apagarlo:
docker compose down
```

## Desarrollo
Una vez dentro del contenedor (o en tu entorno local), ejecuta:
```bash
python src/preprocesamiento.py
```
