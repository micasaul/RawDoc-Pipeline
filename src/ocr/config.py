"""
Configuración del Motor OCR (Tesseract): Resolución de binarios y datos de idiomas.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ocr_config")

# Rutas estándar en Linux/Docker y rutas locales de usuario
RUTAS_CANDIDATAS_BINARIO = [
    os.environ.get("TESSERACT_CMD"),
    shutil.which("tesseract"),
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    str(Path.home() / ".local" / "usr" / "bin" / "tesseract"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

RUTAS_CANDIDATAS_TESSDATA = [
    os.environ.get("TESSDATA_PREFIX"),
    "/usr/share/tesseract-ocr/5/tessdata",
    "/usr/share/tesseract-ocr/4.00/tessdata",
    "/usr/share/tessdata",
    str(Path.home() / ".local" / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata"),
    str(Path.home() / ".local" / "usr" / "share" / "tessdata"),
    r"C:\Program Files\Tesseract-OCR\tessdata"
]

RUTAS_CANDIDATAS_LIB = [
    str(Path.home() / ".local" / "usr" / "lib" / "x86_64-linux-gnu"),
    str(Path.home() / ".local" / "usr" / "lib"),
]


def configurar_entorno_tesseract() -> str:
    """
    Configura las variables de entorno necesarias para pytesseract
    y retorna la ruta ejecutable de Tesseract.
    """
    # 1. Configurar LD_LIBRARY_PATH para bibliotecas en espacio de usuario
    for ruta_lib in RUTAS_CANDIDATAS_LIB:
        if Path(ruta_lib).is_dir():
            ld_actual = os.environ.get("LD_LIBRARY_PATH", "")
            if ruta_lib not in ld_actual:
                os.environ["LD_LIBRARY_PATH"] = f"{ruta_lib}:{ld_actual}" if ld_actual else ruta_lib

    # 2. Configurar TESSDATA_PREFIX
    for candidata in RUTAS_CANDIDATAS_TESSDATA:
        if candidata and Path(candidata).is_dir():
            os.environ["TESSDATA_PREFIX"] = str(Path(candidata).resolve())
            break

    # 3. Detectar binario ejecutable
    for candidata in RUTAS_CANDIDATAS_BINARIO:
        if candidata and Path(candidata).is_file():
            logger.info("Binario de Tesseract detectado en: %s", candidata)
            return str(Path(candidata).resolve())

    raise FileNotFoundError(
        "No se encontró el ejecutable de Tesseract en el sistema ni en ~/.local/usr/bin/tesseract."
    )
