"""
Módulo de Preprocesamiento: Conversión de PDFs a imágenes de alta resolución para OCR.

Características:
- Optimización de memoria O(1) vía streaming directo a disco con Poppler (paths_only=True).
- Salida en formato lossless PNG a 300 DPI nativos.
- Zero-padding en nomenclatura de archivos (ej. pagina_0001.png).
- Procesamiento atómico para evitar datasets inconsistentes o parcialmente procesados.
- Verificación estricta de integridad de páginas mediante pdfinfo.
"""

import os
import re
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional, List

from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("preprocesamiento")

POPPLER_DEFAULT_WINDOWS = r"C:\poppler\Library\bin"


def obtener_ruta_poppler() -> Optional[str]:
    """Determina la ruta de binarios de Poppler según SO y variables de entorno."""
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.name == "nt" and os.path.exists(POPPLER_DEFAULT_WINDOWS):
        return POPPLER_DEFAULT_WINDOWS
    return None


def extraer_numero_pagina(ruta_archivo: Path) -> int:
    """
    Extrae el número de página de los nombres temporales generados por Poppler.
    Ejemplo: 'uuid-01.png' -> 1, 'uuid-10.png' -> 10.
    """
    coincidencia = re.search(r"-(\d+)\.[a-zA-Z0-9]+$", ruta_archivo.name)
    return int(coincidencia.group(1)) if coincidencia else 0


def pdf_a_imagenes(
    ruta_pdf: Path,
    carpeta_salida: Path,
    dpi: int = 300,
    formato: str = "png",
    hilos: int = 4
) -> bool:
    """
    Convierte un documento PDF en imágenes de alta resolución (una por página).

    Utiliza paths_only=True para volcar directamente el renderizado a disco,
    garantizando un consumo de memoria constante O(1) independientemente del
    tamaño o número de páginas del documento.
    """
    if not ruta_pdf.is_file():
        logger.error("El archivo PDF no existe: %s", ruta_pdf)
        return False

    ruta_poppler = obtener_ruta_poppler()

    # 1. Obtener número total de páginas esperado según metadatos del PDF
    try:
        info_pdf = pdfinfo_from_path(str(ruta_pdf), poppler_path=ruta_poppler)
        total_paginas_esperadas = int(info_pdf.get("Pages", 0))
        if total_paginas_esperadas <= 0:
            logger.error("El documento %s reporta 0 páginas.", ruta_pdf.name)
            return False
    except Exception as exc:
        logger.error("No se pudo leer la metadata del PDF %s: %s", ruta_pdf.name, exc)
        return False

    extension = formato.lower().lstrip(".")

    # 2. Comprobar si ya fue procesado íntegramente
    if carpeta_salida.is_dir():
        imagenes_existentes = list(carpeta_salida.glob(f"pagina_*.{extension}"))
        if len(imagenes_existentes) == total_paginas_esperadas:
            logger.info(
                "[SKIP] %s ya procesado (%d/%d páginas presentes).",
                ruta_pdf.name, len(imagenes_existentes), total_paginas_esperadas
            )
            return True
        elif imagenes_existentes:
            logger.warning(
                "[REINTENTO] %s incompleto (%d/%d páginas). Limpiando y reprocesando...",
                ruta_pdf.name, len(imagenes_existentes), total_paginas_esperadas
            )
            shutil.rmtree(carpeta_salida)

    # 3. Directorio temporal de staging para garantizar atomicidad
    directorio_staging = carpeta_salida.parent / f".staging_{carpeta_salida.name}"
    if directorio_staging.exists():
        shutil.rmtree(directorio_staging)
    directorio_staging.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Convirtiendo: %s (%d páginas @ %d DPI, formato %s)...",
        ruta_pdf.name, total_paginas_esperadas, dpi, extension.upper()
    )

    try:
        # Renderizado en streaming directo a disco (sin cargar mapas de bits en Python RAM)
        rutas_temporales_str: List[str] = convert_from_path(
            pdf_path=str(ruta_pdf),
            dpi=dpi,
            output_folder=str(directorio_staging),
            fmt=extension,
            paths_only=True,
            thread_count=hilos,
            poppler_path=ruta_poppler
        )

        archivos_generados = [Path(p) for p in rutas_temporales_str]

        # Ordenar de forma natural según el número de página devuelto por Poppler
        archivos_generados.sort(key=extraer_numero_pagina)

        if len(archivos_generados) != total_paginas_esperadas:
            raise RuntimeError(
                f"Discrepancia en páginas: se esperaban {total_paginas_esperadas}, "
                f"pero se generaron {len(archivos_generados)} archivos."
            )

        # Renombrar con zero-padding (ej. pagina_0001.png) para orden lexicográfico estricto
        for idx, temp_file in enumerate(archivos_generados, start=1):
            destino_final = directorio_staging / f"pagina_{idx:04d}.{extension}"
            temp_file.rename(destino_final)

        # Promover directorio de staging a destino final de forma atómica
        if carpeta_salida.exists():
            shutil.rmtree(carpeta_salida)
        directorio_staging.rename(carpeta_salida)

        logger.info(
            "[OK] %s convertido con éxito -> %s (%d páginas)",
            ruta_pdf.name, carpeta_salida.name, total_paginas_esperadas
        )
        return True

    except Exception as exc:
        logger.exception("Fallo en la conversión de %s: %s", ruta_pdf.name, exc)
        if directorio_staging.exists():
            shutil.rmtree(directorio_staging)
        return False


def procesar_todos_los_pdfs(
    carpeta_entrada: Path,
    carpeta_salida_base: Path,
    dpi: int = 300,
    formato: str = "png",
    hilos: int = 4
) -> dict:
    """Recorre todos los PDFs en carpeta_entrada y los procesa a imágenes de alta calidad."""
    if not carpeta_entrada.exists():
        logger.error("La carpeta de entrada no existe: %s", carpeta_entrada)
        return {"total": 0, "exitosos": 0, "fallidos": 0}

    carpeta_salida_base.mkdir(parents=True, exist_ok=True)

    archivos_pdf = sorted([f for f in carpeta_entrada.glob("*.pdf") if f.is_file()])

    if not archivos_pdf:
        logger.warning("No se encontraron archivos PDF en: %s", carpeta_entrada)
        return {"total": 0, "exitosos": 0, "fallidos": 0}

    logger.info("Encontrados %d documento(s) PDF en %s", len(archivos_pdf), carpeta_entrada)

    exitosos = 0
    fallidos = 0

    for ruta_pdf in archivos_pdf:
        nombre_sin_extension = ruta_pdf.stem
        carpeta_destino = carpeta_salida_base / f"{nombre_sin_extension}-imagenes"
        
        ok = pdf_a_imagenes(
            ruta_pdf=ruta_pdf,
            carpeta_salida=carpeta_destino,
            dpi=dpi,
            formato=formato,
            hilos=hilos
        )
        if ok:
            exitosos += 1
        else:
            fallidos += 1

    logger.info(
        "=== Resumen: %d procesado(s) exitosamente, %d fallido(s) de %d total ===",
        exitosos, fallidos, len(archivos_pdf)
    )
    return {"total": len(archivos_pdf), "exitosos": exitosos, "fallidos": fallidos}


def main():
    parser = argparse.ArgumentParser(description="Conversión de PDFs a imágenes optimizadas para OCR")
    parser.add_argument("--dpi", type=int, default=300, help="Resolución en DPI (default: 300)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpeg", "tiff"], help="Formato de imagen (default: png)")
    parser.add_argument("--threads", type=int, default=4, help="Cantidad de hilos para Poppler (default: 4)")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    carpeta_raw = base_path / "data" / "raw"
    carpeta_processed = base_path / "data" / "processed"

    procesar_todos_los_pdfs(
        carpeta_entrada=carpeta_raw,
        carpeta_salida_base=carpeta_processed,
        dpi=args.dpi,
        formato=args.format,
        hilos=args.threads
    )


if __name__ == "__main__":
    main()