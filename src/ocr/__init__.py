"""Paquete OCR para procesamiento y extracción de texto estructurado."""
from .config import configurar_entorno_tesseract
from .preprocessor import mejorar_imagen_para_ocr, filtrar_regiones_redundantes
from .engine import MotorOCR
from .pipeline import procesar_documento_ocr, ejecutar_pipeline_ocr_batch

__all__ = [
    "configurar_entorno_tesseract",
    "mejorar_imagen_para_ocr",
    "filtrar_regiones_redundantes",
    "MotorOCR",
    "procesar_documento_ocr",
    "ejecutar_pipeline_ocr_batch"
]
