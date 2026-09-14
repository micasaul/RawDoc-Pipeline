"""Módulo de Detección de Layout de Documentos con YOLO."""
from .yolo import (
    cargar_modelo,
    detectar_layout_pagina,
    recortar_region,
    procesar_documento,
    procesar_todos_los_documentos
)

__all__ = [
    "cargar_modelo",
    "detectar_layout_pagina",
    "recortar_region",
    "procesar_documento",
    "procesar_todos_los_documentos"
]
