"""
Módulo de Preprocesamiento de Imágenes para OCR y Filtrado Geométrico de Regiones.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


def mejorar_imagen_para_ocr(
    imagen_bgr: np.ndarray,
    metodo: str = "grayscale"
) -> np.ndarray:
    """
    Aplica técnicas de mejora de imagen sobre un recorte antes de pasarlo a Tesseract.
    
    Métodos disponibles:
    - 'grayscale': Conversión a escala de grises (óptimo para Tesseract 5 LSTM).
    - 'binarize': Binarización con umbral de Otsu.
    - 'clahe': Ecualización de histograma adaptativa (útil para escaneos con sombras).
    """
    if len(imagen_bgr.shape) == 2:
        gris = imagen_bgr
    else:
        gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    if metodo == "binarize":
        # Filtro Gaussiano leve para reducir ruido antes de Otsu
        desenfocado = cv2.GaussianBlur(gris, (3, 3), 0)
        _, binarizada = cv2.threshold(desenfocado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarizada

    elif metodo == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gris)

    # Grayscale estándar
    return gris


def calcular_interseccion_y_areas(
    box_a: List[float],
    box_b: List[float]
) -> Tuple[float, float, float]:
    """Calcula el área de intersección y las áreas individuales de dos cajas [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    return inter_area, area_a, area_b


def filtrar_regiones_redundantes(
    regiones: List[Dict[str, Any]],
    umbral_contencion: float = 0.85,
    umbral_iou: float = 0.70
) -> List[Dict[str, Any]]:
    """
    Elimina cajas delimitadoras duplicadas o anidadas que representan el mismo texto.
    Si una caja pequeña está contenida en más del 85% dentro de una caja mayor, se preserva
    la caja contenedora para evitar extraer líneas de texto repetidas.
    """
    if len(regiones) <= 1:
        return regiones

    # Ordenar por área descendente para priorizar cajas envolventes
    def area_box(r):
        b = r["box_xyxy"]
        return (b[2] - b[0]) * (b[3] - b[1])

    regiones_ordenadas = sorted(regiones, key=area_box, reverse=True)
    indices_a_descartar = set()

    for i in range(len(regiones_ordenadas)):
        if i in indices_a_descartar:
            continue

        box_i = regiones_ordenadas[i]["box_xyxy"]

        for j in range(i + 1, len(regiones_ordenadas)):
            if j in indices_a_descartar:
                continue

            box_j = regiones_ordenadas[j]["box_xyxy"]
            inter, area_i, area_j = calcular_interseccion_y_areas(box_i, box_j)

            if area_j <= 0:
                indices_a_descartar.add(j)
                continue

            # Si la caja menor (j) está casi totalmente dentro de la mayor (i)
            ratio_contencion = inter / area_j
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0

            if ratio_contencion >= umbral_contencion or iou >= umbral_iou:
                indices_a_descartar.add(j)

    regiones_filtradas = [
        r for idx, r in enumerate(regiones_ordenadas)
        if idx not in indices_a_descartar
    ]

    # Restaurar orden de lectura original (top-to-bottom)
    regiones_filtradas.sort(key=lambda r: (r["box_xyxy"][1], r["box_xyxy"][0]))

    # Re-enumerar IDs secuencialmente
    for nuevo_id, reg in enumerate(regiones_filtradas, start=1):
        reg["id"] = nuevo_id

    return regiones_filtradas
