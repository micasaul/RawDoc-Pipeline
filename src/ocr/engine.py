"""
Motor de Reconocimiento Óptico de Caracteres (OCR) basado en Pytesseract y Tesseract 5.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pytesseract

try:
    from .config import configurar_entorno_tesseract
    from .preprocessor import mejorar_imagen_para_ocr
except (ImportError, ValueError):
    from src.ocr.config import configurar_entorno_tesseract
    from src.ocr.preprocessor import mejorar_imagen_para_ocr

logger = logging.getLogger("ocr_engine")


class MotorOCR:
    """Clase envolvente para la ejecución de OCR sobre recortes de documento."""

    def __init__(self, idioma: str = "spa"):
        self.idioma = idioma
        self.tesseract_bin = configurar_entorno_tesseract()
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_bin

        # Validar disponibilidad del idioma solicitado
        try:
            idiomas_disponibles = pytesseract.get_languages()
            if self.idioma not in idiomas_disponibles:
                logger.warning(
                    "Idioma '%s' no encontrado en tessdata (%s). Usando 'eng'.",
                    self.idioma, idiomas_disponibles
                )
                self.idioma = "eng"
        except Exception as exc:
            logger.warning("No se pudieron listar los idiomas de Tesseract: %s", exc)

    def seleccionar_config_psm(self, ocr_type: str) -> str:
        """
        Selecciona el Page Segmentation Mode (PSM) óptimo según el tipo de layout detectado.
        - PSM 6: Bloque uniforme de texto (ideal para párrafos de texto corrido).
        - PSM 4: Columna de texto con tamaños variables (útil para listas y tablas).
        - PSM 7: Línea de texto única (ideal para títulos o encabezados cortos).
        """
        if ocr_type in ("title", "section_header", "page_header", "page_footer"):
            return "--psm 6 --oem 1"
        elif ocr_type == "table":
            return "--psm 6 --oem 1 -c preserve_interword_spaces=1"
        elif ocr_type == "list_item":
            return "--psm 6 --oem 1"
        else:
            return "--psm 6 --oem 1"

    def procesar_recorte(
        self,
        recorte_bgr: np.ndarray,
        ocr_type: str = "text",
        metodo_mejora: str = "grayscale"
    ) -> Dict[str, Any]:
        """
        Ejecuta OCR sobre una imagen recortada de una región y retorna el texto y su confianza.
        """
        if recorte_bgr is None or recorte_bgr.size == 0:
            return {"texto": "", "confianza": 0.0, "caracteres": 0}

        imagen_preprocesada = mejorar_imagen_para_ocr(recorte_bgr, metodo=metodo_mejora)
        tess_config = self.seleccionar_config_psm(ocr_type)

        try:
            # Obtener datos detallados con nivel de confianza por palabra
            data = pytesseract.image_to_data(
                imagen_preprocesada,
                lang=self.idioma,
                config=tess_config,
                output_type=pytesseract.Output.DICT
            )

            palabras = []
            confianzas = []

            for i in range(len(data["text"])):
                texto_palabra = data["text"][i].strip()
                conf_val = float(data["conf"][i])

                if texto_palabra and conf_val >= 0:
                    palabras.append(texto_palabra)
                    confianzas.append(conf_val)

            texto_completo = " ".join(palabras).strip()
            confianza_promedio = (sum(confianzas) / len(confianzas)) if confianzas else 0.0

            return {
                "texto": texto_completo,
                "confianza": round(confianza_promedio, 2),
                "caracteres": len(texto_completo),
                "palabras_detectadas": len(palabras)
            }

        except Exception as exc:
            logger.error("Error al ejecutar Tesseract sobre el recorte: %s", exc)
            return {"texto": "", "confianza": 0.0, "caracteres": 0, "error": str(exc)}
