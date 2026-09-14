"""
Módulo de Detección de Layout con YOLO (DocLayNet) y Exportación Estructurada para OCR.

Este módulo procesa páginas de documentos preprocesadas, detecta las regiones
de layout (Text, Section-header, Table, Picture, etc.) y exporta sus coordenadas
estructuradas (bounding boxes, clases, confianza y aptitud para OCR) en formato JSON.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("layout_yolo")

# Configuración de taxonomía DocLayNet y aptitud para extracción OCR
DOCLAYNET_OCR_CONFIG: Dict[str, Dict[str, Any]] = {
    "Text": {"ocr_eligible": True, "ocr_type": "text"},
    "Section-header": {"ocr_eligible": True, "ocr_type": "section_header"},
    "List-item": {"ocr_eligible": True, "ocr_type": "list_item"},
    "Title": {"ocr_eligible": True, "ocr_type": "title"},
    "Table": {"ocr_eligible": True, "ocr_type": "table"},
    "Caption": {"ocr_eligible": True, "ocr_type": "caption"},
    "Page-header": {"ocr_eligible": True, "ocr_type": "page_header"},
    "Page-footer": {"ocr_eligible": True, "ocr_type": "page_footer"},
    "Footnote": {"ocr_eligible": True, "ocr_type": "footnote"},
    "Picture": {"ocr_eligible": False, "ocr_type": "graphic_non_text"},  # Sellos, firmas, logos
    "Formula": {"ocr_eligible": False, "ocr_type": "formula"}
}

EXTENSIONES_IMAGEN = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def resolver_ruta_modelo(nombre_o_ruta: str, base_dir: Path) -> Path:
    """Busca el archivo de pesos del modelo en rutas relativas o absolutas."""
    p = Path(nombre_o_ruta)
    if p.is_file():
        return p

    candidato_base = base_dir / nombre_o_ruta
    if candidato_base.is_file():
        return candidato_base

    # Fallback común si no existe el solicitado
    modelos_disponibles = list(base_dir.glob("*doclaynet*.pt"))
    if modelos_disponibles:
        logger.warning(
            "Modelo '%s' no encontrado directamente. Usando disponible: %s",
            nombre_o_ruta, modelos_disponibles[0].name
        )
        return modelos_disponibles[0]

    raise FileNotFoundError(
        f"No se encontró el modelo YOLO '{nombre_o_ruta}'. "
        f"Asegúrese de que el archivo .pt exista en {base_dir}"
    )


def cargar_modelo(ruta_modelo: Path, device: Optional[str] = None) -> YOLO:
    """Carga y valida el modelo YOLO para Layout Analysis."""
    logger.info("Cargando modelo YOLO: %s...", ruta_modelo.name)
    modelo = YOLO(str(ruta_modelo))
    if device:
        modelo.to(device)
    logger.info("Modelo cargado con éxito. Clases: %s", list(modelo.names.values()))
    return modelo


def detectar_layout_pagina(
    ruta_imagen: Path,
    modelo: YOLO,
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Ejecuta inferencia YOLO sobre una página individual y extrae las regiones estructuradas.
    Ordena las regiones en orden de lectura natural (top-to-bottom, left-to-right).
    """
    results = modelo(
        str(ruta_imagen),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    result = results[0]
    boxes = result.boxes
    nombres_clases = modelo.names

    orig_shape = result.orig_shape  # (height, width)
    alto_orig, ancho_orig = int(orig_shape[0]), int(orig_shape[1])

    regiones_crudas = []
    if boxes is not None and len(boxes) > 0:
        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            cls_name = nombres_clases.get(cls_id, f"class_{cls_id}")
            conf = float(boxes.conf[idx].item())
            xyxy = [float(coord) for coord in boxes.xyxy[idx].tolist()]
            xyxyn = [float(coord) for coord in boxes.xyxyn[idx].tolist()]

            config_ocr = DOCLAYNET_OCR_CONFIG.get(
                cls_name, {"ocr_eligible": True, "ocr_type": "text"}
            )

            regiones_crudas.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "box_xyxy": [round(c, 1) for c in xyxy],
                "box_xyxyn": [round(c, 4) for c in xyxyn],
                "ocr_eligible": config_ocr["ocr_eligible"],
                "ocr_type": config_ocr["ocr_type"],
                "_sort_key": (xyxy[1], xyxy[0])  # y_min para orden de lectura
            })

    # Ordenar por lectura top-down (eje Y primero, luego eje X)
    regiones_crudas.sort(key=lambda r: r["_sort_key"])

    regiones_finales = []
    for i, reg in enumerate(regiones_crudas, start=1):
        reg_copy = {k: v for k, v in reg.items() if k != "_sort_key"}
        reg_copy["id"] = i
        regiones_finales.append(reg_copy)

    total_ocr_eligible = sum(1 for r in regiones_finales if r["ocr_eligible"])

    return {
        "page_file": ruta_imagen.name,
        "width": ancho_orig,
        "height": alto_orig,
        "total_regions": len(regiones_finales),
        "ocr_eligible_regions": total_ocr_eligible,
        "regions": regiones_finales,
        "_result_obj": result
    }


def guardar_visualizacion(
    result_obj,
    ruta_salida: Path
):
    """Guarda la imagen con las bounding boxes dibujadas para control de calidad visual."""
    annotated_bgr = result_obj.plot(line_width=2, font_size=1)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta_salida), annotated_bgr)


def recortar_region(
    imagen_o_ruta: Any,
    box_xyxy: List[float],
    padding: int = 4
) -> np.ndarray:
    """
    Recorta una región de una página dada su bounding box [xmin, ymin, xmax, ymax].
    Incluye un margen de padding para asegurar que ningún carácter al borde sea cortado.
    """
    if isinstance(imagen_o_ruta, (str, Path)):
        img = cv2.imread(str(imagen_o_ruta))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {imagen_o_ruta}")
    elif isinstance(imagen_o_ruta, np.ndarray):
        img = imagen_o_ruta
    else:
        # Compatibilidad con PIL Image
        img = cv2.cvtColor(np.array(imagen_o_ruta), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = box_xyxy

    x1 = max(0, int(round(xmin)) - padding)
    y1 = max(0, int(round(ymin)) - padding)
    x2 = min(w, int(round(xmax)) + padding)
    y2 = min(h, int(round(ymax)) + padding)

    return img[y1:y2, x1:x2]


def procesar_documento(
    carpeta_documento: Path,
    carpeta_salida: Path,
    modelo: YOLO,
    conf_threshold: float = 0.20,
    save_visuals: bool = True,
    base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Procesa todas las páginas de un documento y genera el archivo layout.json estructurado.
    """
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    if base_dir is None:
        base_dir = carpeta_documento.parents[2]

    archivos_imagen = sorted([
        f for f in carpeta_documento.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONES_IMAGEN
    ])

    if not archivos_imagen:
        logger.warning("No se encontraron imágenes en %s", carpeta_documento)
        return {"document": carpeta_documento.name, "pages": []}

    logger.info("Detectando layout en %s (%d páginas)...", carpeta_documento.name, len(archivos_imagen))

    paginas_resultado = []
    conteo_clases: Dict[str, int] = {}
    total_detecciones = 0

    for idx_pag, img_path in enumerate(archivos_imagen, start=1):
        datos_pag = detectar_layout_pagina(
            ruta_imagen=img_path,
            modelo=modelo,
            conf_threshold=conf_threshold
        )

        result_obj = datos_pag.pop("_result_obj")
        datos_pag["page_number"] = idx_pag
        datos_pag["relative_path"] = str(img_path.relative_to(base_dir))

        if save_visuals:
            ruta_vis = carpeta_salida / f"{img_path.stem}_annotated.jpg"
            guardar_visualizacion(result_obj, ruta_vis)
            datos_pag["annotated_image"] = ruta_vis.name

        for reg in datos_pag["regions"]:
            cls = reg["class_name"]
            conteo_clases[cls] = conteo_clases.get(cls, 0) + 1
            total_detecciones += 1

        paginas_resultado.append(datos_pag)

    datos_documento = {
        "document_name": carpeta_documento.name,
        "conf_threshold": conf_threshold,
        "total_pages": len(paginas_resultado),
        "total_detections": total_detecciones,
        "class_distribution": conteo_clases,
        "pages": paginas_resultado
    }

    # Guardar layout.json estructurado por documento
    ruta_json = carpeta_salida / "layout.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(datos_documento, f, indent=2, ensure_ascii=False)

    logger.info(
        "[OK] %s: %d páginas, %d detecciones -> %s",
        carpeta_documento.name, len(paginas_resultado), total_detecciones, ruta_json.name
    )

    return datos_documento


def procesar_todos_los_documentos(
    carpeta_processed: Path,
    carpeta_resultados: Path,
    modelo: YOLO,
    conf_threshold: float = 0.20,
    save_visuals: bool = True,
    filtro_doc: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de detección de layout sobre todos los documentos.
    Genera un índice general manifest.json para que el motor de OCR conozca todas las entradas.
    """
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    if base_dir is None:
        base_dir = carpeta_processed.parent

    subcarpetas = sorted([
        d for d in carpeta_processed.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if filtro_doc:
        subcarpetas = [d for d in subcarpetas if filtro_doc in d.name]

    if not subcarpetas:
        logger.warning("No hay carpetas de documentos a procesar en %s", carpeta_processed)
        return {}

    logger.info("Iniciando detección de layout sobre %d documento(s)...", len(subcarpetas))

    resumen_global: List[Dict[str, Any]] = []
    conteo_global_clases: Dict[str, int] = {}
    total_global_detecciones = 0
    total_global_paginas = 0

    for subfolder in subcarpetas:
        carpeta_salida_doc = carpeta_resultados / subfolder.name
        res_doc = procesar_documento(
            carpeta_documento=subfolder,
            carpeta_salida=carpeta_salida_doc,
            modelo=modelo,
            conf_threshold=conf_threshold,
            save_visuals=save_visuals,
            base_dir=base_dir
        )

        total_global_paginas += res_doc.get("total_pages", 0)
        total_global_detecciones += res_doc.get("total_detections", 0)
        for c, cant in res_doc.get("class_distribution", {}).items():
            conteo_global_clases[c] = conteo_global_clases.get(c, 0) + cant

        resumen_global.append({
            "document_name": res_doc.get("document_name"),
            "total_pages": res_doc.get("total_pages"),
            "total_detections": res_doc.get("total_detections"),
            "layout_file": str((carpeta_salida_doc / "layout.json").relative_to(base_dir)),
            "class_distribution": res_doc.get("class_distribution")
        })

    manifest = {
        "model_file": Path(modelo.model_name if hasattr(modelo, "model_name") else "yolo").name,
        "conf_threshold": conf_threshold,
        "total_documents": len(subcarpetas),
        "total_pages": total_global_paginas,
        "total_detections": total_global_detecciones,
        "global_class_distribution": conteo_global_clases,
        "documents": resumen_global
    }

    ruta_manifest = carpeta_resultados / "manifest.json"
    with open(ruta_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("=== Resumen Layout Analysis ===")
    logger.info("Documentos: %d | Páginas: %d | Detecciones: %d", len(subcarpetas), total_global_paginas, total_global_detecciones)
    logger.info("Manifest guardado en: %s", ruta_manifest)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Inferencia de layout YOLO y exportación estructurada para OCR")
    parser.add_argument("--model", type=str, default="yolov11m-doclaynet.pt", help="Archivo del modelo .pt")
    parser.add_argument("--conf", type=float, default=0.20, help="Umbral de confianza (default: 0.20)")
    parser.add_argument("--output_name", type=str, default=None, help="Nombre de subcarpeta en runs/detect/")
    parser.add_argument("--no_visuals", action="store_true", help="Desactiva el guardado de imágenes con cajas dibujadas")
    parser.add_argument("--doc", type=str, default=None, help="Filtro para procesar un documento específico")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo de cómputo (ej. 'cpu', 'cuda')")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent
    processed_folder = base_path / "data" / "processed"

    ruta_modelo = resolver_ruta_modelo(args.model, base_path)
    nombre_modelo = ruta_modelo.stem

    nombre_carpeta = args.output_name if args.output_name else f"predict_{nombre_modelo}"
    results_folder = base_path / "runs" / "detect" / nombre_carpeta

    modelo = cargar_modelo(ruta_modelo, device=args.device)

    procesar_todos_los_documentos(
        carpeta_processed=processed_folder,
        carpeta_resultados=results_folder,
        modelo=modelo,
        conf_threshold=args.conf,
        save_visuals=not args.no_visuals,
        filtro_doc=args.doc,
        base_dir=base_path
    )


if __name__ == "__main__":
    main()