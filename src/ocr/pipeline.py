"""
Pipeline Completo de Extracción y Reconstrucción de Documentos vía OCR.

Consume las detecciones de layout estructuradas (layout.json), recorta las regiones
de texto y tablas, ejecuta Tesseract OCR y reconstruye el documento en formatos
estructurados: JSON, Markdown semántico y Texto Plano.
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.YOLO import recortar_region

try:
    from .engine import MotorOCR
    from .preprocessor import filtrar_regiones_redundantes
except (ImportError, ValueError):
    from src.ocr.engine import MotorOCR
    from src.ocr.preprocessor import filtrar_regiones_redundantes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ocr_pipeline")


def formatear_region_markdown(region: Dict[str, Any]) -> str:
    """Formatea una región de texto detectada a sintaxis Markdown según su clase."""
    texto = region.get("ocr_texto", "").strip()
    if not texto:
        return ""

    ocr_type = region.get("ocr_type", "text")

    if ocr_type == "title":
        return f"# {texto}\n"
    elif ocr_type == "section_header":
        return f"\n### {texto}\n"
    elif ocr_type == "list_item":
        return f"- {texto}"
    elif ocr_type == "table":
        return f"\n```\n[TABLA DETECTADA]\n{texto}\n```\n"
    elif ocr_type == "page_header":
        return f"*{texto}*\n"
    elif ocr_type == "page_footer":
        return f"\n_{texto}_\n"
    else:
        # Párrafo de texto corrido
        return f"{texto}\n"


def procesar_documento_ocr(
    ruta_layout_json: Path,
    carpeta_salida_documento: Path,
    motor_ocr: MotorOCR,
    base_dir: Path,
    filtrar_solapamientos: bool = True,
    padding: int = 4
) -> Dict[str, Any]:
    """
    Procesa todas las regiones aptas para OCR de un documento dado su layout.json.
    """
    if not ruta_layout_json.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de layout: {ruta_layout_json}")

    with open(ruta_layout_json, "r", encoding="utf-8") as f:
        datos_layout = json.load(f)

    doc_name = datos_layout.get("document_name", ruta_layout_json.parent.name)
    paginas = datos_layout.get("pages", [])

    logger.info("Iniciando OCR en: %s (%d páginas)...", doc_name, len(paginas))
    start_time = time.time()

    carpeta_salida_documento.mkdir(parents=True, exist_ok=True)

    paginas_procesadas = []
    lineas_markdown_global = [f"# Documento: {doc_name}\n\n---\n"]
    lineas_texto_plano = []

    total_caracteres_doc = 0
    total_palabras_doc = 0
    confianzas_ocr_doc = []

    for pagina in paginas:
        num_pag = pagina.get("page_number", 1)
        rel_path = pagina.get("relative_path")
        ruta_img = base_dir / rel_path

        regiones = pagina.get("regions", [])

        # Filtrado de cajas redundantes para evitar líneas duplicadas
        if filtrar_solapamientos:
            regiones_a_procesar = filtrar_regiones_redundantes(regiones)
        else:
            regiones_a_procesar = regiones

        lineas_markdown_global.append(f"\n<!-- Página {num_pag} -->\n")
        lineas_texto_plano.append(f"\n--- PÁGINA {num_pag} ---\n")

        regiones_ocr_resultado = []
        texto_pagina = []

        for reg in regiones_a_procesar:
            if not reg.get("ocr_eligible", True):
                continue

            box = reg["box_xyxy"]
            ocr_type = reg.get("ocr_type", "text")

            try:
                recorte = recortar_region(ruta_img, box, padding=padding)
                resultado_ocr = motor_ocr.procesar_recorte(recorte, ocr_type=ocr_type)

                texto = resultado_ocr.get("texto", "")
                conf_ocr = resultado_ocr.get("confianza", 0.0)

                if texto:
                    reg_actualizada = dict(reg)
                    reg_actualizada["ocr_texto"] = texto
                    reg_actualizada["ocr_confianza"] = conf_ocr
                    reg_actualizada["ocr_caracteres"] = resultado_ocr.get("caracteres", 0)

                    regiones_ocr_resultado.append(reg_actualizada)
                    texto_pagina.append(texto)
                    confianzas_ocr_doc.append(conf_ocr)
                    total_caracteres_doc += resultado_ocr.get("caracteres", 0)
                    total_palabras_doc += resultado_ocr.get("palabras_detectadas", 0)

                    # Formatear a Markdown
                    md_fragment = formatear_region_markdown(reg_actualizada)
                    if md_fragment:
                        lineas_markdown_global.append(md_fragment)
                        lineas_texto_plano.append(texto)

            except Exception as exc:
                logger.error("Error al procesar región %d de pág %d en %s: %s", reg.get("id"), num_pag, doc_name, exc)

        paginas_procesadas.append({
            "page_number": num_pag,
            "page_file": pagina.get("page_file"),
            "total_regiones_ocr": len(regiones_ocr_resultado),
            "texto_pagina": "\n".join(texto_pagina),
            "regions": regiones_ocr_resultado
        })

    duracion = time.time() - start_time
    confianza_promedio = (sum(confianzas_ocr_doc) / len(confianzas_ocr_doc)) if confianzas_ocr_doc else 0.0

    # 1. Guardar ocr_results.json estructurado
    salida_json = {
        "document_name": doc_name,
        "idioma_ocr": motor_ocr.idioma,
        "total_pages": len(paginas_procesadas),
        "total_caracteres": total_caracteres_doc,
        "total_palabras": total_palabras_doc,
        "confianza_ocr_promedio": round(confianza_promedio, 2),
        "tiempo_segundos": round(duracion, 2),
        "pages": paginas_procesadas
    }

    ruta_json = carpeta_salida_documento / "ocr_results.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(salida_json, f, indent=2, ensure_ascii=False)

    # 2. Guardar documento.md (Markdown estructurado para lectura humana y LLMs)
    ruta_md = carpeta_salida_documento / "documento.md"
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_markdown_global))

    # 3. Guardar documento.txt (Texto limpio para indexación y búsqueda)
    ruta_txt = carpeta_salida_documento / "documento.txt"
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_texto_plano))

    logger.info(
        "[OK OCR] %s: %d palabras, %d carácteres (conf: %.1f%%) en %.1fs -> %s",
        doc_name, total_palabras_doc, total_caracteres_doc, confianza_promedio, duracion, carpeta_salida_documento.name
    )

    return salida_json


def ejecutar_pipeline_ocr_batch(
    ruta_manifest: Path,
    carpeta_salida_ocr: Path,
    motor_ocr: MotorOCR,
    base_dir: Path,
    filtro_doc: Optional[str] = None,
    filtrar_solapamientos: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta el OCR en lote sobre todos los documentos enumerados en el manifest.json de YOLO.
    """
    if not ruta_manifest.is_file():
        raise FileNotFoundError(f"No se encontró el manifest de layout: {ruta_manifest}")

    with open(ruta_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    documentos = manifest.get("documents", [])
    if filtro_doc:
        documentos = [d for d in documentos if filtro_doc in d["document_name"]]

    carpeta_salida_ocr.mkdir(parents=True, exist_ok=True)
    logger.info("Iniciando batch OCR sobre %d documento(s)...", len(documentos))

    resumen_documentos = []
    total_caracteres_batch = 0
    total_palabras_batch = 0
    inicio_batch = time.time()

    for doc_info in documentos:
        rel_layout = doc_info["layout_file"]
        ruta_layout = base_dir / rel_layout
        doc_name = doc_info["document_name"]

        salida_doc = carpeta_salida_ocr / doc_name
        res_doc = procesar_documento_ocr(
            ruta_layout_json=ruta_layout,
            carpeta_salida_documento=salida_doc,
            motor_ocr=motor_ocr,
            base_dir=base_dir,
            filtrar_solapamientos=filtrar_solapamientos
        )

        total_caracteres_batch += res_doc.get("total_caracteres", 0)
        total_palabras_batch += res_doc.get("total_palabras", 0)

        resumen_documentos.append({
            "document_name": doc_name,
            "pages": res_doc.get("total_pages"),
            "caracteres": res_doc.get("total_caracteres"),
            "palabras": res_doc.get("total_palabras"),
            "confianza_promedio": res_doc.get("confianza_ocr_promedio"),
            "output_dir": str(salida_doc.relative_to(base_dir))
        })

    tiempo_total = time.time() - inicio_batch

    manifest_ocr = {
        "idioma": motor_ocr.idioma,
        "total_documentos": len(resumen_documentos),
        "total_caracteres": total_caracteres_batch,
        "total_palabras": total_palabras_batch,
        "tiempo_total_segundos": round(tiempo_total, 2),
        "documentos": resumen_documentos
    }

    ruta_manifest_ocr = carpeta_salida_ocr / "ocr_manifest.json"
    with open(ruta_manifest_ocr, "w", encoding="utf-8") as f:
        json.dump(manifest_ocr, f, indent=2, ensure_ascii=False)

    logger.info("=== Batch OCR Finalizado con Éxito ===")
    logger.info(
        "Procesados: %d docs | Palabras: %d | Caracteres: %d | Tiempo: %.1fs",
        len(resumen_documentos), total_palabras_batch, total_caracteres_batch, tiempo_total
    )
    logger.info("Manifiesto OCR guardado en: %s", ruta_manifest_ocr)

    return manifest_ocr


def main():
    parser = argparse.ArgumentParser(description="Pipeline de OCR sobre regiones de layout detectadas")
    parser.add_argument("--manifest", type=str, default=None, help="Ruta al manifest.json de YOLO")
    parser.add_argument("--output", type=str, default="runs/ocr", help="Directorio de salida para texto y OCR")
    parser.add_argument("--lang", type=str, default="spa", help="Idioma de Tesseract (default: spa)")
    parser.add_argument("--doc", type=str, default=None, help="Filtro para procesar un documento específico")
    parser.add_argument("--no_filter_overlaps", action="store_true", help="Desactiva el filtrado de cajas anidadas")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent

    # Localizar manifest.json por defecto
    if args.manifest:
        ruta_manifest = Path(args.manifest) if Path(args.manifest).is_absolute() else base_path / args.manifest
    else:
        candidatos = list((base_path / "runs" / "detect").glob("*/manifest.json"))
        if not candidatos:
            raise FileNotFoundError("No se encontró ningún manifest.json en runs/detect/. Ejecute primero YOLO.")
        ruta_manifest = sorted(candidatos)[-1]
        logger.info("Usando manifest más reciente: %s", ruta_manifest)

    carpeta_salida = Path(args.output) if Path(args.output).is_absolute() else base_path / args.output
    motor = MotorOCR(idioma=args.lang)

    ejecutar_pipeline_ocr_batch(
        ruta_manifest=ruta_manifest,
        carpeta_salida_ocr=carpeta_salida,
        motor_ocr=motor,
        base_dir=base_path,
        filtro_doc=args.doc,
        filtrar_solapamientos=not args.no_filter_overlaps
    )


if __name__ == "__main__":
    main()
