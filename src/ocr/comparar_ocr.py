"""
Script de Evaluación Experimental: OCR Completo vs. OCR Selectivo por Layout

Compara cuantitativamente el enfoque tradicional (OCR a la página completa)
frente al pipeline propuesto (detección de layout con YOLO11m + OCR selectivo).

Métricas evaluadas:
1. Reducción de ruido estructural (caracteres no alfanuméricos / garabatos en firmas y sellos).
2. Tasa de confianza promedio de Tesseract.
3. Caracteres y palabras extraídas.
4. Coherencia y estructuración (Markdown estructurado vs texto plano caótico).
"""

import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pytesseract

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.engine import MotorOCR
from src.ocr.pipeline import procesar_documento_ocr
from src.YOLO import recortar_region


def calcular_metricas_ruido(texto: str) -> Dict[str, Any]:
    """
    Calcula indicadores de ruido en el texto extraído:
    - Caracteres extraños o sospechosos (símbolos repetidos, caracteres no latinos)
    - Porcentaje de caracteres de puntuación/símbolos sobre el total
    """
    if not texto.strip():
        return {"total_caracteres": 0, "total_palabras": 0, "caracteres_ruido": 0, "pct_ruido": 0.0}

    total_caracteres = len(texto)
    palabras = texto.split()
    total_palabras = len(palabras)

    # Identificar caracteres que típicamente genera Tesseract sobre firmas/sellos
    # Ejemplos: |, ~, _, ^, §, °, ¡, ¿ sueltos, barras, etc.
    caracteres_ruido = len(re.findall(r"[~|_\^§\\\/\[\]\{\}<>»«—–#@$]", texto))
    pct_ruido = (caracteres_ruido / total_caracteres * 100) if total_caracteres > 0 else 0.0

    return {
        "total_caracteres": total_caracteres,
        "total_palabras": total_palabras,
        "caracteres_ruido": caracteres_ruido,
        "pct_ruido": round(pct_ruido, 2)
    }


def ejecutar_ocr_completo_pagina(ruta_imagen: Path, lang: str = "spa") -> Dict[str, Any]:
    """Ejecuta OCR sobre la página completa sin segmentación de layout (enfoque tradicional)."""
    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")

    start_time = time.time()
    # Tesseract en modo automático de página completa (PSM 3)
    data = pytesseract.image_to_data(
        img,
        lang=lang,
        config="--psm 3 --oem 1",
        output_type=pytesseract.Output.DICT
    )
    tiempo = time.time() - start_time

    palabras = []
    confianzas = []
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        c = float(data["conf"][i])
        if t and c >= 0:
            palabras.append(t)
            confianzas.append(c)

    texto_completo = " ".join(palabras)
    conf_promedio = (sum(confianzas) / len(confianzas)) if confianzas else 0.0
    metricas = calcular_metricas_ruido(texto_completo)

    return {
        "texto": texto_completo,
        "confianza_promedio": round(conf_promedio, 2),
        "tiempo_segundos": round(tiempo, 2),
        **metricas
    }


def ejecutar_ocr_selectivo_pagina(
    ruta_imagen: Path,
    regiones_pagina: List[Dict[str, Any]],
    motor_ocr: MotorOCR,
    padding: int = 8
) -> Dict[str, Any]:
    """Ejecuta OCR selectivo únicamente sobre regiones textuales filtradas."""
    start_time = time.time()
    palabras_totales = []
    confianzas = []
    texto_por_bloques = []
    texto_limpio_para_metricas = []

    regiones_filtradas = [r for r in regiones_pagina if r.get("ocr_eligible", True)]
    regiones_excluidas = [r for r in regiones_pagina if not r.get("ocr_eligible", True)]

    for reg in regiones_filtradas:
        box = reg["box_xyxy"]
        ocr_type = reg.get("ocr_type", "text")
        recorte = recortar_region(ruta_imagen, box, padding=padding)
        res = motor_ocr.procesar_recorte(recorte, ocr_type=ocr_type)

        txt = res.get("texto", "").strip()
        conf = res.get("confianza", 0.0)

        if txt:
            texto_por_bloques.append(f"[{reg.get('class_name')}] {txt}")
            texto_limpio_para_metricas.append(txt)
            palabras_totales.extend(txt.split())
            if conf > 0:
                confianzas.append(conf)

    tiempo = time.time() - start_time
    texto_completo = "\n\n".join(texto_por_bloques)
    texto_evaluacion = "\n\n".join(texto_limpio_para_metricas)
    conf_promedio = (sum(confianzas) / len(confianzas)) if confianzas else 0.0
    metricas = calcular_metricas_ruido(texto_evaluacion)

    return {
        "texto": texto_completo,
        "confianza_promedio": round(conf_promedio, 2),
        "tiempo_segundos": round(tiempo, 2),
        "regiones_procesadas": len(regiones_filtradas),
        "regiones_excluidas_firmas_sellos": len(regiones_excluidas),
        **metricas
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comparador experimental: OCR Completo vs. OCR Selectivo")
    parser.add_argument("--doc", type=str, default=None, help="Nombre de carpeta de documento específico (ej. ORD-185-imagenes)")
    parser.add_argument("--page", type=str, default=None, help="Archivo de página específica (ej. pagina_0001.png)")
    parser.add_argument("--padding", type=int, default=8, help="Padding en píxeles para recorte de regiones (default: 8)")
    args = parser.parse_args()

    base_dir = PROJECT_ROOT
    runs_dir = base_dir / "runs"
    layout_base = runs_dir / "detect" / "predict_yolov11m-doclaynet"

    if args.doc and args.page:
        casos_de_prueba = [{
            "doc": args.doc,
            "page_file": args.page,
            "descripcion": f"Caso personalizado: {args.doc} -> {args.page}"
        }]
    else:
        # 3 páginas de prueba de referencia (firmas, membrete/título, ordenanza final)
        casos_de_prueba = [
            {
                "doc": "Res-CS-054-25-27-03-2025-imagenes",
                "page_file": "pagina_0002.png",
                "descripcion": "Resolución con firmas manuscritas y sello al pie (Pág 2)"
            },
            {
                "doc": "ORD-185-imagenes",
                "page_file": "pagina_0001.png",
                "descripcion": "Ordenanza con encabezado normativo y membrete superior (Pág 1)"
            },
            {
                "doc": "ORD-185-imagenes",
                "page_file": "pagina_0008.png",
                "descripcion": "Página final con múltiples firmas institucionales (Pág 8)"
            }
        ]

    print("\n" + "=" * 90)
    print(" 🔬 EXPERIMENTO COMPARATIVO: OCR COMPLETO vs. OCR SELECTIVO POR LAYOUT")
    print("=" * 90)

    motor = MotorOCR(idioma="spa")
    resultados_comparativa = []

    for caso in casos_de_prueba:
        doc_name = caso["doc"]
        page_name = caso["page_file"]
        ruta_img = base_dir / "data" / "processed" / doc_name / page_name
        ruta_layout = layout_base / doc_name / "layout.json"

        if not ruta_img.exists() or not ruta_layout.exists():
            print(f"⚠️ Omitiendo {doc_name}/{page_name}: archivos no encontrados.")
            continue

        with open(ruta_layout, "r", encoding="utf-8") as f:
            layout_data = json.load(f)

        # Buscar regiones de la página específica
        regiones = []
        for pag in layout_data.get("pages", []):
            if pag.get("page_file") == page_name:
                regiones = pag.get("regions", [])
                break

        print(f"\n📄 Analizando: {doc_name} -> {page_name}")
        print(f"   Descripción: {caso['descripcion']}")
        print(f"   Regiones totales detectadas por YOLO: {len(regiones)}")

        # 1. OCR Completo
        res_completo = ejecutar_ocr_completo_pagina(ruta_img, lang="spa")

        # 2. OCR Selectivo (con padding configurable)
        res_selectivo = ejecutar_ocr_selectivo_pagina(ruta_img, regiones, motor, padding=args.padding)

        diff_ruido = res_completo["caracteres_ruido"] - res_selectivo["caracteres_ruido"]
        pct_reduccion_ruido = (diff_ruido / res_completo["caracteres_ruido"] * 100) if res_completo["caracteres_ruido"] > 0 else 0.0

        item = {
            "documento": doc_name,
            "pagina": page_name,
            "descripcion": caso["descripcion"],
            "ocr_completo": res_completo,
            "ocr_selectivo": res_selectivo,
            "reduccion_ruido_pct": round(pct_reduccion_ruido, 2),
            "caracteres_ruido_eliminados": diff_ruido
        }
        resultados_comparativa.append(item)

        print(f"   [OCR Completo]  Palabras: {res_completo['total_palabras']:>4} | Confianza: {res_completo['confianza_promedio']:>5.1f}% | Caracteres de ruido: {res_completo['caracteres_ruido']:>3} ({res_completo['pct_ruido']}%)")
        print(f"   [OCR Selectivo] Palabras: {res_selectivo['total_palabras']:>4} | Confianza: {res_selectivo['confianza_promedio']:>5.1f}% | Caracteres de ruido: {res_selectivo['caracteres_ruido']:>3} ({res_selectivo['pct_ruido']}%)")
        print(f"   ✅ Reducción de ruido en firmas/sellos: {diff_ruido} caracteres eliminados ({pct_reduccion_ruido:.1f}% menos ruido)")

    # Tabla resumen final
    print("\n" + "=" * 90)
    print(" 📊 RESUMEN CUANTITATIVO COMPARATIVO")
    print("=" * 90)
    print(f"{'Caso evaluado':<35} | {'OCR Completo (Ruido)':<22} | {'OCR Selectivo (Ruido)':<22} | {'Mejora Ruido'}")
    print("-" * 90)
    for r in resultados_comparativa:
        c1 = f"{r['ocr_completo']['caracteres_ruido']} chars ({r['ocr_completo']['pct_ruido']}%)"
        c2 = f"{r['ocr_selectivo']['caracteres_ruido']} chars ({r['ocr_selectivo']['pct_ruido']}%)"
        print(f"{r['descripcion'][:35]:<35} | {c1:<22} | {c2:<22} | -{r['reduccion_ruido_pct']:.1f}%")
    print("=" * 90)

    # Guardar reporte JSON del experimento
    out_json = runs_dir / "analisis" / "experimento_ocr_completo_vs_selectivo.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultados_comparativa, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Reporte experimental guardado en: {out_json}\n")


if __name__ == "__main__":
    main()
