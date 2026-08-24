import os
import time
import json
import argparse
from collections import defaultdict
from ultralytics import YOLO

def evaluar_modelo(model_path, processed_folder, conf_threshold=0.20, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n=======================================================")
    print(f"🚀 Evaluando modelo: {model_name} ({model_path})")
    print(f"=======================================================")
    
    start_load = time.time()
    model = YOLO(model_path)
    load_time = time.time() - start_load
    
    nombres_clases = model.names
    conteo_global = defaultdict(int)
    confianzas_por_clase = defaultdict(list)
    detalle_por_documento = {}

    total_paginas = 0
    paginas_sin_deteccion = 0
    total_detecciones = 0

    subfolders = sorted([
        d for d in os.listdir(processed_folder)
        if os.path.isdir(os.path.join(processed_folder, d))
    ])

    start_inference = time.time()

    for subfolder in subfolders:
        subfolder_path = os.path.join(processed_folder, subfolder)
        images = sorted([
            f for f in os.listdir(subfolder_path)
            if f.lower().endswith(extensions)
        ])

        if not images:
            continue

        doc_conteo = defaultdict(int)
        doc_confianzas = []
        doc_paginas = len(images)
        doc_sin_deteccion = 0

        for image in images:
            image_path = os.path.join(subfolder_path, image)
            results = model(image_path, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                doc_sin_deteccion += 1
                paginas_sin_deteccion += 1
                total_paginas += 1
                continue

            for box in boxes:
                clase_id = int(box.cls[0])
                confianza = float(box.conf[0])
                nombre_clase = nombres_clases[clase_id]

                doc_conteo[nombre_clase] += 1
                doc_confianzas.append(confianza)

                conteo_global[nombre_clase] += 1
                confianzas_por_clase[nombre_clase].append(confianza)
                total_detecciones += 1

            total_paginas += 1

        doc_total = sum(doc_conteo.values())
        doc_conf_promedio = sum(doc_confianzas) / len(doc_confianzas) if doc_confianzas else 0

        detalle_por_documento[subfolder] = {
            "paginas": doc_paginas,
            "paginas_sin_deteccion": doc_sin_deteccion,
            "total_detecciones": doc_total,
            "confianza_promedio": round(doc_conf_promedio, 3),
            "detecciones_por_clase": dict(doc_conteo)
        }

    total_time = time.time() - start_inference
    time_per_page = (total_time / total_paginas * 1000) if total_paginas > 0 else 0

    all_confs = [c for confs in confianzas_por_clase.values() for c in confs]
    promedio_global = (sum(all_confs) / len(all_confs)) if all_confs else 0

    estadisticas_clases = {}
    for clase, confs in confianzas_por_clase.items():
        estadisticas_clases[clase] = {
            "cantidad": conteo_global[clase],
            "confianza_promedio": round(sum(confs) / len(confs), 3),
            "confianza_minima": round(min(confs), 3),
            "confianza_maxima": round(max(confs), 3)
        }

    return {
        "model_name": model_name,
        "model_file": os.path.basename(model_path),
        "total_paginas": total_paginas,
        "paginas_sin_deteccion": paginas_sin_deteccion,
        "total_detecciones": total_detecciones,
        "confianza_promedio_global": round(promedio_global, 3),
        "tiempo_total_seg": round(total_time, 2),
        "ms_por_pagina": round(time_per_page, 1),
        "conteo_por_clase": dict(conteo_global),
        "estadisticas_clases": estadisticas_clases,
        "detalle_documentos": detalle_por_documento
    }


def main():
    parser = argparse.ArgumentParser(description="Comparador multi-modelo de YOLO para Document Layout Analysis")
    parser.add_argument("--models", nargs="+", default=["yolov10s-doclaynet.pt", "yolov11s-doclaynet.pt", "yolov11m-doclaynet.pt"], help="Lista de modelos .pt a evaluar")
    parser.add_argument("--conf", type=float, default=0.20, help="Umbral de confianza")
    args = parser.parse_args()

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_folder = os.path.join(base_path, "data", "processed")
    output_folder = os.path.join(base_path, "runs", "analisis")
    os.makedirs(output_folder, exist_ok=True)

    resultados = []
    for m_arg in args.models:
        m_path = os.path.join(base_path, m_arg) if not os.path.isabs(m_arg) else m_arg
        if not os.path.exists(m_path):
            print(f"⚠️ Advertencia: Modelo {m_path} no encontrado. Se omite.")
            continue
        res = evaluar_modelo(m_path, processed_folder, conf_threshold=args.conf)
        resultados.append(res)

    if not resultados:
        print("Error: No se evaluó ningún modelo.")
        return

    # Header formatting
    col_w = 20
    header_models = " | ".join([f"{r['model_name']:<{col_w}}" for r in resultados])
    sep_line = "-" * (25 + len(resultados) * (col_w + 3))

    print("\n" + "=" * (25 + len(resultados) * (col_w + 3)))
    print(" 📊 TABLA COMPARATIVA MULTI-MODELO: BENCHMARK DOCLAYNET EN UADER")
    print("=" * (25 + len(resultados) * (col_w + 3)))
    print(f"{'Métrica / Categoría':<25} | {header_models}")
    print(sep_line)
    
    # Métricas generales
    row_pags = " | ".join([f"{r['total_paginas']:<{col_w}}" for r in resultados])
    print(f"{'Páginas evaluadas':<25} | {row_pags}")

    row_dets = " | ".join([f"{r['total_detecciones']:<{col_w}}" for r in resultados])
    print(f"{'Detecciones totales':<25} | {row_dets}")

    row_conf = " | ".join([f"{r['confianza_promedio_global']:<{col_w}.3f}" for r in resultados])
    print(f"{'Confianza promedio':<25} | {row_conf}")

    row_blank = " | ".join([f"{r['paginas_sin_deteccion']:<{col_w}}" for r in resultados])
    print(f"{'Hojas en blanco (0 det)':<25} | {row_blank}")

    row_time = " | ".join([f"{r['tiempo_total_seg']:<{col_w}.2f}s" for r in resultados])
    print(f"{'Tiempo total (seg)':<25} | {row_time}")

    row_speed = " | ".join([f"{r['ms_por_pagina']:<{col_w}.1f}ms" for r in resultados])
    print(f"{'Velocidad (ms/pág)':<25} | {row_speed}")

    print(sep_line)
    print(" Detecciones por Clase:")

    todas_las_clases = sorted(set([c for r in resultados for c in r['conteo_por_clase'].keys()]))
    for c in todas_las_clases:
        vals = []
        for r in resultados:
            cnt = r['conteo_por_clase'].get(c, 0)
            conf = r['estadisticas_clases'].get(c, {}).get('confianza_promedio', 0.0)
            vals.append(f"{cnt:>4} (conf {conf:.2f})")
        row_c = " | ".join([f"{v:<{col_w}}" for v in vals])
        print(f"  • {c:<22} | {row_c}")

    print("=" * (25 + len(resultados) * (col_w + 3)))

    # Guardar reporte JSON
    reporte_comparativo = {
        "umbral_confianza": args.conf,
        "modelos_evaluados": [r["model_name"] for r in resultados],
        "resultados": resultados
    }
    
    out_file = os.path.join(output_folder, f"benchmark_multimodelo_conf_{args.conf:.2f}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reporte_comparativo, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Benchmark multi-modelo guardado en: {out_file}\n")

if __name__ == "__main__":
    main()
