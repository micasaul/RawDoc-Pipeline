import os
import json
from collections import defaultdict
from ultralytics import YOLO

import argparse

parser = argparse.ArgumentParser(description="Análisis de resultados de inferencia YOLO")
parser.add_argument("--model", type=str, default="yolov11m-doclaynet.pt", help="Archivo del modelo .pt")
parser.add_argument("--conf", type=float, default=0.20, help="Umbral de confianza")
args = parser.parse_args()

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_folder = os.path.join(base_path, "data", "processed")
output_folder = os.path.join(base_path, "runs", "analisis")
model_file = args.model
model_path = os.path.join(base_path, model_file) if not os.path.isabs(model_file) else model_file

EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
UMBRAL_CONFIANZA = args.conf

os.makedirs(output_folder, exist_ok=True)

print(f"Cargando modelo: {model_file}...")
model = YOLO(model_path)
nombres_clases = model.names
print(f"Clases del modelo: {list(nombres_clases.values())}\n")

conteo_global = defaultdict(int)
confianzas_por_clase = defaultdict(list)
detalle_por_documento = {}

total_paginas = 0
paginas_sin_deteccion = 0
total_detecciones = 0
promedio_global = 0.0

subfolders = sorted([
    d for d in os.listdir(processed_folder)
    if os.path.isdir(os.path.join(processed_folder, d))
])

print(f"Analizando {len(subfolders)} documento(s)...\n")
print("=" * 65)

for subfolder in subfolders:
    subfolder_path = os.path.join(processed_folder, subfolder)
    images = sorted([
        f for f in os.listdir(subfolder_path)
        if f.lower().endswith(EXTENSIONES_IMAGEN)
    ])

    if not images:
        continue

    doc_conteo = defaultdict(int)
    doc_confianzas = []
    doc_paginas = len(images)
    doc_sin_deteccion = 0

    for image in images:
        image_path = os.path.join(subfolder_path, image)
        results = model(image_path, conf=UMBRAL_CONFIANZA, verbose=False)
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

    print(f"\n📄 {subfolder}")
    print(f"   Páginas: {doc_paginas} | Sin detección: {doc_sin_deteccion}")
    print(f"   Detecciones: {doc_total} | Confianza promedio: {doc_conf_promedio:.2f}")

    for clase, cantidad in sorted(doc_conteo.items(), key=lambda x: -x[1]):
        print(f"     - {clase}: {cantidad}")

print("\n" + "=" * 65)

print("\n📊 RESUMEN GLOBAL")
print(f"   Documentos analizados:   {len(subfolders)}")
print(f"   Páginas totales:         {total_paginas}")
print(f"   Páginas sin detección:   {paginas_sin_deteccion}")
print(f"   Detecciones totales:     {total_detecciones}")

if total_detecciones > 0:
    conf_todas = []
    for vals in confianzas_por_clase.values():
        conf_todas.extend(vals)
    promedio_global = sum(conf_todas) / len(conf_todas)
    print(f"   Confianza promedio:      {promedio_global:.3f}")

print("\n   Detecciones por categoría:")
for clase, cantidad in sorted(conteo_global.items(), key=lambda x: -x[1]):
    confs = confianzas_por_clase[clase]
    prom = sum(confs) / len(confs)
    minimo = min(confs)
    maximo = max(confs)
    print(f"     {clase:20s}  →  {cantidad:4d} detecciones  "
          f"(conf: prom={prom:.2f}  min={minimo:.2f}  max={maximo:.2f})")

estadisticas_clases = {}
for clase in sorted(conteo_global.keys()):
    confs = confianzas_por_clase[clase]
    estadisticas_clases[clase] = {
        "cantidad": conteo_global[clase],
        "confianza_promedio": round(sum(confs) / len(confs), 3),
        "confianza_minima": round(min(confs), 3),
        "confianza_maxima": round(max(confs), 3)
    }

nombre_modelo_base = os.path.splitext(os.path.basename(model_path))[0]

reporte = {
    "configuracion": {
        "modelo": os.path.basename(model_path),
        "umbral_confianza": UMBRAL_CONFIANZA,
        "clases_modelo": list(nombres_clases.values())
    },
    "resumen_global": {
        "documentos_analizados": len(subfolders),
        "paginas_totales": total_paginas,
        "paginas_sin_deteccion": paginas_sin_deteccion,
        "detecciones_totales": total_detecciones,
        "confianza_promedio_global": round(promedio_global, 3) if total_detecciones > 0 else None
    },
    "estadisticas_por_clase": estadisticas_clases,
    "detalle_por_documento": detalle_por_documento
}

ruta_reporte = os.path.join(output_folder, f"reporte_{nombre_modelo_base}_conf_{UMBRAL_CONFIANZA:.2f}.json")
with open(ruta_reporte, "w", encoding="utf-8") as f:
    json.dump(reporte, f, indent=2, ensure_ascii=False)

print(f"\n✅ Reporte guardado en: {ruta_reporte}")
