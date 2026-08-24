import os
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Inferencia de detección de layout con YOLO")
parser.add_argument("--model", type=str, default="yolov11m-doclaynet.pt", help="Archivo del modelo .pt")
parser.add_argument("--conf", type=float, default=0.20, help="Umbral de confianza")
parser.add_argument("--output_name", type=str, default=None, help="Nombre de subcarpeta dentro de runs/detect/")
args = parser.parse_args()

# Rutas base (relativas a la raíz del proyecto)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
processed_folder = os.path.join(base_path, "data", "processed")

model_file = args.model
model_path = os.path.join(base_path, model_file) if not os.path.isabs(model_file) else model_file
nombre_modelo = os.path.splitext(os.path.basename(model_path))[0]

folder_name = args.output_name if args.output_name else f"predict_{nombre_modelo}"
results_folder = os.path.join(base_path, "runs", "detect", folder_name)

# Extensiones de imagen soportadas
EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# Cargar modelo
print(f"Cargando modelo: {model_path}...")
model = YOLO(model_path)

# Verificar que la carpeta processed existe
if not os.path.exists(processed_folder):
    print(f"Error: La carpeta no existe: {processed_folder}")
    exit(1)

# Recorrer todas las subcarpetas dentro de processed
subfolders = sorted([
    d for d in os.listdir(processed_folder)
    if os.path.isdir(os.path.join(processed_folder, d))
])

if not subfolders:
    print(f"No se encontraron subcarpetas en: {processed_folder}")
    exit(1)

print(f"Se encontraron {len(subfolders)} subcarpeta(s) en processed.\n")

total_images = 0
total_errors = 0

for subfolder in subfolders:
    subfolder_path = os.path.join(processed_folder, subfolder)

    # Filtrar solo archivos de imagen
    images = sorted([
        f for f in os.listdir(subfolder_path)
        if f.lower().endswith(EXTENSIONES_IMAGEN)
    ])

    if not images:
        print(f"[SKIP] {subfolder}: no contiene imágenes.")
        continue

    # Crear carpeta de salida con la estructura: runs/detect/predict/<subfolder>/
    output_folder = os.path.join(results_folder, subfolder)
    os.makedirs(output_folder, exist_ok=True)

    print(f"--- Procesando: {subfolder} ({len(images)} imágenes) ---")

    for image in images:
        image_path = os.path.join(subfolder_path, image)

        try:
            print(f"  Inferencia en: {image}...", end=" ")
            results = model(image_path, save=True, conf=args.conf, project=output_folder, name=".", exist_ok=True)
            print("OK")
            total_images += 1
        except Exception as e:
            print(f"ERROR: {e}")
            total_errors += 1

    print()

print(f"=== Resumen: {total_images} imagen(es) procesada(s), {total_errors} error(es) ===")
print(f"Resultados guardados en: {results_folder}")