from ultralytics import YOLO

img_path = 'data/processed/ORD-185-imagenes/pagina_7.jpg'

# Usamos el modelo preentrenado de DocLayNet
model = YOLO("yolov10s-doclaynet.pt")

print(f"Iniciando inferencia en: {img_path}")


results = model(img_path, save=True, conf=0.20)

for result in results:
    print(f"Proceso terminado. Revisá la carpeta: {result.save_dir}")