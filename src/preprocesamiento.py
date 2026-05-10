import os
from pdf2image import convert_from_path

def pdf_a_imagenes(ruta_pdf, carpeta_salida):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Carpeta creada: {carpeta_salida}")

    if not os.path.exists(ruta_pdf):
        print(f"Error: El archivo PDF no existe en {ruta_pdf}")
        return

    print(f"Cargando el PDF: {ruta_pdf}...")
    
    # Lógica multiplataforma para Poppler
    # En Windows necesitamos la ruta manual, en Linux/Docker suele estar en el PATH global.
    ruta_poppler = r"C:\poppler\Library\bin" if os.name == 'nt' else None
    
    try:
        paginas = convert_from_path(ruta_pdf, poppler_path=ruta_poppler)

        for index, pagina in enumerate(paginas):
            nombre_archivo = os.path.join(carpeta_salida, f"pagina_{index + 1}.jpg")
            pagina.save(nombre_archivo, 'JPEG')
            print(f"[OK] Guardada: {nombre_archivo}")

        print("¡Proceso terminado con éxito!")
    except Exception as e:
        print(f"Error durante la conversión: {e}")

def procesar_todos_los_pdfs(carpeta_entrada, carpeta_salida_base):
    """Recorre todos los PDFs en carpeta_entrada y los convierte a imágenes."""
    if not os.path.exists(carpeta_entrada):
        print(f"Error: La carpeta de entrada no existe: {carpeta_entrada}")
        return

    # Filtramos solo los archivos que terminan en .pdf (ignorando mayúsculas/minúsculas)
    archivos_pdf = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith('.pdf')]

    if not archivos_pdf:
        print(f"No se encontraron archivos PDF en: {carpeta_entrada}")
        return

    print(f"Se encontraron {len(archivos_pdf)} PDF(s) en la carpeta.\n")

    procesados = 0
    salteados = 0

    for archivo in archivos_pdf:
        ruta_pdf = os.path.join(carpeta_entrada, archivo)

        # Creamos una subcarpeta por cada PDF usando su nombre (sin la extensión .pdf)
        nombre_sin_extension = os.path.splitext(archivo)[0]
        carpeta_destino = os.path.join(carpeta_salida_base, f"{nombre_sin_extension}-imagenes")

        # Si la carpeta ya existe y tiene archivos, lo salteamos
        if os.path.exists(carpeta_destino) and os.listdir(carpeta_destino):
            print(f"[SKIP] {archivo} ya fue procesado ({len(os.listdir(carpeta_destino))} imagenes existentes)")
            salteados += 1
            continue

        print(f"--- Procesando: {archivo} ---")
        pdf_a_imagenes(ruta_pdf, carpeta_destino)
        procesados += 1
        print()  # Línea en blanco para separar visualmente cada PDF

    print(f"\n=== Resumen: {procesados} procesado(s), {salteados} salteado(s) ===")


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    carpeta_raw = os.path.join(base_path, "data", "raw")
    carpeta_processed = os.path.join(base_path, "data", "processed")

    procesar_todos_los_pdfs(carpeta_raw, carpeta_processed)