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

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_prueba = os.path.join(base_path, "data", "raw", "ORD-CS-186-25.pdf")
    carpeta_destino = os.path.join(base_path, "data", "processed", "ORD-186-imagenes")
    
    pdf_a_imagenes(pdf_prueba, carpeta_destino)