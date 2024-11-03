import os
import io
from rembg import remove
from PIL import Image

def remove_background(image_path, output_path, output_format="PNG", quality=100):
    # Leer la imagen
    with open(image_path, 'rb') as input_file:
        input_image = input_file.read()

    # Quitar el fondo
    output_image = remove(input_image)

    # Convertir los bytes de la imagen sin fondo a imagen de Pillow
    img_no_bg = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # Crear una nueva imagen con fondo blanco
    background = Image.new("RGBA", img_no_bg.size, (255, 255, 255, 255))
    
    # Combinar la imagen sin fondo con el fondo blanco
    combined_image = Image.alpha_composite(background, img_no_bg)

    # Convertir a formato RGB para guardar como JPG o PNG (sin canal alfa)
    final_image = combined_image.convert("RGB")

    # Guardar la imagen resultante con la calidad ajustada
    if output_format == "JPEG":
        final_image.save(output_path, format=output_format, quality=quality)  # Ajustar calidad si es JPEG
    else:
        final_image.save(output_path, format=output_format)

def process_images(input_folder, output_folder, output_format="PNG", quality=100):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Inicializar contador
    counter = 1

    # Procesar todas las imágenes en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            
            # Generar el nuevo nombre con numeración en formato JPG
            new_filename = f"{counter}.jpg"
            
            output_path = os.path.join(output_folder, new_filename)
            remove_background(image_path, output_path, output_format, quality)
            print(f"Procesada: {filename}")
            
            # Incrementar el contador
            counter += 1

# Rutas de la carpeta de entrada y salida
input_folder = r"S:/U/9S\AprendizajeAutomatico/dataset_papas/tomadas/68"
output_folder = r"S:/U/9S/AprendizajeAutomatico/dataset_papas/limpias/V68"

# Ejecutar el procesamiento de imágenes
process_images(input_folder, output_folder, output_format="PNG", quality=100)  # Cambia a "JPEG" si prefieres JPEG con calidad ajustada