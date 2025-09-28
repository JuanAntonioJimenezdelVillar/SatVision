import math
import os
import rasterio
import numpy as np
from PIL import Image
from rasterio.windows import Window
import subprocess

PIXEL_LIMIT = 10000000
MIN_STD_DEV_THRESHOLD = 10

def extract_gsd(img_path):
    try:
        with rasterio.open(img_path) as src:
            res_x = src.transform[0]  # pixel width
            res_y = -src.transform[4]  # pixel height
            gsd = (abs(res_x) + abs(res_y)) / 2
            print(f"GSD extracted using rasterio: {gsd:.6f} meters/pixel")
            return gsd

    except Exception as e:
        print(f"Error extracting GSD from {img_path}: {str(e)}")
        return None
    
def is_img_informative(img_path, std_dev_threshold):
    try:
        img_pil = Image.open(img_path).convert('L') # Convertir a escala de grises
        img_array = np.array(img_pil)
        std_dev = np.std(img_array)
        return std_dev >= std_dev_threshold

    except Exception as e:
        print(f"Error analizando la informatividad de la tesela {img_path}: {e}")
        return True

def tif_to_png(tif_path, output_dir=None):
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    print(f"Procesando TIF: {tif_path} -> Base name: {base_name}")
    
    # Determinar el directorio de salida para las teselas/imagen
    if output_dir:
        # Las teselas se guardarán directamente en output_dir
        # Si output_dir es, por ejemplo, 'data/iSAID_patches/test/images',
        # las teselas se guardarán como 'data/iSAID_patches/test/images/P000_tile_0_0.png'
        os.makedirs(output_dir, exist_ok=True)
        effective_output_dir = output_dir
    else:
        # Las teselas se guardarán en el mismo directorio que el TIF original
        effective_output_dir = os.path.dirname(tif_path)
        if not effective_output_dir: # Si tif_path es solo un nombre de archivo
            effective_output_dir = "."

    # Calcular la dimensión lateral máxima para las teselas cuadradas
    # para que tile_side * tile_side <= PIXEL_LIMIT
    tile_side_length = math.floor(math.sqrt(PIXEL_LIMIT))
    if tile_side_length == 0: # Evitar tile_side_length de 0 si PIXEL_LIMIT es < 1
        print(f"Error: PIXEL_LIMIT ({PIXEL_LIMIT}) es demasiado bajo para crear teselas.")
        return None

    generated_png_path = None

    try:
        with rasterio.open(tif_path) as src:
            original_width = src.width
            original_height = src.height

            # Si la imagen original es más pequeña o igual que una tesela, convertirla entera
            if original_width <= tile_side_length and original_height <= tile_side_length:
                # print(f"Imagen {tif_path} es suficientemente pequeña, convirtiendo a un solo PNG.")
                single_png_path = os.path.join(effective_output_dir, f"{base_name}.png")
                cmd = [
                    "gdal_translate", "-of", "PNG",
                    "-co", "NUM_THREADS=ALL_CPUS",
                    tif_path, single_png_path
                ]
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode == 0:
                    if is_img_informative(single_png_path, MIN_STD_DEV_THRESHOLD):
                        generated_png_path = single_png_path
                        print(f'Imagen {generated_png_path} es informativa, guardada como PNG.')
                    else:
                        try:
                            os.remove(single_png_path)
                        except OSError as e:
                            print(f"Error al eliminar PNG no informativo {single_png_path}: {e}")
                else:
                    print(f"Error convirtiendo {tif_path} a un solo PNG:")
                    print(f"Stderr: {result.stderr.strip()}")
                    return None # Falló la conversión
            else:
                # La imagen necesita ser teselada
                for r, y_offset in enumerate(range(0, original_height, tile_side_length)):
                    for c, x_offset in enumerate(range(0, original_width, tile_side_length)):
                        current_tile_width = min(tile_side_length, original_width - x_offset)
                        current_tile_height = min(tile_side_length, original_height - y_offset)

                        if current_tile_width == 0 or current_tile_height == 0:
                            continue # Omitir teselas vacías en los bordes

                        tile_filename = f"{base_name}_tile_r{r}_c{c}.png"
                        tile_output_path = os.path.join(effective_output_dir, tile_filename)
                        
                        cmd_tile = [
                            "gdal_translate", "-of", "PNG",
                            "-co", "NUM_THREADS=ALL_CPUS",
                            "-srcwin", str(x_offset), str(y_offset), str(current_tile_width), str(current_tile_height),
                            tif_path, tile_output_path
                        ]
                        result_tile = subprocess.run(cmd_tile, check=False, capture_output=True, text=True)

                        if result_tile.returncode == 0:
                            if is_img_informative(tile_output_path, MIN_STD_DEV_THRESHOLD):
                                generated_png_path = tile_output_path
                            else:
                                try:
                                    os.remove(tile_output_path)
                                except OSError as e:
                                    print(f"Error al eliminar PNG no informativo {tile_output_path}: {e}")
                        else:
                            print(f"Error generando tesela {tile_output_path} desde {tif_path}:")
                            print(f"Stderr: {result_tile.stderr.strip()}")

                            return None 
        
        if not generated_png_path: # Si no se generó nada (ej. imagen original 0x0)
            print(f"Advertencia: No se generaron PNGs para {tif_path}.")
            return [] 
        
        return generated_png_path

    except rasterio.errors.RasterioIOError as e:
        print(f"Error abriendo TIF {tif_path} con Rasterio: {str(e)}")
        return None
    except FileNotFoundError:
        print("Error: gdal_translate no encontrado. ¿Está GDAL instalado y en PATH?")
        return None
    except Exception as e:
        print(f"Error inesperado en tif_to_png para {tif_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_gsd_and_convert(tif_path, output_dir=None):
    gsd = extract_gsd(tif_path)
    png_path = tif_to_png(tif_path, output_dir)
    print(f"PNG path: {png_path}, GSD: {gsd}")
    return png_path, gsd    

