import os
from osgeo import gdal

PIXEL_LIMIT = 5_000_000
input_path = 'data/iSAID/img/images/P00.tif'

def split_tif_to_tiles(input_path, tile_dir=None, pixel_limit=PIXEL_LIMIT):
    ds = gdal.Open(input_path)
    if ds is None:
        print("No se pudo abrir la imagen.")
        return

    width = ds.RasterXSize
    height = ds.RasterYSize
    total_pixels = width * height

    print(f"Dimensiones: {width}x{height} ({total_pixels} píxeles)")

    if total_pixels <= pixel_limit:
        print("La imagen no supera el límite, no se parte en teselas.")
        return

    # Calcular tamaño de tesela cuadrada
    tile_size = int(pixel_limit ** 0.5)
    print(f"Tamaño de tesela: {tile_size}x{tile_size}")

    if tile_dir is None:
        tile_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    tile_num = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            tile_path = os.path.join(tile_dir, f"{base_name}_tile_{tile_num}.tif")
            gdal.Translate(
                tile_path,
                ds,
                srcWin=[x, y, w, h],
                format="GTiff"
            )
            print(f"Tesela guardada: {tile_path} ({w}x{h})")
            tile_num += 1

    ds = None

if __name__ == "__main__":
    split_tif_to_tiles('data/iSAID/img/P00.tif')