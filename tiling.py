# =============================================================
# tiling.py — same tiling logic as your original tiling (1).py,
# adapted to be called as a function instead of a Colab script.
# =============================================================
import os
import rasterio
from rasterio.windows import Window


def tile_tif(input_path: str, output_dir: str, tile_size: int = 512, stride: int = 512):
    """
    Identical logic to your original tile_tif. Splits a large GeoTIFF
    into tile_size x tile_size tiles, skipping incomplete edge tiles.
    Returns the list of output tile paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_paths = []

    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        print(f"[tiling] Image size: {width} x {height}")

        count = 0
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                window = Window(x, y, tile_size, tile_size)
                transform = src.window_transform(window)
                tile = src.read(window=window)

                if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                    continue

                out_path = os.path.join(output_dir, f"tile_{x}_{y}.tif")
                with rasterio.open(
                    out_path, "w",
                    driver="GTiff",
                    height=tile.shape[1],
                    width=tile.shape[2],
                    count=tile.shape[0],
                    dtype=tile.dtype,
                    crs=src.crs,
                    transform=transform,
                    compress="lzw",
                ) as dst:
                    dst.write(tile)

                out_paths.append(out_path)
                count += 1

        print(f"[tiling] Total tiles created: {count}")

    return out_paths
