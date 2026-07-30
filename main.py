# =============================================================
# main.py — FastAPI backend for the village-mapping pipeline.
#
# Run with:
#   uvicorn main:app --host 0.0.0.0 --port 8000
#
# Endpoints:
#   POST /upload/tiff   -> queues a GeoTIFF for tile->infer->GPKG pipeline
#   POST /upload/png    -> queues a plain PNG for mask-overlay-only inference
#   GET  /jobs           -> list all jobs (history)
#   GET  /jobs/{id}       -> single job status (poll this while running)
#   GET  /jobs/{id}/geojson -> all vector layers as GeoJSON (for map display)
#   GET  /jobs/{id}/gpkg     -> download the GPKG file (open in QGIS)
#   GET  /jobs/{id}/overlay  -> download the PNG mask overlay (no-geo path)
# =============================================================
import os
import shutil
import threading
import time
import json
import base64
import io

import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform as warp_transform
from rasterio.enums import Resampling
import geopandas as gpd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from jobs import store, JobStatus
from model import load_model, get_device
import tiling
import inference as inf

# =========================
# CONFIG — adjust paths to where your .pth files live on the Mac
# =========================
SEG_CKPT = os.environ.get("SEG_CKPT", "./checkpoints/best_model.pth")
ROOF_CKPT = os.environ.get("ROOF_CKPT", "./checkpoints/best_roof_model.pth")
WORK_DIR = os.environ.get("WORK_DIR", "./work")
UPLOAD_DIR = os.path.join(WORK_DIR, "uploads")
TILES_DIR = os.path.join(WORK_DIR, "tiles")
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")

for d in [UPLOAD_DIR, TILES_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = get_device()
print(f"[main] Using device: {DEVICE}")

app = FastAPI(title="Village Mapping Pipeline")

# Allow any device on the LAN to call this from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loaded once at startup, shared across requests
MODEL = None


@app.on_event("startup")
def _load_model_on_startup():
    global MODEL
    if not os.path.exists(SEG_CKPT) or not os.path.exists(ROOF_CKPT):
        print(f"[main] WARNING: checkpoint files not found at {SEG_CKPT} / {ROOF_CKPT}")
        print("[main] Set SEG_CKPT and ROOF_CKPT env vars to the correct paths.")
        return
    MODEL = load_model(SEG_CKPT, ROOF_CKPT, DEVICE)


# =========================
# WORKER (single background thread processes queued jobs one at a time —
# safe for a single GPU/MPS device, avoids contention)
# =========================
def _worker_loop():
    while True:
        job_id = store.next_queued()
        if job_id is None:
            time.sleep(0.5)
            continue
        job = store.get(job_id)
        if job is None:
            continue
        try:
            if job.kind == "tiff":
                _process_tiff_job(job)
            else:
                _process_png_job(job)
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.message = f"Failed: {e}"
            print(f"[worker] Job {job.id} failed: {e}")


threading.Thread(target=_worker_loop, daemon=True).start()


def _build_base_mosaic_png(upload_path: str, out_png_path: str, max_dim: int = 2048):
    """
    Downsamples the original GeoTIFF to a PNG + computes its WGS84 bounds,
    so the frontend can show it as a Leaflet ImageOverlay underneath the
    vector layers — i.e. "real image with inference layered on top", same
    idea as a QGIS basemap + layer stack, flattened for the browser.
    """
    with rasterio.open(upload_path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))

        data = src.read(
            [1, 2, 3] if src.count >= 3 else [1, 1, 1],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
        )
        img = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        Image.fromarray(img, mode="RGB").save(out_png_path)

        if src.crs is not None:
            bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        else:
            bounds_wgs84 = src.bounds  # best effort, may not be lat/lon

    # Leaflet wants [[south, west], [north, east]]
    minx, miny, maxx, maxy = bounds_wgs84
    return [[miny, minx], [maxy, maxx]]


def _process_tiff_job(job):
    if MODEL is None:
        raise RuntimeError("Model not loaded — check checkpoint paths on server startup.")

    upload_path = os.path.join(UPLOAD_DIR, f"{job.id}_{job.filename}")
    tile_dir = os.path.join(TILES_DIR, job.id)
    out_gpkg = os.path.join(OUTPUT_DIR, f"{job.id}.gpkg")
    out_base_png = os.path.join(OUTPUT_DIR, f"{job.id}_base.png")

    # 1. Tile
    job.status = JobStatus.TILING
    job.message = "Splitting image into tiles..."
    job.progress = 5
    tile_paths = tiling.tile_tif(upload_path, tile_dir)
    if not tile_paths:
        raise RuntimeError("No tiles produced — image may be smaller than tile size (512x512).")

    # 1b. Base mosaic for "overlay on real image" view (downsampled, georeferenced)
    try:
        base_bounds = _build_base_mosaic_png(upload_path, out_base_png)
        job.base_image_path = out_base_png
        job.base_image_bounds = base_bounds
    except Exception as e:
        print(f"[worker] Base mosaic build failed (non-fatal): {e}")
        job.base_image_path = None
        job.base_image_bounds = None

    # 2. Infer each tile
    job.status = JobStatus.INFERRING
    all_feats = {k: [] for k in ["buildings", "roads", "water", "bridges", "utilities"]}
    crs_out = None
    n = len(tile_paths)
    processed = skipped = blank_skipped = 0
    flagged_tiles = []   # tiles where mean confidence on detected features was low

    for i, tile_path in enumerate(tile_paths):
        job.progress = 5 + (i / n) * 80
        job.message = f"Running inference on tile {i+1}/{n}..."
        try:
            with rasterio.open(tile_path) as src:
                image = np.transpose(
                    src.read([1, 2, 3]) if src.count >= 3 else src.read([1, 1, 1]),
                    (1, 2, 0),
                ).astype(np.uint8)
                tile_tf = src.transform
                tile_crs = src.crs
                tile_bounds = src.bounds
                if crs_out is None:
                    crs_out = tile_crs
        except Exception:
            skipped += 1
            continue

        if inf.is_blank_tile(image):
            blank_skipped += 1
            skipped += 1
            continue

        seg_pred, roof_pred, seg_conf = inf.predict_tile(MODEL, image, DEVICE, skip_blank_check=True)
        if seg_pred is None:
            skipped += 1
            continue

        conf_info = inf.confidence_summary(seg_pred, seg_conf)
        if conf_info["is_low_confidence"]:
            # report tile center in WGS84 so the frontend can show a marker
            try:
                if tile_crs is not None:
                    cx = (tile_bounds.left + tile_bounds.right) / 2
                    cy = (tile_bounds.bottom + tile_bounds.top) / 2
                    lon, lat = warp_transform(tile_crs, "EPSG:4326", [cx], [cy])
                    center = [lat[0], lon[0]]
                else:
                    center = None
            except Exception:
                center = None

            flagged_tiles.append({
                "tile": os.path.basename(tile_path),
                "mean_confidence": conf_info["mean_confidence"],
                "per_class": conf_info["per_class"],
                "center": center,
            })

        feats = inf.extract_features(seg_pred, roof_pred, tile_tf)
        for k, v in feats.items():
            all_feats[k].extend(v)
        processed += 1

    # 3. Build GDFs + write GPKG (same area-in-m2 logic as your original script)
    job.status = JobStatus.BUILDING_OUTPUT
    job.message = "Building vector layers and GeoPackage..."
    job.progress = 90

    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)

    stats = {}
    geojson_layers = {}

    for layer, feat_list in all_feats.items():
        if feat_list:
            gdf = gpd.GeoDataFrame(feat_list, crs=crs_out)
            gdf = gdf[gdf.geometry.is_valid].copy()
        else:
            cols = ["geometry", "class_id", "class_name"]
            if layer == "buildings":
                cols += ["roof_type", "roof_label"]
            gdf = gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs_out)

        if len(gdf) > 0 and gdf.crs is not None and gdf.crs.is_geographic:
            bounds = gdf.total_bounds
            lon_mid = (bounds[0] + bounds[2]) / 2
            lat_mid = (bounds[1] + bounds[3]) / 2
            utm_zone = int((lon_mid + 180) / 6) + 1
            utm_epsg = 32600 + utm_zone if lat_mid >= 0 else 32700 + utm_zone
            gdf_utm = gdf.to_crs(epsg=utm_epsg)
            gdf["area_m2"] = gdf_utm.geometry.area.round(2)
        elif len(gdf) > 0:
            gdf["area_m2"] = gdf.geometry.area.round(2)
        else:
            gdf["area_m2"] = []

        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")

        stats[layer] = {
            "count": int(len(gdf)),
            "area_m2": float(gdf["area_m2"].sum()) if len(gdf) else 0.0,
        }

        # Also produce WGS84 GeoJSON for the web map (Leaflet needs lat/lon)
        if len(gdf) > 0:
            gdf_wgs84 = gdf.to_crs(epsg=4326)
            geojson_layers[layer] = json.loads(gdf_wgs84.to_json())
        else:
            geojson_layers[layer] = {"type": "FeatureCollection", "features": []}

    job.gpkg_path = out_gpkg
    job.geojson_layers = geojson_layers
    job.stats = stats
    job.flagged_tiles = flagged_tiles
    job.status = JobStatus.DONE
    job.message = (f"Done — {processed} tiles processed, {skipped} skipped "
                   f"({blank_skipped} blank/edge), {len(flagged_tiles)} flagged low-confidence.")
    job.progress = 100
    job.finished_at = __import__("datetime").datetime.utcnow().isoformat()

    # Bundle the actual flagged tile .tif files into a downloadable zip
    # BEFORE wiping tile_dir — this is what /jobs/{id}/flagged_tiles_zip serves.
    if flagged_tiles:
        import zipfile
        flagged_zip_path = os.path.join(OUTPUT_DIR, f"{job.id}_flagged_tiles.zip")
        flagged_names = {ft["tile"] for ft in flagged_tiles}
        with zipfile.ZipFile(flagged_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in flagged_names:
                fpath = os.path.join(tile_dir, fname)
                if os.path.exists(fpath):
                    zf.write(fpath, arcname=fname)
        job.flagged_tiles_zip_path = flagged_zip_path

    # cleanup raw tiles to save disk (GPKG + geojson + flagged zip already extracted)
    shutil.rmtree(tile_dir, ignore_errors=True)


def _process_png_job(job):
    if MODEL is None:
        raise RuntimeError("Model not loaded — check checkpoint paths on server startup.")

    upload_path = os.path.join(UPLOAD_DIR, f"{job.id}_{job.filename}")
    out_overlay = os.path.join(OUTPUT_DIR, f"{job.id}_overlay.png")
    out_raw = os.path.join(OUTPUT_DIR, f"{job.id}_raw.png")
    out_mask_only = os.path.join(OUTPUT_DIR, f"{job.id}_mask_only.png")

    job.status = JobStatus.INFERRING
    job.message = "Running inference on image..."
    job.progress = 30

    img = Image.open(upload_path).convert("RGB")
    image_rgb = np.array(img)
    img.save(out_raw)  # keep the unmodified original for the "Inference Only" / no-overlay toggle

    seg_pred, roof_pred, seg_conf = inf.predict_tile(MODEL, image_rgb, DEVICE, skip_blank_check=True)

    job.status = JobStatus.BUILDING_OUTPUT
    job.progress = 80
    job.message = "Building overlay..."

    if seg_pred is None:
        # Only happens now if the model call itself errored, not from the blank-tile
        # heuristic (disabled for direct photo uploads — see skip_blank_check above).
        h, w = image_rgb.shape[:2]
        seg_pred = np.zeros((h, w), dtype=np.int16)
        seg_conf = np.ones((h, w), dtype=np.float32)

    # Pixel-level stats (no real-world area possible without geo-reference)
    stats = {}
    for cls_id, name in inf.ID2LABEL.items():
        if cls_id == 0:
            continue
        px_count = int((seg_pred == cls_id).sum())
        stats[name.lower()] = {"pixel_count": px_count}

    conf_info = inf.confidence_summary(seg_pred, seg_conf)
    job.flagged_tiles = [{
        "tile": job.filename,
        "mean_confidence": conf_info["mean_confidence"],
        "per_class": conf_info["per_class"],
        "center": None,
    }] if conf_info["is_low_confidence"] else []

    # Two outputs: blended (mask on photo) and mask-only (pure inference, old style)
    blended = inf.blend_overlay_on_image(image_rgb, seg_pred)
    Image.fromarray(blended, mode="RGB").save(out_overlay)

    mask_rgba = inf.seg_mask_to_rgba(seg_pred)
    # composite mask-only onto a neutral dark background so it's viewable without the photo
    mask_only_bg = np.full((*seg_pred.shape, 3), 26, dtype=np.uint8)  # matches --ink (#1a1d1a)
    alpha = mask_rgba[..., 3:4].astype(np.float32) / 255.0
    mask_only_rgb = (mask_only_bg * (1 - alpha) + mask_rgba[..., :3] * alpha).astype(np.uint8)
    Image.fromarray(mask_only_rgb, mode="RGB").save(out_mask_only)

    job.overlay_png_path = out_overlay
    job.mask_only_png_path = out_mask_only
    job.stats = stats
    job.status = JobStatus.DONE
    job.message = "Done." if not job.flagged_tiles else "Done — low confidence detected, see flag."
    job.progress = 100
    job.finished_at = __import__("datetime").datetime.utcnow().isoformat()


# =========================
# ROUTES
# =========================
@app.post("/upload/tiff")
async def upload_tiff(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Expected a .tif/.tiff file. Use /upload/png for PNGs.")

    job = store.create(file.filename, kind="tiff")
    upload_path = os.path.join(UPLOAD_DIR, f"{job.id}_{file.filename}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    store.enqueue(job.id)
    return {"job_id": job.id}


@app.post("/upload/png")
async def upload_png(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "Expected a .png/.jpg file. Use /upload/tiff for GeoTIFFs.")

    job = store.create(file.filename, kind="png")
    upload_path = os.path.join(UPLOAD_DIR, f"{job.id}_{file.filename}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    store.enqueue(job.id)
    return {"job_id": job.id}


@app.get("/jobs")
def list_jobs():
    return [j.to_dict() for j in store.all()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/jobs/{job_id}/geojson")
def get_geojson(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.geojson_layers is None:
        raise HTTPException(404, "No vector layers for this job (PNG jobs have no GeoJSON).")
    return JSONResponse(job.geojson_layers)


@app.get("/jobs/{job_id}/flagged_tiles_zip")
def download_flagged_tiles(job_id: str):
    job = store.get(job_id)
    if job is None or job.flagged_tiles_zip_path is None or not os.path.exists(job.flagged_tiles_zip_path):
        raise HTTPException(404, "No flagged-tiles zip available for this job.")
    return FileResponse(job.flagged_tiles_zip_path,
                         filename=os.path.basename(job.flagged_tiles_zip_path),
                         media_type="application/zip")


@app.get("/jobs/{job_id}/base_image")
def get_base_image(job_id: str):
    job = store.get(job_id)
    if job is None or job.base_image_path is None or not os.path.exists(job.base_image_path):
        raise HTTPException(404, "Base image not available for this job.")
    return FileResponse(job.base_image_path, filename=os.path.basename(job.base_image_path),
                         media_type="image/png")


@app.get("/jobs/{job_id}/gpkg")
def download_gpkg(job_id: str):
    job = store.get(job_id)
    if job is None or job.gpkg_path is None or not os.path.exists(job.gpkg_path):
        raise HTTPException(404, "GPKG not available for this job.")
    return FileResponse(job.gpkg_path, filename=os.path.basename(job.gpkg_path),
                         media_type="application/geopackage+sqlite3")


@app.get("/jobs/{job_id}/overlay")
def download_overlay(job_id: str):
    job = store.get(job_id)
    if job is None or job.overlay_png_path is None or not os.path.exists(job.overlay_png_path):
        raise HTTPException(404, "Overlay not available for this job.")
    return FileResponse(job.overlay_png_path, filename=os.path.basename(job.overlay_png_path),
                         media_type="image/png")


@app.get("/jobs/{job_id}/mask_only")
def download_mask_only(job_id: str):
    job = store.get(job_id)
    if job is None or job.mask_only_png_path is None or not os.path.exists(job.mask_only_png_path):
        raise HTTPException(404, "Mask-only image not available for this job.")
    return FileResponse(job.mask_only_png_path, filename=os.path.basename(job.mask_only_png_path),
                         media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": MODEL is not None}


# =========================
# XAI — Integrated Gradients explanation endpoint
# Requires: pip install captum
# Returns a base64-encoded PNG heatmap + per-channel attribution scores
# for a specific tile (identified by tile filename stored in flagged_tiles or
# by tile index), and optionally for a specific predicted class.
#
# Usage from chatbot or direct: GET /jobs/{id}/explain_tile?tile=tile_0_0.tif&class_id=1
# =========================
@app.get("/jobs/{job_id}/explain_tile")
async def explain_tile(job_id: str, tile: str = None, class_id: int = 1):
    """
    Runs Integrated Gradients on a specific tile from a completed TIFF job.
    tile: tile filename (e.g. tile_0_0.tif) — if omitted, uses first flagged tile.
    class_id: which class to explain (1=buildings, 2=roads, 3=water, 4=bridges, 5=utilities).
    Returns JSON with:
      - heatmap_b64: base64 PNG of IG attribution overlaid on the tile image
      - attribution_summary: per-region stats
      - class_name: which class was explained
      - tile: which tile was processed
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded.")

    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    if job.kind != "tiff":
        raise HTTPException(400, "XAI explain only available for TIFF jobs (tiled inference).")
    if job.status.value != "done":
        raise HTTPException(400, "Job not complete yet.")

    # Resolve which tile to explain
    tile_name = tile
    if tile_name is None:
        # Default: first flagged tile, else first tile in output dir
        if job.flagged_tiles:
            tile_name = job.flagged_tiles[0]["tile"]
        else:
            raise HTTPException(400, "No tile specified and no flagged tiles to default to. "
                                     "Pass ?tile=tile_X_Y.tif")

    # Tiles are cleaned up after job finishes (to save disk). If missing, re-tile.
    tile_dir = os.path.join(TILES_DIR, job_id)
    tile_path = os.path.join(tile_dir, tile_name)
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{job.filename}")

    if not os.path.exists(tile_path):
        # Re-tile the original if tiles were cleaned up
        if not os.path.exists(upload_path):
            raise HTTPException(404, f"Original upload not found — cannot re-tile for explanation. "
                                     f"Tile {tile_name} was cleaned up after job completion.")
        try:
            tiling.tile_tif(upload_path, tile_dir)
        except Exception as e:
            raise HTTPException(500, f"Re-tiling failed: {e}")

    if not os.path.exists(tile_path):
        raise HTTPException(404, f"Tile {tile_name} not found even after re-tiling. "
                                 f"Check tile name matches (case-sensitive).")

    try:
        import numpy as np
        from PIL import Image as PILImage
        import io, base64

        # Load tile image
        with rasterio.open(tile_path) as src:
            image = np.transpose(
                src.read([1, 2, 3]) if src.count >= 3 else src.read([1, 1, 1]),
                (1, 2, 0),
            ).astype(np.uint8)

        # Run IG
        ig_result = _run_integrated_gradients(MODEL, image, DEVICE, class_id=class_id)
        if ig_result is None:
            raise HTTPException(500, "Integrated Gradients computation failed.")

        heatmap_b64, attribution_summary = ig_result
        class_name = inf.ID2LABEL.get(class_id, f"class_{class_id}")

        return {
            "tile": tile_name,
            "class_id": class_id,
            "class_name": class_name,
            "heatmap_b64": heatmap_b64,
            "attribution_summary": attribution_summary,
        }
    except ImportError:
        raise HTTPException(503, "Captum not installed. Run: pip install captum")
    except Exception as e:
        raise HTTPException(500, f"XAI explain failed: {e}")


def _run_integrated_gradients(model, image_rgb: "np.ndarray", device: str, class_id: int = 1):
    """
    Runs Captum Integrated Gradients on a single tile image for a given class.
    Returns (heatmap_b64: str, attribution_summary: dict) or None on failure.

    How IG works here:
    - Baseline: black image (all zeros, the "no information" reference)
    - Target: the spatial mean of the logit for `class_id` across the output mask
      (SegFormer outputs per-pixel logits; we reduce to scalar by summing class logit
      over all pixels predicted as that class, so gradient flows back through those
      specific spatial locations → attribution highlights what caused those pixels
      to be classified as `class_id`)
    - Steps: 50 interpolation steps (speed/accuracy tradeoff)
    - Attribution: L2 norm across RGB channels → single heatmap per pixel
    """
    import torch
    import numpy as np
    from PIL import Image as PILImage
    import io, base64

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise

    model.eval()

    # Preprocess exactly as inference.py does
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    img_t = transform(image=image_rgb)["image"].unsqueeze(0).float()

    # Move to CPU for IG (Captum has better support on CPU; MPS IG can silently NaN)
    img_t = img_t.to("cpu")
    model_cpu = model.to("cpu") if device != "cpu" else model

    def forward_fn(x):
        """
        Wrapper: runs the seg head only, returns scalar = sum of logits for
        class_id over pixels where class_id is the argmax prediction.
        This focuses attribution on why *those specific pixels* got that class.
        """
        with torch.enable_grad():
            # Access the segmentation head output (logits before softmax)
            # Assumes model has a forward() returning (seg_logits, roof_logits)
            # or just seg_logits — adapt to your model.py's actual interface.
            out = model_cpu(x)
            if isinstance(out, (tuple, list)):
                seg_logits = out[0]   # (1, num_classes, H, W)
            else:
                seg_logits = out

            # Class logit map for target class
            class_logit = seg_logits[:, class_id, :, :]  # (1, H, W)

            # Mask to pixels predicted as this class (where it's the argmax)
            pred = seg_logits.argmax(dim=1)  # (1, H, W)
            class_mask = (pred == class_id).float()

            # Scalar target: mean logit over predicted pixels, or global mean
            if class_mask.sum() > 0:
                return (class_logit * class_mask).sum() / class_mask.sum()
            else:
                return class_logit.mean()

    img_t.requires_grad_(True)
    baseline = torch.zeros_like(img_t)

    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(img_t, baseline, n_steps=50, return_convergence_delta=False)
    # attributions shape: (1, 3, H, W)
    attr_np = attributions.squeeze(0).detach().cpu().numpy()  # (3, H, W)

    # L2 norm across channels → (H, W) saliency map
    saliency = np.linalg.norm(attr_np, axis=0)

    # Normalize to [0,1]
    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 1e-8:
        saliency_norm = (saliency - s_min) / (s_max - s_min)
    else:
        saliency_norm = np.zeros_like(saliency)

    # Build colormap heatmap (hot colormap: black→red→yellow→white)
    def apply_hot_colormap(s):
        """Simple hot colormap approximation without matplotlib dependency."""
        r = np.clip(s * 3.0, 0, 1)
        g = np.clip(s * 3.0 - 1.0, 0, 1)
        b = np.clip(s * 3.0 - 2.0, 0, 1)
        return np.stack([r, g, b], axis=-1)

    heatmap_rgb = (apply_hot_colormap(saliency_norm) * 255).astype(np.uint8)  # (H, W, 3)

    # Resize heatmap to match original image size
    import PIL.Image
    h_orig, w_orig = image_rgb.shape[:2]
    heatmap_resized = np.array(
        PIL.Image.fromarray(heatmap_rgb).resize((w_orig, h_orig), PIL.Image.BILINEAR)
    )

    # Blend heatmap on original image: 50% original, 50% heatmap
    blended = (image_rgb.astype(np.float32) * 0.5 + heatmap_resized.astype(np.float32) * 0.5).astype(np.uint8)

    # Encode to base64 PNG
    buf = io.BytesIO()
    PIL.Image.fromarray(blended, "RGB").save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Attribution summary: which image regions have highest attribution
    h3, w3 = saliency_norm.shape
    thirds_v = [saliency_norm[:h3//3].mean(), saliency_norm[h3//3:2*h3//3].mean(), saliency_norm[2*h3//3:].mean()]
    thirds_h = [saliency_norm[:, :w3//3].mean(), saliency_norm[:, w3//3:2*w3//3].mean(), saliency_norm[:, 2*w3//3:].mean()]
    region_names_v = ["top", "middle", "bottom"]
    region_names_h = ["left", "center", "right"]
    top_v = region_names_v[int(np.argmax(thirds_v))]
    top_h = region_names_h[int(np.argmax(thirds_h))]

    # Channel importance (which colour band drove the detection)
    ch_importance = np.abs(attr_np).mean(axis=(1, 2))
    ch_importance_norm = ch_importance / (ch_importance.sum() + 1e-8)

    # Move model back to original device
    if device != "cpu":
        model.to(device)

    attribution_summary = {
        "most_attended_region": f"{top_v}-{top_h}",
        "channel_importance": {
            "red": round(float(ch_importance_norm[0]), 3),
            "green": round(float(ch_importance_norm[1]), 3),
            "blue": round(float(ch_importance_norm[2]), 3),
        },
        "mean_attribution": round(float(saliency_norm.mean()), 4),
        "peak_attribution": round(float(saliency_norm.max()), 4),
    }

    return heatmap_b64, attribution_summary


# =========================
# CHATBOT — local Ollama bridge
# Explains inference results for a given job using job stats/flags as context.
# Requires Ollama running locally (ollama serve) with the model pulled.
# =========================
import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:latest")


def _build_job_context(job) -> str:
    """Turns a job's stats into a compact text block the LLM can reason over."""
    if job is None:
        return "No job selected."

    lines = [f"File: {job.filename}", f"Type: {job.kind}", f"Status: {job.status.value}"]

    if job.stats:
        lines.append("Detected features:")
        for layer, s in job.stats.items():
            if "area_m2" in s:
                lines.append(f"  - {layer}: {s['count']} features, {s['area_m2']:.1f} m² total")
            elif "pixel_count" in s:
                lines.append(f"  - {layer}: {s['pixel_count']} pixels")

    if job.flagged_tiles:
        lines.append(f"Low-confidence flags: {len(job.flagged_tiles)} tile(s)/image(s) flagged for review.")
        for ft in job.flagged_tiles[:5]:
            lines.append(f"  - {ft['tile']}: mean confidence {ft['mean_confidence']*100:.1f}%")
    else:
        lines.append("No low-confidence flags — model was confident across all tiles.")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are an explainable AI assistant embedded in a geospatial pipeline that detects \
village features (buildings, roads, water, bridges, utilities, roof types) from aerial/satellite \
imagery using a SegFormer dual-head model.

You explain inference results to the person using the tool — what was detected, confidence flags, \
and WHY the model made specific decisions using Integrated Gradients (IG) attribution.

INTEGRATED GRADIENTS EXPLANATION:
When a user asks WHY the model detected something, or asks about a specific tile or feature's reasoning, \
you should explain using the xai_result if provided in the context. IG works by interpolating between a \
black baseline and the real tile image across 50 steps, computing how much each pixel contributed to \
the model's class prediction. The attribution heatmap shows hot (white/yellow/red) = high model \
attention, cold (black) = low attention.

Key concepts to explain when asked:
- "most_attended_region": spatial area the model focused on most
- "channel_importance": which colour band (R/G/B) drove detection — red dominance often = roof/building \
  texture, green dominance = vegetation boundary near roads/water, blue = water/shadow
- High mean attribution + specific region = confident localised detection
- Low mean attribution across whole tile = model uncertain, matches low-confidence flag

ROOF TYPES: 1=Flat, 2=Gable, 3=Hip, 4=Shed, 5=Complex (your model-specific labels may differ).

Be concise and plain-spoken. Use the provided context block for specifics."""

# Class name for XAI requests — maps common user language to class IDs
_XAI_CLASS_KEYWORDS = {
    "building": 1, "buildings": 1, "house": 1, "houses": 1, "structure": 1,
    "road": 2, "roads": 2, "path": 2, "street": 2,
    "water": 3, "river": 3, "pond": 3, "lake": 3,
    "bridge": 4, "bridges": 4,
    "utility": 5, "utilities": 5, "tower": 5, "pole": 5,
}


def _parse_xai_intent(message: str):
    """
    Detects if the user is asking for XAI/explanation of a tile or feature.
    Returns (is_xai_request, class_id, tile_hint) or (False, None, None).
    """
    msg_lower = message.lower()
    xai_triggers = ["why", "how did", "explain", "what caused", "what made", "reason",
                    "attribution", "gradient", "heatmap", "attention", "focused on",
                    "detect", "classified", "xai", "integrated gradient"]
    is_xai = any(t in msg_lower for t in xai_triggers)

    class_id = 1  # default buildings
    for kw, cid in _XAI_CLASS_KEYWORDS.items():
        if kw in msg_lower:
            class_id = cid
            break

    # Tile hint: look for "tile_X_Y" pattern in message
    import re
    tile_match = re.search(r"tile[_\s]+(\d+)[_\s]+(\d+)", msg_lower)
    tile_hint = f"tile_{tile_match.group(1)}_{tile_match.group(2)}.tif" if tile_match else None

    return is_xai, class_id, tile_hint


@app.post("/chat")
async def chat(payload: dict):
    """
    payload: {"message": str, "job_id": Optional[str], "history": [{"role","content"}, ...]}
    Proxies to local Ollama. Requires `ollama serve` running on the same machine.
    XAI: if message triggers explanation intent, runs Integrated Gradients first and
    injects the attribution result into LLM context so it can explain the model's reasoning.
    Returns: {"reply": str, "xai_heatmap_b64": Optional[str], "xai_tile": Optional[str]}
    """
    message = payload.get("message", "").strip()
    job_id = payload.get("job_id")
    history = payload.get("history", [])

    if not message:
        raise HTTPException(400, "Empty message.")

    job = store.get(job_id) if job_id else None
    context = _build_job_context(job)

    # XAI: detect intent and run IG if applicable
    xai_heatmap_b64 = None
    xai_tile_name = None
    is_xai, xai_class_id, xai_tile_hint = _parse_xai_intent(message)

    if is_xai and job is not None and job.kind == "tiff" and job.status.value == "done":
        try:
            # Resolve tile to explain
            tile_name = xai_tile_hint
            if tile_name is None and job.flagged_tiles:
                tile_name = job.flagged_tiles[0]["tile"]

            if tile_name:
                tile_dir = os.path.join(TILES_DIR, job_id)
                tile_path = os.path.join(tile_dir, tile_name)
                upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{job.filename}")

                # Re-tile if tiles were cleaned up
                if not os.path.exists(tile_path) and os.path.exists(upload_path):
                    tiling.tile_tif(upload_path, tile_dir)

                if os.path.exists(tile_path):
                    with rasterio.open(tile_path) as src:
                        tile_img = np.transpose(
                            src.read([1, 2, 3]) if src.count >= 3 else src.read([1, 1, 1]),
                            (1, 2, 0),
                        ).astype(np.uint8)

                    ig_result = _run_integrated_gradients(MODEL, tile_img, DEVICE, class_id=xai_class_id)
                    if ig_result:
                        xai_heatmap_b64, attr_summary = ig_result
                        xai_tile_name = tile_name
                        class_name = inf.ID2LABEL.get(xai_class_id, f"class_{xai_class_id}")
                        context += f"\n\nXAI Attribution Result (Integrated Gradients):\n"
                        context += f"  Tile: {tile_name}\n"
                        context += f"  Explained class: {class_name} (id={xai_class_id})\n"
                        context += f"  Most attended region: {attr_summary['most_attended_region']}\n"
                        context += f"  Channel importance: R={attr_summary['channel_importance']['red']}, "
                        context += f"G={attr_summary['channel_importance']['green']}, "
                        context += f"B={attr_summary['channel_importance']['blue']}\n"
                        context += f"  Mean attribution: {attr_summary['mean_attribution']}, "
                        context += f"Peak: {attr_summary['peak_attribution']}\n"
                        context += f"  (Heatmap has been sent to the frontend for display)\n"
        except Exception as e:
            # XAI failure is non-fatal — bot still answers from stats context
            context += f"\n\n(XAI attribution attempt failed: {e} — answering from stats only)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCurrent job context:\n" + context}]
    for h in history[-10:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data.get("message", {}).get("content", "").strip()
            return {
                "reply": reply or "(empty response from model)",
                "xai_heatmap_b64": xai_heatmap_b64,
                "xai_tile": xai_tile_name,
            }
    except httpx.ConnectError:
        raise HTTPException(503, f"Can't reach Ollama at {OLLAMA_URL}. Is `ollama serve` running?")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Ollama error: {e.response.text}")
