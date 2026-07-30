# =============================================================
# inference.py — predict_tile + extract_features, same logic as
# your original script, adapted to be reusable + device-agnostic.
# =============================================================
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rasterio.features import shapes
from shapely.geometry import shape

from model import run_inference, ID2LABEL, ROOF_ID2LABEL

IMAGE_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def is_blank_tile(image_rgb: np.ndarray) -> bool:
    """Same blank/white-tile skip logic as your original predict_tile.
    Tuned for GeoTIFF nodata/edge-padding tiles (solid white or solid black).
    Not appropriate for arbitrary photos — see skip_blank_check in predict_tile."""
    if image_rgb.mean() > 248 or image_rgb.mean() < 5:
        return True
    white = ((image_rgb[:, :, 0] > 250) & (image_rgb[:, :, 1] > 250) & (image_rgb[:, :, 2] > 250))
    if white.sum() / white.size > 0.5:
        return True
    return False


def predict_tile(model, image_rgb: np.ndarray, device: str, skip_blank_check: bool = False):
    """Same logic as your original predict_tile, device-agnostic (no hardcoded 'cuda').
    Now also returns a per-pixel confidence map (max softmax prob).

    skip_blank_check: the blank/nodata-tile heuristic below was tuned for GeoTIFF
    edge tiles (solid white/black padding from the tiling step). Direct photo
    uploads (sand, concrete, bright water, snow, etc.) can trip the same
    brightness/white-ratio thresholds and get wrongly skipped, producing an
    all-background result. PNG/JPG uploads pass skip_blank_check=True.
    """
    h, w = image_rgb.shape[:2]
    if not skip_blank_check and is_blank_tile(image_rgb):
        return None, None, None

    img_t = transform(image=image_rgb)["image"].unsqueeze(0)
    seg_pred, roof_pred, seg_conf = run_inference(model, img_t, out_hw=(h, w), device=device)
    return seg_pred, roof_pred, seg_conf


CONFIDENCE_THRESHOLD = 0.6  # below this mean-confidence-on-foreground, tile gets flagged


def confidence_summary(seg_mask: np.ndarray, seg_conf: np.ndarray) -> dict:
    """
    Mean confidence over foreground (non-background) pixels — background is
    usually high-confidence trivially and would mask real uncertainty on the
    classes that matter (buildings/roads/water/etc).
    Returns mean_confidence, is_low_confidence flag, and per-class breakdown.
    """
    fg = seg_mask != 0
    if fg.sum() == 0:
        return {"mean_confidence": 1.0, "is_low_confidence": False, "per_class": {}}

    mean_conf = float(seg_conf[fg].mean())
    per_class = {}
    for cls_id, name in ID2LABEL.items():
        if cls_id == 0:
            continue
        mask = seg_mask == cls_id
        if mask.sum() > 0:
            per_class[name] = round(float(seg_conf[mask].mean()), 4)

    return {
        "mean_confidence": round(mean_conf, 4),
        "is_low_confidence": mean_conf < CONFIDENCE_THRESHOLD,
        "per_class": per_class,
    }


def extract_features(seg_mask, roof_mask, tile_tf, min_area_px=10):
    """Identical logic to your original extract_features."""
    px_area = abs(tile_tf.a * tile_tf.e)
    feats = {k: [] for k in ["buildings", "roads", "water", "bridges", "utilities"]}
    class_map = {1: "buildings", 2: "roads", 3: "water", 4: "bridges", 5: "utilities"}

    for cls_id, layer in class_map.items():
        binary = (seg_mask == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        for geom_d, val in shapes(binary, transform=tile_tf):
            if val == 0:
                continue
            geom = shape(geom_d)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.area < min_area_px * px_area:
                continue

            feat = {"geometry": geom, "class_id": cls_id, "class_name": ID2LABEL[cls_id]}

            if cls_id == 1 and roof_mask is not None:
                bldg_px = (seg_mask == cls_id)
                roof_vals = roof_mask[bldg_px]
                roof_vals = roof_vals[roof_vals > 0]
                if len(roof_vals) > 0:
                    roof_type = int(np.bincount(roof_vals).argmax())
                    feat["roof_type"] = roof_type
                    feat["roof_label"] = ROOF_ID2LABEL.get(roof_type, "Unknown")
                else:
                    feat["roof_type"] = 0
                    feat["roof_label"] = "Unknown"

            feats[layer].append(feat)

    return feats


# Color palette for mask overlay (PNG path) — matches class ids above
CLASS_COLORS = {
    0: (0, 0, 0, 0),        # Background — transparent
    1: (230, 25, 75, 160),   # Building — red
    2: (60, 60, 60, 160),    # Road — dark gray
    3: (0, 130, 200, 160),   # Water — blue
    4: (255, 165, 0, 160),   # Bridge — orange
    5: (255, 225, 25, 160),  # Utility — yellow
}


def seg_mask_to_rgba(seg_mask: np.ndarray) -> np.ndarray:
    """Turns a class-id mask into an RGBA overlay image for the no-geo PNG path."""
    h, w = seg_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgba[seg_mask == cls_id] = color
    return rgba


def blend_overlay_on_image(image_rgb: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Blends the colored mask directly on top of the original photo —
    same idea as a QGIS layer stack, flattened to one RGB image.
    alpha = overlay opacity (0=invisible mask, 1=mask fully opaque).
    """
    overlay = seg_mask_to_rgba(seg_mask)
    out = image_rgb.astype(np.float32).copy()
    fg = seg_mask != 0
    for c in range(3):
        out[..., c][fg] = (1 - alpha) * out[..., c][fg] + alpha * overlay[..., c][fg]
    return out.astype(np.uint8)
