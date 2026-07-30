# =============================================================
# model.py — Dual-head SegFormer, ported for Apple Silicon (MPS)
# Same architecture/weights as your original inference.py.
# =============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

BACKBONE     = "nvidia/mit-b3"
NUM_SEG_CLS  = 6
NUM_ROOF_CLS = 5

ID2LABEL = {0: "Background", 1: "Building", 2: "Road", 3: "Water", 4: "Bridge", 5: "Utility"}
ROOF_ID2LABEL = {0: "Non-building", 1: "Roof-type-1", 2: "Roof-type-2", 3: "Roof-type-3", 4: "Roof-type-4"}


def get_device():
    """MPS if available (M-series GPU), else CPU. Never silently picks CUDA (none on Mac)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RoofHead(nn.Module):
    def __init__(self, in_ch, num_cls):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, num_cls, 1),
        )

    def forward(self, x):
        return self.head(x)


class DualHeadSegFormer(nn.Module):
    def __init__(self, seg_model, num_roof_cls):
        super().__init__()
        self.encoder = seg_model.segformer
        self.seg_head = seg_model.decode_head
        self.roof_head = RoofHead(seg_model.config.hidden_sizes[-1], num_roof_cls)

    def forward(self, pixel_values):
        out = self.encoder(pixel_values, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states
        seg_logits = self.seg_head(hidden)
        roof_logits = self.roof_head(hidden[-1])
        return seg_logits, roof_logits


def load_model(seg_ckpt_path: str, roof_ckpt_path: str, device: str):
    """
    Loads checkpoints exactly like your original script:
    - CPU load first (avoids OOM)
    - config-only base model (no HF pretrained download)
    - strict=False state dict load with warnings
    - moved to target device only after fully assembled
    """
    print(f"[model] Loading checkpoints on CPU first...")
    seg_ckpt = torch.load(seg_ckpt_path, map_location="cpu", weights_only=False)
    roof_ckpt = torch.load(roof_ckpt_path, map_location="cpu", weights_only=False)

    config = SegformerConfig.from_pretrained(
        BACKBONE,
        num_labels=NUM_SEG_CLS,
        ignore_mismatched_sizes=True,
    )
    base = SegformerForSemanticSegmentation(config)

    missing, unexpected = base.load_state_dict(seg_ckpt["model"], strict=False)
    if missing:
        print(f"[model] WARNING — missing keys: {missing}")
    if unexpected:
        print(f"[model] WARNING — unexpected keys: {unexpected}")
    print(f"[model] Seg loaded — best mIoU: {seg_ckpt.get('best_score', 0):.4f}")

    model = DualHeadSegFormer(base, NUM_ROOF_CLS)
    model.roof_head.load_state_dict(roof_ckpt["roof_head"])
    print(f"[model] Roof loaded — best mIoU: {roof_ckpt.get('best_miou', 0):.4f}")

    model = model.to(device)
    model.eval()
    print(f"[model] Ready on device: {device}")
    return model


@torch.no_grad()
def run_inference(model, img_tensor: torch.Tensor, out_hw: tuple, device: str):
    """
    img_tensor: (1, 3, H, W) already normalized.
    out_hw: (height, width) to upsample logits back to.
    Returns: (seg_pred, roof_pred) as int16 numpy arrays.

    MPS NOTE: bf16 autocast is CUDA-specific in the original script.
    On MPS we run plain fp32 — more reliable, MPS bf16/fp16 autocast
    support is still inconsistent across torch versions. If an op is
    unsupported on MPS, we fall back to CPU for that one call rather
    than crashing the whole pipeline.
    """
    img_tensor = img_tensor.to(device)
    try:
        seg_logits, roof_logits = model(img_tensor)
        seg_up = F.interpolate(seg_logits, size=out_hw, mode="bilinear", align_corners=False)
        roof_up = F.interpolate(roof_logits, size=out_hw, mode="bilinear", align_corners=False)
    except (RuntimeError, NotImplementedError) as e:
        # MPS op not implemented yet for this torch version — fall back to CPU for this call.
        print(f"[model] MPS op failed ({e}); retrying this batch on CPU")
        cpu_model = model.to("cpu")
        seg_logits, roof_logits = cpu_model(img_tensor.to("cpu"))
        seg_up = F.interpolate(seg_logits, size=out_hw, mode="bilinear", align_corners=False)
        roof_up = F.interpolate(roof_logits, size=out_hw, mode="bilinear", align_corners=False)
        model.to(device)  # move back for next call

    seg_probs = F.softmax(seg_up, dim=1)
    seg_conf, seg_pred_t = seg_probs.max(1)   # per-pixel max softmax prob = confidence

    seg_pred = seg_pred_t.squeeze(0).cpu().numpy().astype("int16")
    seg_conf = seg_conf.squeeze(0).cpu().numpy().astype("float32")
    roof_pred = roof_up.argmax(1).squeeze(0).cpu().numpy().astype("int16")
    return seg_pred, roof_pred, seg_conf
