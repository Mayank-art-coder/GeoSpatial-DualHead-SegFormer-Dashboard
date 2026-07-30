# Village Survey Pipeline — Setup & Run

## What this is
Upload a GeoTIFF or a plain photo (PNG/JPG) in the browser, your existing
SegFormer dual-head model runs inference, and you get:
- **GeoTIFF →** tiled, inferred, vectorized, written to a `.gpkg` (openable in
  QGIS exactly as before) **and** shown live on an interactive web map with
  layer toggles, areas, and counts.
- **PNG/JPG →** a colored mask overlay of detected classes (no geo-reference,
  so no GPKG — just the visual + pixel counts).

Runs entirely on your MacBook. No PC/QGIS dependency for viewing — QGIS is
still available if you want to open the GPKG there too.

## 1. Folder layout
```
village-pipeline/
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── tiling.py
│   ├── inference.py
│   ├── jobs.py
│   ├── requirements.txt
│   └── checkpoints/          <- put your .pth files here (create this folder)
│       ├── best_model.pth
│       └── best_roof_model.pth
└── frontend/
    └── index.html
```

## 2. Install dependencies (in your existing venv/conda env)
```bash
cd village-pipeline/backend
pip install -r requirements.txt
```

> Note: `torch` here should be a recent version with MPS support
> (2.1+). Check with:
> ```bash
> python -c "import torch; print(torch.backends.mps.is_available())"
> ```
> Should print `True` on M-series Macs.

## 3. Put your checkpoints in place
Copy your two `.pth` files into `backend/checkpoints/`, or point at wherever
they already are via environment variables (see step 4).

## 4. Run the backend
From `backend/`:
```bash
SEG_CKPT=./checkpoints/best_model.pth \
ROOF_CKPT=./checkpoints/best_roof_model.pth \
uvicorn main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is what makes it reachable from other devices on your LAN
(phones, other laptops), not just `localhost`.

On startup it prints which device it's using:
```
[main] Using device: mps
```
If it prints `cpu` instead, MPS isn't available in your torch build — it'll
still work, just slower.

Find your Mac's LAN IP (needed for other devices to connect):
```bash
ipconfig getifaddr en0
```

## 5. Open the frontend
Just open `frontend/index.html` directly in a browser — no build step, no
server needed for it. It auto-detects the backend at whatever host the page
loaded from, port 8000.

- **On the MacBook itself:** double-click `index.html`, or
  `open frontend/index.html`.
- **From another device on the same WiFi:** you need the page itself served
  somewhere reachable, since opening a local file won't have a "hostname".
  Simplest fix — serve the frontend folder too, with a second tiny server:
  ```bash
  cd village-pipeline/frontend
  python3 -m http.server 5173
  ```
  Then on any device on the same network, visit:
  ```
  http://<your-mac-lan-ip>:5173
  ```
  It'll auto-target `http://<your-mac-lan-ip>:8000` for the API.

  If your backend ever runs on a *different* machine than the page, edit the
  `API_BASE` constant near the top of the `<script>` in `index.html` and
  hardcode the IP.

## 6. Using it
- Pick **GEOTIFF** or **PHOTO (PNG)** tab, drop a file in.
- It appears immediately in the **Queue & History** sidebar with live
  progress (tiling → inferring → building output → done).
- Click any job to view its result on the right:
  - GeoTIFF jobs: interactive map, toggle Buildings/Roads/Water/Bridges/
    Utilities layers, click any feature for its attributes (roof type, area
    in m², etc.), download the `.gpkg` to open in QGIS.
  - Photo jobs: the colored mask overlay + pixel counts per class, download
    the overlay PNG.
- History persists for the life of the backend process (in-memory — restart
  clears it, per your choice; easy to add SQLite later if you want it to
  survive restarts).

## 7. Chatbot (local Ollama)
Uses your already-installed `gemma3:latest` via Ollama's local API — no
cloud, no API key.

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
   (if it's already running in the background, you'll get "address already
   in use" — that's fine, means it's up.)
2. Backend talks to it at `http://localhost:11434` by default. Override with:
   ```bash
   OLLAMA_URL=http://localhost:11434 OLLAMA_MODEL=gemma3:latest uvicorn main:app ...
   ```
3. Click the 💬 bubble bottom-right in the web UI. It uses whichever job is
   currently selected as context (detected counts, areas, low-confidence
   flags) so you can ask things like "what got flagged?" or "how much
   building area total?". With no job selected it just answers generally.

If you ever swap models (`ollama pull llama3.2` etc.), just change
`OLLAMA_MODEL` — no code changes needed.

## Diagnosing the TIFF detection drop
If detection still looks empty/sparse after pulling this update, run the
backend and check the terminal for lines like:
```
[DEBUG] tile 0 (tile_0_0.tif): class counts = {0: 240000, 1: 12000}, mean_conf = 0.81
```
- If `class counts` only ever shows `{0: ...}` (just background) across
  tiles → the model itself isn't detecting anything on this image; not a
  pipeline bug, worth re-checking the image/tiles match training
  distribution (resolution, zoom level, etc).
- If counts show real classes here but the GPKG/map still looks empty →
  the bug is downstream in `extract_features` or GPKG writing, not the model.
- Paste a few of these `[DEBUG]` lines back and that pins down exactly which
  stage broke.
These debug prints are temporary — once confirmed fixed, remove the block in
`main.py` starting with `# DEBUG — temporary`.
- Your original script forced `DEVICE = "cpu"` because the training PC's GPU
  was too small (<4GB) and used CUDA-specific `bfloat16` autocast. Both of
  those constraints are PC-specific.
- On the Mac, `model.py` auto-detects MPS and runs in plain fp32 (more
  reliable than autocast on MPS today). If MPS errors on any specific op
  (rare, but a few ops in PyTorch are still CPU-only on MPS depending on
  version), it transparently retries that call on CPU instead of crashing,
  then moves back to MPS for the next tile.
- Model architecture, checkpoint loading logic, tiling logic, and feature
  extraction logic are otherwise untouched from your original scripts.

## What's identical to your original scripts vs. new
**Unchanged logic:** `tile_tif`, `predict_tile`'s blank-tile skip logic,
`extract_features` (polygon extraction, roof-type majority vote, area-in-m²
UTM reprojection), checkpoint loading sequence.

**New:** FastAPI wrapper, in-memory job queue/worker thread, GeoJSON export
for the web map, PNG-only no-geo overlay path, MPS device handling.
