# =============================================================
# jobs.py — in-memory job queue + history (no DB, resets on restart)
# =============================================================
import uuid
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    TILING = "tiling"
    INFERRING = "inferring"
    BUILDING_OUTPUT = "building_output"
    DONE = "done"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, filename: str, kind: str):
        self.id = job_id
        self.filename = filename
        self.kind = kind  # "tiff" or "png"
        self.status = JobStatus.QUEUED
        self.progress = 0.0           # 0–100
        self.message = "Queued"
        self.created_at = datetime.utcnow().isoformat()
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None
        # results
        self.gpkg_path: Optional[str] = None
        self.geojson_layers: Optional[dict] = None   # layer_name -> geojson dict, for map display
        self.overlay_png_path: Optional[str] = None  # for the no-geo PNG path
        self.mask_only_png_path: Optional[str] = None  # pure inference, no photo underneath
        self.stats: Optional[dict] = None             # per-layer counts / areas
        self.base_image_path: Optional[str] = None    # downsampled original TIFF as PNG, for image-overlay view
        self.base_image_bounds: Optional[list] = None # [[south,west],[north,east]] in WGS84
        self.flagged_tiles: list = []                  # low-confidence tiles/images, human-in-the-loop review list
        self.flagged_tiles_zip_path: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "kind": self.kind,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "has_gpkg": self.gpkg_path is not None,
            "has_overlay": self.overlay_png_path is not None,
            "has_mask_only": self.mask_only_png_path is not None,
            "has_base_image": self.base_image_path is not None,
            "base_image_bounds": self.base_image_bounds,
            "stats": self.stats,
            "flagged_tiles": self.flagged_tiles,
            "has_flagged_zip": self.flagged_tiles_zip_path is not None,
        }


class JobStore:
    """Thread-safe in-memory job dict. Simple FIFO worker queue."""
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._queue_lock = threading.Lock()

    def create(self, filename: str, kind: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id, filename, kind)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def enqueue(self, job_id: str):
        with self._queue_lock:
            self._queue.append(job_id)

    def next_queued(self) -> Optional[str]:
        with self._queue_lock:
            if self._queue:
                return self._queue.pop(0)
        return None


store = JobStore()
