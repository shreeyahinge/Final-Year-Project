"""Local prediction service for the waste-bin fill-level model.

Loads model versions from ml/models/<version>/ (best.pt + model-info.json).

Run from the project root:
  uvicorn app:app --port 8000 --app-dir ml/service

Endpoints:
  GET  /health    -> service status and default model version
  GET  /versions  -> model versions available on disk
  POST /predict   -> classify one image (send image_url OR image_base64)
"""

import base64
import io
import json
import os
from pathlib import Path
from threading import Lock

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, model_validator
from ultralytics import YOLO

ML_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ML_DIR / "models"
DEFAULT_VERSION = os.getenv("ML_MODEL_VERSION", "v1")

app = FastAPI(title="SmartBin fill-level model")

_cache: dict = {}
_lock = Lock()


class PredictRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    version: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self):
        if bool(self.image_url) == bool(self.image_base64):
            raise ValueError("Provide exactly one of image_url or image_base64")
        return self


def load_version(version: str):
    """Return (model, info) for a version, loading it once and caching it."""
    with _lock:
        if version in _cache:
            return _cache[version]

        version_dir = MODELS_DIR / version
        weights = version_dir / "best.pt"
        info_path = version_dir / "model-info.json"
        if not weights.exists() or not info_path.exists():
            raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        model = YOLO(str(weights))
        _cache[version] = (model, info)
        return _cache[version]


def read_image(req: PredictRequest) -> Image.Image:
    if req.image_url:
        try:
            response = httpx.get(req.image_url, timeout=10, follow_redirects=True)
            response.raise_for_status()
            data = response.content
        except httpx.HTTPError:
            raise HTTPException(status_code=400, detail="Could not download image_url")
    else:
        raw = req.image_base64.split(",", 1)[-1]  # accept "data:image/...;base64," prefix
        try:
            data = base64.b64decode(raw, validate=True)
        except ValueError:
            raise HTTPException(status_code=400, detail="image_base64 is not valid base64")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")


@app.get("/health")
def health():
    return {"status": "ok", "default_version": DEFAULT_VERSION}


@app.get("/versions")
def versions():
    return sorted(p.name for p in MODELS_DIR.iterdir() if (p / "best.pt").exists())


@app.post("/predict")
def predict(req: PredictRequest):
    version = req.version or DEFAULT_VERSION
    model, info = load_version(version)
    image = read_image(req)

    result = model.predict(image, imgsz=info["imgsz"], verbose=False)[0]
    names = result.names
    probabilities = {names[i]: float(p) for i, p in enumerate(result.probs.data.tolist())}
    predicted = names[int(result.probs.top1)]

    return {
        "version": version,
        "class": predicted,
        "confidence": float(result.probs.top1conf),
        "fill_percent": info["fill_percent"][predicted],
        "probabilities": probabilities,
    }
