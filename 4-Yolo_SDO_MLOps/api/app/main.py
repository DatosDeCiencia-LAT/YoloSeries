import io
import json
import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

import sunpy.map
from sunpy.net import Fido, attrs as a
import astropy.units as u

import os
from google.cloud import bigquery

# ── Configuration ──────────────────────────────────────────────────────
MODEL_PATH  = Path("sdo_best.pt")
LOG_PATH    = Path("logs/predictions.jsonl")
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMGSZ       = 1024
CLASS_NAMES = ["AR", "FL", "CH", "SG"]
LOGGING_BACKEND = os.getenv("LOGGING_BACKEND", "local")

# Initialize BigQuery client only if needed
# Avoids authentication errors when running locally without GCP credentials
if LOGGING_BACKEND == "bigquery":
    BQ_CLIENT = bigquery.Client()
    TABLE_REF = f"{os.getenv('GCP_PROJECT')}.{os.getenv('BQ_DATASET')}.{os.getenv('BQ_TABLE')}"

LOG_PATH.parent.mkdir(exist_ok=True)

app   = FastAPI(title="YOLO Solar Event Detector")
model = YOLO(str(MODEL_PATH))


# ── Helpers ────────────────────────────────────────────────────────────
def run_inference(img_rgb: np.ndarray) -> list:
    results = model.predict(
        source=img_rgb,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMGSZ,
        verbose=False,
    )
    predictions = []
    if results[0].boxes is not None:
        for bbox in results[0].boxes:
            cls_id = int(bbox.cls[0])
            conf   = float(bbox.conf[0])
            coords = list(map(int, bbox.xyxy[0].tolist()))
            predictions.append({
                "class":      CLASS_NAMES[cls_id],
                "confidence": round(conf, 4),
                "bbox":       coords,
            })
    return predictions


def log_prediction(source: str, detections: list) -> dict:
    record = {
        "timestamp":       datetime.datetime.utcnow().isoformat(),
        "source":          source,
        "n_detections":    len(detections),
        "detections":      json.dumps(detections),
        "mean_confidence": round(
            float(np.mean([d["confidence"] for d in detections])), 4
        ) if detections else None,
    }

    if LOGGING_BACKEND == "bigquery":
        # Write to BigQuery for production monitoring
        errors = BQ_CLIENT.insert_rows_json(TABLE_REF, [record])
        if errors:
            print(f"BigQuery insert errors: {errors}")
    else:
        # Write to local JSONL file for development
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

    return record


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    image_bytes = await file.read()
    img_rgb     = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    detections  = run_inference(img_rgb)
    record      = log_prediction(source=file.filename, detections=detections)
    return JSONResponse(record)


@app.get("/predict/realtime")
async def predict_realtime(wavelength: int = 193) -> JSONResponse:
    # Step 1 — search for available images
    end   = datetime.datetime.utcnow()
    start = end - datetime.timedelta(hours=1)

    result = Fido.search(
        a.Time(start.strftime("%Y-%m-%dT%H:%M:%S"),
               end.strftime("%Y-%m-%dT%H:%M:%S")),
        a.Instrument("AIA"),
        a.Wavelength(wavelength * u.angstrom),
    )

    if len(result) == 0:
        return JSONResponse(
            content={"error": "No SDO images found for the requested time range."},
            status_code=404,
        )

    # Step 2 — download most recent image
    downloaded = Fido.fetch(result[0, -1], path="/tmp/sdo/")
    sdo_map    = sunpy.map.Map(downloaded[0])

    img_data = sdo_map.data.astype(np.float32)
    img_norm = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_rgb  = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)

    timestamp  = sdo_map.date.strftime("%Y-%m-%dT%H:%M:%S")
    detections = run_inference(img_rgb)
    record     = log_prediction(
        source=f"SDO_AIA_{wavelength}_{timestamp}",
        detections=detections,
    )

    return JSONResponse(content=record)