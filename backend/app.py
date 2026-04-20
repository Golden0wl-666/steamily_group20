from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional

import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Chicago Crime Forecast API", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

ART_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ART_DIR / "data_v2"
MODEL_DIR = PROJECT_ROOT / "models"

FORECAST_DAYS = 30
CRIME_TYPES_DEFAULT = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "DECEPTIVE PRACTICE"]


def safe_read_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def first_existing(*paths: Path):
    for p in paths:
        if p.exists():
            return p
    return None


def get_slots_per_day(meta: dict) -> int:
    return int(meta.get("slots_per_day", 6)) if meta else 6


def get_model_lookback_steps(meta: dict) -> int:
    if meta and "lookback_steps" in meta:
        return int(meta["lookback_steps"])
    if meta and "lookback" in meta:
        return int(meta["lookback"])
    return 180


def get_crime_types(meta: dict):
    if meta and "crime_types" in meta and isinstance(meta["crime_types"], list):
        return meta["crime_types"]
    return CRIME_TYPES_DEFAULT


def get_grid_shape(meta: dict):
    n_rows = int(meta.get("n_rows", 43))
    n_cols = int(meta.get("n_cols", 35))
    if meta and "n_grids" in meta and n_rows * n_cols != int(meta["n_grids"]):
        return 43, 35
    return n_rows, n_cols

@lru_cache(maxsize=1)
def load_meta():
    meta = safe_read_json(DATA_DIR / "meta.json")
    if meta is None:
        raise FileNotFoundError("meta.json not found")
    return meta


@lru_cache(maxsize=1)
def load_tensor():
    tensor_path = first_existing(DATA_DIR / "tensor.npy", DATA_DIR / "demo_tensor.npy")
    if tensor_path is None:
        raise FileNotFoundError("tensor.npy / demo_tensor.npy not found")
    return np.load(tensor_path, mmap_mode="r")


@lru_cache(maxsize=1)
def load_session():
    onnx_path = MODEL_DIR / "stgcn_best.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def run_inference(session, x_array: np.ndarray):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: x_array.astype(np.float32)})[0]
    pred = np.asarray(pred)

    if pred.ndim == 4 and pred.shape[2] == 1:
        pred = np.squeeze(pred, axis=2)
    if pred.ndim == 3:
        pred = pred[0]

    if pred.ndim != 2:
        raise ValueError(f"Unexpected ONNX output ndim: {pred.ndim}, shape={pred.shape}")

    pred_count = np.expm1(pred)
    pred_count = np.clip(pred_count, 0, None)
    return pred_count.astype(np.float32)


def prepare_model_input(window_lnc: np.ndarray) -> np.ndarray:
    x = np.asarray(window_lnc, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))
    x = np.log1p(x)
    x = np.expand_dims(x, 0)
    return x.astype(np.float32)


@lru_cache(maxsize=1)
def precompute_predictions():
    meta = load_meta()
    tensor = load_tensor()
    session = load_session()

    slots_per_day = get_slots_per_day(meta)
    lookback_steps = get_model_lookback_steps(meta)

    if tensor.shape[0] < lookback_steps:
        raise ValueError(
            f"Tensor has only {tensor.shape[0]} steps, but model needs {lookback_steps}."
        )

    horizon_steps = FORECAST_DAYS * slots_per_day
    window = np.asarray(tensor[-lookback_steps:], dtype=np.float32) 
    step_preds = []

    for _ in range(horizon_steps):
        x_input = prepare_model_input(window)
        y_pred = run_inference(session, x_input) 
        step_preds.append(y_pred)

        next_step = np.transpose(y_pred, (1, 0))
        window = np.concatenate([window[1:], next_step[None, :, :]], axis=0)

    step_preds = np.asarray(step_preds, dtype=np.float32) 

    slot_preds = step_preds.reshape(
        FORECAST_DAYS,
        slots_per_day,
        step_preds.shape[1],
        step_preds.shape[2]
    )

    daily_preds = slot_preds.sum(axis=1)

    return {
        "slot_preds": slot_preds,
        "daily_preds": daily_preds,
        "meta": meta,
    }


class PredictRequest(BaseModel):
    day_index: int = Field(ge=0, le=FORECAST_DAYS - 1)
    slot_index: Optional[int] = Field(default=0, ge=0, le=5)
    aggregate: Literal["slot", "day"] = "slot"


class PredictResponse(BaseModel):
    day_index: int
    slot_index: Optional[int]
    aggregate: str
    crime_types: list[str]
    n_rows: int
    n_cols: int
    values: list[list[float]] 

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    meta = load_meta()
    slots_per_day = get_slots_per_day(meta)
    n_rows, n_cols = get_grid_shape(meta)
    return {
        "forecast_days": FORECAST_DAYS,
        "slots_per_day": slots_per_day,
        "crime_types": get_crime_types(meta),
        "lookback_steps": get_model_lookback_steps(meta),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_grids": int(meta.get("n_grids", n_rows * n_cols)),
        "end_date": meta.get("end_date"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pack = precompute_predictions()
        slot_preds = pack["slot_preds"]
        daily_preds = pack["daily_preds"]
        meta = pack["meta"]
        crime_types = get_crime_types(meta)
        n_rows, n_cols = get_grid_shape(meta)

        if req.aggregate == "slot":
            if req.slot_index is None:
                raise HTTPException(status_code=400, detail="slot_index is required for aggregate='slot'")
            values = slot_preds[req.day_index, req.slot_index] 
        else:
            values = daily_preds[req.day_index]

        return PredictResponse(
            day_index=req.day_index,
            slot_index=req.slot_index if req.aggregate == "slot" else None,
            aggregate=req.aggregate,
            crime_types=crime_types,
            n_rows=n_rows,
            n_cols=n_cols,
            values=values.tolist(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
