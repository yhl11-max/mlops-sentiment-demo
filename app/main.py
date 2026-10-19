"""Sentiment inference service.

Contract with the rest of the project (agreed with the MLOps owner):

    model file  ml/model.joblib
    model type  an sklearn Pipeline (TF-IDF + classifier), so a raw string
                can be passed straight to model.predict(["..."])
    predict     model.predict([text])      -> [0 or 1]     0=neg, 1=pos
    proba       model.predict_proba([text]) -> [[p_neg, p_pos]]  (optional)

The app is built through create_app(model=...). Tests inject a fake model so
they never depend on ml/model.joblib existing -- the CI runner does not have
that file (it is tracked by DVC, not Git).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_PATH_ENV = "MODEL_PATH"
DEFAULT_MODEL_PATH = Path("ml/model.joblib")

MAX_TEXT_LENGTH = 5000
SENTIMENT_LABELS = {0: "negative", 1: "positive"}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)


class PredictResponse(BaseModel):
    text: str
    prediction: int
    sentiment: str
    confidence: float


def model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH)))


def load_model(path: Path | None = None):
    """Load the trained pipeline, or return None when it is not available.

    Returning None (instead of raising at import time) keeps the service
    importable in CI, where DVC-tracked artifacts are absent.
    """
    path = path or model_path()
    if not path.exists():
        return None
    return joblib.load(path)


_UNSET = object()


def create_app(model=_UNSET) -> FastAPI:
    """Build the FastAPI app.

    - create_app()          -> load the model from MODEL_PATH when it exists
    - create_app(model=x)   -> use x (tests inject a fake)
    - create_app(model=None)-> explicitly no model, to exercise the 503 path
    """
    if model is _UNSET:
        model = load_model()

    app = FastAPI(title="Sentiment Model API", version="1.0.0")

    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).resolve().parent / "static"),
        name="static",
    )

    @app.get("/")
    def root() -> dict:
        return {
            "service": "sentiment",
            "endpoints": ["/predict", "/docs", "/ui"],
            "model_loaded": model is not None,
        }

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Run `python ml/train.py` first.",
            )

        text = request.text.strip()
        if not text:
            # A string of spaces passes min_length but carries no content.
            raise HTTPException(status_code=422, detail="text must not be blank")

        prediction = int(model.predict([text])[0])

        confidence = 1.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            confidence = float(max(probabilities))

        return PredictResponse(
            text=text,
            prediction=prediction,
            sentiment=SENTIMENT_LABELS.get(prediction, "unknown"),
            confidence=round(confidence, 4),
        )

    return app


app = create_app()
