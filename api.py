"""FastAPI Microservice for Multimodal Crisis Intelligence Inference.

Exposes REST endpoints:
- GET /health: Health check, device status, and model metadata.
- POST /predict: Multimodal inference accepting multipart text + image or JSON.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from inference import CrisisPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal Crisis Intelligence & Triage API",
    description="Fuses BERTweet and ConvNeXt-Tiny for real-time disaster triage and humanitarian resource estimation.",
    version="1.0.0",
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[CrisisPredictor] = None


@app.on_event("startup")
def startup_event():
    global predictor
    logger.info("Initializing CrisisPredictor instance...")
    predictor = CrisisPredictor()


class TextPredictionRequest(BaseModel):
    text: str
    needs_threshold: float = 0.50


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "multimodal-crisis-triage-api",
        "device": str(predictor.device) if predictor else "uninitialized",
        "checkpoint_exists": predictor.checkpoint_path.exists() if predictor else False,
        "supported_scenarios": ["CYCLONE", "URBAN_FLOODING", "INDUSTRIAL_FIRE", "NOT_DISASTER"],
    }


@app.post("/predict")
async def predict_crisis(
    text: Optional[str] = Form(None),
    needs_threshold: float = Form(0.50),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    """Multipart endpoint for simultaneous text and image inference."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor service is initializing.")

    if not text and not file:
        raise HTTPException(status_code=400, detail="Must provide at least 'text' or an uploaded image file.")

    img_obj = None
    if file:
        try:
            contents = await file.read()
            img_obj = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed decoding image: {e}")

    try:
        result = predictor.predict(text=text, image=img_obj, needs_threshold=needs_threshold)
        return result
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/text-only")
def predict_text_only(req: TextPredictionRequest) -> Dict[str, Any]:
    """Convenience JSON endpoint for text-only queries."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor service is initializing.")
    return predictor.predict(text=req.text, image=None, needs_threshold=req.needs_threshold)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
