"""
FastAPI inference server for crop disease detection.
Accepts image uploads, returns diagnosis + treatment in English, Hausa, Yoruba, Igbo.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="FarmAI — Crop Disease Detector", version="1.0.0")

TRANSLATIONS = {
    "healthy": {
        "hausa": "Amintacciyar tsiro — babu cuta",
        "yoruba": "Irugbin ni ilera — ko si arun",
        "igbo": "Osisi dị mma — ọ dịghị ọrịa",
    },
    "treatment_prefix": {
        "hausa": "Magani: ",
        "yoruba": "Itọju: ",
        "igbo": "Ọgwụ: ",
    },
}

predictor = None


@app.on_event("startup")
async def load_model():
    global predictor
    try:
        from src.models.efficientnet import CropDiseasePredictor
        predictor = CropDiseasePredictor()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Model load failed: {e} — using mock predictor")
        predictor = None


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image = pil_image.resize((224, 224))
        return np.array(pil_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def mock_predict(image: np.ndarray) -> dict:
    return {
        "prediction": "Tomato___Early_blight",
        "confidence": 89.3,
        "crop": "Tomato",
        "disease": "Early blight",
        "is_healthy": False,
        "treatment": "Apply fungicide. Ensure crop rotation next season.",
        "top_5_predictions": [
            {"class": "Tomato___Early_blight", "confidence": 89.3},
            {"class": "Tomato___Late_blight", "confidence": 6.1},
            {"class": "Tomato___healthy", "confidence": 2.8},
        ],
    }


@app.post("/diagnose")
async def diagnose_crop(
    file: UploadFile = File(...),
    language: str = "english",
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    image = preprocess_image(image_bytes)

    if predictor:
        result = predictor.predict(image)
    else:
        result = mock_predict(image)

    if language != "english" and language in TRANSLATIONS["treatment_prefix"]:
        prefix = TRANSLATIONS["treatment_prefix"][language]
        result["treatment_localized"] = f"{prefix}{result['treatment']}"
        if result.get("is_healthy") and language in TRANSLATIONS["healthy"]:
            result["treatment_localized"] = TRANSLATIONS["healthy"][language]

    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/classes")
async def list_classes():
    from src.models.efficientnet import DISEASE_CLASSES
    return {"total_classes": len(DISEASE_CLASSES), "classes": DISEASE_CLASSES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
