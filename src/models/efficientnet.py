"""
EfficientNetV2 fine-tuned for crop disease classification.
Transfer learning from ImageNet. Supports 38 disease classes across 14 crops.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
    "Tomato___Target_Spot", "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___Mosaic_virus", "Tomato___healthy",
    "Cassava___Brown_streak_disease", "Cassava___Mosaic_disease", "Cassava___healthy",
    "Maize___Fall_armyworm", "Maize___Leaf_blight", "Maize___Streak_virus", "Maize___healthy",
    "Yam___Anthracnose", "Yam___healthy",
    "Cowpea___Mosaic_virus", "Cowpea___healthy",
    "Sorghum___Leaf_blight", "Sorghum___healthy",
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(DISEASE_CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(DISEASE_CLASSES)

CROP_FROM_CLASS = {cls: cls.split("___")[0] for cls in DISEASE_CLASSES}
DISEASE_FROM_CLASS = {cls: cls.split("___")[1].replace("_", " ") for cls in DISEASE_CLASSES}

TREATMENT_GUIDE = {
    "Apple_scab": "Apply fungicide (captan or mancozeb). Remove infected leaves.",
    "Black_rot": "Prune infected areas. Apply copper-based fungicide.",
    "Common_rust": "Apply triazole fungicide. Plant resistant varieties.",
    "Late_blight": "Apply mancozeb or chlorothalonil. Remove infected plants immediately.",
    "Early_blight": "Apply fungicide. Ensure crop rotation next season.",
    "Bacterial_spot": "Use copper-based bactericide. Avoid overhead irrigation.",
    "Cassava___Brown_streak_disease": "Remove and destroy infected plants. Use certified disease-free cuttings.",
    "Cassava___Mosaic_disease": "Control whitefly vectors. Use resistant cassava varieties.",
    "Maize___Fall_armyworm": "Apply spinosad or chlorpyrifos early. Monitor traps.",
    "healthy": "No disease detected. Continue standard care.",
}


class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        try:
            import torchvision.models as models
            self.backbone = models.efficientnet_v2_s(
                weights="IMAGENET1K_V1" if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes),
            )
        except Exception:
            logger.warning("torchvision not available — using mock model")
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(224 * 224 * 3, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes),
            )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CropDiseasePredictor:
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CropDiseaseModel(num_classes=NUM_CLASSES, pretrained=False)
        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info(f"Loaded model from {model_path}")
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = 0.6

    def predict(self, image: np.ndarray) -> dict:
        if not torch.is_tensor(image):
            img_tensor = torch.from_numpy(image).float()
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            img_tensor = image.unsqueeze(0) if image.ndim == 3 else image

        img_tensor = img_tensor.to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        img_normalized = (img_tensor / 255.0 - mean) / std

        with torch.no_grad():
            logits = self.model(img_normalized)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

        confidence = confidence.item()
        pred_idx = pred_idx.item()

        if confidence < self.confidence_threshold:
            return {
                "prediction": "UNCERTAIN",
                "confidence": round(confidence * 100, 1),
                "message": "Image quality too low or disease not recognized. Please retake photo.",
                "crop": None,
                "disease": None,
                "treatment": None,
            }

        class_name = IDX_TO_CLASS[pred_idx]
        crop = CROP_FROM_CLASS[class_name]
        disease = DISEASE_FROM_CLASS[class_name]
        key = class_name.split("___")[1]
        treatment = TREATMENT_GUIDE.get(key, TREATMENT_GUIDE.get("healthy", "Consult local agronomist."))

        top5_probs = probs[0].topk(5)
        top5 = [
            {"class": IDX_TO_CLASS[i.item()], "confidence": round(p.item() * 100, 1)}
            for p, i in zip(top5_probs.values, top5_probs.indices)
        ]

        return {
            "prediction": class_name,
            "confidence": round(confidence * 100, 1),
            "crop": crop,
            "disease": disease,
            "is_healthy": "healthy" in class_name,
            "treatment": treatment,
            "top_5_predictions": top5,
        }

    def save(self, path: str = "models/crop_disease_model.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved → {path}")
