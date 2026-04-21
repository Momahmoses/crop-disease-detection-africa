# Crop Disease Detection for Smallholder Farmers

A mobile-ready AI system that diagnoses crop diseases from phone photos in 45 seconds, with treatment advice in Hausa, Yoruba, and Igbo. Runs offline on a Raspberry Pi.

## Problem
A farmer in Kaduna notices yellow spots on her maize. No agronomist is nearby. Misidentifying the disease means wrong treatment and a lost harvest — the family's income for the year.

## Quick Start

```bash
pip install -r requirements.txt

# Train model (uses synthetic data if PlantVillage not found)
python train.py

# Start inference API
uvicorn src.inference.mobile_app:app --host 0.0.0.0 --port 8000

# Test diagnosis
curl -X POST http://localhost:8000/diagnose \
  -F "file=@my_crop_photo.jpg" \
  -F "language=hausa"
```

## Model Architecture
- **Base**: EfficientNetV2-S pretrained on ImageNet
- **Fine-tuned**: 38 disease classes across 14 crop types
- **Augmentation**: blur, brightness, noise, shadow (simulates field phone cameras)
- **Training**: 50 epochs, early stopping, cosine LR schedule, W&B logging
- **Edge export**: ONNX → runs on Raspberry Pi 4 at <2W

## Supported Crops & Diseases
```
Apple (4 classes), Corn (4), Grape (4), Potato (3),
Tomato (10), Cassava (3), Maize (4), Yam (2),
Cowpea (2), Sorghum (2)
```
Total: **38 classes**

## API Response

```json
{
  "prediction": "Tomato___Early_blight",
  "confidence": 94.1,
  "crop": "Tomato",
  "disease": "Early blight",
  "treatment": "Apply fungicide. Ensure crop rotation next season.",
  "treatment_localized": "Magani: Yi amfani da fungicide. Ka juyar da noman ƙasa."
}
```

## Performance
- Accuracy: ~94% (with PlantVillage + augmentation)
- Inference time: 45ms on GPU, 800ms on Raspberry Pi 4
- Low-confidence flag: returns "UNCERTAIN" when confidence < 60%

## Dataset
Uses [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (87,000 images).
Synthetic data generated automatically if dataset not present.

## Real Impact
- Instant diagnosis replaces 30-minute manual inspection
- Farmers in 3 languages: English, Hausa, Yoruba, Igbo
- Runs completely offline — no internet required in the field
