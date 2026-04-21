"""
Training entry point: fine-tune EfficientNetV2 on crop disease dataset.
Uses PlantVillage or synthetic data if real dataset not found.
"""
import torch
from torch.utils.data import DataLoader, random_split
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.models.efficientnet import CropDiseaseModel, NUM_CLASSES
from src.training.trainer import CropDiseaseDataset, Trainer
from src.data.augmentation import get_train_transforms, get_val_transforms


def main():
    DATA_DIR = os.getenv("DATA_DIR", "data/raw/PlantVillage")
    EPOCHS = int(os.getenv("EPOCHS", "30"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    MODEL_PATH = "models/crop_disease_model.pt"

    train_transform = get_train_transforms(224)
    val_transform = get_val_transforms(224)

    dataset = CropDiseaseDataset(DATA_DIR, transform=train_transform)
    val_size = max(50, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CropDiseaseModel(num_classes=NUM_CLASSES, pretrained=True)
    trainer = Trainer(model, lr=1e-4)

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples")
    print(f"Classes: {NUM_CLASSES} | Device: {trainer.device}")

    best_acc = trainer.fit(train_loader, val_loader, epochs=EPOCHS, save_path=MODEL_PATH)
    print(f"\nFinal best validation accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")


if __name__ == "__main__":
    main()
