"""
Training loop with early stopping, LR scheduling, and W&B logging.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


class CropDiseaseDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_names = []

        if self.root_dir.exists():
            class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_names = [d.name for d in class_dirs]
            class_to_idx = {c: i for i, c in enumerate(self.class_names)}
            for class_dir in class_dirs:
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append(str(img_path))
                    self.labels.append(class_to_idx[class_dir.name])
                for img_path in class_dir.glob("*.png"):
                    self.samples.append(str(img_path))
                    self.labels.append(class_to_idx[class_dir.name])
        else:
            logger.info(f"Data directory {root_dir} not found — using synthetic data")
            self._generate_synthetic(200)

    def _generate_synthetic(self, n: int):
        from src.models.efficientnet import NUM_CLASSES, IDX_TO_CLASS
        self.class_names = list(IDX_TO_CLASS.values())
        for i in range(n):
            self.samples.append(None)
            self.labels.append(i % NUM_CLASSES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        sample_path = self.samples[idx]

        if sample_path and os.path.exists(sample_path):
            from PIL import Image
            image = np.array(Image.open(sample_path).convert("RGB").resize((224, 224)))
        else:
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.best_val_acc = 0
        self.patience = 7
        self.no_improve = 0

    def train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
        return total_loss / total, correct / total

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 50, save_path: str = "models/crop_disease_model.pt"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if WANDB:
            wandb.init(project="crop-disease-detection", name="efficientnetv2-finetune")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.eval_epoch(val_loader)
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            if WANDB:
                wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                           "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                self.no_improve = 0
                logger.info(f"  Best model saved (val_acc={val_acc:.4f})")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if WANDB:
            wandb.finish()

        logger.info(f"Training complete. Best val accuracy: {self.best_val_acc:.4f}")
        return self.best_val_acc
