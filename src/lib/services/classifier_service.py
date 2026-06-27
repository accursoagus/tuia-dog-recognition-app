from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import onnxruntime

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lib.config import Settings


logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        settings: "Settings",
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.settings = settings
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model
    

    def _build_transforms(self, train: bool) -> transforms.Compose:
        base = [transforms.Resize((self.image_size, self.image_size))]
        if train:
            base += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        base += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        return transforms.Compose(base)

    def _build_resnet18_finetuned(self, n_classes: int) -> nn.Module:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model.to(self.device)

    def _build_cnn_custom(self, n_classes: int) -> nn.Module:
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        return model.to(self.device)

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        train_tf = self._build_transforms(train=True)
        valid_tf = self._build_transforms(train=False)

        train_ds = datasets.ImageFolder(self.dataset_path / "train", transform=train_tf)
        valid_ds = datasets.ImageFolder(self.dataset_path / "valid", transform=valid_tf)

        if train_ds.classes != valid_ds.classes:
            only_train = set(train_ds.classes) - set(valid_ds.classes)
            only_valid = set(valid_ds.classes) - set(train_ds.classes)
            raise ValueError(
                "Las clases de train y valid no coinciden exactamente "
                "(probable inconsistencia de nombres de carpeta, ej. espacios "
                f"extra). Solo en train: {only_train}. Solo en valid: {only_valid}. "
                "Normalizar nombres de carpeta antes de entrenar."
            )

        n_classes = len(train_ds.classes)
        train_loader = DataLoader(train_ds, batch_size=self.settings.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.settings.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        if self.active_model_name == "resnet18_finetuned":
            model = self._build_resnet18_finetuned(n_classes)
            optimizer = torch.optim.Adam([
                {"params": model.layer4.parameters(), "lr": self.settings.lr_backbone},
                {"params": model.fc.parameters(), "lr": self.settings.lr_head},
            ])
        elif self.active_model_name == "cnn_custom":
            model = self._build_cnn_custom(n_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.settings.lr_head)
        else:
            raise ValueError(f"Modelo desconocido para entrenamiento: {self.active_model_name}")

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.settings.step_size, gamma=self.settings.gamma
        )

        best_valid_loss = float("inf")
        epochs_without_improvement = 0
        history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}

        for epoch in range(self.settings.max_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            model.eval()
            v_running_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    v_running_loss += loss.item() * images.size(0)
                    v_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    v_total += labels.size(0)

            valid_loss = v_running_loss / v_total
            valid_acc = v_correct / v_total
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            history["train_acc"].append(train_acc)
            history["valid_acc"].append(valid_acc)

            logger.info(
                "epoch %d/%d - train_loss=%.4f train_acc=%.4f valid_loss=%.4f valid_acc=%.4f",
                epoch + 1, self.settings.max_epochs, train_loss, train_acc, valid_loss, valid_acc,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_without_improvement = 0
                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": train_ds.classes,
                    "model_name": self.active_model_name,
                    "history": history,
                }, self.active_checkpoint)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.settings.patience:
                    logger.info("Early stopping en epoch %d (sin mejora en %d epochs).", epoch + 1, self.settings.patience)
                    break

        self._loaded.pop(self.active_model_name, None)

    def evaluate_classifier(self) -> dict[str, float]:
        checkpoint = self.load_model()
        class_names = checkpoint["class_names"]
        n_classes = len(class_names)

        if self.active_model_name == "resnet18_finetuned":
            model = self._build_resnet18_finetuned(n_classes)
        else:
            model = self._build_cnn_custom(n_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_tf = self._build_transforms(train=False)
        test_ds = datasets.ImageFolder(self.dataset_path / "test", transform=test_tf)

        if test_ds.classes != class_names:
            raise ValueError(
                "Las clases de test no coinciden con las del checkpoint entrenado "
                "(probable inconsistencia de nombres de carpeta)."
            )

        test_loader = DataLoader(test_ds, batch_size=self.settings.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = float((all_preds == all_labels).mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )

        cm = confusion_matrix(all_labels, all_preds)
        specificities = []
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specificities))

        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "specificity": specificity,
            "f1": float(f1),
        }
