from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
from PIL import Image

from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando un modelo YOLO
        pre-entrenado (ej: YOLOv8n via ultralytics). No es necesario entrenar
        el detector.

        Sugerencias:
          - self.yolo_model_name, self.conf_threshold y self.dog_class_id
            (clase 'dog' = 16 en COCO) vienen de la configuracion (.env).
          - Debe funcionar con un perro, multiples perros y escenas complejas.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "No está instalada la librería ultralytics. "
                "Agregá 'ultralytics' a requirements.txt e instalá las dependencias."
            ) from exc

        if image is None or image.size == 0:
            logger.warning("detect_dogs recibió una imagen vacía.")
            return []

        height, width = image.shape[:2]

        # Carga lazy del modelo YOLO: se carga una sola vez y queda cacheado.
        if not hasattr(self, "_yolo_model"):
            logger.info("Cargando modelo YOLO: %s", self.yolo_model_name)
            self._yolo_model = YOLO(self.yolo_model_name)

        results = self._yolo_model.predict(
            source=image,
            conf=float(self.conf_threshold),
            classes=[int(self.dog_class_id)],
            verbose=False,
        )

        detections: list[tuple[tuple[int, int, int, int], float]] = []

        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            conf = float(box.conf[0].detach().cpu().item())

            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            x1, y1, x2, y2 = self._clip_xyxy(x1, y1, x2, y2, height, width)

            detections.append(((x1, y1, x2, y2), conf))

        return detections
        


    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        if crop is None or crop.size == 0:
            logger.warning("classify_detected_dog recibió un recorte vacío.")
            return "unknown", 0.0

        checkpoint = self.classifier.load_model()
        class_names = checkpoint["class_names"]
        n_classes = len(class_names)

        # Cache del modelo de clasificación ya reconstruido.
        cache_key = f"classification_model{self.classifier.active_model_name}"

        if not hasattr(self, cache_key):
            if self.classifier.active_model_name == "resnet18_finetuned":
                model = self.classifier._build_resnet18_finetuned(n_classes)
            elif self.classifier.active_model_name == "cnn_custom":
                model = self.classifier._build_cnn_custom(n_classes)
            else:
                raise ValueError(
                    f"Modelo de clasificación desconocido: "
                    f"{self.classifier.active_model_name}"
                )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.classifier.device)
            model.eval()

            setattr(self, cache_key, model)

        model = getattr(self, cache_key)

        # El crop viene en BGR porque se leyó con OpenCV.
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(crop_rgb)

        transform = self.classifier._build_transforms(train=False)
        tensor = transform(pil_image).unsqueeze(0).to(self.classifier.device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            score, pred_idx = torch.max(probs, dim=1)

        breed = class_names[int(pred_idx.item())]
        confidence = float(score.item())

        return breed, confidence


    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        if model_name:
            self.classifier.set_active_model(model_name)
        breed, score = self.classify_detected_dog(image)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)