from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn as nn             
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el informe.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver


        # Device: usa GPU si esta disponible (Colab), CPU en local.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelo baseline: ResNet18 pre-entrenado en ImageNet, sin la capa
        # final de clasificacion (fc). Se carga una sola vez en el
        # constructor, no en cada llamada a extract_embedding, para no
        # pagar el costo de carga del modelo por cada imagen.
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Quitamos la ultima capa (fc) y nos quedamos con el average pooling
        # global como salida -> vector de 512 dimensiones (penultima capa).
        self._embedding_model = nn.Sequential(*list(backbone.children())[:-1])
        self._embedding_model.eval()  # modo inferencia: desactiva dropout/batchnorm-update
        self._embedding_model.to(self.device)

        # Transformaciones: resize a image_size, conversion a tensor,
        # normalizacion con las estadisticas de ImageNet (las mismas con las
        # que se entreno ResNet18 originalmente).
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando ResNet18 pre-entrenado en
        ImageNet (penultima capa, sin clasificador final).
        """
        # OpenCV entrega BGR; los modelos de torchvision se entrenaron con
        # imagenes RGB. Si no se convierte, el modelo procesa los canales de
        # color invertidos y las metricas degradan sin tirar ningun error.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self._transform(image_rgb)          # (3, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W) -> batch de 1

        with torch.no_grad():  # no se necesita gradiente: solo inferencia
            output = self._embedding_model(tensor)    # (1, 512, 1, 1)

        embedding = output.squeeze().cpu().numpy()     # (512,)
        return embedding.tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera las top_k imagenes mas similares de la base vectorial.
        Rama segun el backend configurado (pgvector vs JSON), porque cada
        uno expone una interfaz distinta (self.store.search vs self.store.all).
        """
        # hasattr detecta si el store es PgVectorEmbeddingStore (tiene
        # metodo search nativo) o EmbeddingStore (solo expone all()).
        if hasattr(self.store, "search"):
            records = self.store.search(embedding, top_k)
            # pgvector con operador <=> devuelve distancia coseno (menor =
            # mas similar); se convierte a score de similitud (mayor = mejor)
            # para que sea consistente con la rama JSON.
            neighbors = []
            for record in records:
                score = self.similarity(embedding, record.embedding)
                neighbors.append(
                    Neighbor(path=record.path, breed=record.breed, score=round(score, 4))
                )
            return neighbors

        # Backend JSON: no hay busqueda nativa, se calcula similitud contra
        # todos los registros y se ordena manualmente.
        all_records = self.store.all()
        scored = [
            Neighbor(path=r.path, breed=r.breed, score=round(self.similarity(embedding, r.embedding), 4))
            for r in all_records
        ]
        # Orden descendente: mayor score = mas similar, en ambas metricas
        # (cosine y la l2 transformada en similarity()).
        scored.sort(key=lambda n: n.score, reverse=True)
        return scored[:top_k]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Voto mayoritario ponderado por score entre los vecinos recuperados.
        Si el mejor score no supera el threshold, se considera "unknown"
        (la imagen no se parece lo suficiente a nada conocido).
        """
        if not results:
            return "unknown", 0.0

        best_score = results[0].score
        if best_score < self.similarity_threshold:
            return "unknown", best_score

        # Voto ponderado: cada vecino aporta su score a la raza que propone,
        # en vez de contar votos simples. Esto evita que 6 vecinos con score
        # bajo le ganen a 4 vecinos con score muy alto de otra raza.
        votes: dict[str, float] = {}
        for neighbor in results:
            votes[neighbor.breed] = votes.get(neighbor.breed, 0.0) + neighbor.score

        predicted_breed = max(votes, key=votes.get)
        # El score final reportado es el del vecino mas cercano de la raza
        # ganadora (mas interpretable que la suma de votos).
        breed_best_score = max(n.score for n in results if n.breed == predicted_breed)
        return predicted_breed, breed_best_score

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial."""
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(Path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
