"""Construccion de los servicios del sistema (wiring compartido entre la API y los scripts)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from lib.config import Settings
from lib.services.classifier_service import ClassifierService
from lib.services.detection_service import DetectionService
from lib.services.similarity_service import SimilarityService
from lib.storage.base import EmbeddingStoreProtocol
from lib.storage.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)

StoreType = Union[EmbeddingStoreProtocol, EmbeddingStore]
UrlResolver = Callable[[Path], Optional[str]]


@dataclass
class ServiceContainer:
    store: StoreType
    similarity: SimilarityService
    classifier: ClassifierService
    detection: DetectionService


def build_store(settings: Settings) -> StoreType:
    if settings.use_pgvector:
        logger.info("Using PostgreSQL vector store")
        # Import lazy: psycopg/pgvector solo se requieren con USE_PGVECTOR=true
        # (permite usar el resto de los servicios en entornos sin postgres, ej: Colab).
        from lib.storage.pgvector_store import PgVectorEmbeddingStore

        return PgVectorEmbeddingStore(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            embedding_dim=settings.embedding_dim,
        )
    logger.info("Using JSON file vector store")
    return EmbeddingStore(settings.embeddings_path)


def build_classifier(settings: Settings) -> ClassifierService:
    return ClassifierService(
        checkpoints={
            "resnet18_finetuned": settings.model_path / settings.resnet18_model_name,
            "cnn_custom": settings.model_path / settings.cnn_custom_model_name,
        },
        image_size=settings.image_size,
        dataset_path=settings.dataset_path,
        output_path=settings.output_path,
        settings=settings,
    )


def build_similarity(
    settings: Settings,
    store: StoreType,
    url_resolver: Optional[UrlResolver] = None,
) -> SimilarityService:
    return SimilarityService(
        store=store,
        similarity_metric=settings.similarity_metric,
        similarity_threshold=settings.similarity_threshold,
        top_k=settings.top_k,
        image_size=settings.image_size,
        model_name=settings.embedding_model,
        url_resolver=url_resolver,
    )


def build_detection(settings: Settings, classifier: ClassifierService) -> DetectionService:
    return DetectionService(
        classifier=classifier,
        yolo_model=settings.yolo_model,
        conf_threshold=settings.yolo_conf_threshold,
        dog_class_id=settings.yolo_dog_class_id,
    )


def build_services(
    settings: Settings,
    url_resolver: Optional[UrlResolver] = None,
) -> ServiceContainer:
    store = build_store(settings)
    classifier = build_classifier(settings)
    similarity = build_similarity(settings, store, url_resolver)
    detection = build_detection(settings, classifier)
    return ServiceContainer(
        store=store,
        similarity=similarity,
        classifier=classifier,
        detection=detection,
    )
