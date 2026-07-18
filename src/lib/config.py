from functools import lru_cache
import logging
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Configuración general del proyecto.

    Los valores pueden definirse mediante variables de entorno
    o mediante el archivo src/.env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------
    # Aplicación
    # ---------------------------------------------------------

    app_name: str = "Dog Breed Recognition TP2"

    cors_origins: str = Field(
        default="*",
        description=(
            "Orígenes permitidos separados por comas, "
            "o * para permitir cualquiera."
        ),
    )

    max_workers: int = 2

    # ---------------------------------------------------------
    # Modelos de clasificación y embeddings
    # ---------------------------------------------------------

    # Modelo usado para generar embeddings:
    # baseline | resnet18_finetuned | cnn_custom
    embedding_model: str = "baseline"

    # Clasificador utilizado en la Etapa 3:
    # resnet18_finetuned | cnn_custom
    classifier_model: str = "resnet18_finetuned"

    model_path: Path = Path("models")

    resnet18_model_name: str = (
        "resnet18_finetuned.pth"
    )

    cnn_custom_model_name: str = (
        "cnn_custom.pth"
    )

    image_size: int = 224
    embedding_dim: int = 512

    # ---------------------------------------------------------
    # Entrenamiento — Etapa 2
    # ---------------------------------------------------------

    batch_size: int = 32
    max_epochs: int = 15
    patience: int = 4

    lr_head: float = 1e-3
    lr_backbone: float = 1e-4

    step_size: int = 5
    gamma: float = 0.5

    # ---------------------------------------------------------
    # Búsqueda por similitud — Etapa 1
    # ---------------------------------------------------------

    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.55
    top_k: int = 10

    # ---------------------------------------------------------
    # YOLO — Etapa 3
    # ---------------------------------------------------------

    yolo_model: str = "yolov8n.pt"

    yolo_conf_threshold: float = 0.25

    yolo_nms_iou_threshold: float = 0.70

    yolo_image_size: int = 960

    # Clase dog en el dataset COCO.
    yolo_dog_class_id: int = 16

    # Una detección se considera duplicada cuando su caja
    # más pequeña está cubierta en esta proporción por otra.
    duplicate_overlap_threshold: float = 0.90

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    embeddings_path: Path = Path(
        "data/embeddings.json"
    )

    data_path: Path = Path("data")
    dataset_path: Path = Path("data/dataset")
    output_path: Path = Path("output")

    # ---------------------------------------------------------
    # PostgreSQL / pgvector
    # ---------------------------------------------------------

    use_pgvector: bool = True

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dogs"
    postgres_user: str = "dogs_user"
    postgres_password: str = "dogs_pass"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()