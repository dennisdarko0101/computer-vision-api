"""Image embedding and similarity search.

ImageEmbedder wraps a torchvision backbone (imported lazily) to produce L2-normalized
feature vectors. SimilaritySearch is a dependency-free in-memory cosine index; because
embeddings are L2-normalized, cosine similarity reduces to a dot product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class SimilarityResult:
    """A single nearest-neighbor hit."""

    image_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageEmbedder:
    """Produce L2-normalized embeddings from images via a torchvision backbone.

    The backbone is built lazily; the final classification layer is replaced with
    Identity so the model outputs feature vectors. A model can be injected for tests.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        device: str = "cpu",
        embedding_dim: int | None = None,
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self._model = model
        self._transform = None

    def _build_transform(self):
        if self._transform is not None:
            return self._transform
        from torchvision import transforms

        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return self._transform

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        model = tv_models.resnet50(weights="DEFAULT")
        self.embedding_dim = model.fc.in_features
        model.fc = nn.Identity()
        model.eval()
        self._model = model.to(torch.device(self.device))
        return self._model

    @property
    def model(self) -> Any:
        return self._ensure_model()

    def get_torch_model(self) -> Any:
        return self._ensure_model()

    def _preprocess(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self._build_transform()(image)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, image) -> np.ndarray:
        import torch

        model = self._ensure_model()
        tensor = self._preprocess(image).unsqueeze(0).to(torch.device(self.device))
        with torch.no_grad():
            out = model(tensor)
        vec = out.cpu().numpy().flatten().astype(np.float32)
        return self._l2_normalize(vec)

    def embed_batch(self, images: list) -> np.ndarray:
        import torch

        model = self._ensure_model()
        tensors = torch.stack([self._preprocess(img) for img in images]).to(
            torch.device(self.device)
        )
        with torch.no_grad():
            out = model(tensors)
        vecs = out.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return vecs / norms


class SimilaritySearch:
    """In-memory cosine-similarity vector index.

    Vectors are stored as L2-normalized float32 arrays, so cosine similarity equals
    the dot product. Indexing the same id again overwrites (upsert). This avoids any
    hard dependency on an external vector database while matching the search API the
    rest of the service expects.
    """

    def __init__(
        self,
        collection_name: str = "images",
        persist_directory: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._ids: list[str] = []
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _normalize(embedding) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def index_image(
        self,
        image_id: str,
        embedding,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or overwrite a single image embedding (upsert by id)."""
        if image_id not in self._vectors:
            self._ids.append(image_id)
        self._vectors[image_id] = self._normalize(embedding)
        self._metadata[image_id] = dict(metadata or {})

    def index_batch(
        self,
        image_ids: list[str],
        embeddings,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add or overwrite multiple embeddings."""
        embeddings = np.asarray(embeddings)
        metadatas = metadatas or [{} for _ in image_ids]
        for image_id, emb, meta in zip(image_ids, embeddings, metadatas):
            self.index_image(image_id, emb, meta)

    def search(self, query_embedding, k: int = 10) -> list[SimilarityResult]:
        """Return the up-to-k most similar images by cosine similarity."""
        if not self._ids:
            return []
        query = self._normalize(query_embedding)
        scored: list[SimilarityResult] = []
        for image_id in self._ids:
            score = float(np.dot(query, self._vectors[image_id]))
            scored.append(
                SimilarityResult(
                    image_id=image_id,
                    score=score,
                    metadata=dict(self._metadata.get(image_id, {})),
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def delete(self, image_id: str) -> None:
        """Remove a single image from the index."""
        if image_id in self._vectors:
            del self._vectors[image_id]
            del self._metadata[image_id]
            self._ids.remove(image_id)

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        """Clear the entire collection."""
        self._ids.clear()
        self._vectors.clear()
        self._metadata.clear()
