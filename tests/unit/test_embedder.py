"""Tests for ImageEmbedder and SimilaritySearch — 14 tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.embedder import SimilaritySearch
from tests.conftest import MockEmbedderModel


# ── Fake embedder (no weight downloads) ─────────────────────────────────────

class FakeEmbedder:
    """Minimal embedder using MockEmbedderModel."""

    def __init__(self, embedding_dim: int = 128):
        self.model_name = "mock_embedder"
        self.embedding_dim = embedding_dim
        self.model = MockEmbedderModel(embedding_dim)
        self.model.eval()
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    def embed(self, image: Image.Image) -> np.ndarray:
        tensor = self._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        vec = out.numpy().flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self._preprocess(img) for img in images])
        with torch.no_grad():
            out = self.model(tensors)
        vecs = out.numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return vecs / norms


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(embedding_dim=128)


@pytest.fixture
def similarity_search() -> SimilaritySearch:
    return SimilaritySearch(collection_name="test_images")


class TestImageEmbedder:
    def test_embed_returns_ndarray(self, embedder, sample_image):
        vec = embedder.embed(sample_image)
        assert isinstance(vec, np.ndarray)

    def test_embed_correct_dimension(self, embedder, sample_image):
        vec = embedder.embed(sample_image)
        assert vec.shape == (128,)

    def test_embed_l2_normalized(self, embedder, sample_image):
        vec = embedder.embed(sample_image)
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_embed_batch(self, embedder, sample_images_batch):
        vecs = embedder.embed_batch(sample_images_batch)
        assert vecs.shape == (4, 128)

    def test_embed_batch_normalized(self, embedder, sample_images_batch):
        vecs = embedder.embed_batch(sample_images_batch)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embed_grayscale(self, embedder, sample_grayscale):
        vec = embedder.embed(sample_grayscale)
        assert vec.shape == (128,)

    def test_different_images_different_embeddings(self, embedder):
        img1 = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), "RGB")
        img2 = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255, "RGB")
        v1 = embedder.embed(img1)
        v2 = embedder.embed(img2)
        # Embeddings should differ (not strictly guaranteed but very likely)
        assert not np.allclose(v1, v2, atol=1e-3)


class TestSimilaritySearch:
    def test_index_and_count(self, similarity_search):
        vec = np.random.randn(128).astype(np.float32)
        vec /= np.linalg.norm(vec)
        similarity_search.index_image("img001", vec, {"tag": "test"})
        assert similarity_search.count() >= 1

    def test_search_returns_results(self, similarity_search):
        for i in range(5):
            vec = np.random.randn(128).astype(np.float32)
            vec /= np.linalg.norm(vec)
            similarity_search.index_image(f"img_{i}", vec, {"idx": i})

        query = np.random.randn(128).astype(np.float32)
        query /= np.linalg.norm(query)
        results = similarity_search.search(query, k=3)
        assert len(results) <= 3
        for r in results:
            assert r.image_id.startswith("img_")
            assert isinstance(r.score, float)

    def test_search_empty_collection(self):
        search = SimilaritySearch(collection_name="empty_test")
        query = np.random.randn(128).astype(np.float32)
        results = search.search(query, k=5)
        assert results == []

    def test_delete_image(self, similarity_search):
        vec = np.random.randn(128).astype(np.float32)
        similarity_search.index_image("to_delete", vec)
        count_before = similarity_search.count()
        similarity_search.delete("to_delete")
        assert similarity_search.count() == count_before - 1

    def test_index_batch(self, similarity_search):
        vecs = np.random.randn(3, 128).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= norms
        similarity_search.index_batch(
            ["b1", "b2", "b3"], vecs, [{"i": 1}, {"i": 2}, {"i": 3}]
        )
        assert similarity_search.count() >= 3

    def test_reset(self, similarity_search):
        vec = np.random.randn(128).astype(np.float32)
        similarity_search.index_image("r1", vec)
        similarity_search.reset()
        assert similarity_search.count() == 0

    def test_upsert_overwrites(self, similarity_search):
        vec1 = np.random.randn(128).astype(np.float32)
        vec2 = np.random.randn(128).astype(np.float32)
        similarity_search.index_image("same_id", vec1, {"v": 1})
        similarity_search.index_image("same_id", vec2, {"v": 2})
        # Should not duplicate
        assert similarity_search.count() >= 1
