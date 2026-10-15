"""End-to-end tests for every API endpoint (Assignments 1 and 2)."""
import pytest

from helper_lib.data_loader import CIFAR10_CLASSES


# ---------- Assignment 2: /classify ----------
@pytest.mark.parametrize("name", CIFAR10_CLASSES)
def test_classify_every_sample_image(client, sample_dir, name):
    with open(sample_dir / f"{name}.png", "rb") as f:
        r = client.post("/classify", files={"file": (f"{name}.png", f, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in CIFAR10_CLASSES
    assert body["predicted_class"] == CIFAR10_CLASSES[body["class_index"]]
    assert 0 <= body["confidence"] <= 1
    assert body["top_predictions"][0]["class"] == body["predicted_class"]


def test_classify_top_k_sorted(client, sample_dir):
    with open(sample_dir / "cat.png", "rb") as f:
        r = client.post("/classify?top_k=5", files={"file": ("cat.png", f, "image/png")})
    probs = [p["probability"] for p in r.json()["top_predictions"]]
    assert len(probs) == 5
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0001


def test_classify_rejects_non_image(client):
    r = client.post("/classify", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_classify_rejects_bad_top_k(client, sample_dir):
    with open(sample_dir / "cat.png", "rb") as f:
        r = client.post("/classify?top_k=0", files={"file": ("cat.png", f, "image/png")})
    assert r.status_code == 422


def test_classify_requires_file(client):
    assert client.post("/classify").status_code == 422


def test_classify_classes(client):
    body = client.get("/classify/classes").json()
    assert body["classes"] == CIFAR10_CLASSES
    assert body["test_accuracy"] > 50


# ---------- Assignment 1 and Module 3 ----------
def test_root(client):
    assert client.get("/").json() == {"Hello": "World"}


def test_embedding(client):
    body = client.post("/embedding", json={"word": "king"}).json()
    assert body["in_vocabulary"] is True
    assert body["dimension"] == 300
    assert len(body["embedding"]) == 300


def test_embedding_rejects_multiple_words(client):
    assert client.post("/embedding", json={"word": "two words"}).status_code == 400


def test_similarity(client):
    body = client.post("/similarity", json={"word1": "king", "word2": "queen"}).json()
    assert 0 < body["similarity"] <= 1


def test_generate(client):
    r = client.post("/generate", json={"start_word": "the", "length": 5})
    assert r.status_code == 200
    assert r.json()["generated_text"].startswith("the")
