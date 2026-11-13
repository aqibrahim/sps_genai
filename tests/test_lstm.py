"""Tests for the LSTM text generator (Module 6/7 activity)."""
from pathlib import Path

import pytest
import torch

from helper_lib.generator import generate_with_rnn
from helper_lib.model import LSTMLanguageModel
from helper_lib.text import NextWordDataset, Vocabulary, detokenize, tokenize
from helper_lib.trainer import train_lstm

TEXT = "The Count of Monte Cristo is a novel. Edmond Dantès is falsely imprisoned, and later seeks revenge!"


def test_tokenize_and_detokenize():
    tokens = tokenize(TEXT)
    assert tokens[:4] == ["the", "count", "of", "monte"] and "dantès" in tokens and "." in tokens
    assert detokenize(["hello", ",", "world", "."]) == "hello, world."


def test_vocabulary_round_trip():
    vocab = Vocabulary.build(tokenize(TEXT))
    assert vocab.itos[0] == "<unk>" and vocab.encode(["zzz"]) == [0]
    assert vocab.decode(vocab.encode(["count", "novel"])) == ["count", "novel"]


def test_dataset_targets_are_shifted():
    x, y = NextWordDataset(list(range(100)), seq_len=10, stride=5)[0]
    assert x.tolist() == list(range(10)) and y.tolist() == list(range(1, 11))


def test_lstm_shapes():
    model = LSTMLanguageModel(vocab_size=50)
    logits, (h, c) = model(torch.randint(0, 50, (3, 7)))
    assert logits.shape == (3, 7, 50) and h.shape == (2, 3, 256)


def test_train_and_generate():
    vocab = Vocabulary.build(tokenize(TEXT * 5))
    ids = vocab.encode(tokenize(TEXT * 5))
    loader = torch.utils.data.DataLoader(NextWordDataset(ids, seq_len=8, stride=4), batch_size=4)
    model = LSTMLanguageModel(len(vocab), embedding_dim=16, hidden_dim=32, num_layers=1)
    model = train_lstm(model, loader, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), epochs=1)
    assert len(model.history["train_loss"]) == 1
    text = generate_with_rnn(model, vocab, "the count", length=6, seed=0)
    assert text.startswith("the count") and "<unk>" not in text
    assert text == generate_with_rnn(model, vocab, "the count", length=6, seed=0)  # same seed, same text


@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "models" / "lstm_text.pth").exists(),
                    reason="LSTM not trained yet")
def test_api_generate_with_rnn(client):
    from app import main
    assert isinstance(main.lstm_model, LSTMLanguageModel)
    r = client.post("/generate_with_rnn", json={"start_word": "the count", "length": 15, "seed": 1})
    assert r.status_code == 200 and r.json()["generated_text"].startswith("the count")
    assert client.post("/generate_with_rnn", json={"start_word": "edmond", "length": 0}).status_code == 422
