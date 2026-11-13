"""Train the LSTM text generator (Module 6/7 RNN activity) on "The Count of Monte Cristo".

Usage:  uv run --with matplotlib python train_lstm_text.py            (default 10 epochs)
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from helper_lib.generator import generate_with_rnn
from helper_lib.model import LSTMLanguageModel
from helper_lib.text import NextWordDataset, Vocabulary, load_corpus, tokenize
from helper_lib.trainer import train_lstm
from helper_lib.utils import get_device

MODEL_PATH = Path("models/lstm_text.pth")
METRICS_PATH = Path("models/lstm_metrics.json")
FIG_PATH = Path("docs/screenshots/lstm_training_curves.png")
CONFIG = dict(embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2)
PROMPTS = ["the count", "edmond", "the prisoner", "mercedes"]


def plot(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(epochs, history["train_perplexity"], marker="o", markersize=3, label="Training")
    ax.plot(epochs, history["val_perplexity"], marker="o", markersize=3, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Perplexity (lower is better)"); ax.set_title("LSTM language model")
    ax.legend(); fig.tight_layout(); FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=300_000)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    tokens = tokenize(load_corpus())[: args.max_tokens]
    vocab = Vocabulary.build(tokens, max_size=args.vocab_size)
    ids = vocab.encode(tokens)
    split = int(len(ids) * 0.95)  # last 5% of the text held out for validation
    train_loader = torch.utils.data.DataLoader(NextWordDataset(ids[:split]), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(NextWordDataset(ids[split:], stride=30), batch_size=args.batch_size)
    print(f"Corpus: {len(tokens):,} tokens, vocabulary {len(vocab):,} words, {len(train_loader.dataset):,} training windows")

    model = LSTMLanguageModel(vocab_size=len(vocab), **CONFIG)
    print(f"LSTM parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.time()
    model = train_lstm(model, train_loader, criterion, optimizer, device=device, epochs=args.epochs,
                       val_loader=val_loader)
    model = model.to("cpu")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "itos": vocab.itos, "config": CONFIG, "epochs": args.epochs}, MODEL_PATH)
    samples = {p: generate_with_rnn(model, vocab, p, length=25, seed=1) for p in PROMPTS}
    for p, text in samples.items():
        print(f"  [{p}] {text}")
    plot(model.history)
    METRICS_PATH.write_text(json.dumps({
        "model": "LSTM word-level language model", "corpus": "The Count of Monte Cristo (Project Gutenberg #1184)",
        "tokens": len(tokens), "vocab_size": len(vocab), "config": CONFIG, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.lr, "sequence_length": 30,
        "parameters": sum(p.numel() for p in model.parameters()),
        "final_train_perplexity": round(model.history["train_perplexity"][-1], 1),
        "final_val_perplexity": round(model.history["val_perplexity"][-1], 1),
        "training_minutes": round((time.time() - start) / 60, 1),
        "samples_temperature_0.8": samples,
        "history": {k: [round(v, 4) for v in vals] for k, vals in model.history.items()},
    }, indent=2))
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
