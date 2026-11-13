"""Text utilities for word-level language models (Module 6/7 RNN activity)."""
import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch

UNK = "<unk>"
# Project Gutenberg #1184: "The Count of Monte Cristo" by Alexandre Dumas (public domain),
# the same book as the Module 3 bigram corpus.
MONTE_CRISTO_URL = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"


def tokenize(text):
    """Lowercase words (including accented letters and apostrophes) and basic punctuation."""
    return re.findall(r"[a-zà-öø-ÿ]+(?:'[a-z]+)?|[.,!?;:]", text.lower())


def detokenize(tokens):
    """Join tokens with spaces, attaching punctuation to the previous word."""
    return re.sub(r" ([.,!?;:])", r"\1", " ".join(tokens))


def load_corpus(data_dir="data", url=MONTE_CRISTO_URL):
    """Download (once) and return the book text without the Project Gutenberg header and footer."""
    path = Path(data_dir) / "monte_cristo.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (course project)"})
        path.write_bytes(urllib.request.urlopen(request, timeout=60).read())
    text = path.read_text(encoding="utf-8", errors="ignore")
    start, end = text.find("*** START OF"), text.find("*** END OF")
    if start != -1 and end != -1:
        text = text[text.find("\n", start) + 1:end]
    return text


class Vocabulary:
    """Maps words to integer ids (and back). Ids are ordered by word frequency; 0 is <unk>."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, tokens, max_size=10000):
        most_common = [w for w, _ in Counter(tokens).most_common(max_size - 1)]
        return cls([UNK] + most_common)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, 0) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


class NextWordDataset(torch.utils.data.Dataset):
    """Windows of seq_len token ids; the target is the same window shifted one word to the right."""

    def __init__(self, ids, seq_len=30, stride=15):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.starts = list(range(0, len(ids) - seq_len - 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        chunk = self.ids[s:s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]
