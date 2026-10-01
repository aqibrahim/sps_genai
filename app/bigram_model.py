"""Bigram language model (moved from the Module 2 notebook)."""
import random
import re
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self, corpus: list[str], seed: int | None = None):
        self.rng = random.Random(seed)
        self.bigram_probs = self._build_bigram_probs(corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # lowercase, keep words (incl. accented letters) and basic punctuation
        return re.findall(r"\b\w+\b|[.,!?;]", text.lower())

    def _build_bigram_probs(self, corpus: list[str]) -> dict[str, dict[str, float]]:
        counts: dict[str, Counter] = defaultdict(Counter)
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for w1, w2 in zip(tokens, tokens[1:]):
                counts[w1][w2] += 1
        # P(w2 | w1) = count(w1, w2) / count(w1, *)
        probs = {}
        for w1, nexts in counts.items():
            total = sum(nexts.values())
            probs[w1] = {w2: c / total for w2, c in nexts.items()}
        return probs

    def generate_text(self, start_word: str, length: int = 10) -> str:
        current = start_word.lower()
        words = [current]
        for _ in range(max(length, 1) - 1):
            nexts = self.bigram_probs.get(current)
            if not nexts:  # dead end: no known continuation
                break
            current = self.rng.choices(list(nexts), weights=list(nexts.values()))[0]
            words.append(current)
        return " ".join(words)
