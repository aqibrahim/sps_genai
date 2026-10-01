"""Word embeddings with spaCy (Module 2 word-embeddings functionality)."""
import spacy


class EmbeddingModel:
    def __init__(self, model_name: str = "en_core_web_md"):
        # en_core_web_md ships 300-dimensional static word vectors
        self.nlp = spacy.load(model_name)

    def get_embedding(self, word: str) -> dict:
        token = self.nlp(word.strip())[0]
        return {
            "word": token.text,
            "in_vocabulary": bool(token.has_vector),
            "dimension": len(token.vector),
            "embedding": token.vector.tolist(),
        }

    def similarity(self, word1: str, word2: str) -> float:
        t1, t2 = self.nlp(word1.strip())[0], self.nlp(word2.strip())[0]
        if not (t1.has_vector and t2.has_vector):
            return 0.0
        return float(t1.similarity(t2))
