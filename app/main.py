from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(title="SPS GenAI API", version="1.0.0")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()  # loaded once at startup


# ---------- Request schemas ----------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(10, ge=1, le=100)


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["king"])


class SimilarityRequest(BaseModel):
    word1: str = Field(..., min_length=1, examples=["king"])
    word2: str = Field(..., min_length=1, examples=["queen"])


# ---------- Endpoints ----------
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    """Return the spaCy word embedding (300-d vector) for the query word."""
    word = request.word.strip()
    if not word or len(word.split()) > 1:
        raise HTTPException(status_code=400, detail="Please provide a single word.")
    return embedding_model.get_embedding(word)


@app.get("/embedding/{word}")
def get_embedding_by_path(word: str):
    """Same as POST /embedding, convenient for testing in a browser."""
    return get_embedding(EmbeddingRequest(word=word))


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    """Cosine similarity between the embeddings of two words."""
    score = embedding_model.similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": score}
