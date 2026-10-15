from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import DEFAULT_MODEL_PATH, ImageClassifier

app = FastAPI(title="SPS GenAI API", version="2.0.0")

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
# CIFAR-10 CNN (Assignment 2); None if the weights file has not been trained yet
image_classifier = ImageClassifier() if DEFAULT_MODEL_PATH.exists() else None


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


# ---------- Assignment 2: CIFAR-10 image classification ----------
@app.get("/classify/classes")
def get_classes():
    """The 10 CIFAR-10 classes the model can predict."""
    if image_classifier is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run: uv run python train_cifar10.py")
    return {"classes": image_classifier.classes, "test_accuracy": image_classifier.test_accuracy}


@app.post("/classify")
async def classify_image(
    file: UploadFile = File(..., description="Image file (PNG or JPEG)"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top predictions to return"),
):
    """Upload an image and get its predicted CIFAR-10 class."""
    if image_classifier is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run: uv run python train_cifar10.py")
    contents = await file.read()
    try:
        result = image_classifier.predict(contents, top_k=top_k)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image (use PNG or JPEG).")
    return {"filename": file.filename, **result}
