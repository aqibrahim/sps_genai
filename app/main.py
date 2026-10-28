from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.digit_generator import GENERATOR_PATH, DigitGenerator
from app.image_classifier import DEFAULT_MODEL_PATH, ImageClassifier

app = FastAPI(title="SPS GenAI API", version="3.0.0")

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
# MNIST GAN (Assignment 3); None if the generator has not been trained yet
digit_generator = DigitGenerator() if GENERATOR_PATH.exists() else None


# ---------- Request schemas ----------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(10, ge=1, le=100)


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["king"])


class DigitGenerationRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64, description="How many digits to generate")
    seed: int | None = Field(None, description="Random seed; the same seed gives the same digits")
    scale: int = Field(4, ge=1, le=8, description="Upscaling factor for the returned PNG")


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


# ---------- Assignment 3: MNIST digit generation with a GAN ----------
def _require_generator():
    if digit_generator is None:
        raise HTTPException(status_code=503, detail="GAN not trained. Run: uv run python train_gan_mnist.py")
    return digit_generator


@app.get(
    "/gan/generate",
    responses={200: {"content": {"image/png": {}}, "description": "PNG grid of generated digits"}},
    response_class=Response,
)
def generate_digits_image(
    num_images: int = Query(16, ge=1, le=64, description="How many digits to generate"),
    seed: int | None = Query(None, description="Random seed; the same seed gives the same digits"),
    scale: int = Query(4, ge=1, le=8, description="Upscaling factor for the PNG"),
):
    """Generate hand-written digits and return them directly as a PNG image (opens in a browser)."""
    gen = _require_generator()
    images = gen.generate(num_images, seed=seed)
    return Response(content=gen.to_png(images, scale=scale), media_type="image/png")


@app.post("/gan/generate")
def generate_digits(request: DigitGenerationRequest):
    """Generate hand-written digits; returns the PNG grid as base64 plus the discriminator's
    probability that each generated digit is real."""
    gen = _require_generator()
    images = gen.generate(request.num_images, seed=request.seed)
    return {
        "num_images": request.num_images,
        "seed": request.seed,
        "image_shape": [1, 28, 28],
        "image_format": "png",
        "image_base64": gen.to_base64(gen.to_png(images, scale=request.scale)),
        "discriminator_scores": gen.realness_scores(images),
    }


@app.get("/gan/info")
def gan_info():
    """Architecture summary of the trained GAN."""
    gen = _require_generator()
    return {
        "dataset": "MNIST",
        "latent_dim": gen.generator.latent_dim,
        "output_shape": [1, 28, 28],
        "epochs_trained": gen.epochs,
        "generator": "Linear(100->6272) -> reshape 128x7x7 -> ConvT(128->64,k4,s2,p1)+BN+ReLU -> "
                     "ConvT(64->1,k4,s2,p1)+Tanh",
        "discriminator": "Conv(1->64,k4,s2,p1)+LeakyReLU(0.2) -> Conv(64->128,k4,s2,p1)+BN+"
                         "LeakyReLU(0.2) -> Flatten -> Linear(6272->1)+Sigmoid",
    }
