import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier
from helper_lib.generator import discriminator_scores, generate_gan_samples, images_to_png
from helper_lib.model import CNN, LATENT_DIM, Discriminator, Generator
from helper_lib.utils import load_model

app = FastAPI(title="SPS GenAI API", version="3.1.0")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CNN_PATH = MODELS_DIR / "cifar10_cnn.pth"
GAN_GENERATOR_PATH = MODELS_DIR / "gan_generator.pth"
GAN_DISCRIMINATOR_PATH = MODELS_DIR / "gan_discriminator.pth"

# ---------- Module 3: bigram model ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

# ---------- Assignment 1: spaCy word embeddings ----------
embedding_model = EmbeddingModel()  # loaded once at startup

# ---------- Assignment 2: CNN from helper_lib (Part 1 architecture) ----------
# helper_lib.model.CNN: 64x64x3 input -> Conv(16, 3x3, s1, p1) -> ReLU -> MaxPool(2, 2)
#   -> Conv(32, 3x3, s1, p1) -> ReLU -> MaxPool(2, 2) -> Flatten -> FC(100) -> ReLU -> FC(10)
image_classifier = None
if CNN_PATH.exists():
    cnn, cnn_checkpoint = load_model(CNN(num_classes=10), CNN_PATH)  # trained CIFAR-10 weights
    image_classifier = ImageClassifier(
        cnn,
        classes=cnn_checkpoint.get("classes"),
        test_accuracy=cnn_checkpoint.get("test_accuracy"),
    )

# ---------- Assignment 3: GAN Generator and Discriminator from helper_lib (Part 1 architecture) ----------
# helper_lib.model.Generator: noise (B, 100) -> FC to 7x7x128 -> reshape
#   -> ConvTranspose2d(128->64, k4, s2, p1) + BatchNorm2d + ReLU (14x14)
#   -> ConvTranspose2d(64->1, k4, s2, p1) + Tanh (28x28)
# helper_lib.model.Discriminator: image (1, 28, 28)
#   -> Conv2d(1->64, k4, s2, p1) + LeakyReLU(0.2) (14x14)
#   -> Conv2d(64->128, k4, s2, p1) + BatchNorm2d + LeakyReLU(0.2) (7x7)
#   -> Flatten -> Linear(6272->1) + Sigmoid (real/fake probability)
generator = discriminator = None
gan_epochs = None
if GAN_GENERATOR_PATH.exists():
    generator, gen_checkpoint = load_model(Generator(latent_dim=LATENT_DIM), GAN_GENERATOR_PATH)
    gan_epochs = gen_checkpoint.get("epochs")
if GAN_DISCRIMINATOR_PATH.exists():
    discriminator, _ = load_model(Discriminator(), GAN_DISCRIMINATOR_PATH)


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


# ---------- Assignment 3: MNIST digit generation with the GAN ----------
def _require_generator():
    if generator is None:
        raise HTTPException(status_code=503, detail="GAN not trained. Run: uv run python train_gan_mnist.py")
    return generator


def _layers(module):
    """The model's actual layers, as PyTorch prints them (shows the configured parameters)."""
    return [str(m) for m in module.modules() if not list(m.children())]


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
    """Generate hand-written digits with the trained Generator and return them as a PNG image."""
    images = generate_gan_samples(_require_generator(), num_samples=num_images, seed=seed)
    return Response(content=images_to_png(images, scale=scale), media_type="image/png")


@app.post("/gan/generate")
def generate_digits(request: DigitGenerationRequest):
    """Generate hand-written digits with the trained Generator; returns the PNG grid as base64 plus
    the trained Discriminator's probability that each generated digit is real."""
    images = generate_gan_samples(_require_generator(), num_samples=request.num_images, seed=request.seed)
    png = images_to_png(images, scale=request.scale)
    return {
        "num_images": request.num_images,
        "seed": request.seed,
        "image_shape": [1, 28, 28],
        "image_format": "png",
        "image_base64": base64.b64encode(png).decode("ascii"),
        "discriminator_scores": discriminator_scores(discriminator, images) if discriminator else None,
    }


@app.get("/gan/info")
def gan_info():
    """The loaded Generator and Discriminator, layer by layer, with parameter counts."""
    gen = _require_generator()
    return {
        "dataset": "MNIST",
        "latent_dim": gen.latent_dim,
        "output_shape": [1, 28, 28],
        "epochs_trained": gan_epochs,
        "generator": {
            "class": "helper_lib.model.Generator",
            "parameters": sum(p.numel() for p in gen.parameters()),
            "layers": _layers(gen),
        },
        "discriminator": None if discriminator is None else {
            "class": "helper_lib.model.Discriminator",
            "parameters": sum(p.numel() for p in discriminator.parameters()),
            "layers": _layers(discriminator),
        },
    }
