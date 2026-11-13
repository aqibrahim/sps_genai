import base64
from pathlib import Path

import torch

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier
from helper_lib.generator import (
    discriminator_scores,
    generate_diffusion_samples,
    generate_ebm_samples,
    generate_gan_samples,
    generate_with_rnn,
    images_to_png,
)
from helper_lib.model import (
    CNN,
    LATENT_DIM,
    DiffusionModel,
    Discriminator,
    EnergyModel,
    Generator,
    LSTMLanguageModel,
    UNet,
)
from helper_lib.text import Vocabulary
from helper_lib.utils import load_model

app = FastAPI(title="SPS GenAI API", version="4.0.0")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CNN_PATH = MODELS_DIR / "cifar10_cnn.pth"
GAN_GENERATOR_PATH = MODELS_DIR / "gan_generator.pth"
GAN_DISCRIMINATOR_PATH = MODELS_DIR / "gan_discriminator.pth"
DIFFUSION_PATH = MODELS_DIR / "diffusion_unet.pth"
EBM_PATH = MODELS_DIR / "ebm.pth"
LSTM_PATH = MODELS_DIR / "lstm_text.pth"

# ---------- Module 3: bigram model ----------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)

# ---------- Module 6/7 activity: LSTM text generator from helper_lib (replaces the bigram model) ----------
# helper_lib.model.LSTMLanguageModel: token ids -> Embedding(vocab, 128) -> LSTM(128 -> 256, 2 layers)
#   -> Linear(256 -> vocab): next-word logits. Trained on "The Count of Monte Cristo".
lstm_model = lstm_vocab = None
if LSTM_PATH.exists():
    lstm_checkpoint = torch.load(LSTM_PATH, map_location="cpu", weights_only=True)
    lstm_vocab = Vocabulary(lstm_checkpoint["itos"])
    lstm_model = LSTMLanguageModel(vocab_size=len(lstm_vocab), embedding_dim=128, hidden_dim=256, num_layers=2,
                                   dropout=0.2)
    lstm_model.load_state_dict(lstm_checkpoint["state_dict"])
    lstm_model.eval()

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

# ---------- Assignment 4: Diffusion model from helper_lib (U-Net noise predictor) ----------
# helper_lib.model.UNet: noisy image x_t (3x32x32) + timestep t
#   -> sinusoidal time embedding (32 dims), upsampled to 32x32 and concatenated with the image features
#   -> DownBlocks 32x32 (32 ch) -> 16x16 (64 ch) -> 8x8 (96 ch) -> 4x4, bottleneck 128 ch,
#   -> UpBlocks with skip connections back to 32x32 -> predicted noise (3x32x32)
# helper_lib.model.DiffusionModel: DDPM objective, cosine schedule over 1,000 timesteps; sampled with DDIM
unet = UNet(image_size=32, in_channels=3, widths=(32, 64, 96, 128), block_depth=2, embedding_dim=32)
diffusion = None
diffusion_epochs = None
if DIFFUSION_PATH.exists():
    unet, diffusion_checkpoint = load_model(unet, DIFFUSION_PATH)  # trained (EMA) weights
    diffusion = DiffusionModel(unet, timesteps=1000).eval()
    diffusion_epochs = diffusion_checkpoint.get("epochs")

# ---------- Assignment 4: Energy-based model from helper_lib ----------
# helper_lib.model.EnergyModel: image (3x32x32) -> Conv(3->16, k5, s2) -> Conv(16->32, k3, s2)
#   -> Conv(32->64, k3, s2) -> Conv(64->64, k3, s2), Swish activations -> Linear(256->64) -> Linear(64->1): energy
# Samples are generated with Langevin dynamics: gradient descent on the input pixels, weights frozen.
energy_model = None
ebm_epochs = None
if EBM_PATH.exists():
    energy_model, ebm_checkpoint = load_model(EnergyModel(in_channels=3), EBM_PATH)
    ebm_epochs = ebm_checkpoint.get("epochs")


# ---------- Request schemas ----------
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(10, ge=1, le=100)
    temperature: float = Field(0.8, gt=0, le=2.0, description="Sampling temperature (generate_with_rnn only)")
    seed: int | None = Field(None, description="Random seed (generate_with_rnn only)")


class EmbeddingRequest(BaseModel):
    word: str = Field(..., min_length=1, examples=["king"])


class DigitGenerationRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64, description="How many digits to generate")
    seed: int | None = Field(None, description="Random seed; the same seed gives the same digits")
    scale: int = Field(4, ge=1, le=8, description="Upscaling factor for the returned PNG")


class DiffusionGenerationRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64, description="How many images to generate")
    steps: int = Field(50, ge=5, le=1000, description="DDIM denoising steps (more = slower, slightly sharper)")
    seed: int | None = Field(None, description="Random seed; the same seed gives the same images")
    scale: int = Field(4, ge=1, le=8, description="Upscaling factor for the returned PNG")


class EBMGenerationRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64, description="How many images to generate")
    steps: int = Field(200, ge=1, le=1000, description="Langevin dynamics steps starting from noise")
    step_size: float = Field(10.0, gt=0, le=100, description="Langevin step size (gradient multiplier)")
    seed: int | None = Field(None, description="Random seed; the same seed gives the same images")
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


@app.post("/generate_with_rnn")
def generate_with_rnn_endpoint(request: TextGenerationRequest):
    """Generate text with the LSTM (Module 6/7 activity): `length` new words after `start_word`."""
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM not trained. Run: uv run python train_lstm_text.py")
    generated_text = generate_with_rnn(lstm_model, lstm_vocab, request.start_word, length=request.length,
                                       temperature=request.temperature, seed=request.seed)
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


# ---------- Assignment 4: diffusion model ----------
def _require_diffusion():
    if diffusion is None:
        raise HTTPException(status_code=503, detail="Diffusion model not trained. Run: uv run python train_diffusion_cifar10.py")
    return diffusion


@app.get(
    "/diffusion/generate",
    responses={200: {"content": {"image/png": {}}, "description": "PNG grid of generated CIFAR-10-style images"}},
    response_class=Response,
)
def diffusion_generate_image(
    num_images: int = Query(16, ge=1, le=64, description="How many images to generate"),
    steps: int = Query(50, ge=5, le=1000, description="DDIM denoising steps"),
    seed: int | None = Query(None, description="Random seed; the same seed gives the same images"),
    scale: int = Query(4, ge=1, le=8, description="Upscaling factor for the PNG"),
):
    """Generate images by reverse diffusion (DDIM) from pure noise, returned as a PNG grid."""
    images = generate_diffusion_samples(_require_diffusion(), num_samples=num_images, steps=steps, seed=seed)
    return Response(content=images_to_png(images, scale=scale), media_type="image/png")


@app.post("/diffusion/generate")
def diffusion_generate(request: DiffusionGenerationRequest):
    """Generate images by reverse diffusion; returns the PNG grid as base64 plus the settings used."""
    images = generate_diffusion_samples(_require_diffusion(), num_samples=request.num_images,
                                        steps=request.steps, seed=request.seed)
    return {
        "num_images": request.num_images,
        "steps": request.steps,
        "seed": request.seed,
        "image_shape": [3, 32, 32],
        "image_format": "png",
        "image_base64": base64.b64encode(images_to_png(images, scale=request.scale)).decode("ascii"),
    }


@app.get("/diffusion/info")
def diffusion_info():
    """The loaded U-Net and noise schedule."""
    model = _require_diffusion()
    return {
        "dataset": "CIFAR-10",
        "image_shape": [3, 32, 32],
        "objective": "DDPM: predict the added noise, MSE loss",
        "timesteps": model.timesteps,
        "schedule": "cosine",
        "sampler": "DDIM (deterministic)",
        "epochs_trained": diffusion_epochs,
        "unet": {
            "class": "helper_lib.model.UNet",
            "parameters": sum(p.numel() for p in model.unet.parameters()),
            "time_embedding": "sinusoidal, 32 dimensions, max period 10000",
            "resolutions": [32, 16, 8, 4],
            "layers": _layers(model.unet),
        },
    }


# ---------- Assignment 4: energy-based model ----------
def _require_ebm():
    if energy_model is None:
        raise HTTPException(status_code=503, detail="Energy model not trained. Run: uv run python train_ebm_cifar10.py")
    return energy_model


@app.get(
    "/ebm/generate",
    responses={200: {"content": {"image/png": {}}, "description": "PNG grid of Langevin samples"}},
    response_class=Response,
)
def ebm_generate_image(
    num_images: int = Query(16, ge=1, le=64, description="How many images to generate"),
    steps: int = Query(200, ge=1, le=1000, description="Langevin dynamics steps starting from noise"),
    step_size: float = Query(10.0, gt=0, le=100, description="Langevin step size"),
    seed: int | None = Query(None, description="Random seed; the same seed gives the same images"),
    scale: int = Query(4, ge=1, le=8, description="Upscaling factor for the PNG"),
):
    """Generate images with Langevin dynamics on the trained energy function, returned as a PNG grid."""
    images, _, _ = generate_ebm_samples(_require_ebm(), num_samples=num_images, steps=steps,
                                        step_size=step_size, seed=seed)
    return Response(content=images_to_png(images, scale=scale), media_type="image/png")


@app.post("/ebm/generate")
def ebm_generate(request: EBMGenerationRequest):
    """Generate images with Langevin dynamics; returns the PNG grid as base64 plus each image's
    energy before (random noise) and after sampling, showing that Langevin dynamics lowers the energy."""
    images, before, after = generate_ebm_samples(_require_ebm(), num_samples=request.num_images,
                                                 steps=request.steps, step_size=request.step_size,
                                                 seed=request.seed)
    return {
        "num_images": request.num_images,
        "steps": request.steps,
        "step_size": request.step_size,
        "seed": request.seed,
        "image_shape": [3, 32, 32],
        "image_format": "png",
        "image_base64": base64.b64encode(images_to_png(images, scale=request.scale)).decode("ascii"),
        "energy_before": [round(e, 4) for e in before],
        "energy_after": [round(e, 4) for e in after],
        "mean_energy_before": round(sum(before) / len(before), 4),
        "mean_energy_after": round(sum(after) / len(after), 4),
    }


@app.get("/ebm/info")
def ebm_info():
    """The loaded energy model, layer by layer."""
    model = _require_ebm()
    return {
        "dataset": "CIFAR-10",
        "image_shape": [3, 32, 32],
        "training": "contrastive divergence with Langevin negative samples and a replay buffer",
        "sampling": "Langevin dynamics: x <- x + noise; x <- x - step_size * dE/dx (weights frozen)",
        "epochs_trained": ebm_epochs,
        "energy_model": {
            "class": "helper_lib.model.EnergyModel",
            "parameters": sum(p.numel() for p in model.parameters()),
            "layers": _layers(model),
        },
    }
