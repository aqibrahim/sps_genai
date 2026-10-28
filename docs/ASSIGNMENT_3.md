# Assignment 3: Image Generation with a GAN

**Author:** Aqib Rahim · **Course:** Applied Generative AI · **Snapshot:** [`assignment-3` tag](https://github.com/aqibrahim/sps_genai/tree/assignment-3)

## What was asked

1. **Practice: GAN architecture.** Implement a GAN in PyTorch matching the given generator (noise 100 → FC 7×7×128 → ConvT 128→64 + BN + ReLU → ConvT 64→1 + Tanh) and discriminator (Conv 1→64 + LeakyReLU(0.2) → Conv 64→128 + BN + LeakyReLU(0.2) → Flatten → Linear to one real/fake output).
2. **Practice: model deployment.** Train the GAN on MNIST to generate hand-written digits, add it to the API, and commit to GitHub.
3. **Theory: building blocks of GANs.** Submitted as a PDF on Canvas.

## Where to find it

| Part | File |
| --- | --- |
| Generator and discriminator | [`helper_lib/model.py`](../helper_lib/model.py): classes `Generator`, `Discriminator`, `GAN` (`get_model("GAN")`) |
| MNIST data loader | [`helper_lib/data_loader.py`](../helper_lib/data_loader.py): `get_mnist_loader` (pixels scaled to [-1, 1]) |
| GAN training loop | [`helper_lib/trainer.py`](../helper_lib/trainer.py): `train_gan` |
| Sampling and image grids | [`helper_lib/generator.py`](../helper_lib/generator.py) |
| Training script | [`train_gan_mnist.py`](../train_gan_mnist.py) |
| Trained models | [`models/gan_generator.pth`](../models), [`models/gan_discriminator.pth`](../models), [`models/gan_metrics.json`](../models/gan_metrics.json) |
| API wrapper and endpoints | [`app/digit_generator.py`](../app/digit_generator.py), [`app/main.py`](../app/main.py): `GET /gan/generate`, `POST /gan/generate`, `GET /gan/info` |
| Tests | [`tests/test_gan.py`](../tests/test_gan.py) |

The architecture tables (every layer with its output shape) are in the [README](../README.md#assignment-3-gan-for-hand-written-digits).

## How to test

```bash
git clone https://github.com/aqibrahim/sps_genai.git
cd sps_genai
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open **http://127.0.0.1:8000/gan/generate?num_images=16&seed=42** in a browser to see 16 generated digits, or use **GET /gan/generate** and **POST /gan/generate** in http://127.0.0.1:8000/docs.

## Step-by-step: how it was built

### Step 1: Data pipeline

`get_mnist_loader` in [`helper_lib/data_loader.py`](../helper_lib/data_loader.py) loads the 60,000 MNIST training images (28×28 grayscale digits) through torchvision.

1. `ToTensor()` converts each image to a `1×28×28` tensor with pixel values in **[0, 1]**.
2. `Normalize((0.5,), (0.5,))` applies `(x − 0.5) / 0.5`, mapping pixels to **[−1, 1]**. This matters because the generator ends with **Tanh**, whose output range is also [−1, 1]: real and generated images must live in the same range, otherwise the discriminator could tell them apart by brightness alone.
3. Batches of 128 are shuffled every epoch; `drop_last=True` keeps every batch the same size, which keeps the batch-norm statistics consistent.

Labels are not used: a GAN learns the distribution of the images themselves (unsupervised).

### Step 2: Generator, layer by layer

The generator turns a random noise vector `z ~ N(0, I)` of size 100 into a 28×28 image. Output sizes follow the transposed-convolution formula from Part 3, **O = (I − 1)·S − 2P + K**:

| Step | Operation | Calculation | Output |
| --- | --- | --- | --- |
| 1 | Input noise | sampled from a standard normal | (B, 100) |
| 2 | `Linear(100 → 6272)` | 6272 = 7 · 7 · 128 | (B, 6272) |
| 3 | Reshape | 6272 values → 128 channels of 7×7 | (B, 128, 7, 7) |
| 4 | `ConvTranspose2d(128→64, k=4, s=2, p=1)` | (7 − 1)·2 − 2·1 + 4 = **14** | (B, 64, 14, 14) |
| 5 | `BatchNorm2d(64)` + `ReLU` | shape unchanged | (B, 64, 14, 14) |
| 6 | `ConvTranspose2d(64→1, k=4, s=2, p=1)` | (14 − 1)·2 − 2·1 + 4 = **28** | (B, 1, 28, 28) |
| 7 | `Tanh` | pixels squashed to [−1, 1] | (B, 1, 28, 28) |

Each transposed convolution with kernel 4, stride 2, padding 1 **exactly doubles** the spatial size (7 → 14 → 28), while the channels shrink (128 → 64 → 1) as the features become a single grayscale image.

**Parameters (765,761):** Linear 100·6272 + 6272 = 633,472; ConvT₁ 128·64·4·4 + 64 = 131,136; BatchNorm 2·64 = 128; ConvT₂ 64·1·4·4 + 1 = 1,025.

### Step 3: Discriminator, layer by layer

The discriminator is a binary classifier: image in, probability "real" out. Output sizes follow the convolution formula **O = ⌊(I − K + 2P) / S⌋ + 1**:

| Step | Operation | Calculation | Output |
| --- | --- | --- | --- |
| 1 | Input image | real (from MNIST) or generated | (B, 1, 28, 28) |
| 2 | `Conv2d(1→64, k=4, s=2, p=1)` + `LeakyReLU(0.2)` | ⌊(28 − 4 + 2)/2⌋ + 1 = **14** | (B, 64, 14, 14) |
| 3 | `Conv2d(64→128, k=4, s=2, p=1)` | ⌊(14 − 4 + 2)/2⌋ + 1 = **7** | (B, 128, 7, 7) |
| 4 | `BatchNorm2d(128)` + `LeakyReLU(0.2)` | shape unchanged | (B, 128, 7, 7) |
| 5 | `Flatten` | 128 · 7 · 7 = 6272 | (B, 6272) |
| 6 | `Linear(6272 → 1)` + `Sigmoid` | single score → probability in (0, 1) | (B, 1) |

The discriminator mirrors the generator: each convolution **halves** the size (28 → 14 → 7) while the channels grow (1 → 64 → 128). The specification asks for a single real/fake *probability*, so a `Sigmoid` turns the linear layer's output into a value between 0 and 1 (paired with binary cross-entropy loss).

**Parameters (138,817):** Conv₁ 1·64·4·4 + 64 = 1,088; Conv₂ 64·128·4·4 + 128 = 131,200; BatchNorm 2·128 = 256; Linear 6272 + 1 = 6,273.

### Step 4: Training algorithm

`train_gan` in [`helper_lib/trainer.py`](../helper_lib/trainer.py) alternates two updates on every batch of real images `x`:

**(a) Discriminator update.** Sample noise `z`, generate fakes `G(z)`, and train D to output 1 for real and 0 for fake:

L_D = BCE(D(x), 1) + BCE(D(G(z)), 0) = −log D(x) − log(1 − D(G(z)))

The fakes are passed with `.detach()`, so this step updates only the discriminator's weights.

**(b) Generator update.** Pass the same fakes through D again (now *without* detaching) and train G so that D labels them as real:

L_G = BCE(D(G(z)), 1) = −log D(G(z))

The gradient flows back *through* the discriminator into the generator, but only the generator's optimizer takes a step. This is the **non-saturating** generator loss from the Module 6 lecture: early in training D easily rejects the fakes (D(G(z)) ≈ 0), and the original minimax loss log(1 − D(G(z))) would then have an almost flat gradient, while −log D(G(z)) still gives a strong signal.

Per epoch the trainer records the average D loss, G loss, D(x) (mean score on real images) and D(G(z)) (mean score on fakes). After every epoch, `train_gan_mnist.py` saves a grid generated from the **same fixed noise** at epochs 1, 5, 10, 20 and 30, so the progress images are directly comparable.

### Step 5: Design choices

| Choice | Value | Why |
| --- | --- | --- |
| Optimizer | Adam, lr 0.0002, betas (0.5, 0.999) | The standard DCGAN setting; β₁ = 0.5 (instead of 0.9) reduces momentum, which makes the adversarial updates less likely to oscillate |
| Loss | Binary cross-entropy | Matches the sigmoid probability output of the discriminator |
| Generator loss | Non-saturating, −log D(G(z)) | Strong gradients even when D confidently rejects the fakes |
| Weight initialization | N(0, 0.02) for conv/linear; N(1, 0.02) for batch-norm scale | DCGAN initialization; keeps early activations small and training stable |
| Batch size | 128 | Large enough for stable batch-norm statistics |
| Epochs | 30 | Digits are clear by epoch 10–20; further epochs refine the strokes |
| Random seed | 42 | Reproducible training run and progress grids |
| Pixel range | [−1, 1] | Matches the Tanh output of the generator |

### Step 6: Serving the model through the API

When the API starts, [`app/digit_generator.py`](../app/digit_generator.py) loads `models/gan_generator.pth` (and the discriminator) once and switches both to **eval mode**, so batch norm uses the running statistics learned during training instead of per-request batch statistics. Each request to `/gan/generate` then:

1. creates a random-number generator from `seed` (if given), so the **same seed always returns the same digits**;
2. samples `num_images` noise vectors of size 100;
3. runs the generator under `torch.no_grad()` (no gradients needed for inference);
4. maps the Tanh output from [−1, 1] back to [0, 1] pixel values;
5. arranges the digits in a near-square grid (e.g. 16 → 4×4) and upscales it by `scale` with nearest-neighbour resizing, so the 28×28 digits stay sharp;
6. **GET** returns the PNG directly (`image/png`, viewable in a browser); **POST** returns JSON with the PNG as base64 plus `discriminator_scores`: the trained discriminator's probability that each generated digit is real.

Invalid input (for example `num_images` outside 1–64 or `scale` outside 1–8) is rejected with a 422 error before the model runs, and if the model file were missing the endpoint would return 503 with instructions instead of crashing.

## Training

30 epochs on the 60,000 MNIST training images (Apple M-series GPU): Adam (lr 0.0002, betas 0.5/0.999), BCE loss with the non-saturating generator loss from the Module 6 lecture, batch size 128, DCGAN weight initialization, random seed 42.

![Training output](screenshots/gan_training_output.png)

| Metric (epoch 30) | Value |
| --- | --- |
| Discriminator loss | 0.3207 |
| Generator loss | 3.2217 |
| D(x): mean output on real digits | 0.886 |
| D(G(z)): mean output on generated digits | 0.114 |
| Generator / discriminator parameters | 765,761 / 138,817 |

Full per-epoch history: [`models/gan_metrics.json`](../models/gan_metrics.json).

### Per-epoch metrics (all 30 epochs)

| Epoch | D loss | G loss | D(x) | D(G(z)) |
| --- | --- | --- | --- | --- |
| 1 | 0.5296 | 2.3542 | 0.812 | 0.196 |
| 2 | 0.5311 | 1.9017 | 0.791 | 0.209 |
| 3 | 0.6266 | 1.8364 | 0.769 | 0.231 |
| 4 | 0.6607 | 1.7783 | 0.760 | 0.240 |
| 5 | 0.6696 | 1.8129 | 0.758 | 0.242 |
| 6 | 0.5921 | 1.9305 | 0.782 | 0.218 |
| 7 | 0.6518 | 1.9200 | 0.768 | 0.231 |
| 8 | 0.5155 | 2.1665 | 0.811 | 0.189 |
| 9 | 0.5078 | 2.2355 | 0.818 | 0.183 |
| 10 | 0.4651 | 2.3349 | 0.830 | 0.170 |
| 11 | 0.4414 | 2.4555 | 0.839 | 0.161 |
| 12 | 0.4640 | 2.4783 | 0.836 | 0.164 |
| 13 | 0.4223 | 2.5875 | 0.849 | 0.151 |
| 14 | 0.5064 | 2.5904 | 0.834 | 0.165 |
| 15 | 0.4268 | 2.5225 | 0.840 | 0.161 |
| 16 | 0.4171 | 2.6786 | 0.851 | 0.148 |
| 17 | 0.4259 | 2.7296 | 0.853 | 0.147 |
| 18 | 0.3988 | 2.8324 | 0.862 | 0.139 |
| 19 | 0.3610 | 2.9198 | 0.871 | 0.129 |
| 20 | 0.4322 | 2.8706 | 0.854 | 0.146 |
| 21 | 0.4266 | 2.9348 | 0.857 | 0.143 |
| 22 | 0.4177 | 2.9321 | 0.862 | 0.138 |
| 23 | 0.3638 | 2.9641 | 0.871 | 0.130 |
| 24 | 0.3498 | 3.0025 | 0.875 | 0.126 |
| 25 | 0.3669 | 3.1464 | 0.878 | 0.121 |
| 26 | 0.3067 | 3.1199 | 0.887 | 0.113 |
| 27 | 0.3168 | 3.2326 | 0.885 | 0.114 |
| 28 | 0.3480 | 3.2277 | 0.885 | 0.115 |
| 29 | 0.3701 | 3.1937 | 0.876 | 0.123 |
| 30 | 0.3207 | 3.2217 | 0.886 | 0.114 |

How to read it: D(x) close to 1 and D(G(z)) close to 0 mean the discriminator separates real from fake easily; both near 0.5 would mean it cannot tell them apart. The lowest generator loss (1.7783) and highest D(G(z)) (0.242) occur around epochs 4–5, when the generator briefly caught up; after that the discriminator gradually regained the advantage while the sample quality kept improving.

## Results

### Training progress (same 64 noise vectors at every snapshot)

| Epoch 1 | Epoch 5 | Epoch 10 | Epoch 20 | Epoch 30 |
| --- | --- | --- | --- | --- |
| ![](screenshots/gan_epoch_01.png) | ![](screenshots/gan_epoch_05.png) | ![](screenshots/gan_epoch_10.png) | ![](screenshots/gan_epoch_20.png) | ![](screenshots/gan_epoch_30.png) |

Because the input noise is fixed, each grid position shows how the *same* sample evolves. After epoch 1 the outputs are blurry, noisy blobs with only rough digit outlines. By epoch 5 most samples are recognizable digits, and from epoch 10 to 30 the strokes become thinner, smoother and more consistent, with fewer broken or extra marks. Early on, some samples still change which digit they show (the top-left sample is a 3 at epoch 5 and a 5 at epoch 10), but by epoch 20 each sample has settled on a stable identity (the top-left becomes an 8 and stays an 8 at epoch 30, just cleaner). This shows the generator converging to a stable mapping from noise to digits.

### Final samples (64 new digits, different noise)

![Final samples](screenshots/gan_samples_final.png)

All ten digits (0–9) appear with varied slant, thickness and style, so the generator has **not mode-collapsed** (mode collapse would show the same few digits repeated). Some samples are still imperfect, for example ambiguous 4/9 shapes or incomplete loops, which is expected for a small two-layer generator trained for 30 epochs.

### Training curves

![Training curves](screenshots/gan_training_curves.png)

**Interpretation.**

- **Two phases.** In the first 5 epochs the generator catches up quickly: D(G(z)) rises from 0.20 to its peak of 0.24 and D(x) dips from 0.81 to 0.76, so the discriminator is briefly less sure. After that the discriminator pulls ahead again: by epoch 30, D(x) = 0.89 and D(G(z)) = 0.11, so it separates real from generated digits more confidently. A perfect equilibrium would put both near 0.5.
- **The generator loss rises, but the images improve.** The generator loss, −log D(G(z)), falls from 2.35 to its lowest point of 1.78 at epoch 4 and then grows to 3.22 by epoch 30 because the discriminator keeps getting better at spotting fakes. In a GAN the losses measure each network *against the other*, not absolute quality: the generator is improving, but the discriminator improves slightly faster. The sample grids are the real measure of quality, and they get steadily better.
- **Training is stable.** There is no collapse (the discriminator loss never goes to 0 and D(G(z)) never goes to 0), no oscillation, and no mode collapse. The non-saturating loss is what keeps the generator learning here: even with D(G(z)) ≈ 0.11, −log D(G(z)) still gives a strong gradient, whereas the original log(1 − D(G(z))) loss would give a very weak one (the vanishing-gradient problem from the lecture).
- **Possible improvements:** one-sided label smoothing (real label 0.9) or a slightly larger generator would balance the two networks; longer training would sharpen the remaining imperfect digits.

## Reproducing the results

```bash
git clone https://github.com/aqibrahim/sps_genai.git
cd sps_genai
uv sync                                              # exact dependencies from uv.lock
uv run --with matplotlib python train_gan_mnist.py   # downloads MNIST (~12 MB), trains 30 epochs
uv run --with pytest pytest -v                       # 46 tests
docker build -t sps-genai . && docker run -p 8000:80 sps-genai
```

Training with seed 42 regenerates the models, metrics and progress grids. Exact numbers can differ slightly between hardware (GPU kernels are not bit-for-bit deterministic), but the behaviour and image quality should match.

## API tests through Docker

**Browser:** `GET /gan/generate?num_images=16&seed=42` returns a PNG grid of new digits:

![GET /gan/generate in the browser](screenshots/gan_browser_docker.png)

**Swagger, GET /gan/generate:** the same seed returns the identical grid, showing generation is reproducible:

![GET /gan/generate in Swagger](screenshots/gan_get_swagger.png)

**Swagger, POST /gan/generate** with `{"num_images": 9, "seed": 42}`: returns `200` with the PNG as base64, plus the discriminator's probability that each digit is real:

![POST /gan/generate in Swagger](screenshots/gan_post_swagger.png)

## Automated tests and continuous integration

[![CI](https://github.com/aqibrahim/sps_genai/actions/workflows/ci.yml/badge.svg)](https://github.com/aqibrahim/sps_genai/actions/workflows/ci.yml)

Every push is also checked by GitHub Actions on x86-64 Linux: the full test suite, plus a fresh Docker build that starts the container and queries `GET` and `POST /gan/generate` (and every earlier assignment's endpoints). So the deployment is verified on both Apple Silicon (screenshots above) and standard x86-64 machines.


`uv run --with pytest pytest -v` runs **46 tests, all passing**. The 15 new Assignment 3 tests check every generator and discriminator layer's output shape and settings against the spec (kernel 4, stride 2, padding 1, LeakyReLU slope 0.2, BatchNorm placement), the Tanh output range, the sigmoid probability output, the parameter counts, seed reproducibility, and the `/gan` endpoints (PNG size, JSON format, input validation).

![pytest: 46 passed](screenshots/gan_pytest_results.png)
