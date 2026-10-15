# Test Results

## Assignment 2: CIFAR-10 CNN classifier

### Training output

```
Using device: mps
Epoch 1/10 - loss: 1.5771 - train acc: 42.53%
Epoch 2/10 - loss: 1.2939 - train acc: 53.16%
Epoch 3/10 - loss: 1.1660 - train acc: 58.42%
Epoch 4/10 - loss: 1.0922 - train acc: 60.91%
Epoch 5/10 - loss: 1.0412 - train acc: 62.95%
Epoch 6/10 - loss: 0.9998 - train acc: 64.59%
Epoch 7/10 - loss: 0.9658 - train acc: 65.68%
Epoch 8/10 - loss: 0.9470 - train acc: 66.43%
Epoch 9/10 - loss: 0.9278 - train acc: 67.42%
Epoch 10/10 - loss: 0.9141 - train acc: 67.78%
Test loss: 0.8595 - Test accuracy: 69.70%
Saved model to models/cifar10_cnn.pth and metrics to models/cifar10_cnn_metrics.json
Saved 10 sample images to sample_images/
```

**Result:** the Assignment 2 CNN reaches **69.70% accuracy** on the 10,000 CIFAR-10 test images (random guessing would be 10%). ✅

### Sample images for testing `/classify`

One CIFAR-10 test image per class, in [`sample_images/`](../sample_images):

| | | | | |
| --- | --- | --- | --- | --- |
| ![airplane](../sample_images/airplane.png)<br>airplane | ![automobile](../sample_images/automobile.png)<br>automobile | ![bird](../sample_images/bird.png)<br>bird | ![cat](../sample_images/cat.png)<br>cat | ![deer](../sample_images/deer.png)<br>deer |
| ![dog](../sample_images/dog.png)<br>dog | ![frog](../sample_images/frog.png)<br>frog | ![horse](../sample_images/horse.png)<br>horse | ![ship](../sample_images/ship.png)<br>ship | ![truck](../sample_images/truck.png)<br>truck |

---

# Assignment 1 Results

The API was built and run with Docker on **October 1, 2026** (macOS, Apple Silicon) using:

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

All requests below were sent through the Swagger UI at http://127.0.0.1:8000/docs.

## Swagger UI running from the Docker container

![Swagger UI served by the Docker container](screenshots/swagger_ui_docker.png)

---

## 1. `POST /embedding` (Assignment 1)

**Request**

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "king"}'
```

**Response: `200 OK`**

```json
{
  "word": "king",
  "in_vocabulary": true,
  "dimension": 300,
  "embedding": [
    -0.606440007686615,
    -0.5120499730110168,
    0.006492100190371275,
    -0.29194000363349915,
    -0.5651500225067139,
    "... 290 more values ...",
    0.2286199927330017,
    0.21859000623226166,
    -0.19043999910354614,
    -0.10253000259399414
  ]
}
```

Full 300-value response: [`embedding_king_response.json`](embedding_king_response.json)

**Screenshots (Swagger UI, Docker)**

Request:

![POST /embedding request](screenshots/embedding_request.png)

Response (code 200, 300-dimensional vector):

![POST /embedding response](screenshots/embedding_response.png)

**Result:** the endpoint returns spaCy's 300-dimensional `en_core_web_md` vector for the query word. ✅

---

## 2. `POST /generate` (bigram model, Module 3)

**Request**

```json
{
  "start_word": "the",
  "length": 10
}
```

**Response: `200 OK`** (two separate runs)

```json
{ "generated_text": "the count of edmond dantès , who is falsely imprisoned" }
```

```json
{ "generated_text": "the story of edmond dantès , who is another example" }
```

**Screenshots (Swagger UI, Docker)**

Request:

![POST /generate request](screenshots/generate_request.png)

Response (code 200):

![POST /generate response](screenshots/generate_response.png)

**Result:** text is generated word by word from bigram probabilities. After "is", the model switched from the Monte Cristo sentence to "this is another example sentence", since "is" is followed by different words in the corpus. Output is random, so it varies between calls, as the two runs above show. ✅

---

## 3. `POST /similarity` (extra)

**Request**

```json
{
  "word1": "king",
  "word2": "queen"
}
```

**Response: `200 OK`**

```json
{
  "word1": "king",
  "word2": "queen",
  "similarity": 0.38253089785575867
}
```

**Result:** cosine similarity between the two word vectors. ✅

---

## 4. Input validation

| Request | Response |
| --- | --- |
| `POST /embedding` with `{"word": "two words"}` | `400` – `{"detail": "Please provide a single word."}` |
| `POST /embedding` with `{}` | `422` – validation error (missing `word`) |

## Summary

| Endpoint | Status |
| --- | --- |
| `GET /` | ✅ 200 |
| `POST /embedding` | ✅ 200, 300-d vector |
| `GET /embedding/{word}` | ✅ 200 |
| `POST /similarity` | ✅ 200 |
| `POST /generate` | ✅ 200 |
