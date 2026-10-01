# SPS GenAI – Assignment 1

FastAPI service with two models:

- **Bigram text generator** (Module 3 class activity) – `POST /generate`
- **spaCy word embeddings** (Module 2, Assignment 1) – `POST /embedding`, `GET /embedding/{word}`, `POST /similarity`

Embeddings come from spaCy's `en_core_web_md` model (300-dimensional vectors). The model is a locked dependency in `pyproject.toml`, so `uv sync` installs it automatically – no separate `spacy download` step is needed.

## Project structure

```
sps_genai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app and endpoints
│   ├── bigram_model.py      # Bigram language model
│   └── embedding_model.py   # spaCy word-embedding wrapper
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── probability_solutions.py # Part 2 calculations (not part of the API)
```

## Run locally (uv)

```bash
uv sync
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs.

## Example requests

```bash
# Health check
curl http://127.0.0.1:8000/

# Bigram text generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 8}'

# Word embedding (POST)
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "king"}'

# Word embedding (GET – works in a browser)
curl http://127.0.0.1:8000/embedding/apple

# Similarity between two words
curl -X POST http://127.0.0.1:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"word1": "king", "word2": "queen"}'
```

### `/embedding` response

```json
{
  "word": "king",
  "in_vocabulary": true,
  "dimension": 300,
  "embedding": [-0.6064, -0.5120, 0.0065, ...]
}
```

`in_vocabulary` is `false` when the word has no vector in the model (the embedding is then all zeros). Multi-word input returns HTTP 400.
