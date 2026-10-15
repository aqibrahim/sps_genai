# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download the latest uv installer, run it, then remove it
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the dependency files first (better layer caching)
COPY pyproject.toml uv.lock README.md /code/

# Install dependencies (spaCy model, CPU-only PyTorch, ...) exactly as locked
RUN uv sync --frozen

# Copy the application code, the helper library and the trained model weights
COPY ./helper_lib /code/helper_lib
COPY ./app /code/app
COPY ./models /code/models

# Command to run the application
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
