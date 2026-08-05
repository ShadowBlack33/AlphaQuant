FROM python:3.11-slim

WORKDIR /app

# FIX (Fase 0): previous Dockerfile did `COPY ia-financiera /app/ia-financiera`,
# a folder that doesn't exist in this repo (leftover from the old project name).
# Copy the actual repo contents instead.
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY config ./config
COPY tests ./tests

RUN pip install --no-cache-dir -e .

# Default command: run tests (safe for CI / image validation)
CMD ["pytest", "-q"]
