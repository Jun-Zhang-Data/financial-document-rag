# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY src ./src
COPY data ./data
COPY eval_questions.json ./eval_questions.json

# Production clusters can bake the embedding model into the image so pods do not need
# outbound access to Hugging Face at startup. Local/CI builds can leave this disabled.
ARG PRELOAD_EMBEDDING_MODEL=0
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN if [ "$PRELOAD_EMBEDDING_MODEL" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}').save('/opt/embedding-model')"; \
    fi

RUN chown -R app:app /app /home/app /opt/embedding-model 2>/dev/null || chown -R app:app /app /home/app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=2)"

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
