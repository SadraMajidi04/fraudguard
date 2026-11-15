FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install -r requirements.txt && pip install .
# Train inside the image so the model matches the installed library versions.
RUN python -m fraudguard.cli train --n 60000 --out-dir /app/models
EXPOSE 8000
# bind to all interfaces INSIDE the container; publish the port explicitly with -p
CMD ["python", "-m", "fraudguard.cli", "serve", "--model", "/app/models/model.joblib", "--host", "0.0.0.0", "--port", "8000"]
