# ── Stage 1: train the model ─────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN python train.py

# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY templates/ templates/

# Copy trained model from the trainer stage
COPY --from=trainer /app/savedmodel.pth .

ENV MODEL_PATH=savedmodel.pth
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["python", "app.py"]
