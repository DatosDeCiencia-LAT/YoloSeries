# ── Base image ─────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── System dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code and model ─────────────────────────────────────────
COPY app/ ./app/
COPY sdo_best.pt .

# ── Runtime ────────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]