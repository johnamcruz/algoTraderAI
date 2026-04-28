FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# git required for futures-foundation-model pip dependency
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Application source
COPY algoTrader.py backtest.py ./
COPY bots/       ./bots/
COPY strategies/ ./strategies/
COPY utils/      ./utils/

# Models baked into image (never change between runs)
COPY models/ ./models/

# Runtime directories created at build time so volume mounts work
RUN mkdir -p logs backtest_logs configs

ENTRYPOINT ["python", "algoTrader.py"]
CMD ["--help"]
