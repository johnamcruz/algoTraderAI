# Use a Python base image with necessary compiler tools
FROM python:3.12-slim

# Set environment variables to prevent issues and specify working directory
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# 1. Install system dependencies required by some libraries (like numpy/pandas-ta)
# Run apt update/install in a single layer to minimize image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements and install Python dependencies
# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# 3. Copy application core files
COPY algoTrader.py /app/
COPY strategy_base.py /app/
COPY strategy_factory.py /app/
COPY config_loader.py /app/

# 4. Copy Strategy Implementation Files
COPY strategy_pivot_reversal_3min.py /app/
COPY strategy_pivot_reversal_5min.py /app/
COPY strategy_squeeze.py /app/
COPY strategy_vwap_3min.py /app/

# 5. Copy the entire 'models' directory and its contents
COPY models/ /app/models/

# Define entrypoint (optional, but good practice)
ENTRYPOINT ["python", "algoTrader.py"]

# Default command (will be overridden by docker-compose)
CMD ["--help"]