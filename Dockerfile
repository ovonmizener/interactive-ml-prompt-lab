# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/uploads

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Interactive ML & Prompting Playground..."\n\
echo "FastAPI will be available at http://localhost:8000"\n\
echo "Streamlit will be available at http://localhost:8501"\n\
\n\
# Start FastAPI in background\n\
python app.py &\n\
FASTAPI_PID=$!\n\
\n\
# Wait a moment for FastAPI to start\n\
sleep 3\n\
\n\
# Start Streamlit\n\
streamlit run frontend/1_data_explorer.py --server.port 8501 --server.address 0.0.0.0 --server.headless true\n\
\n\
# Wait for background process\n\
wait $FASTAPI_PID\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["/app/start.sh"] 