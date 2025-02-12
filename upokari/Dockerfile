# Stage 1: Build dlib
FROM python:3.11-slim AS build-stage

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    libboost-python-dev \
    libboost-system-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dlib
RUN pip install dlib

# Stage 2: Final image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (including libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dlib from build stage
COPY --from=build-stage /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the rest of the application
COPY upokari /app/

# Copy .env for environment variables
COPY .env /app/.env

# Install Python dependencies
COPY upokari/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port and run the app
EXPOSE 6000
CMD ["python", "manage.py", "runserver", "0.0.0.0:6000"]
