<<<<<<< HEAD
# Use a lightweight, modern Python 3.11 image 
FROM python:3.11-slim

# Create a non-root user that Hugging Face Spaces expects
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy your requirements.txt first (this makes future uploads much faster)
COPY --chown=user requirements.txt requirements.txt

# Install all the python libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the rest of your files (app.py, utils/, model/, etc.) into the container
COPY --chown=user . /app

# Hugging Face Docker Spaces strictly require apps to broadcast on port 7860
EXPOSE 7860

# The command to launch your Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
=======
FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
>>>>>>> f8797673035db62e2d5f24f43efb94fc15349ebb
