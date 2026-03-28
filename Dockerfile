FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    Pillow matplotlib scikit-learn tqdm numpy opencv-python-headless gradio

# Copy project files
COPY app.py .
COPY predict.py .
COPY models/ ./models/

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]