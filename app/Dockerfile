FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch keeps the image small enough for a free Space
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
