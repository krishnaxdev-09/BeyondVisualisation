---
title: Chest X-ray Reader
emoji: 🫁
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Chest X-ray reader

This is a web demo for the paper **"Beyond Visual Explanations"**. Upload a chest X-ray to see:

- the probability of pneumonia from **ResNet50** and from **DenseNet121**,
- **Grad-CAM** heatmaps for both models, with a divider you can drag to compare them on the same image,
- an on-demand **Integrated Gradients** map that shows detail at the pixel level.

This is a research demo, not a medical device.

## Files

```
app.py              FastAPI backend: predictions, Grad-CAM, Integrated Gradients
static/index.html   The interface (one HTML file with its own CSS and JavaScript)
models/             ResNet50.pth and DenseNet121.pth, copied from the notebook's results/models/
samples/            6–8 test images from Data/sample_images/ (optional, shown as one-click examples)
Dockerfile          Build instructions for the Hugging Face Space
requirements.txt    Python packages the app needs
```

## Run locally

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --port 7860
# open http://localhost:7860
```

