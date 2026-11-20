"""Chest X-ray reader — FastAPI backend for the pneumonia XAI demo.

Serves the single-page UI and two endpoints:
  POST /api/analyze  -> predictions + Grad-CAM overlays from ResNet50 and DenseNet121
  POST /api/ig       -> Integrated Gradients overlay for one model (on demand, slower)
"""
import base64, io, os, uuid
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(HERE, "models")
SAMPLE_DIR = os.path.join(HERE, "samples")
CLASSES    = ["NORMAL", "PNEUMONIA"]
THRESHOLD  = 0.5
IG_STEPS   = int(os.environ.get("IG_STEPS", 48))
torch.set_num_threads(max(1, os.cpu_count() or 1))
device = torch.device("cpu")

# ── Models (identical architecture to the training notebook) ──────────────
def build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 512), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(512, 2))
    return m

class DenseNetXAI(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features
        self.classifier = nn.Linear(base.classifier.in_features, 2)
    def forward(self, x):
        out = F.relu(self.features(x), inplace=False)
        return self.classifier(F.adaptive_avg_pool2d(out, 1).flatten(1))

def load(name, builder, layer_fn):
    m = builder()
    sd = torch.load(os.path.join(MODEL_DIR, f"{name}.pth"), map_location="cpu")
    m.load_state_dict(sd)
    m.eval()
    return dict(name=name, model=m, layer=layer_fn(m))

MODELS = [
    load("ResNet50", build_resnet50, lambda m: m.layer4[-1]),
    load("DenseNet121", DenseNetXAI, lambda m: m.features.denseblock4),
]

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])

# ── Colour map (inferno, transparent where attention is low) ──────────────
_INFERNO = np.array([
    [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85],
    [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]], dtype=np.float32)

def colorize(heat, size):
    """heat: HxW in [0,1] -> RGBA PNG data URL, resized to `size`."""
    h = Image.fromarray((heat * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    v = np.asarray(h, dtype=np.float32) / 255.0
    pos = v * (len(_INFERNO) - 1)
    lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, len(_INFERNO) - 1)
    t = (pos - lo)[..., None]
    rgb = _INFERNO[lo] * (1 - t) + _INFERNO[hi] * t
    alpha = np.clip((v - 0.15) / 0.55, 0, 1) ** 0.8 * 235
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return png_url(Image.fromarray(rgba, "RGBA"))

def png_url(img):
    buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def jpeg_url(img):
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ── Explanations ──────────────────────────────────────────────────────────
def grad_cam(entry, x, target):
    store = {}
    def fwd(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))
    h = entry["layer"].register_forward_hook(fwd)
    try:
        x = x.clone().requires_grad_(True)
        logits = entry["model"](x)
        entry["model"].zero_grad(set_to_none=True)
        logits[0, target].backward()
    finally:
        h.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    return cam / (cam.max() + 1e-8)

def integrated_gradients(entry, x, target, steps=IG_STEPS):
    baseline = (torch.zeros_like(x) - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
    total = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps + 1)[1:]
    for chunk in alphas.split(16):
        path = baseline + chunk.view(-1, 1, 1, 1) * (x - baseline)
        path.requires_grad_(True)
        out = entry["model"](path)[:, target].sum()
        g, = torch.autograd.grad(out, path)
        total += g.sum(0, keepdim=True)
    attr = ((x - baseline) * total / steps)[0].sum(0).abs().numpy()
    # light smoothing so the map is readable at display size
    attr = F.avg_pool2d(torch.from_numpy(attr)[None, None], 5, 1, 2)[0, 0].numpy()
    hi = np.percentile(attr, 99.5)
    return np.clip(attr / (hi + 1e-8), 0, 1)

# ── Image cache so IG can reuse an uploaded image ────────────────────────
CACHE: "OrderedDict[str, dict]" = OrderedDict()
def remember(item):
    key = uuid.uuid4().hex
    CACHE[key] = item
    while len(CACHE) > 64:
        CACHE.popitem(last=False)
    return key

def display_size(img, longest=720):
    s = longest / max(img.size)
    return (max(1, round(img.width * s)), max(1, round(img.height * s)))

# ── API ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Chest X-ray reader")

@app.get("/api/samples")
def samples():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(SAMPLE_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            label = "Pneumonia" if fn.upper().startswith(("PERSON", "PNEUMONIA")) else "Normal"
            out.append(dict(name=fn, url=f"/samples/{fn}", label=label))
    return out

@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), sample: str | None = Form(None)):
    try:
        if file is not None:
            img = Image.open(io.BytesIO(await file.read()))
        elif sample:
            path = os.path.join(SAMPLE_DIR, os.path.basename(sample))
            if not os.path.isfile(path):
                raise HTTPException(404, "That sample image is not available.")
            img = Image.open(path)
        else:
            raise HTTPException(400, "Upload an image or pick a sample.")
        img = img.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(415, "This file is not an image. Upload a JPG or PNG chest X-ray.")

    size = display_size(img)
    x = to_tensor(img).unsqueeze(0)
    readings = []
    for entry in MODELS:
        with torch.no_grad():
            p = torch.softmax(entry["model"](x), 1)[0]
        prob = float(p[1])
        target = int(prob >= THRESHOLD)
        readings.append(dict(name=entry["name"], prob_pneumonia=round(prob, 4),
                             label=CLASSES[target], cam=colorize(grad_cam(entry, x, target), size)))
    key = remember(dict(x=x, size=size, targets=[CLASSES.index(r["label"]) for r in readings]))
    return JSONResponse(dict(id=key, image=jpeg_url(img.resize(size)), width=size[0], height=size[1],
                             threshold=THRESHOLD, models=readings))

@app.post("/api/ig")
def ig(id: str = Form(...), model: str = Form(...)):
    item = CACHE.get(id)
    if item is None:
        raise HTTPException(410, "This image has expired. Analyze it again.")
    idx = next((i for i, e in enumerate(MODELS) if e["name"] == model), None)
    if idx is None:
        raise HTTPException(404, "Unknown model.")
    heat = integrated_gradients(MODELS[idx], item["x"], item["targets"][idx])
    return dict(model=model, overlay=colorize(heat, item["size"]))

app.mount("/samples", StaticFiles(directory=SAMPLE_DIR), name="samples")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))
