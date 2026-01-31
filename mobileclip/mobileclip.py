import time
import torch
import open_clip
from urllib.request import urlopen
from PIL import Image
from timm.utils import reparameterize_model

# -------------------------
# Model Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, transforms, and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S4', pretrained='dfndr2b')
model = model.to(device)
model.eval()

# Reparameterize for better inference performance
model = reparameterize_model(model)

tokenizer = open_clip.get_tokenizer('MobileCLIP2-S4')

# Load Image
image = Image.open("assets/img/beach_rocks.jpg")
labels_list = ["A photo of Rocks", "a cat", "a donut", "a beignet"]

# -------------------------
# Prepare inputs once
# -------------------------
image_tensor = preprocess(image).unsqueeze(0).to(device)
text_tensor = tokenizer(labels_list).to(device)

# -------------------------
# Warmup
# -------------------------
warmup_iters = 5
# Note: MobileCLIP uses standard CLIP normalization via the encode functions
with torch.no_grad(), torch.amp.autocast(device_type=device.type):
    for _ in range(warmup_iters):
        _ = model.encode_image(image_tensor, normalize=True)
        _ = model.encode_text(text_tensor, normalize=True)

if device.type == "cuda":
    torch.cuda.synchronize()

# -------------------------
# Timed image preprocess + embed
# -------------------------
t0 = time.perf_counter()

# Note: preprocessing happens on CPU usually, then move to device
image_tensor = preprocess(image).unsqueeze(0).to(device)

if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

with torch.no_grad(), torch.amp.autocast(device_type=device.type):
    image_features = model.encode_image(image_tensor, normalize=True)

if device.type == "cuda":
    torch.cuda.synchronize()
t2 = time.perf_counter()

# -------------------------
# Timed text tokenize + embed
# -------------------------
t3 = time.perf_counter()

text_tensor = tokenizer(labels_list).to(device)

if device.type == "cuda":
    torch.cuda.synchronize()
t4 = time.perf_counter()

with torch.no_grad(), torch.amp.autocast(device_type=device.type):
    text_features = model.encode_text(text_tensor, normalize=True)

if device.type == "cuda":
    torch.cuda.synchronize()
t5 = time.perf_counter()

# --- DEBUG COMPARISON ---
# 3. Text Embeddings
print(f"Text Embeds[0] - Mean: {text_features[0].mean():.6f}, Std: {text_features[0].std():.6f}")
print(f"Text Embeds[0] (first 5): {text_features[0][:5].tolist()}")

# 4. Image Embeddings (First Image)
print(f"Image Embeds[0] - Mean: {image_features[0].mean():.6f}, Std: {image_features[0].std():.6f}")
print(f"Image Embeds[0] (first 5): {image_features[0][:5].tolist()}")
# ------------------------

# -------------------------
# Similarity
# -------------------------
# CLIP models typically use Softmax, whereas SigLIP uses Sigmoid.
# Using the model's internal logit_scale (usually 100.0 for MobileCLIP)
with torch.no_grad():
    logits_per_image = image_features @ text_features.T * model.logit_scale.exp()
    text_probs = logits_per_image.softmax(dim=-1)

zipped_list = list(zip(labels_list, [100 * round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities:", zipped_list)

print(f"\nPerformance Metrics:")
print(f"Image preprocess: {(t1 - t0) * 1e3:.2f} ms")
print(f"Image embedding:  {(t2 - t1) * 1e3:.2f} ms")
print(f"Text tokenize:    {(t4 - t3) * 1e3:.2f} ms")
print(f"Text embedding:   {(t5 - t4) * 1e3:.2f} ms")