import os
import time
import torch
import open_clip
import torch.nn.functional as F
from PIL import Image
from timm.utils import reparameterize_model

# -------------------------
# Configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ASSETS_IMG_DIR = "assets/img"
QUERY_TEXT = "A photo of Rocks"
IMAGE_FILES = [
    "beach_rocks.jpg",
    "beetle_car.jpg",
    "cat_face.jpg",
    "dark_sunset.jpg",
    "palace.jpg",
    "rocky_coast.jpg",
    "stacked_plates.jpg",
    "verdant_cliff.jpg",
]

# -------------------------
# Model Setup
# -------------------------
print("🚀 Loading model...")
model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S4', pretrained='dfndr2b')
model = model.to(device)
model.eval()

# Critical for MobileCLIP performance and matching ONNX behavior
model = reparameterize_model(model)
tokenizer = open_clip.get_tokenizer('MobileCLIP2-S4')

# -------------------------
# Load and Preprocess Data
# -------------------------
# Load query
text_input = tokenizer([QUERY_TEXT]).to(device)

# Load images
loaded_images = []
valid_names = []
for name in IMAGE_FILES:
    path = os.path.join(ASSETS_IMG_DIR, name)
    if os.path.exists(path):
        img = Image.open(path)
        loaded_images.append(preprocess(img))
        valid_names.append(name)

image_input = torch.stack(loaded_images).to(device)

# -------------------------
# Warmup
# -------------------------
with torch.no_grad():
    for _ in range(3):
        _ = model.encode_image(image_input, normalize=True)
        _ = model.encode_text(text_input, normalize=True)

# -------------------------
# Inference & Timing
# -------------------------
print(f"🧠 Embedding {len(valid_names)} images...")
t_start = time.perf_counter()

with torch.no_grad(), torch.amp.autocast(device_type=device.type):
    # Encode Text
    text_features = model.encode_text(text_input, normalize=True)
    # Encode Images
    image_features = model.encode_image(image_input, normalize=True)

if device.type == "cuda":
    torch.cuda.synchronize()
t_end = time.perf_counter()

# -------------------------
# Debug Comparison Stats
# -------------------------
print(f"⚡ Inference completed in {(t_end - t_start) * 1000:.2f} ms")
print(f"\n--- DEBUG INFO ---")
print(f"Text Embeds[0]  - Mean: {text_features[0].mean():.6f}, Std: {text_features[0].std():.6f}")
print(f"Text Embeds[0]  (first 5): {text_features[0][:5].tolist()}")
print(f"Image Embeds[0] - Mean: {image_features[0].mean():.6f}, Std: {image_features[0].std():.6f}")
print(f"Image Embeds[0] (first 5): {image_features[0][:5].tolist()}")

# -------------------------
# Similarity & Probabilities
# -------------------------
scale = model.logit_scale.exp()
bias = getattr(model, 'logit_bias', 0.0)
bias = 0.0 if bias is None else bias

with torch.no_grad():
    # Dot product: [Num_Images, D] @ [D, 1] -> [Num_Images]
    similarities = image_features @ text_features.T

    # Apply scale and bias
    logits = (similarities * scale) + bias

    # Softmax over the images (matching your Rust Softmax logic)
    probs = F.softmax(logits, dim=0).squeeze()

# -------------------------
# Display Results
# -------------------------
results = sorted(zip(valid_names, probs.tolist()), key=lambda x: x[1], reverse=True)

print(f"\n🔍 SEARCH RESULTS")
print(f"Query: \"{QUERY_TEXT}\"")
print(f"Logit Scale: {scale.item():.4f} | Logit Bias: {bias if isinstance(bias, float) else bias.item():.4f}")
print("-" * 60)

for i, (name, prob) in enumerate(results):
    marker = "★ [BEST]" if i == 0 else "  "
    print(f"{marker} {name:<20} | {prob * 100:>6.2f}%")
