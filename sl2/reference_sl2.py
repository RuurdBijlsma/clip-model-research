import os
import open_clip
import torch
from PIL import Image

# --- CONFIG & PATHS ---
# Using the SigLIP 2 model from reference.py
MODEL_ID = 'hf-hub:timm/ViT-SO400M-16-SigLIP2-384'
IMAGE_DIR = "assets/img"
QUERY_TEXT = "A photo of Rocks"

IMAGE_FILES = [
    "beach_rocks.jpg",
    "beetle_car.jpg",
    "cat_face.jpg",
    "dark_sunset.jpg",
    "palace.jpg",
    "rocky_coast.jpg",
    "stacked_plates.jpg",
    "verdant_cliff.jpg"
]

# 1. Load Model, Preprocess, and Tokenizer
print(f"Loading {MODEL_ID}...")
model, preprocess = open_clip.create_model_from_pretrained(MODEL_ID)
tokenizer = open_clip.get_tokenizer(MODEL_ID)

device = 'cpu'
model = model.to(device)
model.eval()

# Extract SigLIP specific scales
logit_scale = model.logit_scale.exp()
logit_bias = model.logit_bias

# 2. Process Images
print(f"Processing {len(IMAGE_FILES)} images...")
valid_names = []
image_tensors = []

for name in IMAGE_FILES:
    path = os.path.join(IMAGE_DIR, name)
    if os.path.exists(path):
        img = Image.open(path).convert("RGB")
        # preprocess returns (C, H, W), we need to collect them
        image_tensors.append(preprocess(img))
        valid_names.append(name)

# Stack images into a single batch (B, C, H, W)
image_input_batch = torch.stack(image_tensors).to(device)

with torch.no_grad():
    # encode_image with normalize=True performs L2 normalization internally
    image_embeds = model.encode_image(image_input_batch, normalize=True)

# 3. Process Text
print(f"Encoding query: '{QUERY_TEXT}'...")
query = QUERY_TEXT.lower()
text_input_ids = tokenizer([query], context_length=model.context_length).to(device)

with torch.no_grad():
    # encode_text with normalize=True performs L2 normalization internally
    text_embeds = model.encode_text(text_input_ids, normalize=True)

# --- DEBUG COMPARISON (Matching reference.py style) ---
print("\n--- DEBUG: PYTORCH VALUES (SigLIP 2) ---")
# 1. Text Input IDs
print(f"Text Input IDs (first 10): {text_input_ids[0][:10].tolist()}")

# 2. Image Input Tensors (First image in batch)
pix = image_input_batch[0]
print(f"Image Pixel Values - Mean: {pix.mean():.6f}, Std: {pix.std():.6f}")
print(f"Image Pixel Values (slice): {pix[0, 0, :5].tolist()}")

# 3. Text Embeddings
print(f"Text Embeds[0] - Mean: {text_embeds[0].mean():.6f}, Std: {text_embeds[0].std():.6f}")
print(f"Text Embeds[0] (first 5): {text_embeds[0][:5].tolist()}")

# 4. Image Embeddings (First Image)
print(f"Image Embeds[0] - Mean: {image_embeds[0].mean():.6f}, Std: {image_embeds[0].std():.6f}")
print(f"Image Embeds[0] (first 5): {image_embeds[0][:5].tolist()}")
# ------------------------

# 4. Calculate Similarities
# SigLIP Logic: sigmoid(image_embeds @ text_embeds.T * scale + bias)
with torch.no_grad():
    # (Batch_Img, Dim) @ (1, Dim).T -> (Batch_Img, 1)
    logits = (image_embeds @ text_embeds.T) * logit_scale + logit_bias

    # SigLIP is trained with sigmoid for pairwise matching
    # We use squeeze() to get a 1D array of probabilities for our images
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

# 5. Rank and Print Results
results = sorted(zip(valid_names, probs), key=lambda x: x[1], reverse=True)

print("\n--- SIGLIP 2 REFERENCE RESULTS ---")
print(f"Query: '{QUERY_TEXT}'")
for i, (name, prob) in enumerate(results):
    marker = "⭐ [BEST]" if i == 0 else "  "
    # SigLIP probabilities are absolute (0-1), not relative (softmax)
    print(f"{marker} {name}: {prob * 100:.2f}")
