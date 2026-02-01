import torch
import json
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- CONFIG & PATHS ---
MODEL_ID = "openai/clip-vit-base-patch32"
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

# 1. Load Model and Processor
print(f"Loading {MODEL_ID}...")
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

# Extract logit scale (this is what you saved to model_config.json)
logit_scale = model.logit_scale.exp().item()

# 2. Process Images
print(f"Processing {len(IMAGE_FILES)} images...")
images = []
valid_names = []

for name in IMAGE_FILES:
    path = os.path.join(IMAGE_DIR, name)
    if os.path.exists(path):
        images.append(Image.open(path).convert("RGB"))
        valid_names.append(name)

# Prepare inputs
image_inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
    # Get visual embeddings
    # Note: get_image_features does not automatically L2-normalize,
    # but your ONNX wrapper does, so we do it here manually for a 1:1 match.
    vision_outputs = model.get_image_features(**image_inputs)
    image_embeds = vision_outputs / vision_outputs.norm(p=2, dim=-1, keepdim=True)

# 3. Process Text
print(f"Encoding query: '{QUERY_TEXT}'...")
text_inputs = processor(
    text=[QUERY_TEXT],
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    # Get text embeddings and L2-normalize
    text_outputs = model.get_text_features(**text_inputs)
    text_embeds = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)

# --- DEBUG COMPARISON ---
print("\n--- DEBUG: PYTORCH VALUES ---")
# 1. Text Input IDs (Overview)
print(f"Text Input IDs (first 10): {text_inputs['input_ids'][0][:10].tolist()}")

# 2. Image Input Tensors (Overview of the first image)
pix = image_inputs['pixel_values'][0]
print(f"Image Pixel Values - Mean: {pix.mean():.6f}, Std: {pix.std():.6f}")
print(f"Image Pixel Values (slice): {pix[0, 0, :5].tolist()}") # First 5 pixels of top row

# 3. Text Embeddings
print(f"Text Embeds - Mean: {text_embeds.mean():.6f}, Std: {text_embeds.std():.6f}")
print(f"Text Embeds (first 5): {text_embeds[0][:5].tolist()}")

# 4. Image Embeddings (First Image)
print(f"Image Embeds[0] - Mean: {image_embeds[0].mean():.6f}, Std: {image_embeds[0].std():.6f}")
print(f"Image Embeds[0] (first 5): {image_embeds[0][:5].tolist()}")
# ------------------------

# 4. Calculate Similarities
# Logic: (Batch_Img, Dim) @ (1, Dim).T -> (Batch_Img, 1)
raw_similarities = (image_embeds @ text_embeds.T).squeeze()
scaled_scores = raw_similarities * logit_scale

# Softmax across the images to see which image matches the text best
probs = torch.nn.functional.softmax(scaled_scores, dim=0).numpy()

# 5. Rank and Print Results
results = sorted(zip(valid_names, probs), key=lambda x: x[1], reverse=True)

print("\n--- PYTORCH REFERENCE RESULTS ---")
print(f"Query: '{QUERY_TEXT}'")
for i, (name, prob) in enumerate(results):
    marker = "⭐ [BEST]" if i == 0 else "  "
    print(f"{marker} {name}: {prob*100:.2f}%")