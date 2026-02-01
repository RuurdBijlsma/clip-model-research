import torch
import os
import open_clip
from PIL import Image

# --- CONFIG & PATHS ---
# Using the specific Marqo Fashion SigLIP model
MODEL_ID = 'hf-hub:Marqo/marqo-fashionSigLIP'
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
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
tokenizer = open_clip.get_tokenizer(MODEL_ID)
model = model.to(device)
model.eval()

# 2. Process Images
print(f"Processing {len(IMAGE_FILES)} images...")
processed_images = []
valid_names = []

for name in IMAGE_FILES:
    path = os.path.join(IMAGE_DIR, name)
    if os.path.exists(path):
        # open_clip's preprocess expects a PIL image and returns a tensor
        img = Image.open(path).convert("RGB")
        processed_images.append(preprocess(img))
        valid_names.append(name)

# Stack all images into a single batch tensor [Batch, 3, 224, 224]
image_input = torch.stack(processed_images).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    # encode_image with normalize=True handles L2 normalization automatically
    image_features = model.encode_image(image_input, normalize=True)

# 3. Process Text
print(f"Encoding query: '{QUERY_TEXT}'...")
text_input = tokenizer([QUERY_TEXT]).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    # encode_text with normalize=True handles L2 normalization automatically
    text_features = model.encode_text(text_input, normalize=True)

# --- DEBUG COMPARISON ---
print("\n--- DEBUG: OPEN_CLIP VALUES ---")
print(f"Image Features Shape: {list(image_features.shape)}")
print(f"Text Features Shape: {list(text_features.shape)}")

# Image Embeddings (First Image)
img_0 = image_features[0]
print(f"Image Embeds[0] - Mean: {img_0.mean():.6f}, Std: {img_0.std():.6f}")
print(f"Image Embeds[0] (first 5): {img_0[:5].tolist()}")

# Text Embeddings
txt_0 = text_features[0]
print(f"Text Embeds - Mean: {txt_0.mean():.6f}, Std: {txt_0.std():.6f}")
print(f"Text Embeds (first 5): {txt_0[:5].tolist()}")
# ------------------------

# 4. Calculate Similarities
with torch.no_grad():
    # Logic: (Batch_Img, Dim) @ (1, Dim).T -> (Batch_Img, 1)
    # The example code for Marqo uses 100.0 as the scale factor
    logits_per_image = 100.0 * image_features @ text_features.T

    # Softmax across the images (dim=0) to see which image matches the text best
    probs = torch.nn.functional.softmax(logits_per_image.squeeze(), dim=0).cpu().numpy()

# 5. Rank and Print Results
results = sorted(zip(valid_names, probs), key=lambda x: x[1], reverse=True)

print("\n--- FASHION SIGLIP RESULTS ---")
print(f"Query: '{QUERY_TEXT}'")
for i, (name, prob) in enumerate(results):
    marker = "⭐ [BEST]" if i == 0 else "  "
    print(f"{marker} {name}: {prob * 100:.2f}%")