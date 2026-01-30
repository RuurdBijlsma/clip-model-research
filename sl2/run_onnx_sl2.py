import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

# --- CONFIG & PATHS ---
MODEL_DIR = "assets/model"
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 1. Load Config and Tokenizer
with open(os.path.join(MODEL_DIR, "model_config.json"), "r") as f:
    config = json.load(f)

logit_scale = config["logit_scale"]
logit_bias = config["logit_bias"]
img_size = config["image_size"]
context_length = config["context_length"]

print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
# Ensure tokenizer pads/truncates to the model's context length
tokenizer.enable_padding(length=context_length, pad_id=0)  # <pad> is 0
tokenizer.enable_truncation(max_length=context_length)

# 2. Initialize ONNX Sessions
providers = ['CPUExecutionProvider']
print("Initializing ONNX sessions...")
visual_session = ort.InferenceSession(os.path.join(MODEL_DIR, "visual.onnx"), providers=providers)
text_session = ort.InferenceSession(os.path.join(MODEL_DIR, "text.onnx"), providers=providers)


# 3. Preprocess Images
def preprocess_image(image_path, size):
    img = Image.open(image_path).convert("RGB")
    # SigLIP 2 uses 'squash' resize (non-aspect-ratio preserving)
    img = img.resize((size, size), resample=Image.BICUBIC)

    img_array = np.array(img).astype(np.float32) / 255.0
    # Normalize: (val - 0.5) / 0.5
    img_array = (img_array - 0.5) / 0.5
    # HWC to CHW
    img_array = img_array.transpose(2, 0, 1)
    return img_array


print(f"Processing {len(IMAGE_FILES)} images...")
valid_names = []
image_arrays = []

for name in IMAGE_FILES:
    path = os.path.join(IMAGE_DIR, name)
    if os.path.exists(path):
        image_arrays.append(preprocess_image(path, img_size))
        valid_names.append(name)

image_input_batch = np.stack(image_arrays)

# 4. Image Inference
print("Running Vision ONNX model...")
image_embeds = visual_session.run(
    ["image_embeddings"],
    {"pixel_values": image_input_batch}
)[0]

# 5. Text Inference
print(f"Encoding query: '{QUERY_TEXT}'...")
# Gemma tokenizer behavior: usually prepend BOS, append EOS
# The tokenizer.json usually has these rules baked in, but we handle the IDs here
encoded = tokenizer.encode(QUERY_TEXT.lower())
text_input_ids = np.array([encoded.ids], dtype=np.int64)

print("Running Text ONNX model...")
text_embeds = text_session.run(
    ["text_embeddings"],
    {"input_ids": text_input_ids}
)[0]

# --- DEBUG COMPARISON ---
print("\n--- DEBUG: ONNX VALUES (SigLIP 2) ---")
print(f"Text Input IDs (first 10): {text_input_ids[0][:10].tolist()}")

pix = image_input_batch[0]
print(f"Image Pixel Values - Mean: {pix.mean():.6f}, Std: {pix.std():.6f}")
print(f"Image Pixel Values (slice): {pix[0, 0, :5].tolist()}")

print(f"Text Embeds[0] - Mean: {text_embeds[0].mean():.6f}, Std: {text_embeds[0].std():.6f}")
print(f"Text Embeds[0] (first 5): {text_embeds[0][:5].tolist()}")

print(f"Image Embeds[0] - Mean: {image_embeds[0].mean():.6f}, Std: {image_embeds[0].std():.6f}")
print(f"Image Embeds[0] (first 5): {image_embeds[0][:5].tolist()}")

# 6. Calculate Similarities
# SigLIP Logic: sigmoid(image_embeds @ text_embeds.T * scale + bias)
# (Batch_Img, Dim) @ (Dim, 1) -> (Batch_Img, 1)
logits = (image_embeds @ text_embeds.T) * logit_scale + logit_bias
probs = sigmoid(logits).flatten()

# 7. Rank and Print Results
results = sorted(zip(valid_names, probs), key=lambda x: x[1], reverse=True)

print("\n--- SIGLIP 2 ONNX RESULTS ---")
print(f"Query: '{QUERY_TEXT}'")
for i, (name, prob) in enumerate(results):
    marker = "⭐ [BEST]" if i == 0 else "  "
    print(f"{marker} {name}: {prob * 100:.2f}")