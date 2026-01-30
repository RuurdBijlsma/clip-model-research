import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPProcessor

# Config
CONFIG_PATH = "assets/model_openai/model_config.json"
VISUAL_MODEL_PATH = "assets/model_openai/visual.onnx"
TEXT_MODEL_PATH = "assets/model_openai/text.onnx"
PROCESSOR_PATH = "assets/model_openai"
IMAGE_DIR = "assets/img"
QUERY_TEXT = "a photo of rocks"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    LOGIT_SCALE = config.get("logit_scale", 100.0)

# List of specific files to check
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

# 1. Load Components
print(f"Loading processor and ONNX models...")
processor = CLIPProcessor.from_pretrained(PROCESSOR_PATH)

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
visual_session = ort.InferenceSession(VISUAL_MODEL_PATH, providers=providers)
text_session = ort.InferenceSession(TEXT_MODEL_PATH, providers=providers)


def get_image_embeddings(image_paths):
    """Loads images and returns a batch of embeddings."""
    images = []
    valid_paths = []

    for path in image_paths:
        full_path = os.path.join(IMAGE_DIR, path)
        if os.path.exists(full_path):
            images.append(Image.open(full_path).convert("RGB"))
            valid_paths.append(path)
        else:
            print(f"Warning: {full_path} not found.")

    if not images:
        return None, []

    # Preprocess all images at once
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].numpy()

    # Run ONNX inference (batched)
    onnx_inputs = {visual_session.get_inputs()[0].name: pixel_values}
    embeddings = visual_session.run(None, onnx_inputs)[0]

    return embeddings, valid_paths


def get_text_embedding(text):
    """Returns the embedding for a single text query."""
    inputs = processor(
        text=[text],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )

    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }

    embedding = text_session.run(None, onnx_inputs)[0]
    return embedding


# 2. Main Search Logic
print(f"Processing {len(IMAGE_FILES)} images...")
img_embs, image_names = get_image_embeddings(IMAGE_FILES)

print(f"Encoding query: '{QUERY_TEXT}'...")
text_emb = get_text_embedding(QUERY_TEXT)

# 3. Calculate Similarities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# We use dot product because vectors were L2-normalized during export.
# (Batch, Dim) @ (Dim, 1) -> (Batch, 1)
raw_similarities = (img_embs @ text_emb.T).flatten()
scaled_scores = raw_similarities * LOGIT_SCALE
probs = softmax(scaled_scores)

# 4. Rank and Print Results
results = sorted(zip(image_names, probs), key=lambda x: x[1], reverse=True)

print("\n--- SEARCH RESULTS ---")
print(f"\nQuery: '{QUERY_TEXT}'")
for i, (name, prob) in enumerate(results):
    marker = "⭐ [BEST]" if i == 0 else "  "
    print(f"{marker} {name}: {prob*100:.2f}%")
