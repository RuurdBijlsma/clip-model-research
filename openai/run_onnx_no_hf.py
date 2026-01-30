import json
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

# --- CONFIG & PATHS ---
ASSETS_DIR = "assets/model_openai"
CONFIG_PATH = os.path.join(ASSETS_DIR, "model_config.json")
VISUAL_MODEL_PATH = os.path.join(ASSETS_DIR, "visual.onnx")
TEXT_MODEL_PATH = os.path.join(ASSETS_DIR, "text.onnx")
TOKENIZER_JSON = os.path.join(ASSETS_DIR, "tokenizer.json")
IMAGE_DIR = "assets/img"

# Constants from preprocessor_config.json
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711])
IMAGE_SIZE = 224

QUERY_TEXT = "a photo of rocks"
IMAGE_FILES = [
    "beach_rocks.jpg", "beetle_car.jpg", "cat_face.jpg", "dark_sunset.jpg",
    "palace.jpg", "rocky_coast.jpg", "stacked_plates.jpg", "verdant_cliff.jpg"
]

# --- 1. LOAD COMPONENTS ---

# Load Logit Scale
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
    LOGIT_SCALE = config.get("logit_scale", 100.0)

# Load Tokenizer (Decoupled from transformers)
tokenizer = Tokenizer.from_file(TOKENIZER_JSON)
# Configure padding and truncation to match CLIP's 77 token limit
tokenizer.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
tokenizer.enable_truncation(max_length=77)

# Load ONNX Sessions
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
visual_session = ort.InferenceSession(VISUAL_MODEL_PATH, providers=providers)
text_session = ort.InferenceSession(TEXT_MODEL_PATH, providers=providers)


# --- 2. IMAGE PREPROCESSING (Manual) ---

def preprocess_image(image_path):
    """Manual implementation of CLIPImageProcessor."""
    img = Image.open(image_path).convert("RGB")

    # 1. Resize shortest edge to 224
    w, h = img.size
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = IMAGE_SIZE, int(IMAGE_SIZE * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # 2. Center Crop to 224x224
    left = (new_w - IMAGE_SIZE) / 2
    top = (new_h - IMAGE_SIZE) / 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # 3. To Numpy & Normalize
    pixel_values = np.array(img).astype(np.float32) / 255.0
    pixel_values = (pixel_values - IMAGE_MEAN) / IMAGE_STD

    # 4. Transpose to (Channels, Height, Width)
    return pixel_values.transpose(2, 0, 1)


def get_image_embeddings(image_filenames):
    processed_images = []
    valid_names = []

    for name in image_filenames:
        full_path = os.path.join(IMAGE_DIR, name)
        if os.path.exists(full_path):
            processed_images.append(preprocess_image(full_path))
            valid_names.append(name)

    if not processed_images:
        return None, []

    # Batch images: (Batch, 3, 224, 224)
    batch_data = np.stack(processed_images).astype(np.float32)

    onnx_inputs = {visual_session.get_inputs()[0].name: batch_data}
    embeddings = visual_session.run(None, onnx_inputs)[0]
    return embeddings, valid_names


# --- 3. TEXT PREPROCESSING (Manual) ---

def get_text_embedding(text):
    """Manual implementation using the Tokenizer library."""
    # The tokenizer handles <|startoftext|> and <|endoftext|> automatically 
    # if tokenizer.json is configured correctly (which HF's usually is).
    output = tokenizer.encode(text)

    # Convert to numpy and add batch dimension
    input_ids = np.array([output.ids], dtype=np.int64)
    attention_mask = np.array([output.attention_mask], dtype=np.int64)

    onnx_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    embedding = text_session.run(None, onnx_inputs)[0]
    return embedding


# --- 4. MAIN LOGIC ---

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


print(f"Processing {len(IMAGE_FILES)} images...")
img_embs, image_names = get_image_embeddings(IMAGE_FILES)

print(f"Encoding query: '{QUERY_TEXT}'...")
text_emb = get_text_embedding(QUERY_TEXT)

# --- DEBUG COMPARISON ---
print("\n--- DEBUG: ONNX (NO-HF) VALUES ---")
# 1. Text Input IDs
# Re-encoding for debug display
debug_ids = tokenizer.encode(QUERY_TEXT).ids[:10]
print(f"Text Input IDs (first 10): {debug_ids}")

# 2. Image Input Tensors (Manual Preprocessing check)
# img_embs comes from batch_data in get_image_embeddings.
# Let's check the manual preprocess output of the first image
pix = preprocess_image(os.path.join(IMAGE_DIR, IMAGE_FILES[0]))
print(f"Image Pixel Values - Mean: {pix.mean():.6f}, Std: {pix.std():.6f}")
print(f"Image Pixel Values (slice): {pix[0, 0, :5].tolist()}")

# 3. Text Embeddings
print(f"Text Embeds - Mean: {text_emb.mean():.6f}, Std: {text_emb.std():.6f}")
print(f"Text Embeds (first 5): {text_emb[0][:5].tolist()}")

# 4. Image Embeddings (First Image)
print(f"Image Embeds[0] - Mean: {img_embs[0].mean():.6f}, Std: {img_embs[0].std():.6f}")
print(f"Image Embeds[0] (first 5): {img_embs[0][:5].tolist()}")
# ------------------------

# Calculate similarities (embeddings are pre-normalized by the ONNX model)
raw_similarities = (img_embs @ text_emb.T).flatten()
scaled_scores = raw_similarities * LOGIT_SCALE
probs = softmax(scaled_scores)

results = sorted(zip(image_names, probs), key=lambda x: x[1], reverse=True)

print("\n--- SEARCH RESULTS ---")
print(f"Query: '{QUERY_TEXT}'")
for i, (name, prob) in enumerate(results):
    marker = "⭐ [BEST]" if i == 0 else "  "
    print(f"{marker} {name}: {prob * 100:.2f}%")