# /// script
# dependencies = [
#   "requests",
#   "numpy",
#   "pillow",
# ]
# ///
from pathlib import Path
from PIL import Image
import io
import requests
import base64
import os
import numpy as np
import time

TEXT_PROMPT = "A photo of Rocks"
IMAGES = [
    "beach_rocks.jpg",
    "beetle_car.jpg",
    "cat_face.jpg",
    "dark_sunset.jpg",
    "palace.jpg",
    "rocky_coast.jpg",
    "stacked_plates.jpg",
    "verdant_cliff.jpg",
]

TEMPERATURE = 0.05
API_URL = "http://127.0.0.1:8080/embedding"
MODEL_ID = "qwen-embed"
IMAGE_FOLDER = 'img'
TEXT_PREFIX = "Query: "
VISION_PROMPT = "<|vision_start|><|image_pad|><__media__><|vision_end|>"
TARGET_HEIGHT = 600

session = requests.Session()


def resize_and_encode_image(image_path, target_height=350):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return None
    try:
        with Image.open(image_path) as img:
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            if img_resized.mode in ("RGBA", "P"):
                img_resized = img_resized.convert("RGB")
            buffer = io.BytesIO()
            img_resized.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def get_embedding(content_payload):
    payload = {"model": MODEL_ID, "content": content_payload}

    try:
        response = session.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        raw_data = None
        if "embedding" in data:
            raw_data = data["embedding"]
        elif isinstance(data, list) and "embedding" in data[0]:
            raw_data = data[0]["embedding"]

        if not raw_data: return None

        embedding_array = np.array(raw_data)

        if embedding_array.ndim == 2:
            pooled = embedding_array[-1]
            return pooled
        else:
            return embedding_array

    except Exception as e:
        print(f"Request failed: {e}")
        return None


def get_image_embedding(image_path):
    base64_image = resize_and_encode_image(image_path, TARGET_HEIGHT)
    if not base64_image: return None

    return get_embedding({
        "prompt_string": VISION_PROMPT,
        "multimodal_data": [base64_image]
    })


def cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(vec_a / norm_a, vec_b / norm_b)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main():
    print(f"--- Embedding Comparison (Mean Pooling) ---")
    query_text = f"{TEXT_PREFIX}{TEXT_PROMPT}"
    print(f"Query: '{query_text}'\n")

    print(f"Embedding text...")
    text_emb = get_embedding(query_text)
    if text_emb is None: return

    img_data = []

    print(f"Processing images...")
    for img_filename in IMAGES:
        path = Path(IMAGE_FOLDER) / img_filename

        if not path.exists(): continue

        start = time.perf_counter()
        img_emb = get_image_embedding(path)
        elapsed = time.perf_counter() - start

        if img_emb is not None:
            score = cosine_similarity(text_emb, img_emb)
            img_data.append((img_filename, score))
            print(f".", end="", flush=True)
        else:
            print(f"x", end="", flush=True)

    print(f"\n\n--- Results ---")

    if not img_data:
        print("No images processed.")
        return

    scores = np.array([item[1] for item in img_data])
    probs = softmax(scores / TEMPERATURE)

    final_results = []
    for i, (name, score) in enumerate(img_data):
        final_results.append((name, score, probs[i]))

    final_results.sort(key=lambda x: x[1], reverse=True)

    print(f"{'IMAGE':<25} | {'SIMILARITY':<10} | {'CONFIDENCE'}")
    print("-" * 55)
    for name, score, prob in final_results:
        print(f"{name:<25} | {score:.4f}     | {prob:.2%}")

    print(f"\n🏆 Best Match: {final_results[0][0].upper()}")


if __name__ == "__main__":
    main()
