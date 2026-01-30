import time
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import CLIPProcessor

# Config
VISUAL_MODEL_PATH = "assets/model_openai/visual.onnx"
TEXT_MODEL_PATH = "assets/model_openai/text.onnx"
PROCESSOR_PATH = "assets/model_openai"  # Where we saved the processor files
IMAGE_PATH = "assets/img/beach_rocks.jpg"
ITERS = 10

# 1. Load Components
print("Loading Processor...")
processor = CLIPProcessor.from_pretrained(PROCESSOR_PATH)

print("Loading ONNX Models...")
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
visual_session = ort.InferenceSession(VISUAL_MODEL_PATH, providers=providers)
text_session = ort.InferenceSession(TEXT_MODEL_PATH, providers=providers)


def run_image_pipeline():
    start = time.perf_counter()

    # Preprocess Image (returns numpy by default if we ask)
    image = Image.open(IMAGE_PATH)
    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"]

    # Run Inference
    onnx_inputs = {visual_session.get_inputs()[0].name: pixel_values}
    img_output = visual_session.run(None, onnx_inputs)

    # Print first run only
    if ITERS == 1:
        print("Image Embedding [0:20]:", img_output[0][0, :20].tolist())

    return time.perf_counter() - start


def run_text_pipeline():
    start = time.perf_counter()

    # Preprocess Text
    text = "a photo of rocks"
    inputs = processor(text=[text], padding=True, return_tensors="np")

    # ONNX Runtime expects specific input names
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    text_output = text_session.run(None, onnx_inputs)

    # Print first run only
    if ITERS == 1:
        print("Text Embedding [0:20]:", text_output[0][0, :20].tolist())

    return time.perf_counter() - start


# --- Execution ---

print("Starting Warmup...")
run_image_pipeline()
run_text_pipeline()

print(f"Starting Benchmark ({ITERS} iterations)...")
img_times = []
text_times = []

# Doing a loop to simulate real benchmark
for _ in range(ITERS):
    img_times.append(run_image_pipeline())
    text_times.append(run_text_pipeline())


def to_ms(sec_list):
    avg_sec = sum(sec_list) / len(sec_list)
    return round(avg_sec * 1000, 2)


print("\n--- OPENAI CLIP ONNX RESULTS (AVG PER RUN) ---")
print(f"Image Pipeline: {to_ms(img_times)}ms")
print(f"Text Pipeline:  {to_ms(text_times)}ms")

# Verification: Calculate Dot Product (Similarity)
# Since we normalized in the ONNX export, Dot Product == Cosine Similarity
print("\n--- SIMILARITY CHECK ---")
img_emb = visual_session.run(None, {
    visual_session.get_inputs()[0].name: processor(images=Image.open(IMAGE_PATH), return_tensors="np")[
        "pixel_values"]})[0]
txt_emb = text_session.run(None, {
    "input_ids": processor(text=["a photo of rocks"], return_tensors="np")["input_ids"],
    "attention_mask": processor(text=["a photo of rocks"], return_tensors="np")["attention_mask"]
})[0]

# Calculate dot product
score = (img_emb @ txt_emb.T)[0][0]
print(f"Similarity score for 'beach_rocks.jpg' vs 'a photo of rocks': {score:.4f}")