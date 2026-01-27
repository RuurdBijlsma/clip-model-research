import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import time
import os

# --- 1. CONFIGURATION & SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

model_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'

# Load Model and Preprocess
# We move the model to GPU immediately if available
print(f"Loading model: {model_name}...")
model, preprocess = create_model_from_pretrained(model_name)
model.to(device)
model.eval()

tokenizer = get_tokenizer(model_name)

# --- 2. DATA PREPARATION ---
# Defined paths based on your request
image_paths = [
    "assets/img/beach_rocks.jpg",
    "assets/img/beetle_car.jpg",
    "assets/img/cat_face.jpg",
    "assets/img/dark_sunset.jpg",
    "assets/img/palace.jpg",
    "assets/img/rocky_coast.jpg",
    "assets/img/stacked_plates.jpg",
    "assets/img/verdant_cliff.jpg",
]

query_text = "campsite"

print("Loading and preprocessing images...")
images = []
valid_filenames = []

# Load images (skipping ones that don't exist to prevent crash)
for path in image_paths:
    if os.path.exists(path):
        try:
            img = Image.open(path).convert('RGB')
            images.append(preprocess(img))
            valid_filenames.append(os.path.basename(path))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    else:
        print(f"Warning: File not found {path}")

if not images:
    print("No images found. Exiting.")
    exit()

# Stack images into a single tensor: (N, 3, 384, 384)
image_tensor = torch.stack(images).to(device)

# Tokenize text
text_tokens = tokenizer([query_text], context_length=model.context_length).to(device)

# --- 3. INFERENCE (PROBABILITY CALCULATION) ---
print("\n--- Running Inference ---")

# Context manager for mixed precision (faster on GPU)
autocast_ctx = torch.cuda.amp.autocast() if device == "cuda" else torch.no_grad()

with torch.no_grad(), autocast_ctx:
    # Encode
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tokens)

    # Normalize
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Calculate Probabilities (Sigmoid for SigLIP)
    # (N_images, Dim) @ (Dim, 1) -> (N_images, 1)
    logits = (image_features @ text_features.T) * model.logit_scale.exp() + model.logit_bias
    probs = torch.sigmoid(logits)

# Process results for printing
probs_list = probs.flatten().cpu().tolist()  # Move back to CPU for printing
results = list(zip(valid_filenames, probs_list))

# Sort by probability (Highest to Lowest)
results.sort(key=lambda x: x[1], reverse=True)

# Print Table
print(f"Query: '{query_text}'\n")
print(f"{'Image File':<25} | {'Probability':<10}")
print("-" * 40)
for filename, prob in results:
    print(f"{filename:<25} | {prob:.4f}")

# --- 4. BENCHMARKING SPEED ---
print("\n--- Benchmarking Speed ---")


def run_benchmark(label, func, input_data, iterations=50):
    """
    Runs a timing benchmark with warm-up and synchronization.
    """
    # 1. Warm up (run a few times to initialize memory/caches)
    for _ in range(5):
        with torch.no_grad(), autocast_ctx:
            func(input_data)

    # Sync CUDA to ensure warm-up is actually done
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    # 2. Actual Timing Loop
    for _ in range(iterations):
        with torch.no_grad(), autocast_ctx:
            func(input_data)

    # Sync CUDA again before stopping timer
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    print(f"{label}: {avg_time_ms:.2f} ms per embedding")


# Prepare Single Inputs for Benchmark
# Slice one image from the tensor to get shape (1, 3, 384, 384)
single_image_input = image_tensor[0].unsqueeze(0)
# Create a single text token input
single_text_input = tokenizer(["Benchmark text"], context_length=model.context_length).to(device)

run_benchmark("Single Image Embedding", model.encode_image, single_image_input)
run_benchmark("Single Text Embedding ", model.encode_text, single_text_input)