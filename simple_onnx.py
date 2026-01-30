import time
import onnxruntime as ort
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

VISUAL_MODEL_PATH = "assets/model/visual.onnx"
TEXT_MODEL_PATH = "assets/model/text.onnx"
IMAGE_PATH = "assets/img/beach_rocks.jpg"
MODEL_NAME = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'
ITERS = 1

# Loaders
_, preprocess = create_model_from_pretrained(MODEL_NAME)
tokenizer = get_tokenizer(MODEL_NAME)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
visual_session = ort.InferenceSession(VISUAL_MODEL_PATH, providers=providers)
text_session = ort.InferenceSession(TEXT_MODEL_PATH, providers=providers)


print("print(preprocess)")
print(preprocess)

def run_image_pipeline():
    start = time.perf_counter()
    image_tensor = preprocess(Image.open(IMAGE_PATH)).unsqueeze(0)
    image_np = image_tensor.detach().cpu().numpy()
    visual_inputs = {visual_session.get_inputs()[0].name: image_np}
    img_output = visual_session.run(None, visual_inputs)
    print("Image Embedding [0:20]")
    print(img_output[0][0, :20].tolist())

    return time.perf_counter() - start


def run_text_pipeline():
    start = time.perf_counter()
    text_tensor = tokenizer(["rocks in the rock business"])
    text_np = text_tensor.detach().cpu().numpy()
    text_inputs = {text_session.get_inputs()[0].name: text_np}
    text_output = text_session.run(None, text_inputs)
    print("Text Embedding [0:20]")
    print(text_output[0][0, :20].tolist())
    return time.perf_counter() - start


print("Starting Warmup...")
run_image_pipeline()
run_text_pipeline()

print(f"Starting Benchmark ({ITERS} iterations)...")

img_times = [run_image_pipeline() for _ in range(ITERS)]
text_times = [run_text_pipeline() for _ in range(ITERS)]


def to_ms(sec_list):
    avg_sec = sum(sec_list) / len(sec_list)
    return round(avg_sec * 1000, 2)


print("\n--- PYTHON RESULTS (AVG PER RUN) ---")
print(f"Image Pipeline: {to_ms(img_times)}ms")
print(f"Text Pipeline:  {to_ms(text_times)}ms")
