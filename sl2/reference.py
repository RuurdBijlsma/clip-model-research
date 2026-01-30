import time
import torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

model, preprocess = create_model_from_pretrained(
    'hf-hub:timm/ViT-SO400M-16-SigLIP2-384'
)
tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-16-SigLIP2-384')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
labels_list = ["a dog", "a cat", "a donut", "a beignet"]

# -------------------------
# Prepare inputs once
# -------------------------
image_tensor = preprocess(image).unsqueeze(0).to(device)
text_tensor = tokenizer(
    labels_list,
    context_length=model.context_length
).to(device)

# -------------------------
# Warmup
# -------------------------
warmup_iters = 5
with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
    for _ in range(warmup_iters):
        _ = model.encode_image(image_tensor, normalize=True)
        _ = model.encode_text(text_tensor, normalize=True)

if device.type == "cuda":
    torch.cuda.synchronize()

# -------------------------
# Timed image preprocess + embed
# -------------------------
t0 = time.perf_counter()

image_tensor = preprocess(image).unsqueeze(0).to(device)

if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
    image_features = model.encode_image(image_tensor, normalize=True)

if device.type == "cuda":
    torch.cuda.synchronize()
t2 = time.perf_counter()

# -------------------------
# Timed text tokenize + embed
# -------------------------
t3 = time.perf_counter()

text_tensor = tokenizer(
    labels_list,
    context_length=model.context_length
).to(device)

if device.type == "cuda":
    torch.cuda.synchronize()
t4 = time.perf_counter()

with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
    text_features = model.encode_text(text_tensor, normalize=True)

if device.type == "cuda":
    torch.cuda.synchronize()
t5 = time.perf_counter()

# -------------------------
# Similarity
# -------------------------
text_probs = torch.sigmoid(
    image_features @ text_features.T * model.logit_scale.exp() + model.logit_bias
)

zipped_list = list(zip(labels_list, [100 * round(p.item(), 3) for p in text_probs[0]]))
print("Label probabilities:", zipped_list)

print(f"Image preprocess: {(t1 - t0) * 1e3:.2f} ms")
print(f"Image embedding:  {(t2 - t1) * 1e3:.2f} ms")
print(f"Text tokenize:    {(t4 - t3) * 1e3:.2f} ms")
print(f"Text embedding:   {(t5 - t4) * 1e3:.2f} ms")
