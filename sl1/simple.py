import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# Setup
device = "cuda"
model_name = 'hf-hub:timm/ViT-SO400M-14-SigLIP-384'

# Load Model
model, preprocess = create_model_from_pretrained(model_name)
model.to(device)
model.eval()
tokenizer = get_tokenizer(model_name)

# Prepare Data
image = preprocess(Image.open("assets/img/beach_rocks.jpg")).unsqueeze(0).to(device)
text = tokenizer(["A photo of Rocks"]).to(device)

# Embed
with torch.no_grad():
    image_emb = model.encode_image(image, normalize=True)
    text_emb = model.encode_text(text, normalize=True)

# Print

print("Image pixel slice")
print(image[0][0][0][:30].cpu().tolist())

print("Text input IDs")
print(text.cpu().tolist())

print("\nText Embedding [0:50]:")
print(text_emb[0, :50].tolist())
print("Image Embedding [0:50]:")
print(image_emb[0, :50].tolist())