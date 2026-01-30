import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# 1. Load Model and Tokenizer
model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
tokenizer = get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')

# 2. Setup your data
# I added a filenames list just so we can print the results nicely later
filenames = [
    "beach_rocks.jpg", "beetle_car.jpg", "cat_face.jpg", "dark_sunset.jpg",
    "palace.jpg", "rocky_coast.jpg", "stacked_plates.jpg", "verdant_cliff.jpg"
]

images = [Image.open(f"sl1/assets/img/{x}") for x in filenames]

query_text = "An old car on a campsite"

# 3. Preprocess Input
# Stack all images into a single tensor of shape (N_images, Channels, Height, Width)
image_tensor = torch.stack([preprocess(img) for img in images])

# Tokenize text (put in a list to keep dimensions correct)
text_tokens = tokenizer([query_text], context_length=model.context_length)

# 4. Inference
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tokens)

    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Calculate Logits
    # Shape: (N_images, D) @ (D, 1) -> (N_images, 1)
    # We use sigmoid because SigLIP is trained with sigmoid loss (binary classification per pair)
    logits = (image_features @ text_features.T) * model.logit_scale.exp() + model.logit_bias
    probs = torch.sigmoid(logits)

# 5. Format and Print Results
# Flatten probs from (N, 1) to (N)
probs_list = probs.flatten().tolist()

# Zip filenames with their probabilities
results = list(zip(filenames, probs_list))

# Sort by probability (Highest to Lowest)
results.sort(key=lambda x: x[1], reverse=True)

print(f"Query: '{query_text}'\n")
print(f"{'Image File':<25} | {'Probability':<10}")
print("-" * 40)
for filename, prob in results:
    print(f"{filename:<25} | {prob:.4f}")
