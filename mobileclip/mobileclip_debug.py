import torch
import torch.nn as nn
import open_clip
import os
import numpy as np
from PIL import Image
from timm.utils import reparameterize_model
import torch.nn.functional as F


def get_stats(tensor):
    """Equivalent to Rust get_stats: returns (mean, std)"""
    mean = tensor.mean().item()
    std = tensor.std(unbiased=False).item()
    return mean, std


def save_debug_image(pixel_tensor, mean, std, filename):
    """
    Equivalent to Rust save_debug_image.
    Reverses normalization and saves as an image.
    """
    # pixel_tensor shape: [1, 3, H, W]
    img = pixel_tensor.squeeze(0).cpu().numpy()

    # Re-order and un-normalize
    # Rust: (pix * std + mean) * 255
    for c in range(3):
        img[c] = (img[c] * std[c]) + mean[c]

    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    # CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    Image.fromarray(img).save(filename)
    print(f"📸 Debug image (reconstructed) saved to: {filename}")


def main():
    # Paths
    asset_dir = "assets"
    img_path = os.path.join(asset_dir, "img/beach_rocks.jpg")
    query_text = "A photo of Rocks"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚀 Loading MobileCLIP2-S4...")
    # Using the same model as your search script
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S4', pretrained='dfndr2b')
    model = model.to(device)
    model.eval()

    # CRITICAL: Reparameterize to match ONNX behavior
    model = reparameterize_model(model)
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S4')

    # Get config stats from the model/preprocess
    # Note: open_clip's preprocess contains the mean/std
    mean = preprocess.transforms[-1].mean
    std = preprocess.transforms[-1].std

    print("\n--- DEBUG: OPEN_CLIP PYTHON REFERENCE ---")

    # 1. Text Preprocessing
    tokens = tokenizer([query_text]).to(device)
    # OpenCLIP tokenizers usually return [Batch, CtxLen]
    print("\n[TEXT PREPROCESSING]")
    print(f"Input IDs (first 10): {tokens[0, :10].tolist()}")
    # OpenCLIP/CLIP usually doesn't use a separate mask tensor in the same way 
    # as BERT/SigLIP, but the padding is handled inside the IDs.

    # 2. Image Preprocessing
    raw_img = Image.open(img_path)
    # preprocess(raw_img) returns [3, H, W]
    pixel_tensor = preprocess(raw_img).unsqueeze(0).to(device)

    save_debug_image(pixel_tensor, mean, std, "debug_python_input.png")

    pix_mean, pix_std = get_stats(pixel_tensor)
    print("\n[IMAGE PREPROCESSING]")
    print(f"Pixel Stats - Mean: {pix_mean:.6f}, Std: {pix_std:.6f}")
    # Slice first 10 pixels of the first row (y=0) of the Red channel (c=0)
    # Rust: pixel_tensor.slice(s![0, 0, 0, ..10])
    pixel_slice = pixel_tensor[0, 0, 0, :10].tolist()
    print(f"Pixel Slice (ch0, row0): {pixel_slice}")

    # 3. Inference
    with torch.no_grad():
        # normalize=True is essential as your Rust export wrapper uses it
        image_embeds = model.encode_image(pixel_tensor, normalize=True)
        text_embeds = model.encode_text(tokens, normalize=True)

    t_row = text_embeds[0]
    t_mean, t_std = get_stats(t_row)
    print("\n[INFERENCE RESULTS]")
    print(f"Text Embeds  - Mean: {t_mean:.6f}, Std: {t_std:.6f}")
    print(f"Text Embeds  (first 5): {t_row[:5].tolist()}")

    i_row = image_embeds[0]
    i_mean, i_std = get_stats(i_row)
    print(f"Image Embeds - Mean: {i_mean:.6f}, Std: {i_std:.6f}")
    print(f"Image Embeds (first 5): {i_row[:5].tolist()}")

    # 4. Scoring
    similarity = torch.dot(i_row, t_row).item()
    print("\n[SCORING CHECK]")
    print(f"Raw Dot Product (Similarity): {similarity:.4f}")


if __name__ == "__main__":
    main()