# apple-clip

Research and reference repository for CLIP models used in Ruurd Photos. 

This repository contains scripts for benchmarking CLIP models and converting them to ONNX for integration into the Ruurd Photos Rust-based backend.

## Overview

- Purpose: research various CLIP models (SigLIP, SigLIP2, OpenAI).
- Models:
  - SigLIP2: `timm/ViT-SO400M-16-SigLIP2-384` (In `./sl2`)
  - SigLIP: `timm/ViT-SO400M-14-SigLIP-384` (In `./sl1`)
  - OpenAI CLIP: `openai/clip-vit-base-patch32` (In `./openai`)
- Goal: Export and optimize models for Rust execution via ONNX Runtime.

## Project Structure

- `main.py`: Basic PyTorch inference demonstration.
- `bench_embedder.py`: Performance benchmarking for image and text embeddings.
- `compare.py`, `compare_input_pixels.py`: Validation scripts to ensure parity between models and onnx run scripts.

## Setup

The project uses `uv` for dependency management.

```bash
uv sync
```

## Usage

### Run Basic Inference
```bash
python main.py
```

### Benchmark Embeddings
```bash
python bench_embedder.py
```

### Export to ONNX
To export the latest SigLIP2 model to the `assets/model/` directory:
```bash
cd sl2
uv run export_onnx_sl2.py
```

This produces:

* model_config.json
* special_tokens_map.json
* text.onnx
* text.onnx.data
* tokenizer.json
* tokenizer_config.json
* visual.onnx
* visual.onnx.data
