
# Sharp MLX

A high-performance port of [Sharp](https://github.com/apple/ml-tract) (3D Gaussian Splatting from a single image) to **MLX**, available in **Python** and **Swift**.

Running entirely on Apple Silicon and achieves faster inference speeds compared to the PyTorch original.

## Usage

### Download & Convert Weights

1. **Download the official PyTorch checkpoint:**
```bash
mkdir -p checkpoints
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt -O checkpoints/sharp.pt
```

2. **Convert to MLX format:**
```bash
cd python
# This generates checkpoints/sharp.safetensors
python convert.py -i ../checkpoints/sharp.pt -o ../checkpoints/sharp.safetensors
```

### Python

```bash
cd python
# Install dependencies
pip install -r requirements.txt

# Run inference
python generate.py --input ../inputs/image.jpg --output ../output.ply --checkpoint ../checkpoints/sharp.safetensors
```

### Swift

```bash
cd swift
# Build
xcodebuild build -scheme generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet

# Run inference
.build/DerivedData/Build/Products/Release/generate --input ../inputs/image.jpg --output ../output.ply --weights ../checkpoints/sharp.safetensors
```

## Directory Structure

- `python/` - Pure MLX Python implementation
- `swift/` - Pure MLX Swift implementation
- `checkpoints/` - Model weights

## License

This project is licensed under the Apache License 2.0.

Based on [Apple's Sharp](https://github.com/apple/ml-tract) (Copyright (C) 2025 Apple Inc.)