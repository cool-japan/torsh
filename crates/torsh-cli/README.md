# torsh-cli

Command-line tools for the ToRSh deep learning framework with PyTorch-compatible CLI interface.

## Overview

This crate provides a comprehensive command-line interface for ToRSh, enabling model training, inference, conversion, benchmarking, and management directly from the terminal. It offers a familiar experience for users coming from PyTorch while leveraging Rust's performance and safety.

## Features

- **Model Training**: Train models with configuration files or command-line arguments
- **Inference**: Run predictions on trained models with various input formats
- **Model Conversion**: Convert between different model formats (ONNX, TorchScript, etc.)
- **Benchmarking**: Profile and benchmark model performance
- **Model Hub**: Download and manage pre-trained models
- **Dataset Management**: Download and prepare datasets for training
- **Quantization**: Quantize models for deployment
- **Profiling**: Analyze model performance and memory usage
- **Interactive Mode**: REPL for quick experimentation

## Installation

```bash
# Install from source
cargo install --path crates/torsh-cli

# Install with all features
cargo install --path crates/torsh-cli --features full
```

## Usage

### Basic Commands

```bash
# Display help
torsh --help

# Train a model
torsh train --config config.yaml

# Run inference
torsh infer --model model.torsh --input data.json

# Benchmark a model
torsh bench --model model.torsh --batch-size 32

# Convert model format
torsh convert --input model.pth --output model.onnx --format onnx

# Download pre-trained model from hub
torsh hub download --model resnet50 --output ./models/
```

### Training

#### From Configuration File

```bash
# Train with YAML config
torsh train --config training_config.yaml

# Override config parameters
torsh train --config config.yaml --epochs 100 --lr 0.001
```

Example `training_config.yaml`:

```yaml
model:
  type: resnet
  num_classes: 10
  pretrained: false

data:
  train_path: ./data/train
  val_path: ./data/val
  batch_size: 64
  num_workers: 4

training:
  epochs: 50
  learning_rate: 0.01
  optimizer: sgd
  momentum: 0.9
  weight_decay: 0.0001

scheduler:
  type: step
  step_size: 30
  gamma: 0.1

checkpoints:
  save_dir: ./checkpoints
  save_interval: 5
```

#### Direct Command-Line Arguments

```bash
torsh train \
  --model resnet50 \
  --data ./data/imagenet \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --device cuda:0
```

### Inference

```bash
# Single file inference
torsh infer \
  --model model.torsh \
  --input image.jpg \
  --output predictions.json

# Batch inference
torsh infer \
  --model model.torsh \
  --input-dir ./images/ \
  --output-dir ./predictions/ \
  --batch-size 32

# Streaming inference
torsh infer \
  --model model.torsh \
  --stream \
  --format json
```

### Benchmarking

```bash
# Benchmark model throughput
torsh bench \
  --model model.torsh \
  --batch-size 1,8,32,128 \
  --device cuda:0 \
  --warmup 10 \
  --iterations 100

# Profile memory usage
torsh bench \
  --model model.torsh \
  --profile memory \
  --output profile.json

# Compare multiple models
torsh bench \
  --models model1.torsh,model2.torsh,model3.torsh \
  --compare \
  --output comparison.csv
```

### Model Conversion

```bash
# Convert PyTorch to ONNX
torsh convert \
  --input model.pth \
  --output model.onnx \
  --format onnx \
  --opset 14

# Convert to TorchScript
torsh convert \
  --input model.pth \
  --output model.pt \
  --format torchscript

# Quantize during conversion
torsh convert \
  --input model.pth \
  --output model_int8.torsh \
  --quantize int8
```

### Model Hub

```bash
# List available models
torsh hub list

# Search for models
torsh hub search --query "resnet"

# Download model
torsh hub download \
  --model resnet50 \
  --variant imagenet \
  --output ./models/

# Upload model to hub
torsh hub upload \
  --model my_model.torsh \
  --name my-awesome-model \
  --description "My custom model"
```

### Dataset Management

```bash
# List available datasets
torsh data list

# Download dataset
torsh data download --dataset cifar10 --output ./data/

# Prepare custom dataset
torsh data prepare \
  --input ./raw_data/ \
  --output ./processed_data/ \
  --format imagefolder

# Split dataset
torsh data split \
  --input ./data/ \
  --train 0.8 \
  --val 0.1 \
  --test 0.1
```

### Quantization

```bash
# Quantize model to INT8
torsh quantize \
  --model model.torsh \
  --output model_int8.torsh \
  --precision int8 \
  --calibration-data ./calib_data/

# Dynamic quantization
torsh quantize \
  --model model.torsh \
  --output model_dynamic.torsh \
  --mode dynamic

# QAT (Quantization-Aware Training)
torsh quantize \
  --model model.torsh \
  --mode qat \
  --data ./data/ \
  --epochs 10
```

### Profiling

```bash
# Profile model execution
torsh profile \
  --model model.torsh \
  --input-shape 1,3,224,224 \
  --device cuda:0

# Generate detailed report
torsh profile \
  --model model.torsh \
  --report-format html \
  --output profile_report.html

# Layer-wise profiling
torsh profile \
  --model model.torsh \
  --layer-wise \
  --output layers.json
```

### Interactive Mode

```bash
# Start interactive REPL
torsh repl

# Load model in REPL
torsh repl --model model.torsh
```

Within the REPL:

```python
>>> import numpy as np
>>> x = tensor([[1, 2], [3, 4]])
>>> y = x @ x.t()
>>> print(y)
Tensor([[5, 11], [11, 25]], dtype=i32)

>>> model = load("model.torsh")
>>> output = model(x)
```

## Configuration

### Global Configuration

The CLI uses a global configuration file located at `~/.torsh/config.toml`:

```toml
[defaults]
device = "cuda:0"
dtype = "float32"
num_workers = 4

[hub]
cache_dir = "~/.torsh/hub"
api_url = "https://hub.torsh.ai"

[logging]
level = "info"
format = "text"

[performance]
num_threads = 8
enable_cudnn = true
```

### Environment Variables

```bash
# Set default device
export TORSH_DEVICE=cuda:0

# Set cache directory
export TORSH_CACHE_DIR=/path/to/cache

# Set log level
export TORSH_LOG=debug

# Number of threads
export TORSH_NUM_THREADS=8
```

## Advanced Features

### Custom Scripts

```bash
# Run Python-like script
torsh run script.torsh

# With arguments
torsh run script.torsh --arg1 value1 --arg2 value2
```

### Model Inspection

```bash
# Show model architecture
torsh info --model model.torsh

# Show detailed layer information
torsh info --model model.torsh --verbose

# Export to dot format for visualization
torsh info --model model.torsh --format dot --output model.dot
```

### Distributed Training

```bash
# Launch distributed training
torsh train \
  --config config.yaml \
  --distributed \
  --world-size 4 \
  --rank 0 \
  --master-addr localhost \
  --master-port 29500
```

## Shell Completion

Generate shell completion scripts:

```bash
# Bash
torsh completion bash > /etc/bash_completion.d/torsh

# Zsh
torsh completion zsh > ~/.zfunc/_torsh

# Fish
torsh completion fish > ~/.config/fish/completions/torsh.fish

# PowerShell
torsh completion powershell > torsh.ps1
```

## Examples

### Train ResNet on CIFAR-10

```bash
torsh data download --dataset cifar10 --output ./data/
torsh train \
  --model resnet18 \
  --data ./data/cifar10 \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --scheduler cosine
```

### Fine-tune Pre-trained Model

```bash
torsh hub download --model resnet50 --output ./models/
torsh train \
  --model ./models/resnet50.torsh \
  --fine-tune \
  --data ./custom_data/ \
  --epochs 20 \
  --lr 0.001
```

### Export for Production

```bash
torsh convert \
  --input model.torsh \
  --output model.onnx \
  --format onnx \
  --optimize \
  --opset 15

torsh quantize \
  --model model.onnx \
  --output model_int8.onnx \
  --precision int8
```

## Integration with SciRS2

This crate leverages the SciRS2 ecosystem for:

- High-performance tensor operations through `scirs2-core`
- Neural network implementations via `scirs2-neural`
- Optimization algorithms from `scirs2-optimize` and `optirs`
- Metrics and evaluation through `scirs2-metrics`

All operations follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for consistent, maintainable code.

## Development

### Building

```bash
# Build CLI
cargo build --package torsh-cli

# Build with all features
cargo build --package torsh-cli --all-features

# Release build
cargo build --package torsh-cli --release
```

### Testing

```bash
# Run tests
cargo test --package torsh-cli

# Integration tests
cargo test --package torsh-cli --test integration_tests
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.
