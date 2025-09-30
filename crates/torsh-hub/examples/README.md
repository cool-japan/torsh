# ToRSh Hub Examples

This directory contains comprehensive examples demonstrating the capabilities of ToRSh Hub.

## Examples

### Basic Usage

- **`basic_model_loading.rs`** - Demonstrates basic model loading from various sources
  ```bash
  cargo run --example basic_model_loading
  ```

- **`model_registry_search.rs`** - Shows how to search and discover models in the registry
  ```bash
  cargo run --example model_registry_search
  ```

### Advanced Features

- **`model_publishing.rs`** - Complete guide to publishing models to ToRSh Hub
  ```bash
  cargo run --example model_publishing
  ```

- **`onnx_integration.rs`** - Working with ONNX models: loading, optimization, deployment
  ```bash
  cargo run --example onnx_integration
  ```

- **`fine_tuning.rs`** - Fine-tuning pre-trained models with various strategies
  ```bash
  cargo run --example fine_tuning
  ```

## Prerequisites

Ensure you have the required dependencies installed:

```toml
[dependencies]
torsh-hub = { path = ".." }
torsh-core = { path = "../../torsh-core" }
torsh-tensor = { path = "../../torsh-tensor" }
torsh-nn = { path = "../../torsh-nn" }
```

## Features

Some examples require specific features to be enabled:

```bash
# For TensorFlow integration
cargo run --example onnx_integration --features tensorflow

# For all features
cargo run --example basic_model_loading --all-features
```

## Environment Setup

For authenticated operations, set your ToRSh Hub token:

```bash
export TORSH_HUB_TOKEN=your_token_here
```

Or create a config file at `~/.config/torsh/hub_token`.

## Example Data

Some examples expect model files in the `examples/models/` directory:

```
examples/
├── models/
│   ├── resnet18.onnx
│   ├── bert-base-uncased/
│   └── saved_model/
└── *.rs
```

You can download example models or the examples will create synthetic data for demonstration.