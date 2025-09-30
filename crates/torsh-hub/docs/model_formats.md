# ToRSh Hub Model Formats

This document describes the various model formats supported by ToRSh Hub and how to work with them.

## Overview

ToRSh Hub supports multiple model formats to maximize compatibility with the broader machine learning ecosystem:

- **Native ToRSh Models**: Pure Rust implementations with the best performance
- **ONNX Models**: Cross-platform neural network representation
- **TensorFlow Models**: SavedModel and frozen graph formats
- **HuggingFace Models**: Automatic conversion from HuggingFace Hub
- **PyTorch State Dicts**: Weight loading from PyTorch checkpoints
- **GitHub Repository Models**: Models hosted in git repositories

## Native ToRSh Models

### Format Structure

Native ToRSh models are the preferred format, offering the best performance and full Rust safety guarantees.

```
model_name/
├── config.toml           # Model configuration
├── model.safetensors     # Weights in SafeTensors format
├── tokenizer.json        # Tokenizer configuration (for NLP models)
├── vocab.txt            # Vocabulary file (if applicable)
└── README.md           # Model documentation
```

### Configuration Format

The `config.toml` file contains model metadata and architecture parameters:

```toml
[model]
name = "bert-base-uncased"
architecture = "BERT"
version = "1.0.0"
author = "torsh-team"
license = "MIT"
description = "BERT Base Uncased model for English"

[architecture]
vocab_size = 30522
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
max_position_embeddings = 512
type_vocab_size = 2

[hardware]
min_ram_gb = 2.0
recommended_ram_gb = 4.0
supports_cpu = true
supports_gpu = true
supports_mixed_precision = true

[metrics]
accuracy = 0.895
f1_score = 0.912
inference_time_ms = 12.5
model_size_mb = 440.0
```

### Loading Native Models

```rust
use torsh_hub::load;

// Load from ToRSh Hub
let model = load("torsh-models/bert-base-uncased", "", true, None)?;

// Load from local path
let model = load_from_path("./models/my-model")?;
```

## ONNX Models

### Supported ONNX Features

- **Execution Providers**: CPU, CUDA, DirectML, CoreML
- **Data Types**: FP32, FP16, INT8, INT16, INT32, INT64
- **Dynamic Shapes**: Variable batch sizes and sequence lengths
- **Optimization Levels**: None, Basic, Extended, All

### Loading ONNX Models

```rust
use torsh_hub::{load_onnx_model, onnx::OnnxConfig};

// Basic ONNX loading
let model = load_onnx_model("model.onnx", None)?;

// With custom configuration
let config = OnnxConfig {
    execution_provider: Some("CUDAExecutionProvider".to_string()),
    optimization_level: Some("all".to_string()),
    enable_profiling: false,
    log_severity_level: 2,
    inter_op_num_threads: Some(4),
    intra_op_num_threads: Some(4),
};

let model = load_onnx_model("model.onnx", Some(config))?;

// From bytes
let model_bytes = std::fs::read("model.onnx")?;
let model = load_onnx_model_from_bytes(&model_bytes, None)?;

// Download and load
let model = load_onnx_model_from_url("https://example.com/model.onnx", None)?;
```

### ONNX Metadata

ONNX models automatically extract metadata:

```rust
let metadata = model.metadata();
println!("Model name: {}", metadata.name);
println!("Input shapes: {:?}", metadata.inputs);
println!("Output shapes: {:?}", metadata.outputs);
```

## TensorFlow Models

### Supported TensorFlow Formats

- **SavedModel**: TensorFlow's recommended format
- **Frozen Graphs**: Legacy .pb format
- **Checkpoints**: Variable checkpoint files
- **Lite Models**: TensorFlow Lite format

### Loading TensorFlow Models

```rust
#[cfg(feature = "tensorflow")]
use torsh_hub::{load_tensorflow_model, tensorflow::TfConfig};

// Load SavedModel
let model = load_tensorflow_model("saved_model_dir", None)?;

// With custom configuration
let config = TfConfig {
    execution_provider: "GPU".to_string(),
    allow_growth: true,
    memory_limit_mb: Some(2048),
    enable_xla: true,
};

let model = load_tensorflow_model("saved_model_dir", Some(config))?;

// Load frozen graph
let model = load_tensorflow_frozen_graph("model.pb", &["input:0"], &["output:0"])?;
```

## HuggingFace Models

### Automatic Conversion

ToRSh Hub can automatically download and convert models from HuggingFace Hub:

```rust
use torsh_hub::huggingface::{HuggingFaceHub, HfToTorshConverter};

// Search models
let hub = HuggingFaceHub::new()?;
let models = hub.search_models("bert", None)?;

// Download and convert
let converter = HfToTorshConverter::new();
let model = converter.convert_model("bert-base-uncased")?;

// Load specific revision
let model = converter.convert_model_with_revision("bert-base-uncased", "main")?;
```

### Supported HuggingFace Architectures

- **NLP**: BERT, RoBERTa, GPT-2, T5, BART, DistilBERT
- **Vision**: ViT, DETR, YOLOS, ResNet, EfficientNet
- **Audio**: Wav2Vec2, Whisper, HuBERT
- **Multimodal**: CLIP, ALIGN, LayoutLM

## GitHub Repository Models

### Repository Structure

Models hosted in GitHub repositories should follow this structure:

```
repository/
├── hubconf.rs           # Model definitions (Rust equivalent of hubconf.py)
├── models/             # Model implementation files
│   ├── resnet.rs
│   └── bert.rs
├── weights/            # Pre-trained weights
│   ├── resnet18.safetensors
│   └── bert_base.safetensors
└── README.md          # Documentation
```

### hubconf.rs Format

```rust
use torsh_nn::Module;
use torsh_core::error::Result;

/// Load ResNet-18 model
/// 
/// # Arguments
/// * `pretrained` - Whether to load pre-trained weights
pub fn resnet18(pretrained: bool) -> Result<Box<dyn Module>> {
    let mut model = ResNet18::new()?;
    
    if pretrained {
        let weights = load_weights("weights/resnet18.safetensors")?;
        model.load_state_dict(&weights)?;
    }
    
    Ok(Box::new(model))
}

/// List available models in this repository
pub fn list_models() -> Vec<&'static str> {
    vec!["resnet18", "resnet34", "resnet50"]
}

/// Get documentation for a model
pub fn help(model_name: &str) -> Result<String> {
    match model_name {
        "resnet18" => Ok("ResNet-18 model architecture".to_string()),
        _ => Err("Model not found".into()),
    }
}
```

### Loading Repository Models

```rust
use torsh_hub::{load, list, help};

// Load model from repository
let model = load("pytorch/vision", "resnet18", true, None)?;

// List available models
let models = list("pytorch/vision", None)?;

// Get model documentation
let doc = help("pytorch/vision", "resnet18", None)?;
```

## PyTorch State Dicts

### Loading PyTorch Weights

```rust
use torsh_hub::load_state_dict_from_url;

// Download and load state dict
let state_dict = load_state_dict_from_url(
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    None,
    None,
    true
)?;

// Apply to model
model.load_state_dict(&state_dict)?;
```

### State Dict Format

ToRSh uses a JSON-based state dict format for compatibility:

```json
{
  "version": "1.0",
  "tensors": {
    "conv1.weight": {
      "shape": [64, 3, 7, 7],
      "dtype": "f32",
      "data": "base64_encoded_tensor_data"
    },
    "conv1.bias": {
      "shape": [64],
      "dtype": "f32", 
      "data": "base64_encoded_tensor_data"
    }
  }
}
```

## Model Metadata

### Registry Entry

All models in ToRSh Hub have associated metadata stored in the registry:

```rust
use torsh_hub::registry::{ModelRegistry, RegistryEntry};

let registry = ModelRegistry::load()?;
let entry = registry.get_model("bert-base-uncased")?;

println!("Model: {}", entry.name);
println!("Author: {}", entry.author);
println!("Downloads: {}", entry.downloads);
println!("License: {}", entry.license);
println!("Model size: {:?} MB", entry.model_size_mb);
```

### Model Cards

Detailed model information is stored in model cards:

```rust
use torsh_hub::model_info::{ModelCard, ModelCardBuilder};

// Create model card
let card = ModelCardBuilder::new()
    .name("My Model")
    .description("A custom model for classification")
    .author("researcher@university.edu")
    .license("MIT")
    .add_training_data("ImageNet", "1M images")
    .add_metric("Top-1 Accuracy", 0.785)
    .add_limitation("May not generalize to out-of-domain data")
    .build()?;

// Render to markdown
let markdown = card.to_markdown()?;
```

## Best Practices

### Model Organization

1. **Use semantic versioning** for model versions
2. **Include comprehensive metadata** in config files
3. **Provide model cards** with training details and limitations
4. **Use SafeTensors format** for weight storage
5. **Include benchmark results** and inference metrics

### Performance Optimization

1. **Choose appropriate precision** (FP32, FP16, INT8)
2. **Enable hardware acceleration** when available
3. **Use model optimization** techniques (pruning, quantization)
4. **Implement caching** for frequently used models
5. **Monitor memory usage** and optimize batch sizes

### Security Considerations

1. **Verify model signatures** before loading
2. **Use trusted repositories** and authors
3. **Scan for vulnerabilities** in model files
4. **Implement access controls** for sensitive models
5. **Audit model usage** and downloads

## Examples

### Loading Different Model Types

```rust
use torsh_hub::*;

// Native ToRSh model
let bert = load("torsh-models/bert-base", "", true, None)?;

// ONNX model
let onnx_model = load_onnx_model("model.onnx", None)?;

// TensorFlow model (if feature enabled)
#[cfg(feature = "tensorflow")]
let tf_model = load_tensorflow_model("saved_model", None)?;

// HuggingFace model
let hf_model = huggingface::HfToTorshConverter::new()
    .convert_model("distilbert-base-uncased")?;

// Repository model
let repo_model = load("pytorch/vision", "resnet50", true, None)?;
```

### Model Registry Operations

```rust
use torsh_hub::registry::{ModelRegistry, SearchQuery, SortBy};

let registry = ModelRegistry::load()?;

// Search models
let query = SearchQuery {
    text: Some("image classification".to_string()),
    category: Some(ModelCategory::Vision),
    sort_by: SortBy::Downloads,
    limit: 10,
    ..Default::default()
};

let results = registry.search(&query)?;

for model in results {
    println!("{}: {} downloads", model.name, model.downloads);
}
```

## Troubleshooting

### Common Issues

1. **Model not found**: Check repository name and model name spelling
2. **Format not supported**: Verify the model format is compatible
3. **Memory errors**: Reduce batch size or use model quantization
4. **Device mismatch**: Ensure model and tensors are on the same device
5. **Version conflicts**: Update to compatible model and framework versions

### Debug Mode

Enable verbose logging for troubleshooting:

```rust
let config = HubConfig {
    verbose: true,
    ..Default::default()
};

let model = load("repo/model", "model_name", true, Some(config))?;
```

## Future Formats

ToRSh Hub is actively developing support for additional formats:

- **MLX Models**: Apple Silicon optimized models
- **WebGPU Models**: Browser-compatible models
- **Quantized Models**: INT4 and sub-byte precision formats
- **Sparse Models**: Pruned and sparse neural networks
- **Federated Models**: Privacy-preserving distributed models

For the latest format support, check the ToRSh Hub documentation and release notes.