# torsh-hub

Model hub integration for ToRSh, providing easy access to pre-trained models and datasets.

## Overview

ToRSh Hub is a repository of pre-trained models that facilitates research reproducibility. It provides:

- **Model Repository**: Access to pre-trained models
- **Easy Loading**: Simple API to load models with weights
- **Model Cards**: Detailed information about each model
- **Version Control**: Model versioning and updates
- **Caching**: Local caching for offline use

## Usage

### Loading Pre-trained Models

```rust
use torsh_hub::prelude::*;

// Load a pre-trained ResNet model
let model = hub::load("cooljapan/resnet50", pretrained=true)?;

// Load with specific revision
let model = hub::load_with_config(
    "cooljapan/resnet50",
    LoadConfig::default()
        .revision("v2.0")
        .map_location(Device::cuda(0))
        .force_reload(false),
)?;

// Load from URL
let model = hub::load_state_dict_from_url(
    "https://example.com/models/resnet50.pt",
    Some("~/.cache/torsh/hub"),
    true,  // check hash
)?;
```

### Model Discovery

```rust
// List available models
let models = hub::list_models(
    ModelFilter::default()
        .task("image-classification")
        .dataset("imagenet")
        .framework("torsh"),
)?;

for model in models {
    println!("{}: {}", model.id, model.description);
}

// Get model info
let info = hub::get_model_info("cooljapan/resnet50")?;
println!("Downloads: {}", info.downloads);
println!("Tags: {:?}", info.tags);
```

### Publishing Models

```rust
use torsh_hub::publish::*;

// Create model card
let model_card = ModelCard::new()
    .name("My Awesome Model")
    .description("A model trained on custom dataset")
    .task("image-classification")
    .dataset("my-dataset")
    .metrics(vec![
        ("accuracy", 0.95),
        ("top5_accuracy", 0.99),
    ])
    .add_tag("vision")
    .add_tag("resnet");

// Publish model
hub::publish(
    model,
    "myusername/my-model",
    model_card,
    Some("Initial release"),
)?;
```

### Custom Model Registry

```rust
// Define custom model
#[derive(ModelRegistry)]
#[model(name = "custom_resnet", task = "image-classification")]
struct CustomResNet {
    // model fields
}

impl CustomResNet {
    fn from_pretrained(name: &str, config: LoadConfig) -> Result<Self> {
        // Load logic
    }
}

// Register with hub
hub::register_model::<CustomResNet>();

// Now loadable via hub
let model = hub::load("username/custom_resnet", pretrained=true)?;
```

### Model Configuration

```rust
use torsh_hub::config::*;

// Load model with configuration
let config = AutoConfig::from_pretrained("cooljapan/bert-base")?;
let model = AutoModel::from_config(config)?;

// Modify configuration
let mut config = ResNetConfig::default();
config.num_classes = 100;
config.dropout = 0.5;

let model = ResNet::from_config(config)?;
```

### Caching and Offline Mode

```rust
// Set cache directory
hub::set_dir("~/my_cache/torsh/hub")?;

// Download for offline use
hub::download("cooljapan/resnet50", include_patterns=None)?;

// Use in offline mode
hub::set_offline_mode(true);
let model = hub::load("cooljapan/resnet50", pretrained=true)?; // Uses cache
```

### Integration with HuggingFace Hub

```rust
// Load from HuggingFace
let model = hub::from_huggingface(
    "microsoft/resnet-50",
    Some("main"),
    None,
)?;

// Convert and upload
let torch_model = load_pytorch_model("model.pt")?;
let torsh_model = convert_from_pytorch(torch_model)?;
hub::push_to_huggingface(
    torsh_model,
    "myusername/converted-model",
    token,
)?;
```

### Model Versioning

```rust
// Load specific version
let model_v1 = hub::load("cooljapan/model:v1.0", pretrained=true)?;
let model_latest = hub::load("cooljapan/model:latest", pretrained=true)?;

// Update model
hub::update_model(
    "cooljapan/model",
    updated_weights,
    UpdateConfig::default()
        .version("v2.0")
        .changelog("Improved accuracy, fixed bug in layer3"),
)?;
```

### Model Zoo

```rust
use torsh_hub::zoo::*;

// Vision models
let resnet = zoo::vision::resnet50(pretrained=true)?;
let efficientnet = zoo::vision::efficientnet_b0(pretrained=true)?;
let vit = zoo::vision::vit_base_patch16_224(pretrained=true)?;

// NLP models
let bert = zoo::nlp::bert_base_uncased(pretrained=true)?;
let gpt2 = zoo::nlp::gpt2_small(pretrained=true)?;

// Audio models
let wav2vec = zoo::audio::wav2vec2_base(pretrained=true)?;
```

### Security and Verification

```rust
// Verify model integrity
let verification = hub::verify_model("cooljapan/resnet50")?;
assert!(verification.is_trusted);
assert!(verification.signature_valid);

// Load with security checks
let model = hub::load_secure(
    "cooljapan/resnet50",
    SecurityConfig::default()
        .require_signature(true)
        .allowed_ops(vec!["aten::*", "torsh::*"])
        .max_file_size(500 * 1024 * 1024), // 500MB
)?;
```

## Environment Variables

- `TORSH_HUB_DIR`: Override default cache directory
- `TORSH_HUB_OFFLINE`: Enable offline mode
- `TORSH_HUB_TOKEN`: Authentication token for private models

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.