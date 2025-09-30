# torsh-models

Pre-built model architectures and model zoo for ToRSh.

## Overview

This crate provides ready-to-use model architectures for various domains:

- **Computer Vision**: ResNet, EfficientNet, Vision Transformer, etc.
- **Natural Language Processing**: BERT, GPT, T5, etc.
- **Audio Processing**: Wav2Vec2, Whisper, etc.
- **Multimodal**: CLIP, DALL-E, etc.
- **Model Utilities**: Weight initialization, model surgery, pruning

## Usage

### Vision Models

```rust
use torsh_models::vision::*;

// ResNet variants
let resnet18 = resnet::resnet18(pretrained=true, num_classes=1000)?;
let resnet50 = resnet::resnet50(pretrained=true, num_classes=1000)?;
let resnet101 = resnet::resnet101(pretrained=true, num_classes=1000)?;

// Custom configuration
let custom_resnet = resnet::ResNet::new(
    resnet::ResNetConfig {
        layers: vec![3, 4, 6, 3],
        num_classes: 100,
        groups: 1,
        width_per_group: 64,
        norm_layer: Some(BatchNorm2d),
    }
)?;

// EfficientNet family
let efficientnet_b0 = efficientnet::efficientnet_b0(pretrained=true)?;
let efficientnet_b7 = efficientnet::efficientnet_b7(pretrained=true)?;

// Vision Transformer
let vit_b_16 = vit::vit_base_patch16_224(pretrained=true)?;
let vit_l_32 = vit::vit_large_patch32_384(pretrained=true)?;

// Object Detection
let faster_rcnn = detection::fasterrcnn_resnet50_fpn(pretrained=true)?;
let mask_rcnn = detection::maskrcnn_resnet50_fpn(pretrained=true)?;

// Segmentation
let deeplabv3 = segmentation::deeplabv3_resnet101(pretrained=true)?;
let fcn = segmentation::fcn_resnet50(pretrained=true)?;
```

### NLP Models

```rust
use torsh_models::nlp::*;

// BERT variants
let bert_base = bert::bert_base_uncased(pretrained=true)?;
let bert_large = bert::bert_large_cased(pretrained=true)?;

// Custom BERT configuration
let custom_bert = bert::BertModel::new(
    bert::BertConfig {
        vocab_size: 30522,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        hidden_dropout_prob: 0.1,
        attention_probs_dropout_prob: 0.1,
        max_position_embeddings: 512,
        ..Default::default()
    }
)?;

// GPT models
let gpt2 = gpt::gpt2(pretrained=true)?;
let gpt2_medium = gpt::gpt2_medium(pretrained=true)?;

// T5 models
let t5_small = t5::t5_small(pretrained=true)?;
let t5_base = t5::t5_base(pretrained=true)?;

// For specific tasks
let bert_classifier = bert::BertForSequenceClassification::new(
    bert_base,
    num_labels=2,
)?;
```

### Audio Models

```rust
use torsh_models::audio::*;

// Wav2Vec2
let wav2vec2 = wav2vec2::wav2vec2_base(pretrained=true)?;

// Whisper
let whisper_base = whisper::whisper_base(pretrained=true)?;
let whisper_large = whisper::whisper_large(pretrained=true)?;

// Audio classification
let audio_classifier = AudioClassifier::new(
    wav2vec2,
    num_classes=50,
)?;
```

### Multimodal Models

```rust
use torsh_models::multimodal::*;

// CLIP
let clip = clip::clip_vit_base_patch32(pretrained=true)?;
let (image_features, text_features) = clip.forward(images, texts)?;

// Flamingo
let flamingo = flamingo::flamingo_base(pretrained=true)?;
```

### Model Utilities

```rust
use torsh_models::utils::*;

// Weight initialization
init_weights(&mut model, InitType::KaimingNormal)?;

// Model surgery
let pruned_model = prune_model(&model, amount=0.3, structured=true)?;

// Knowledge distillation
let student = distill_model(&teacher, &student_arch, temperature=3.0)?;

// Model conversion
let quantized = quantize_model(&model, bits=8)?;
let onnx_model = export_onnx(&model, &example_input)?;
```

### Transfer Learning

```rust
use torsh_models::transfer::*;

// Fine-tune pre-trained model
let base_model = resnet50(pretrained=true)?;
let feature_extractor = remove_head(&base_model);
let new_model = add_custom_head(&feature_extractor, num_classes=10)?;

// Freeze base layers
freeze_layers(&mut new_model, until_layer="layer3")?;

// Progressive unfreezing
let scheduler = UnfreezeScheduler::new(&model)
    .unfreeze_at(epoch=5, layer="layer3")
    .unfreeze_at(epoch=10, layer="layer2");
```

### Model Configuration

```rust
use torsh_models::config::*;

// Load configuration
let config = ModelConfig::from_pretrained("cooljapan/bert-base")?;

// Modify configuration
config.hidden_size = 1024;
config.num_attention_heads = 16;

// Create model from config
let model = AutoModel::from_config(config)?;

// Save configuration
config.save("my_model_config.json")?;
```

### Model Registry

```rust
use torsh_models::registry::*;

// Register custom model
register_model("my_custom_resnet", || {
    resnet::ResNet::new(custom_config)
});

// List available models
let models = list_models()?;
for (name, info) in models {
    println!("{}: {}", name, info.description);
}

// Load by name
let model = load_model("my_custom_resnet", pretrained=false)?;
```

### Benchmarking

```rust
use torsh_models::benchmark::*;

// Benchmark inference speed
let results = benchmark_model(&model, input_shape, num_runs=100)?;
println!("Average latency: {:?}", results.mean_latency);
println!("Throughput: {} samples/sec", results.throughput);

// Compare models
let comparison = compare_models(
    vec![&model1, &model2, &model3],
    input_shape,
    metrics=vec!["latency", "memory", "flops"],
)?;
```

## Tutorials

### Tutorial 1: Image Classification with Pre-trained ResNet

```rust
use torsh_models::prelude::*;
use torsh_models::registry::get_global_registry;
use torsh_tensor::Tensor;

// Load a pre-trained ResNet model
let registry = get_global_registry();
let model_handle = registry.get_model_handle("resnet18", Some("1.0.0"))?;

// Create the model
let mut model = resnet::resnet18(true, 1000)?;
model.eval(); // Set to evaluation mode

// Prepare input (batch of RGB images)
let batch_size = 4;
let input = Tensor::randn(&[batch_size, 3, 224, 224])?;

// Forward pass
let output = model.forward(&input)?;
let predictions = output.softmax(-1)?;

// Get top-5 predictions
let (top5_probs, top5_indices) = predictions.topk(5, -1, true, true)?;
println!("Top 5 predictions for each image in batch:");
for i in 0..batch_size {
    println!("Image {}: {:?}", i, top5_indices.select(0, i)?);
}
```

### Tutorial 2: Text Classification with BERT

```rust
use torsh_models::prelude::*;
use torsh_tensor::Tensor;

// Create BERT model for sequence classification
let config = bert::BertConfig {
    vocab_size: 30522,
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    num_labels: 2, // Binary classification
    ..Default::default()
};

let mut model = bert::BertForSequenceClassification::new(config)?;
model.eval();

// Prepare tokenized input (token IDs)
let batch_size = 2;
let seq_len = 128;
let input_ids = Tensor::randint(0, 30522, &[batch_size, seq_len])?;
let attention_mask = Tensor::ones(&[batch_size, seq_len])?;

// Forward pass
let output = model.forward(&input_ids, Some(&attention_mask), None)?;
let logits = output.logits;
let predictions = logits.softmax(-1)?;

// Get predicted classes
let predicted_classes = predictions.argmax(-1, false)?;
println!("Predicted classes: {:?}", predicted_classes);
```

### Tutorial 3: Speech Recognition with Whisper

```rust
use torsh_models::prelude::*;
use torsh_tensor::Tensor;

// Load Whisper model
let config = whisper::WhisperConfig::base();
let mut model = whisper::WhisperForConditionalGeneration::new(config)?;
model.eval();

// Prepare mel spectrogram input
let batch_size = 1;
let n_mels = 80;
let seq_len = 3000;
let input_features = Tensor::randn(&[batch_size, n_mels, seq_len])?;

// Generate transcription
let decoder_input_ids = Tensor::tensor(&[[50258]])?: // Start token
let output = model.generate(&input_features, Some(&decoder_input_ids), None)?;
println!("Generated tokens: {:?}", output);
```

### Tutorial 4: Vision-Language Understanding with CLIP

```rust
use torsh_models::prelude::*;
use torsh_tensor::Tensor;

// Load CLIP model
let config = clip::CLIPConfig::default();
let mut model = clip::CLIPModel::new(config)?;
model.eval();

// Prepare inputs
let batch_size = 4;
let image_input = Tensor::randn(&[batch_size, 3, 224, 224])?;
let text_input = Tensor::randint(0, 49408, &[batch_size, 77])?; // Tokenized text

// Get embeddings
let image_features = model.get_image_features(&image_input)?;
let text_features = model.get_text_features(&text_input)?;

// Compute similarity
let similarity = image_features.matmul(&text_features.transpose(-2, -1)?)?;
let probs = similarity.softmax(-1)?;
println!("Image-text similarity matrix: {:?}", probs.shape());
```

### Tutorial 5: Fine-tuning for Custom Dataset

```rust
use torsh_models::prelude::*;
use torsh_optim::SGD;
use torsh_nn::functional as F;

// Load pre-trained model and modify for custom task
let mut base_model = resnet::resnet18(true, 1000)?;

// Replace classifier head for custom number of classes
let num_custom_classes = 10;
let in_features = 512; // ResNet18 final layer input size
let custom_head = Linear::new(in_features, num_custom_classes, true)?;
base_model.fc = custom_head;

// Set up optimizer
let mut optimizer = SGD::new(base_model.parameters(), 0.001)?;

// Training loop
for epoch in 0..10 {
    base_model.train();
    
    // Prepare batch (replace with actual data loading)
    let batch_images = Tensor::randn(&[32, 3, 224, 224])?;
    let batch_labels = Tensor::randint(0, num_custom_classes, &[32])?;
    
    // Forward pass
    let outputs = base_model.forward(&batch_images)?;
    let loss = F::cross_entropy(&outputs, &batch_labels)?;
    
    // Backward pass
    optimizer.zero_grad();
    loss.backward()?;
    optimizer.step()?;
    
    if epoch % 2 == 0 {
        println!("Epoch {}: Loss = {:.4}", epoch, loss.item::<f32>()?);
    }
}
```

### Tutorial 6: Model Quantization and Optimization

```rust
use torsh_models::prelude::*;
use torsh_models::quantization::*;

// Load pre-trained model
let mut model = resnet::resnet50(true, 1000)?;
model.eval();

// Prepare calibration data
let calibration_data = vec![
    Tensor::randn(&[1, 3, 224, 224])?,
    Tensor::randn(&[1, 3, 224, 224])?,
    // ... more calibration samples
];

// Configure quantization
let quant_config = QuantizationConfig {
    strategy: QuantizationStrategy::PostTrainingQuantization,
    dtype: QuantizationDType::Int8,
    granularity: QuantizationGranularity::PerChannel,
    observer_type: ObserverType::MinMax,
    ..Default::default()
};

// Quantize the model
let mut quantizer = ModelQuantizer::new(quant_config);
let quantized_model = quantizer.quantize(&model, &calibration_data)?;

// Benchmark original vs quantized
let input = Tensor::randn(&[1, 3, 224, 224])?;

let start = std::time::Instant::now();
let output1 = model.forward(&input)?;
let original_time = start.elapsed();

let start = std::time::Instant::now();
let output2 = quantized_model.forward(&input)?;
let quantized_time = start.elapsed();

println!("Original model: {:.2}ms", original_time.as_millis());
println!("Quantized model: {:.2}ms", quantized_time.as_millis());
println!("Speedup: {:.2}x", original_time.as_millis() as f32 / quantized_time.as_millis() as f32);
```

### Tutorial 7: Model Ensembling

```rust
use torsh_models::prelude::*;
use torsh_models::ensembling::*;

// Create multiple models
let model1 = resnet::resnet18(true, 1000)?;
let model2 = resnet::resnet34(true, 1000)?;
let model3 = efficientnet::efficientnet_b0(true)?;

// Create ensemble
let ensemble_config = EnsembleConfig {
    method: EnsembleMethod::WeightedAverage,
    weights: vec![0.4, 0.4, 0.2],
    diversity_regularization: true,
    ..Default::default()
};

let mut ensemble = ModelEnsemble::new(ensemble_config);
ensemble.add_model(Box::new(model1))?;
ensemble.add_model(Box::new(model2))?;
ensemble.add_model(Box::new(model3))?;

// Inference with ensemble
let input = Tensor::randn(&[4, 3, 224, 224])?;
let ensemble_output = ensemble.forward(&input)?;
let predictions = ensemble_output.softmax(-1)?;

println!("Ensemble predictions shape: {:?}", predictions.shape());
```

## Migration Guide

### Migrating from PyTorch

ToRSh models are designed to be similar to PyTorch for easy migration. Here are common patterns:

#### Model Creation

**PyTorch:**
```python
import torch
import torchvision.models as models

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()
```

**ToRSh:**
```rust
use torsh_models::vision::resnet;

// Load pre-trained model
let mut model = resnet::resnet18(true, 1000)?;
model.eval();
```

#### Forward Pass

**PyTorch:**
```python
with torch.no_grad():
    output = model(input_tensor)
    predictions = torch.softmax(output, dim=1)
```

**ToRSh:**
```rust
let output = model.forward(&input_tensor)?;
let predictions = output.softmax(1)?;
```

#### Model Configuration

**PyTorch:**
```python
from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)
model = BertModel(config)
```

**ToRSh:**
```rust
use torsh_models::nlp::bert;

let config = bert::BertConfig {
    vocab_size: 30522,
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    ..Default::default()
};
let model = bert::BertModel::new(config)?;
```

#### Training Loop

**PyTorch:**
```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch.data)
        loss = criterion(outputs, batch.labels)
        loss.backward()
        optimizer.step()
```

**ToRSh:**
```rust
use torsh_optim::SGD;
use torsh_nn::functional as F;

let mut optimizer = SGD::new(model.parameters(), 0.001)?;

for epoch in 0..num_epochs {
    model.train();
    for batch in dataloader {
        optimizer.zero_grad();
        let outputs = model.forward(&batch.data)?;
        let loss = F::cross_entropy(&outputs, &batch.labels)?;
        loss.backward()?;
        optimizer.step()?;
    }
}
```

### Migrating from TensorFlow/Keras

#### Sequential Model

**TensorFlow/Keras:**
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**ToRSh:**
```rust
use torsh_nn::prelude::*;

let model = Sequential::new()
    .add(Conv2d::new(3, 32, 3, ConvConfig::default())?)
    .add(BatchNorm2d::new(32)?)
    .add(ReLU::new())
    .add(MaxPool2d::new(2, 2))
    .add(Flatten::new())
    .add(Linear::new(flatten_size, 10, true)?)
    .add(Softmax::new(-1));
```

#### Model Compilation and Training

**TensorFlow/Keras:**
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```

**ToRSh:**
```rust
use torsh_optim::Adam;
use torsh_nn::functional as F;

let mut optimizer = Adam::new(model.parameters(), 0.001)?;

for epoch in 0..10 {
    // Training phase
    model.train();
    let train_loss = training_step(&mut model, &mut optimizer, &train_data)?;
    
    // Validation phase
    model.eval();
    let val_accuracy = validate(&model, &val_data)?;
    
    println!("Epoch {}: Train Loss = {:.4}, Val Acc = {:.4}", 
             epoch, train_loss, val_accuracy);
}
```

### Common Migration Patterns

#### Error Handling

**Python (with exceptions):**
```python
try:
    model = load_model('path/to/model')
    output = model(input_data)
except Exception as e:
    print(f"Error: {e}")
```

**Rust (with Result types):**
```rust
match load_model("path/to/model") {
    Ok(model) => {
        match model.forward(&input_data) {
            Ok(output) => println!("Success: {:?}", output.shape()),
            Err(e) => println!("Forward error: {}", e),
        }
    },
    Err(e) => println!("Load error: {}", e),
}

// Or using the ? operator for cleaner code:
let model = load_model("path/to/model")?;
let output = model.forward(&input_data)?;
```

#### Device Management

**PyTorch:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)
```

**ToRSh:**
```rust
use torsh_core::Device;

let device = if Device::cuda_is_available() {
    Device::Cuda(0)
} else {
    Device::Cpu
};

let model = model.to_device(&device)?;
let input_tensor = input_tensor.to_device(&device)?;
```

#### Model Saving and Loading

**PyTorch:**
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
```

**ToRSh:**
```rust
// Save
model.save("model.safetensors")?;

// Load
let model = ModelType::load("model.safetensors")?;
```

### Key Differences to Note

1. **Memory Safety**: ToRSh provides compile-time memory safety guarantees
2. **Error Handling**: Rust uses `Result<T, E>` instead of exceptions
3. **Ownership**: Rust's ownership system requires explicit handling of data movement
4. **Immutability**: Variables are immutable by default, use `mut` for mutable variables
5. **Type Safety**: Strong static typing catches errors at compile time
6. **Performance**: Zero-cost abstractions and no garbage collection

### Best Practices for Migration

1. **Start Small**: Begin with simple models and gradually increase complexity
2. **Use Type Annotations**: Leverage Rust's type system for better code clarity
3. **Handle Errors Properly**: Use `?` operator and proper error handling patterns
4. **Leverage Rust Tools**: Use `cargo clippy` for linting and `cargo fmt` for formatting
5. **Test Thoroughly**: Write unit tests to ensure model behavior matches expectations
6. **Use the Registry**: Take advantage of the built-in model registry for pretrained models

## Available Models

### Vision
- ResNet (18, 34, 50, 101, 152)
- ResNeXt (50, 101)
- Wide ResNet
- EfficientNet (B0-B7)
- MobileNet (V2, V3)
- VGG (11, 13, 16, 19)
- DenseNet (121, 161, 169, 201)
- Vision Transformer (ViT)
- Swin Transformer
- ConvNeXt

### NLP
- BERT (Base, Large)
- RoBERTa
- GPT-2 (Small, Medium, Large, XL)
- T5 (Small, Base, Large)
- BART
- XLNet
- ELECTRA

### Audio
- Wav2Vec2
- Whisper
- HuBERT
- WavLM

### Detection & Segmentation
- Faster R-CNN
- Mask R-CNN
- YOLO (v5, v8)
- DETR
- DeepLabV3
- U-Net

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.