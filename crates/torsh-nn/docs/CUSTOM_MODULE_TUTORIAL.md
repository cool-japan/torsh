# Custom Module Tutorial

This tutorial walks you through building custom neural network modules in ToRSh-NN with practical, real-world examples.

## Table of Contents

1. [Introduction](#introduction)
2. [Tutorial 1: Simple MLP Classifier](#tutorial-1-simple-mlp-classifier)
3. [Tutorial 2: CNN for Image Classification](#tutorial-2-cnn-for-image-classification)
4. [Tutorial 3: Transformer Block](#tutorial-3-transformer-block)
5. [Tutorial 4: ResNet Block](#tutorial-4-resnet-block)
6. [Tutorial 5: Custom Loss Function](#tutorial-5-custom-loss-function)
7. [Tutorial 6: Composite Models](#tutorial-6-composite-models)
8. [Advanced Patterns](#advanced-patterns)

---

## Introduction

In ToRSh-NN, modules are the building blocks of neural networks. This tutorial teaches you to:
- Build custom layers from scratch
- Combine existing layers into models
- Create reusable components
- Handle complex architectures

### Prerequisites

```rust
use torsh_core::error::Result;
use torsh_tensor::{Tensor, creation::*};
use torsh_nn::{Module, Parameter, ParameterCollection, Linear, Conv2d, Sequential};
use torsh_nn::functional::*;
use torsh_nn::init::*;
```

---

## Tutorial 1: Simple MLP Classifier

Let's build a Multi-Layer Perceptron for classification.

### Step 1: Define the Structure

```rust
/// Simple 3-layer MLP for classification
pub struct MLPClassifier {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    dropout: Dropout,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}
```

### Step 2: Implement Constructor

```rust
impl MLPClassifier {
    /// Create new MLP classifier
    ///
    /// # Arguments
    /// * `input_dim` - Input feature dimension
    /// * `hidden_dim` - Hidden layer dimension
    /// * `output_dim` - Number of output classes
    /// * `dropout_rate` - Dropout probability (0.0-1.0)
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        dropout_rate: f32,
    ) -> Result<Self> {
        // Initialize layers with proper sizes
        let fc1 = Linear::new(input_dim, hidden_dim, true);
        let fc2 = Linear::new(hidden_dim, hidden_dim, true);
        let fc3 = Linear::new(hidden_dim, output_dim, true);
        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            fc1,
            fc2,
            fc3,
            dropout,
            input_dim,
            hidden_dim,
            output_dim,
        })
    }

    /// Create with custom initialization
    pub fn with_init(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        dropout_rate: f32,
        init_method: InitMethod,
    ) -> Result<Self> {
        let mut model = Self::new(input_dim, hidden_dim, output_dim, dropout_rate)?;

        // Apply custom initialization to all linear layers
        for (name, param) in model.parameters().iter_mut() {
            if name.contains("weight") {
                let shape = param.tensor().read().shape().dims().to_vec();
                let initialized = init_method.initialize(&shape)?;
                *param = Parameter::new(initialized, true);
            }
        }

        Ok(model)
    }
}
```

### Step 3: Implement Forward Pass

```rust
impl Module for MLPClassifier {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // First layer + ReLU + Dropout
        let mut x = self.fc1.forward(input)?;
        x = relu(&x)?;
        x = self.dropout.forward(&x)?;

        // Second layer + ReLU + Dropout
        x = self.fc2.forward(&x)?;
        x = relu(&x)?;
        x = self.dropout.forward(&x)?;

        // Output layer (no activation - will use cross entropy loss)
        x = self.fc3.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.dropout.is_training()
    }
}
```

### Step 4: Usage Example

```rust
fn main() -> Result<()> {
    // Create model
    let mut model = MLPClassifier::new(784, 256, 10, 0.5)?;

    // Training mode
    model.train();
    let train_input = randn::<f32>(&[32, 784])?; // Batch of 32 images
    let logits = model.forward(&train_input)?;
    println!("Training output shape: {:?}", logits.shape().dims());

    // Evaluation mode
    model.eval();
    let test_input = randn::<f32>(&[1, 784])?; // Single image
    let logits = model.forward(&test_input)?;
    let probs = softmax(&logits, Some(1))?;
    println!("Prediction probabilities: {:?}", probs.to_vec()?);

    Ok(())
}
```

---

## Tutorial 2: CNN for Image Classification

Build a Convolutional Neural Network for image classification.

### Complete Implementation

```rust
use torsh_nn::{Conv2d, MaxPool2d, Flatten, BatchNorm2d};

/// CNN for image classification (similar to LeNet-5)
pub struct ImageCNN {
    // Convolutional layers
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    pool: MaxPool2d,

    // Fully connected layers
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,

    // Regularization
    dropout: Dropout,

    // Config
    num_classes: usize,
}

impl ImageCNN {
    /// Create new CNN
    ///
    /// # Arguments
    /// * `num_classes` - Number of output classes
    /// * `dropout_rate` - Dropout probability
    pub fn new(num_classes: usize, dropout_rate: f32) -> Result<Self> {
        // First conv block: 1 -> 32 channels
        let conv1 = Conv2d::new(1, 32, 5, 1, 2, 1, 1, true)?; // 28x28 -> 28x28
        let bn1 = BatchNorm2d::new(32, 1e-5, 0.1, true, true)?;

        // Second conv block: 32 -> 64 channels
        let conv2 = Conv2d::new(32, 64, 5, 1, 2, 1, 1, true)?; // 14x14 -> 14x14
        let bn2 = BatchNorm2d::new(64, 1e-5, 0.1, true, true)?;

        // Pooling
        let pool = MaxPool2d::new(2, 2, 0)?; // 28x28 -> 14x14, 14x14 -> 7x7

        // Fully connected layers
        // After conv1 + pool: 14x14x32
        // After conv2 + pool: 7x7x64 = 3136
        let fc1 = Linear::new(7 * 7 * 64, 512, true);
        let fc2 = Linear::new(512, 128, true);
        let fc3 = Linear::new(128, num_classes, true);

        let dropout = Dropout::new(dropout_rate);

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            pool,
            fc1,
            fc2,
            fc3,
            dropout,
            num_classes,
        })
    }

    /// Forward through convolutional layers
    fn forward_conv(&self, x: &Tensor) -> Result<Tensor> {
        // First conv block
        let mut x = self.conv1.forward(x)?;
        x = self.bn1.forward(&x)?;
        x = relu(&x)?;
        x = self.pool.forward(&x)?;

        // Second conv block
        x = self.conv2.forward(&x)?;
        x = self.bn2.forward(&x)?;
        x = relu(&x)?;
        x = self.pool.forward(&x)?;

        Ok(x)
    }

    /// Forward through fully connected layers
    fn forward_fc(&self, x: &Tensor) -> Result<Tensor> {
        // First FC layer
        let mut x = self.fc1.forward(x)?;
        x = relu(&x)?;
        x = self.dropout.forward(&x)?;

        // Second FC layer
        x = self.fc2.forward(&x)?;
        x = relu(&x)?;
        x = self.dropout.forward(&x)?;

        // Output layer
        x = self.fc3.forward(&x)?;

        Ok(x)
    }
}

impl Module for ImageCNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Convolutional feature extraction
        let features = self.forward_conv(input)?;

        // Flatten for FC layers
        let batch_size = features.shape().dims()[0];
        let flattened = features.view(&[batch_size as i32, -1])?;

        // Classification head
        let output = self.forward_fc(&flattened)?;

        Ok(output)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.dropout.is_training()
    }
}

// Usage
fn train_cnn() -> Result<()> {
    let mut model = ImageCNN::new(10, 0.5)?;
    model.train();

    // MNIST-like input: batch_size=32, channels=1, height=28, width=28
    let input = randn::<f32>(&[32, 1, 28, 28])?;
    let output = model.forward(&input)?;

    println!("Output shape: {:?}", output.shape().dims()); // [32, 10]

    Ok(())
}
```

---

## Tutorial 3: Transformer Block

Implement a Transformer encoder block with multi-head attention.

### Complete Implementation

```rust
/// Transformer Encoder Block
pub struct TransformerBlock {
    // Multi-head self-attention
    self_attn: MultiHeadAttention,
    attn_norm: LayerNorm,
    attn_dropout: Dropout,

    // Feed-forward network
    ff_linear1: Linear,
    ff_linear2: Linear,
    ff_norm: LayerNorm,
    ff_dropout: Dropout,

    // Config
    d_model: usize,
    nhead: usize,
    dim_feedforward: usize,
}

impl TransformerBlock {
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout: f32,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, nhead, dropout)?,
            attn_norm: LayerNorm::new(vec![d_model], 1e-5)?,
            attn_dropout: Dropout::new(dropout),

            ff_linear1: Linear::new(d_model, dim_feedforward, true),
            ff_linear2: Linear::new(dim_feedforward, d_model, true),
            ff_norm: LayerNorm::new(vec![d_model], 1e-5)?,
            ff_dropout: Dropout::new(dropout),

            d_model,
            nhead,
            dim_feedforward,
        })
    }

    /// Forward pass with optional attention mask
    pub fn forward_with_mask(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Multi-head self-attention with residual connection
        let attn_out = self.self_attn.forward_with_mask(x, mask)?;
        let attn_out = self.attn_dropout.forward(&attn_out)?;
        let x = x.add(&attn_out)?; // Residual connection
        let x = self.attn_norm.forward(&x)?; // Post-norm

        // Feed-forward network with residual connection
        let mut ff_out = self.ff_linear1.forward(&x)?;
        ff_out = gelu(&ff_out)?; // GELU activation (common in transformers)
        ff_out = self.ff_dropout.forward(&ff_out)?;
        ff_out = self.ff_linear2.forward(&ff_out)?;
        ff_out = self.ff_dropout.forward(&ff_out)?;

        let output = x.add(&ff_out)?; // Residual connection
        let output = self.ff_norm.forward(&output)?; // Post-norm

        Ok(output)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(input, None)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.attn_norm.parameters());
        params.extend(self.ff_linear1.parameters());
        params.extend(self.ff_linear2.parameters());
        params.extend(self.ff_norm.parameters());
        params
    }

    fn train(&mut self) {
        self.self_attn.train();
        self.attn_dropout.train();
        self.ff_dropout.train();
    }

    fn eval(&mut self) {
        self.self_attn.eval();
        self.attn_dropout.eval();
        self.ff_dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.attn_dropout.is_training()
    }
}

/// Complete Transformer Encoder
pub struct TransformerEncoder {
    blocks: Vec<TransformerBlock>,
    num_layers: usize,
}

impl TransformerEncoder {
    pub fn new(
        num_layers: usize,
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout: f32,
    ) -> Result<Self> {
        let blocks: Result<Vec<_>> = (0..num_layers)
            .map(|_| TransformerBlock::new(d_model, nhead, dim_feedforward, dropout))
            .collect();

        Ok(Self {
            blocks: blocks?,
            num_layers,
        })
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        for (i, block) in self.blocks.iter().enumerate() {
            for (name, param) in block.parameters().iter() {
                params.add(&format!("block{}.{}", i, name), param.clone());
            }
        }
        params
    }

    fn train(&mut self) {
        for block in &mut self.blocks {
            block.train();
        }
    }

    fn eval(&mut self) {
        for block in &mut self.blocks {
            block.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.blocks.first().map(|b| b.is_training()).unwrap_or(true)
    }
}
```

---

## Tutorial 4: ResNet Block

Implement a residual block with skip connections.

```rust
/// Basic ResNet Block with skip connection
pub struct ResNetBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
    stride: usize,
}

impl ResNetBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        // First convolution
        let conv1 = Conv2d::new(
            in_channels,
            out_channels,
            3,        // kernel_size
            stride,   // stride
            1,        // padding
            1,        // dilation
            1,        // groups
            false,    // bias (BatchNorm handles this)
        )?;
        let bn1 = BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true)?;

        // Second convolution
        let conv2 = Conv2d::new(
            out_channels,
            out_channels,
            3,
            1,
            1,
            1,
            1,
            false,
        )?;
        let bn2 = BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true)?;

        // Downsample for skip connection if needed
        let downsample = if stride != 1 || in_channels != out_channels {
            let mut seq = Sequential::new();
            seq.add_module(
                "conv",
                Box::new(Conv2d::new(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    0,
                    1,
                    1,
                    false,
                )?),
            );
            seq.add_module(
                "bn",
                Box::new(BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true)?),
            );
            Some(seq)
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
            stride,
        })
    }
}

impl Module for ResNetBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let identity = input.clone();

        // Main path
        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = relu(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        // Skip connection
        let residual = if let Some(downsample) = &self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

        // Add residual and activate
        out = out.add(&residual)?;
        out = relu(&out)?;

        Ok(out)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());

        if let Some(downsample) = &self.downsample {
            params.extend(downsample.parameters());
        }

        params
    }

    fn train(&mut self) {
        self.bn1.train();
        self.bn2.train();
        if let Some(ds) = &mut self.downsample {
            ds.train();
        }
    }

    fn eval(&mut self) {
        self.bn1.eval();
        self.bn2.eval();
        if let Some(ds) = &mut self.downsample {
            ds.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.bn1.is_training()
    }
}

/// Complete ResNet-18 Architecture
pub struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    maxpool: MaxPool2d,

    layer1: Vec<ResNetBlock>,
    layer2: Vec<ResNetBlock>,
    layer3: Vec<ResNetBlock>,
    layer4: Vec<ResNetBlock>,

    avgpool: AdaptiveAvgPool2d,
    fc: Linear,

    num_classes: usize,
}

impl ResNet18 {
    pub fn new(num_classes: usize) -> Result<Self> {
        // Initial convolution
        let conv1 = Conv2d::new(3, 64, 7, 2, 3, 1, 1, false)?;
        let bn1 = BatchNorm2d::new(64, 1e-5, 0.1, true, true)?;
        let maxpool = MaxPool2d::new(3, 2, 1)?;

        // ResNet layers (2 blocks each for ResNet-18)
        let layer1 = vec![
            ResNetBlock::new(64, 64, 1)?,
            ResNetBlock::new(64, 64, 1)?,
        ];

        let layer2 = vec![
            ResNetBlock::new(64, 128, 2)?,
            ResNetBlock::new(128, 128, 1)?,
        ];

        let layer3 = vec![
            ResNetBlock::new(128, 256, 2)?,
            ResNetBlock::new(256, 256, 1)?,
        ];

        let layer4 = vec![
            ResNetBlock::new(256, 512, 2)?,
            ResNetBlock::new(512, 512, 1)?,
        ];

        // Classification head
        let avgpool = AdaptiveAvgPool2d::new(1, 1)?;
        let fc = Linear::new(512, num_classes, true);

        Ok(Self {
            conv1,
            bn1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
            num_classes,
        })
    }

    fn forward_layer(&self, x: &Tensor, blocks: &[ResNetBlock]) -> Result<Tensor> {
        let mut x = x.clone();
        for block in blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

impl Module for ResNet18 {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Initial convolution
        let mut x = self.conv1.forward(input)?;
        x = self.bn1.forward(&x)?;
        x = relu(&x)?;
        x = self.maxpool.forward(&x)?;

        // ResNet blocks
        x = self.forward_layer(&x, &self.layer1)?;
        x = self.forward_layer(&x, &self.layer2)?;
        x = self.forward_layer(&x, &self.layer3)?;
        x = self.forward_layer(&x, &self.layer4)?;

        // Classification head
        x = self.avgpool.forward(&x)?;
        let batch_size = x.shape().dims()[0];
        x = x.view(&[batch_size as i32, -1])?;
        x = self.fc.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());

        for (i, block) in self.layer1.iter().enumerate() {
            for (name, param) in block.parameters().iter() {
                params.add(&format!("layer1.{}.{}", i, name), param.clone());
            }
        }

        // Similar for layer2, layer3, layer4...
        params.extend(self.fc.parameters());

        params
    }

    fn train(&mut self) {
        self.bn1.train();
        for block in &mut self.layer1 {
            block.train();
        }
        for block in &mut self.layer2 {
            block.train();
        }
        for block in &mut self.layer3 {
            block.train();
        }
        for block in &mut self.layer4 {
            block.train();
        }
    }

    fn eval(&mut self) {
        self.bn1.eval();
        for block in &mut self.layer1 {
            block.eval();
        }
        for block in &mut self.layer2 {
            block.eval();
        }
        for block in &mut self.layer3 {
            block.eval();
        }
        for block in &mut self.layer4 {
            block.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.bn1.is_training()
    }
}
```

---

## Tutorial 5: Custom Loss Function

Create a custom loss function with the `CustomLoss` trait.

```rust
use torsh_nn::functional::loss_advanced::{CustomLoss, Reduction};

/// Focal Loss for handling class imbalance
pub struct FocalLoss {
    alpha: f32,
    gamma: f32,
    reduction: Reduction,
}

impl FocalLoss {
    pub fn new(alpha: f32, gamma: f32, reduction: Reduction) -> Self {
        Self {
            alpha,
            gamma,
            reduction,
        }
    }
}

impl CustomLoss for FocalLoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Apply softmax to get probabilities
        let probs = softmax(predictions, Some(1))?;

        // Get probabilities for true class
        let pt = probs.gather(1, targets)?;

        // Compute focal term: (1 - pt)^gamma
        let one_minus_pt = Tensor::from_data(vec![1.0], vec![1], pt.device())?
            .sub(&pt)?;
        let focal_term = one_minus_pt.pow_scalar(self.gamma)?;

        // Compute loss: -alpha * (1 - pt)^gamma * log(pt)
        let log_pt = pt.log()?;
        let loss = focal_term
            .mul_scalar(-self.alpha)?
            .mul(&log_pt)?;

        Ok(loss)
    }

    fn reduction(&self) -> &Reduction {
        &self.reduction
    }
}

// Usage
fn use_focal_loss() -> Result<()> {
    let loss_fn = FocalLoss::new(0.25, 2.0, Reduction::Mean);

    let predictions = randn::<f32>(&[32, 10])?;
    let targets = randint(0, 10, &[32])?;

    let loss = loss_fn.compute_loss(&predictions, &targets)?;
    println!("Focal loss: {:?}", loss.to_vec()?);

    Ok(())
}
```

---

## Tutorial 6: Composite Models

Build complex models by composing smaller modules.

```rust
/// Encoder-Decoder architecture for sequence-to-sequence tasks
pub struct Seq2SeqModel {
    encoder: TransformerEncoder,
    decoder: TransformerEncoder,
    embedding: Embedding,
    positional_encoding: Parameter,
    output_projection: Linear,

    d_model: usize,
    vocab_size: usize,
    max_seq_len: usize,
}

impl Seq2SeqModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        nhead: usize,
        num_layers: usize,
        dim_feedforward: usize,
        max_seq_len: usize,
        dropout: f32,
    ) -> Result<Self> {
        let encoder = TransformerEncoder::new(
            num_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
        )?;

        let decoder = TransformerEncoder::new(
            num_layers,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
        )?;

        let embedding = Embedding::new(vocab_size, d_model)?;

        // Learnable positional encoding
        let pos_encoding = randn::<f32>(&[max_seq_len, d_model])?;
        let positional_encoding = Parameter::new(pos_encoding, true);

        let output_projection = Linear::new(d_model, vocab_size, true);

        Ok(Self {
            encoder,
            decoder,
            embedding,
            positional_encoding,
            output_projection,
            d_model,
            vocab_size,
            max_seq_len,
        })
    }

    pub fn encode(&self, src: &Tensor) -> Result<Tensor> {
        // Embed and add positional encoding
        let embedded = self.embedding.forward(src)?;
        let seq_len = embedded.shape().dims()[1];

        let pos_enc = self.positional_encoding
            .tensor()
            .read()
            .narrow(0, 0, seq_len as i64)?;

        let x = embedded.add(&pos_enc)?;

        // Encode
        self.encoder.forward(&x)
    }

    pub fn decode(&self, tgt: &Tensor, memory: &Tensor) -> Result<Tensor> {
        // Embed and add positional encoding
        let embedded = self.embedding.forward(tgt)?;
        let seq_len = embedded.shape().dims()[1];

        let pos_enc = self.positional_encoding
            .tensor()
            .read()
            .narrow(0, 0, seq_len as i64)?;

        let mut x = embedded.add(&pos_enc)?;

        // Decode with cross-attention to encoder output
        x = self.decoder.forward(&x)?;

        // Project to vocabulary
        self.output_projection.forward(&x)
    }
}

impl Module for Seq2SeqModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For seq2seq, you typically need separate encode/decode calls
        // This is a simplified version
        let memory = self.encode(input)?;
        self.decode(input, &memory)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params.extend(self.embedding.parameters());
        params.add("positional_encoding", self.positional_encoding.clone());
        params.extend(self.output_projection.parameters());
        params
    }

    fn train(&mut self) {
        self.encoder.train();
        self.decoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
        self.decoder.eval();
    }

    fn is_training(&self) -> bool {
        self.encoder.is_training()
    }
}
```

---

## Advanced Patterns

### Pattern 1: Dynamic Module Selection

```rust
pub enum BackboneType {
    ResNet18,
    ResNet34,
    MobileNetV2,
}

pub struct FlexibleClassifier {
    backbone: Box<dyn Module>,
    classifier: Linear,
}

impl FlexibleClassifier {
    pub fn new(backbone_type: BackboneType, num_classes: usize) -> Result<Self> {
        let backbone: Box<dyn Module> = match backbone_type {
            BackboneType::ResNet18 => Box::new(ResNet18::new(1000)?),
            BackboneType::ResNet34 => Box::new(ResNet34::new(1000)?),
            BackboneType::MobileNetV2 => Box::new(MobileNetV2::new(1000)?),
        };

        let classifier = Linear::new(1000, num_classes, true);

        Ok(Self {
            backbone,
            classifier,
        })
    }
}
```

### Pattern 2: Module Registry

```rust
use std::collections::HashMap;

pub struct ModuleRegistry {
    modules: HashMap<String, Box<dyn Module>>,
}

impl ModuleRegistry {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, module: Box<dyn Module>) {
        self.modules.insert(name, module);
    }

    pub fn get(&self, name: &str) -> Option<&Box<dyn Module>> {
        self.modules.get(name)
    }

    pub fn forward_all(&self, input: &Tensor) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();
        for (name, module) in &self.modules {
            outputs.insert(name.clone(), module.forward(input)?);
        }
        Ok(outputs)
    }
}
```

### Pattern 3: Conditional Computation

```rust
pub struct AdaptiveModel {
    early_exit_threshold: f32,
    blocks: Vec<ResNetBlock>,
    classifiers: Vec<Linear>,
}

impl Module for AdaptiveModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        for (i, (block, classifier)) in self.blocks.iter()
            .zip(self.classifiers.iter())
            .enumerate()
        {
            x = block.forward(&x)?;

            if self.is_training() {
                // During training, compute all exits
                continue;
            } else {
                // During inference, early exit if confident
                let logits = classifier.forward(&x)?;
                let probs = softmax(&logits, Some(1))?;
                let max_prob = probs.max()?.item()?;

                if max_prob > self.early_exit_threshold {
                    return Ok(logits);
                }
            }
        }

        // Final classifier
        self.classifiers.last().unwrap().forward(&x)
    }

    // ... other trait methods
}
```

---

## Best Practices Summary

1. **Modular Design**: Break complex models into reusable components
2. **Parameter Management**: Always register all learnable parameters
3. **Training Mode**: Handle training/eval modes for layers like Dropout and BatchNorm
4. **Input Validation**: Check input shapes and provide clear error messages
5. **Memory Efficiency**: Avoid unnecessary clones, use views when possible
6. **Initialization**: Use appropriate initialization methods for your architecture
7. **Documentation**: Document module parameters, input/output shapes, and usage
8. **Testing**: Write tests for each custom module
9. **Composition**: Build complex models by composing simpler modules
10. **Flexibility**: Use enums and traits for configurable architectures

---

## Next Steps

- Explore the `examples/` directory for more complete examples
- Read the [Layer Implementation Guide](LAYER_IMPLEMENTATION_GUIDE.md) for detailed API reference
- Check out [PyTorch Migration Guide](PYTORCH_MIGRATION_GUIDE.md) for converting existing models
- Join the community and contribute your custom modules!

For questions or contributions, visit: https://github.com/cool-japan/torsh
