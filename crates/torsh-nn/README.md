# torsh-nn

Neural network modules for ToRSh with PyTorch-compatible API, powered by scirs2-neural.

## Overview

This crate provides comprehensive neural network building blocks including:

- Common layers (Linear, Conv2d, BatchNorm, etc.)
- Activation functions (ReLU, Sigmoid, GELU, etc.)
- Container modules (Sequential, ModuleList, ModuleDict)
- Parameter initialization utilities
- Functional API for stateless operations

## Usage

### Basic Module Definition

```rust
use torsh_nn::prelude::*;
use torsh_tensor::prelude::*;

// Define a simple neural network
struct SimpleNet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SimpleNet {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 128, true),
            fc2: Linear::new(128, 64, true),
            fc3: Linear::new(64, 10, true),
        }
    }
}

impl Module for SimpleNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(input)?;
        let x = F::relu(&x);
        let x = self.fc2.forward(&x)?;
        let x = F::relu(&x);
        self.fc3.forward(&x)
    }
    
    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
    
    // ... other trait methods
}
```

### Using Sequential Container

```rust
use torsh_nn::prelude::*;

let model = Sequential::new()
    .add(Linear::new(784, 128, true))
    .add(ReLU::new(false))
    .add(Dropout::new(0.5, false))
    .add(Linear::new(128, 64, true))
    .add(ReLU::new(false))
    .add(Linear::new(64, 10, true));

let output = model.forward(&input)?;
```

### Functional API

```rust
use torsh_nn::functional as F;

// Activation functions
let x = F::relu(&input);
let x = F::gelu(&x);
let x = F::softmax(&x, -1)?;

// Pooling
let x = F::max_pool2d(&input, (2, 2), None, None, None)?;
let x = F::global_avg_pool2d(&x)?;

// Loss functions
let loss = F::cross_entropy(&logits, &targets, None, "mean", None)?;
let loss = F::mse_loss(&predictions, &targets, "mean")?;
```

### Parameter Initialization

```rust
use torsh_nn::init;

// Xavier/Glorot initialization
let weight = init::xavier_uniform(&[128, 784]);

// Kaiming/He initialization for ReLU
let weight = init::kaiming_normal(&[64, 128], "fan_out");

// Initialize existing tensor
let mut tensor = zeros(&[10, 10]);
init::init_tensor(&mut tensor, "orthogonal", Some(1.0), None);
```

### Common Layers

#### Linear Layer
```rust
let linear = Linear::new(in_features, out_features, bias);
```

#### Convolutional Layer
```rust
let conv = Conv2d::new(
    in_channels,
    out_channels,
    (3, 3),        // kernel_size
    Some((1, 1)),  // stride
    Some((1, 1)),  // padding
    None,          // dilation
    None,          // groups
    true,          // bias
);
```

#### Batch Normalization
```rust
let bn = BatchNorm2d::new(
    num_features,
    Some(1e-5),    // eps
    Some(0.1),     // momentum
    true,          // affine
    true,          // track_running_stats
);
```

#### LSTM
```rust
let lstm = LSTM::new(
    input_size,
    hidden_size,
    Some(2),       // num_layers
    true,          // bias
    false,         // batch_first
    Some(0.2),     // dropout
    false,         // bidirectional
);
```

### Container Modules

#### ModuleList
```rust
let mut layers = ModuleList::new();
layers.append(Linear::new(10, 20, true));
layers.append(Linear::new(20, 30, true));

// Access by index
if let Some(layer) = layers.get(0) {
    let output = layer.forward(&input)?;
}
```

#### ModuleDict
```rust
let mut blocks = ModuleDict::new();
blocks.insert("encoder".to_string(), Linear::new(784, 128, true));
blocks.insert("decoder".to_string(), Linear::new(128, 784, true));

// Access by key
if let Some(encoder) = blocks.get("encoder") {
    let encoded = encoder.forward(&input)?;
}
```

### Parameter Management

```rust
use torsh_nn::parameter::utils;

// Count parameters
let total = utils::count_parameters(&model.parameters());
let trainable = utils::count_trainable_parameters(&model.parameters());

// Freeze/unfreeze parameters
utils::freeze_parameters(&encoder.parameters());
utils::unfreeze_parameters(&decoder.parameters());

// Get parameter statistics
let stats = utils::parameter_stats(&model.parameters());
println!("{}", stats);

// Gradient clipping
utils::clip_grad_norm_(&mut model.parameters(), 1.0, 2.0);
utils::clip_grad_value_(&mut model.parameters(), 0.5);
```

## Integration with SciRS2

This crate leverages scirs2-neural for:

- Optimized layer implementations
- Automatic differentiation support
- Hardware acceleration
- Memory-efficient operations

All modules are designed to work seamlessly with ToRSh's autograd system while benefiting from scirs2's performance optimizations.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.