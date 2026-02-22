# PyTorch to ToRSh Migration Guide

A comprehensive guide for migrating PyTorch models and code to ToRSh-NN.

## Table of Contents

1. [Quick Start Comparison](#quick-start-comparison)
2. [Core Concepts](#core-concepts)
3. [Layer-by-Layer Migration](#layer-by-layer-migration)
4. [Training Loop Migration](#training-loop-migration)
5. [Common Patterns](#common-patterns)
6. [API Differences](#api-differences)
7. [Migration Checklist](#migration-checklist)
8. [Complete Examples](#complete-examples)

---

## Quick Start Comparison

### Simple Neural Network

#### PyTorch
```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(784, 256, 10)
```

#### ToRSh-NN
```rust
use torsh_nn::{Module, Linear, ParameterCollection};
use torsh_nn::functional::relu;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

pub struct SimpleNet {
    fc1: Linear,
    fc2: Linear,
}

impl SimpleNet {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(input_dim, hidden_dim, true),
            fc2: Linear::new(hidden_dim, output_dim, true),
        })
    }
}

impl Module for SimpleNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(input)?;
        let x = relu(&x)?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { true }
}

let model = SimpleNet::new(784, 256, 10)?;
```

---

## Core Concepts

### Tensor Operations

| PyTorch | ToRSh | Notes |
|---------|-------|-------|
| `torch.tensor([1, 2, 3])` | `tensor_1d(&[1.0, 2.0, 3.0])?` | Explicit error handling |
| `torch.randn(2, 3)` | `randn::<f32>(&[2, 3])?` | Type parameter required |
| `x.shape` | `x.shape().dims()` | Returns slice of dimensions |
| `x.view(-1, 784)` | `x.view(&[-1, 784])?` | Similar reshaping |
| `x.requires_grad = True` | `Parameter::new(x, true)` | Parameters track gradients |
| `x.detach()` | `x.detach()` | Same concept |
| `x.clone()` | `x.clone()` | Same method |

### Module Definition

| Aspect | PyTorch | ToRSh |
|--------|---------|-------|
| **Base Class** | `nn.Module` | `Module` trait |
| **Constructor** | `__init__` | `new()` (by convention) |
| **Forward** | `def forward(self, x)` | `fn forward(&self, input: &Tensor) -> Result<Tensor>` |
| **Parameters** | Automatic registration | Explicit via `parameters()` method |
| **Training Mode** | `model.train()` / `model.eval()` | `model.train()` / `model.eval()` |

---

## Layer-by-Layer Migration

### Linear (Fully Connected) Layers

#### PyTorch
```python
import torch.nn as nn

# Create layer
linear = nn.Linear(in_features=128, out_features=64, bias=True)

# Forward pass
output = linear(input)

# Access weights
weights = linear.weight
bias = linear.bias
```

#### ToRSh-NN
```rust
use torsh_nn::Linear;

// Create layer
let linear = Linear::new(128, 64, true);

// Forward pass
let output = linear.forward(&input)?;

// Access weights (through parameters)
let params = linear.parameters();
let weight = params.get("weight").unwrap();
let bias = params.get("bias").unwrap();
```

### Convolutional Layers

#### PyTorch
```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True
)
output = conv(input)
```

#### ToRSh-NN
```rust
use torsh_nn::Conv2d;

let conv = Conv2d::new(
    3,      // in_channels
    64,     // out_channels
    3,      // kernel_size
    1,      // stride
    1,      // padding
    1,      // dilation
    1,      // groups
    true,   // bias
)?;
let output = conv.forward(&input)?;
```

### Normalization Layers

#### PyTorch
```python
# Batch Normalization
bn = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1)
output = bn(input)

# Layer Normalization
ln = nn.LayerNorm(normalized_shape=[256], eps=1e-5)
output = ln(input)
```

#### ToRSh-NN
```rust
use torsh_nn::{BatchNorm2d, LayerNorm};

// Batch Normalization
let bn = BatchNorm2d::new(
    64,     // num_features
    1e-5,   // eps
    0.1,    // momentum
    true,   // affine
    true,   // track_running_stats
)?;
let output = bn.forward(&input)?;

// Layer Normalization
let ln = LayerNorm::new(vec![256], 1e-5)?;
let output = ln.forward(&input)?;
```

### Activation Functions

#### PyTorch (Functional)
```python
import torch.nn.functional as F

x = F.relu(x)
x = F.gelu(x)
x = F.sigmoid(x)
x = F.softmax(x, dim=1)
x = F.dropout(x, p=0.5, training=True)
```

#### ToRSh-NN (Functional)
```rust
use torsh_nn::functional::*;

let x = relu(&x)?;
let x = gelu(&x)?;
let x = sigmoid(&x)?;
let x = softmax(&x, Some(1))?;
let x = dropout(&x, 0.5, true)?;
```

#### PyTorch (Module)
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
```

#### ToRSh-NN (Module)
```rust
pub struct MyModel {
    dropout: Dropout,
}

impl MyModel {
    pub fn new() -> Result<Self> {
        Ok(Self {
            dropout: Dropout::new(0.5),
        })
    }
}

impl Module for MyModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = relu(input)?;
        let x = self.dropout.forward(&x)?;
        Ok(x)
    }
    // ... other methods
}
```

### Pooling Layers

#### PyTorch
```python
# Max pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
output = maxpool(input)

# Average pooling
avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
output = avgpool(input)

# Adaptive average pooling
adaptive = nn.AdaptiveAvgPool2d((1, 1))
output = adaptive(input)
```

#### ToRSh-NN
```rust
use torsh_nn::{MaxPool2d, AvgPool2d, AdaptiveAvgPool2d};

// Max pooling
let maxpool = MaxPool2d::new(2, 2, 0)?;
let output = maxpool.forward(&input)?;

// Average pooling
let avgpool = AvgPool2d::new(2, 2, 0)?;
let output = avgpool.forward(&input)?;

// Adaptive average pooling
let adaptive = AdaptiveAvgPool2d::new(1, 1)?;
let output = adaptive.forward(&input)?;
```

### Recurrent Layers

#### PyTorch
```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(input, (h_0, c_0))

gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
output, h_n = gru(input, h_0)
```

#### ToRSh-NN
```rust
use torsh_nn::{LSTM, GRU};

let lstm = LSTM::new(128, 256, 2, true, 0.0, true)?;
let (output, (h_n, c_n)) = lstm.forward_with_state(&input, Some((&h_0, &c_0)))?;

let gru = GRU::new(128, 256, 2, true, 0.0, true)?;
let (output, h_n) = gru.forward_with_state(&input, Some(&h_0))?;
```

---

## Training Loop Migration

### Basic Training Loop

#### PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
```

#### ToRSh-NN
```rust
use torsh_nn::{Module, MyModel};
use torsh_nn::functional::cross_entropy;
use torsh_optim::{Adam, Optimizer};

let mut model = MyModel::new()?;
let mut optimizer = Adam::new(model.parameters(), 0.001);

model.train();
for epoch in 0..num_epochs {
    for (batch_idx, (data, target)) in train_loader.enumerate() {
        // Forward pass
        let output = model.forward(&data)?;
        let loss = cross_entropy(&output, &target, None, "mean")?;

        // Backward pass (automatic differentiation)
        loss.backward()?;

        // Update weights
        optimizer.step()?;
        optimizer.zero_grad();

        if batch_idx % 100 == 0 {
            let loss_val = loss.item::<f32>()?;
            println!("Epoch: {}, Loss: {:.4}", epoch, loss_val);
        }
    }
}
```

### Evaluation Loop

#### PyTorch
```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

accuracy = 100. * correct / total
print(f'Accuracy: {accuracy:.2f}%')
```

#### ToRSh-NN
```rust
model.eval();
let mut correct = 0;
let mut total = 0;

for (data, target) in test_loader {
    let output = model.forward(&data)?;
    let pred = output.argmax(1, false)?;

    // Compare predictions with targets
    let matches = pred.eq(&target)?;
    correct += matches.sum()?.item::<i64>()?;
    total += target.numel();
}

let accuracy = 100.0 * correct as f32 / total as f32;
println!("Accuracy: {:.2}%", accuracy);
```

---

## Common Patterns

### Sequential Models

#### PyTorch
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

#### ToRSh-NN
```rust
use torsh_nn::Sequential;

let mut model = Sequential::new();
model.add_module("fc1", Box::new(Linear::new(784, 256, true)));
model.add_module("relu1", Box::new(ReLU::new()));
model.add_module("dropout", Box::new(Dropout::new(0.5)));
model.add_module("fc2", Box::new(Linear::new(256, 128, true)));
model.add_module("relu2", Box::new(ReLU::new()));
model.add_module("fc3", Box::new(Linear::new(128, 10, true)));
```

### ModuleList / ModuleDict

#### PyTorch
```python
# ModuleList
self.layers = nn.ModuleList([
    nn.Linear(128, 128) for _ in range(5)
])

# ModuleDict
self.layers = nn.ModuleDict({
    'linear': nn.Linear(128, 256),
    'conv': nn.Conv2d(3, 64, 3),
})

# Forward
for layer in self.layers:
    x = layer(x)
```

#### ToRSh-NN
```rust
// ModuleList
pub struct MyModel {
    layers: Vec<Linear>,
}

impl MyModel {
    pub fn new() -> Result<Self> {
        let layers: Result<Vec<_>> = (0..5)
            .map(|_| Linear::new(128, 128, true))
            .collect();

        Ok(Self {
            layers: layers?,
        })
    }
}

impl Module for MyModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
    // ... other methods
}

// ModuleDict equivalent
use std::collections::HashMap;

pub struct MyModel {
    layers: HashMap<String, Box<dyn Module>>,
}
```

### Custom Backward Pass

#### PyTorch
```python
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
output = CustomFunction.apply(input)
```

#### ToRSh-NN
```rust
// ToRSh handles automatic differentiation
// Custom backward passes are implemented through the autograd system

use torsh_autograd::{Function, Context};

pub struct CustomFunction;

impl Function for CustomFunction {
    fn forward(&self, ctx: &mut Context, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let input = &inputs[0];
        ctx.save_for_backward(&[input.clone()]);
        Ok(vec![input.clamp_min(0.0)?])
    }

    fn backward(&self, ctx: &Context, grad_outputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let saved = ctx.get_saved_tensors();
        let input = &saved[0];
        let grad_output = &grad_outputs[0];

        let mut grad_input = grad_output.clone();
        let mask = input.lt(0.0)?;
        grad_input = grad_input.masked_fill(&mask, 0.0)?;

        Ok(vec![grad_input])
    }
}
```

---

## API Differences

### Key Differences

| Feature | PyTorch | ToRSh |
|---------|---------|-------|
| **Error Handling** | Exceptions | `Result<T, Error>` |
| **Memory Management** | Reference counting | Ownership + `Rc<RefCell<>>` for parameters |
| **Dynamic Graphs** | Always dynamic | Dynamic through autograd |
| **Device Management** | `.to(device)` | `Tensor::to_device(device)` |
| **Dtypes** | `torch.float32`, etc. | `DType::F32`, etc. |
| **Parameter Access** | Direct attributes | Through `parameters()` method |
| **In-place Operations** | `x.add_(y)` | `x.add_(&y)?` |
| **Broadcasting** | Automatic | Automatic (via SciRS2) |

### Initialization

#### PyTorch
```python
# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
```

#### ToRSh-NN
```rust
use torsh_nn::init::*;

// During layer creation
impl MyLayer {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        // Initialize with Xavier uniform
        let weight_tensor = xavier_uniform(&[out_features, in_features])?;
        let weight = Parameter::new(weight_tensor, true);

        // Initialize bias with constants
        let bias_tensor = constant(&[out_features], 0.01)?;
        let bias = Parameter::new(bias_tensor, true);

        Ok(Self { weight, bias, in_features, out_features })
    }

    // Or using InitMethod
    pub fn with_init(
        in_features: usize,
        out_features: usize,
        init_method: InitMethod
    ) -> Result<Self> {
        let weight_tensor = init_method.initialize(&[out_features, in_features])?;
        let weight = Parameter::new(weight_tensor, true);

        // ...
        Ok(Self { weight, bias, in_features, out_features })
    }
}
```

### Save and Load Models

#### PyTorch
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
```

#### ToRSh-NN
```rust
// Save
model.save("model.safetensors")?;

// Load
model.load("model.safetensors")?;

// Or use serialization
use torsh_nn::serialization::{save_model, load_model};

save_model(&model, "model.json")?;
let loaded_model = load_model::<MyModel>("model.json")?;
```

---

## Migration Checklist

### Before Migration
- [ ] Review PyTorch model architecture
- [ ] Document layer types and configurations
- [ ] List custom operations or layers
- [ ] Identify training hyperparameters
- [ ] Note any custom initialization schemes

### During Migration
- [ ] Convert layer definitions to ToRSh equivalents
- [ ] Implement `Module` trait for custom modules
- [ ] Register all parameters in `parameters()` method
- [ ] Handle training/eval modes if needed
- [ ] Add `Result<>` error handling
- [ ] Test forward pass with sample data
- [ ] Verify output shapes match PyTorch

### After Migration
- [ ] Compare output numerically with PyTorch
- [ ] Test training loop
- [ ] Verify gradient computation
- [ ] Check memory usage
- [ ] Benchmark performance
- [ ] Add unit tests
- [ ] Document any differences or limitations

---

## Complete Examples

### Example 1: ResNet Block

#### PyTorch
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out
```

#### ToRSh-NN
```rust
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
}

impl BasicBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Result<Self> {
        let conv1 = Conv2d::new(in_channels, out_channels, 3, stride, 1, 1, 1, false)?;
        let bn1 = BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true)?;
        let conv2 = Conv2d::new(out_channels, out_channels, 3, 1, 1, 1, 1, false)?;
        let bn2 = BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true)?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let mut seq = Sequential::new();
            seq.add_module("conv", Box::new(
                Conv2d::new(in_channels, out_channels, 1, stride, 0, 1, 1, false)?
            ));
            seq.add_module("bn", Box::new(
                BatchNorm2d::new(out_channels, 1e-5, 0.1, true, true)?
            ));
            Some(seq)
        } else {
            None
        };

        Ok(Self { conv1, bn1, conv2, bn2, downsample })
    }
}

impl Module for BasicBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let identity = input.clone();

        let mut out = self.conv1.forward(input)?;
        out = self.bn1.forward(&out)?;
        out = relu(&out)?;

        out = self.conv2.forward(&out)?;
        out = self.bn2.forward(&out)?;

        let residual = if let Some(downsample) = &self.downsample {
            downsample.forward(&identity)?
        } else {
            identity
        };

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
        if let Some(ds) = &self.downsample {
            params.extend(ds.parameters());
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
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Error Handling
```rust
// ❌ WRONG
let output = model.forward(&input);

// ✅ CORRECT
let output = model.forward(&input)?;
// or
let output = model.forward(&input).expect("Forward pass failed");
```

### Pitfall 2: Not Registering Parameters
```rust
// ❌ WRONG
impl Module for MyModel {
    fn parameters(&self) -> ParameterCollection {
        ParameterCollection::new() // Empty!
    }
}

// ✅ CORRECT
impl Module for MyModel {
    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.linear.parameters());
        params
    }
}
```

### Pitfall 3: Incorrect Shape Handling
```rust
// ❌ WRONG - assuming specific dimensions
let batch_size = input.shape().dims()[0];
let features = input.shape().dims()[1];

// ✅ CORRECT - validate shape first
let shape = input.shape().dims();
if shape.len() != 2 {
    return Err(TorshError::InvalidArgument("Expected 2D input".to_string()));
}
let batch_size = shape[0];
let features = shape[1];
```

---

## Additional Resources

- **ToRSh Documentation**: Full API reference
- **Examples Directory**: See `examples/` for complete working examples
- **Layer Implementation Guide**: Detailed guide for creating custom layers
- **Custom Module Tutorial**: Step-by-step tutorials

For questions or issues, visit: https://github.com/cool-japan/torsh

---

## Quick Reference Table

| PyTorch | ToRSh-NN | Notes |
|---------|----------|-------|
| `nn.Module` | `Module` trait | Base for all modules |
| `nn.Linear` | `Linear` | Fully connected layer |
| `nn.Conv2d` | `Conv2d` | 2D convolution |
| `nn.BatchNorm2d` | `BatchNorm2d` | Batch normalization |
| `nn.Dropout` | `Dropout` | Dropout regularization |
| `F.relu` | `relu` | ReLU activation |
| `F.softmax` | `softmax` | Softmax function |
| `nn.CrossEntropyLoss` | `cross_entropy` | Cross entropy loss |
| `optim.Adam` | `Adam` | Adam optimizer |
| `torch.save` | `save_model` | Save model weights |
| `model.train()` | `model.train()` | Training mode |
| `model.eval()` | `model.eval()` | Evaluation mode |
| `with torch.no_grad():` | N/A | Gradients handled by `requires_grad` |
| `loss.backward()` | `loss.backward()?` | Backward pass |
| `optimizer.step()` | `optimizer.step()?` | Update weights |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` | Reset gradients |

Happy migrating! 🦀
