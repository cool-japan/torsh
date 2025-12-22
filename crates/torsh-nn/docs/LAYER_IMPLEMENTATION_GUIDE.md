# Layer Implementation Guide

This guide shows you how to implement custom neural network layers in ToRSh-NN following best practices and conventions.

## Table of Contents

1. [Basic Layer Structure](#basic-layer-structure)
2. [Module Trait Implementation](#module-trait-implementation)
3. [Parameter Management](#parameter-management)
4. [Forward Pass Implementation](#forward-pass-implementation)
5. [Initialization](#initialization)
6. [Training vs Evaluation Modes](#training-vs-evaluation-modes)
7. [Advanced Features](#advanced-features)
8. [Testing Your Layer](#testing-your-layer)
9. [Complete Examples](#complete-examples)

---

## Basic Layer Structure

Every layer in ToRSh-NN follows a consistent structure:

```rust
use torsh_core::error::Result;
use torsh_tensor::Tensor;
use torsh_nn::{Module, Parameter, ParameterCollection};

/// Custom layer description
pub struct MyCustomLayer {
    // Learnable parameters
    weight: Parameter,
    bias: Option<Parameter>,

    // Configuration (non-learnable)
    in_features: usize,
    out_features: usize,

    // Mode tracking
    training: bool,
}
```

### Key Components

1. **Parameters**: Learnable tensors (weights, biases)
2. **Configuration**: Layer hyperparameters (sizes, rates, etc.)
3. **State**: Training mode, internal caches, etc.

---

## Module Trait Implementation

All layers must implement the `Module` trait:

```rust
impl Module for MyCustomLayer {
    /// Forward pass computation
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implement your layer's computation
        todo!()
    }

    /// Return all trainable parameters
    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.add("weight", self.weight.clone());
        if let Some(bias) = &self.bias {
            params.add("bias", bias.clone());
        }
        params
    }

    /// Switch to training mode
    fn train(&mut self) {
        self.training = true;
    }

    /// Switch to evaluation mode
    fn eval(&mut self) {
        self.training = false;
    }

    /// Check if in training mode
    fn is_training(&self) -> bool {
        self.training
    }
}
```

### Required Methods

- `forward()`: Compute layer output
- `parameters()`: Return all learnable parameters
- `train()`: Enable training mode
- `eval()`: Enable evaluation mode
- `is_training()`: Query current mode

---

## Parameter Management

### Creating Parameters

```rust
use torsh_nn::Parameter;
use torsh_nn::init::{InitMethod, xavier_uniform};

impl MyCustomLayer {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Result<Self> {
        // Initialize weight parameter
        let weight_tensor = xavier_uniform(&[out_features, in_features])?;
        let weight = Parameter::new(weight_tensor, true); // requires_grad=true

        // Initialize bias (optional)
        let bias_param = if bias {
            let bias_tensor = zeros(&[out_features])?;
            Some(Parameter::new(bias_tensor, true))
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
        })
    }
}
```

### Initialization Methods

ToRSh-NN provides multiple initialization strategies:

```rust
// Xavier/Glorot initialization (good for sigmoid/tanh)
let weight = xavier_uniform(&[out_features, in_features])?;
let weight = xavier_normal(&[out_features, in_features])?;

// Kaiming/He initialization (good for ReLU)
let weight = kaiming_uniform(&[out_features, in_features], "fan_in")?;
let weight = kaiming_normal(&[out_features, in_features], "fan_in")?;

// Using InitMethod enum with builder pattern
let init = InitMethod::kaiming_normal()
    .with_fan_mode(FanMode::FanOut)
    .with_nonlinearity(Nonlinearity::ReLU);
let weight = init.initialize(&[out_features, in_features])?;

// Special initializations
let weight = orthogonal_init(&[out_features, in_features], 1.0)?;
let weight = dirac_init(&[out_channels, in_channels, kernel_size])?;
let weight = siren_init(&[out_features, in_features], 6.0, 1.0)?;
```

---

## Forward Pass Implementation

### Basic Linear Layer Example

```rust
impl Module for MyCustomLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Validate input shape
        let input_shape = input.shape().dims();
        if input_shape[input_shape.len() - 1] != self.in_features {
            return Err(TorshError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![input_shape[input_shape.len() - 1]],
            });
        }

        // Get weight tensor
        let weight_tensor = self.weight.tensor().read();

        // Matrix multiplication
        let mut output = input.matmul(&weight_tensor.t()?)?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            let bias_tensor = bias.tensor().read();
            output = output.add(&bias_tensor)?;
        }

        Ok(output)
    }
}
```

### Handling Batch Dimensions

```rust
fn forward(&self, input: &Tensor) -> Result<Tensor> {
    let input_shape = input.shape().dims();

    // Handle both 2D (batch_size, features) and 3D (batch_size, seq_len, features)
    let output = if input_shape.len() == 2 {
        // Standard 2D input
        self.forward_2d(input)?
    } else if input_shape.len() == 3 {
        // Sequential data: reshape to 2D, process, reshape back
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let features = input_shape[2];

        let reshaped = input.view(&[(batch_size * seq_len) as i32, features as i32])?;
        let processed = self.forward_2d(&reshaped)?;
        processed.view(&[batch_size as i32, seq_len as i32, self.out_features as i32])?
    } else {
        return Err(TorshError::InvalidArgument(
            format!("Expected 2D or 3D input, got {}D", input_shape.len())
        ));
    };

    Ok(output)
}
```

---

## Initialization

### Reset Parameters Method

Implement parameter initialization with proper defaults:

```rust
impl MyCustomLayer {
    /// Reset parameters using default initialization
    pub fn reset_parameters(&mut self) -> Result<()> {
        // Weight initialization (Kaiming for ReLU networks)
        let weight_tensor = kaiming_uniform(
            &[self.out_features, self.in_features],
            "fan_in"
        )?;
        self.weight = Parameter::new(weight_tensor, true);

        // Bias initialization (zeros or small uniform)
        if let Some(bias) = &mut self.bias {
            let bias_tensor = zeros(&[self.out_features])?;
            *bias = Parameter::new(bias_tensor, true);
        }

        Ok(())
    }

    /// Custom initialization with specific method
    pub fn init_with(&mut self, method: InitMethod) -> Result<()> {
        let weight_tensor = method.initialize(&[self.out_features, self.in_features])?;
        self.weight = Parameter::new(weight_tensor, true);
        Ok(())
    }
}
```

---

## Training vs Evaluation Modes

Some layers behave differently during training and evaluation:

```rust
impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.is_training() {
            // Training: apply dropout
            dropout(input, self.p, true)
        } else {
            // Evaluation: no dropout
            Ok(input.clone())
        }
    }
}

impl Module for BatchNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.is_training() {
            // Training: compute batch statistics and update running stats
            self.forward_training(input)
        } else {
            // Evaluation: use running statistics
            self.forward_eval(input)
        }
    }
}
```

---

## Advanced Features

### 1. Layer with Multiple Inputs

```rust
pub struct AttentionLayer {
    // ... parameters
}

impl AttentionLayer {
    /// Forward pass with query, key, and value
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Scaled dot-product attention
        let scores = query.matmul(&key.transpose(-2, -1)?)?;
        let scaled = scores.div_scalar(self.scale)?;

        // Apply mask if provided
        let masked = if let Some(mask) = mask {
            scaled.masked_fill(mask, f32::NEG_INFINITY)?
        } else {
            scaled
        };

        // Apply softmax and compute output
        let attention_weights = softmax(&masked, Some(-1))?;
        let output = attention_weights.matmul(value)?;

        Ok(output)
    }
}
```

### 2. Residual Connections

```rust
impl Module for ResidualBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Main path
        let out = self.conv1.forward(input)?;
        let out = relu(&out)?;
        let out = self.conv2.forward(&out)?;

        // Residual connection
        let residual = if self.downsample.is_some() {
            self.downsample.as_ref().unwrap().forward(input)?
        } else {
            input.clone()
        };

        // Add and activate
        let out = out.add(&residual)?;
        let out = relu(&out)?;

        Ok(out)
    }
}
```

### 3. Stateful Layers (RNN)

```rust
pub struct LSTMCell {
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias_ih: Option<Parameter>,
    bias_hh: Option<Parameter>,
    hidden_size: usize,
}

impl LSTMCell {
    /// Forward pass with hidden state
    pub fn forward_with_state(
        &self,
        input: &Tensor,
        hidden: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let (h_prev, c_prev) = if let Some((h, c)) = hidden {
            (h.clone(), c.clone())
        } else {
            // Initialize hidden state
            let batch_size = input.shape().dims()[0];
            let h = zeros(&[batch_size, self.hidden_size])?;
            let c = zeros(&[batch_size, self.hidden_size])?;
            (h, c)
        };

        // LSTM computation
        // ... (gates computation)

        Ok((h_next, c_next))
    }
}
```

### 4. Configurable Activation

```rust
pub struct ConvBlock {
    conv: Conv2d,
    activation: ActivationType,
}

#[derive(Clone)]
pub enum ActivationType {
    ReLU,
    LeakyReLU(f32),
    GELU,
    Swish,
}

impl ActivationType {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::ReLU => relu(x),
            Self::LeakyReLU(slope) => leaky_relu(x, *slope),
            Self::GELU => gelu(x),
            Self::Swish => swish(x),
        }
    }
}

impl Module for ConvBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(input)?;
        self.activation.apply(&out)
    }
}
```

---

## Testing Your Layer

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_layer_creation() {
        let layer = MyCustomLayer::new(10, 5, true);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_forward_pass() {
        let layer = MyCustomLayer::new(10, 5, true).unwrap();
        let input = randn::<f32>(&[2, 10]).unwrap();

        let output = layer.forward(&input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_parameter_count() {
        let layer = MyCustomLayer::new(10, 5, true).unwrap();
        let params = layer.parameters();

        // Should have weight (10*5=50) and bias (5) = 55 parameters
        let total: usize = params.iter().map(|(_, p)| {
            p.tensor().read().numel()
        }).sum();

        assert_eq!(total, 55);
    }

    #[test]
    fn test_training_mode() {
        let mut layer = MyCustomLayer::new(10, 5, true).unwrap();

        assert!(layer.is_training());

        layer.eval();
        assert!(!layer.is_training());

        layer.train();
        assert!(layer.is_training());
    }

    #[test]
    fn test_gradient_flow() {
        let layer = MyCustomLayer::new(10, 5, true).unwrap();
        let input = randn::<f32>(&[2, 10]).unwrap();

        // Ensure parameters require gradients
        let params = layer.parameters();
        for (_, param) in params.iter() {
            assert!(param.requires_grad());
        }
    }
}
```

### Integration Tests

```rust
#[test]
fn test_with_optimizer() {
    use torsh_optim::{SGD, Optimizer};

    let mut layer = MyCustomLayer::new(10, 5, true).unwrap();
    let mut optimizer = SGD::new(layer.parameters(), 0.01);

    // Training step
    let input = randn::<f32>(&[2, 10]).unwrap();
    let target = randn::<f32>(&[2, 5]).unwrap();

    let output = layer.forward(&input).unwrap();
    let loss = mse_loss(&output, &target, "mean").unwrap();

    // Backward and step
    // loss.backward();
    // optimizer.step();

    // Verify loss decreased
    // ...
}
```

---

## Complete Examples

### Example 1: Simple Activation Layer

```rust
use torsh_core::error::Result;
use torsh_tensor::Tensor;
use torsh_nn::{Module, ParameterCollection};

/// PReLU (Parametric ReLU) activation
pub struct PReLU {
    alpha: Parameter,
    num_parameters: usize,
}

impl PReLU {
    /// Create new PReLU layer
    ///
    /// # Arguments
    /// * `num_parameters` - Number of alpha parameters (1 for shared, or num_channels for per-channel)
    /// * `init` - Initial value for alpha (default: 0.25)
    pub fn new(num_parameters: usize, init: f32) -> Result<Self> {
        let alpha_tensor = constant(&[num_parameters], init)?;
        let alpha = Parameter::new(alpha_tensor, true);

        Ok(Self {
            alpha,
            num_parameters,
        })
    }
}

impl Module for PReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let alpha = self.alpha.tensor().read();

        // PReLU: f(x) = max(0, x) + alpha * min(0, x)
        let positive = input.clamp_min(0.0)?;
        let negative = input.clamp_max(0.0)?;
        let scaled_negative = negative.mul(&alpha)?;

        positive.add(&scaled_negative)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.add("alpha", self.alpha.clone());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { true }
}
```

### Example 2: Normalization Layer

```rust
/// Layer Normalization
pub struct LayerNorm {
    normalized_shape: Vec<usize>,
    weight: Parameter,
    bias: Parameter,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Result<Self> {
        let num_features: usize = normalized_shape.iter().product();

        let weight = Parameter::new(ones(&[num_features])?, true);
        let bias = Parameter::new(zeros(&[num_features])?, true);

        Ok(Self {
            normalized_shape,
            weight,
            bias,
            eps,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.weight.tensor().read();
        let bias = self.bias.tensor().read();

        // Compute mean and variance
        let mean = input.mean_dim(&[-1], true)?;
        let var = input.var_dim(&[-1], true, true)?;

        // Normalize
        let normalized = input
            .sub(&mean)?
            .div(&var.add_scalar(self.eps)?.sqrt()?)?;

        // Scale and shift
        normalized
            .mul(&weight)?
            .add(&bias)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.add("weight", self.weight.clone());
        params.add("bias", self.bias.clone());
        params
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn is_training(&self) -> bool { true }
}
```

### Example 3: Attention Mechanism

```rust
/// Multi-Head Self-Attention
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    dropout: Dropout,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, dropout: f32) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(TorshError::InvalidArgument(
                "embed_dim must be divisible by num_heads".to_string()
            ));
        }

        let head_dim = embed_dim / num_heads;

        Ok(Self {
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim, true),
            k_proj: Linear::new(embed_dim, embed_dim, true),
            v_proj: Linear::new(embed_dim, embed_dim, true),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            dropout: Dropout::new(dropout),
        })
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().dims();
        let batch_size = shape[0];
        let seq_len = shape[1];

        // (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        let reshaped = x.view(&[
            batch_size as i32,
            seq_len as i32,
            self.num_heads as i32,
            self.head_dim as i32,
        ])?;

        // Transpose to (batch, num_heads, seq_len, head_dim)
        reshaped.permute(&[0, 2, 1, 3])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape().dims()[0];
        let seq_len = input.shape().dims()[1];

        // Project Q, K, V
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Split heads
        let q = self.split_heads(&q)?;
        let k = self.split_heads(&k)?;
        let v = self.split_heads(&v)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)?)?.div_scalar(scale)?;
        let attn_weights = softmax(&scores, Some(-1))?;
        let attn_weights = self.dropout.forward(&attn_weights)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Merge heads
        let merged = attn_output
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .view(&[
                batch_size as i32,
                seq_len as i32,
                (self.num_heads * self.head_dim) as i32,
            ])?;

        // Final projection
        self.out_proj.forward(&merged)
    }

    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
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

---

## Best Practices Checklist

- [ ] **Proper parameter initialization**: Use appropriate initialization methods
- [ ] **Input validation**: Check input shapes and types
- [ ] **Memory efficiency**: Avoid unnecessary clones
- [ ] **Error handling**: Provide clear error messages
- [ ] **Documentation**: Add doc comments with examples
- [ ] **Unit tests**: Test creation, forward pass, parameters
- [ ] **Training/eval modes**: Implement if behavior differs
- [ ] **Parameter registration**: Include all learnable parameters
- [ ] **Shape consistency**: Verify output shapes match expectations
- [ ] **Numerical stability**: Add epsilon where needed (division, log, sqrt)

---

## Common Pitfalls

### 1. Forgetting to Register Parameters

```rust
// ❌ WRONG: Parameter not registered
impl Module for MyLayer {
    fn parameters(&self) -> ParameterCollection {
        ParameterCollection::new() // Missing parameters!
    }
}

// ✅ CORRECT: All parameters registered
impl Module for MyLayer {
    fn parameters(&self) -> ParameterCollection {
        let mut params = ParameterCollection::new();
        params.add("weight", self.weight.clone());
        params.add("bias", self.bias.clone());
        params
    }
}
```

### 2. Ignoring Training Mode

```rust
// ❌ WRONG: Always applies dropout
impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        dropout(input, 0.5, true) // Always training=true!
    }
}

// ✅ CORRECT: Respects training mode
impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        dropout(input, 0.5, self.is_training())
    }
}
```

### 3. Shape Mismatches

```rust
// ❌ WRONG: No shape validation
impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.linear.forward(input) // May fail on wrong shapes!
    }
}

// ✅ CORRECT: Validate shapes
impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape().dims();
        if input_shape[input_shape.len() - 1] != self.in_features {
            return Err(TorshError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![input_shape[input_shape.len() - 1]],
            });
        }
        self.linear.forward(input)
    }
}
```

---

## Additional Resources

- **API Documentation**: Full API docs at `cargo doc --open`
- **Examples**: See `examples/` directory for complete examples
- **Tests**: See `tests/` directory for integration tests
- **PyTorch Migration**: See `PYTORCH_MIGRATION_GUIDE.md` for PyTorch equivalents

---

## Next Steps

1. Start with a simple layer (activation, normalization)
2. Add unit tests to verify correctness
3. Test integration with optimizers
4. Benchmark performance if needed
5. Contribute to torsh-nn!

For questions or contributions, visit: https://github.com/cool-japan/torsh
