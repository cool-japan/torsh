# ToRSh-NN Best Practices

A comprehensive guide to writing efficient, maintainable, and idiomatic ToRSh-NN code.

## Table of Contents

1. [Code Organization](#code-organization)
2. [Memory Management](#memory-management)
3. [Error Handling](#error-handling)
4. [Performance Optimization](#performance-optimization)
5. [Testing Strategies](#testing-strategies)
6. [API Design](#api-design)
7. [Documentation](#documentation)
8. [Common Anti-Patterns](#common-anti-patterns)

---

## Code Organization

### Module Structure

**✅ DO**: Organize code into logical modules

```rust
pub mod layers {
    pub mod conv;
    pub mod linear;
    pub mod attention;
}

pub mod models {
    pub mod resnet;
    pub mod transformer;
}

pub mod utils {
    pub mod initialization;
    pub mod metrics;
}
```

**❌ DON'T**: Put everything in one file

```rust
// src/lib.rs with 5000+ lines
pub struct Linear { ... }
pub struct Conv2d { ... }
pub struct ResNet { ... }
// ... hundreds more
```

### Layer Organization

**✅ DO**: Separate concerns clearly

```rust
pub struct MyLayer {
    // Learnable parameters
    weight: Parameter,
    bias: Option<Parameter>,

    // Configuration (immutable)
    in_features: usize,
    out_features: usize,

    // State (mutable)
    training: bool,
}
```

**❌ DON'T**: Mix concerns

```rust
pub struct MyLayer {
    weight: Parameter,
    config: LayerConfig,  // Opaque configuration
    cache: Vec<Tensor>,   // Unclear purpose
}
```

---

## Memory Management

###  1: Avoid Unnecessary Clones

**✅ DO**: Use references when possible

```rust
impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.weight.tensor().read();
        input.matmul(&weight.t()?)  // No clone needed
    }
}
```

**❌ DON'T**: Clone unnecessarily

```rust
impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_clone = input.clone();  // Unnecessary!
        let weight_clone = self.weight.tensor().read().clone();  // Unnecessary!
        input_clone.matmul(&weight_clone.t()?)
    }
}
```

### 2: Use Views for Reshaping

**✅ DO**: Use views to avoid data copying

```rust
fn flatten(&self, x: &Tensor) -> Result<Tensor> {
    let batch_size = x.shape().dims()[0];
    x.view(&[batch_size as i32, -1])  // No copy
}
```

**❌ DON'T**: Reshape by copying data

```rust
fn flatten(&self, x: &Tensor) -> Result<Tensor> {
    let data = x.to_vec()?;  // Copies all data!
    let batch_size = x.shape().dims()[0];
    let features = data.len() / batch_size;
    Tensor::from_vec(data, &[batch_size, features])
}
```

### 3: Reuse Buffers

**✅ DO**: Reuse pre-allocated tensors

```rust
pub struct CachedLayer {
    buffer: RefCell<Option<Tensor>>,
    output_shape: Vec<usize>,
}

impl CachedLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut buffer = self.buffer.borrow_mut();
        if buffer.is_none() {
            *buffer = Some(zeros(&self.output_shape)?);
        }
        // Reuse buffer
        Ok(buffer.as_ref().unwrap().clone())
    }
}
```

### 4: Parameter Sharing

**✅ DO**: Share parameters when appropriate

```rust
pub struct SiameseNetwork {
    encoder: Linear,  // Shared encoder
}

impl SiameseNetwork {
    fn forward_pair(&self, x1: &Tensor, x2: &Tensor) -> Result<(Tensor, Tensor)> {
        let h1 = self.encoder.forward(x1)?;  // Same encoder
        let h2 = self.encoder.forward(x2)?;  // Same encoder
        Ok((h1, h2))
    }
}
```

---

## Error Handling

### 1: Provide Context in Errors

**✅ DO**: Add helpful error messages

```rust
fn forward(&self, input: &Tensor) -> Result<Tensor> {
    let shape = input.shape().dims();

    if shape[shape.len() - 1] != self.in_features {
        return Err(TorshError::ShapeMismatch {
            expected: vec![self.in_features],
            got: vec![shape[shape.len() - 1]],
        }.with_context(format!(
            "Layer '{}' expected {} input features, got {}",
            self.name(), self.in_features, shape[shape.len() - 1]
        )));
    }

    // ... rest of implementation
}
```

**❌ DON'T**: Use generic errors

```rust
fn forward(&self, input: &Tensor) -> Result<Tensor> {
    let shape = input.shape().dims();

    if shape[shape.len() - 1] != self.in_features {
        return Err(TorshError::RuntimeError("Bad shape".to_string()));
    }

    // ... rest
}
```

### 2: Validate Inputs Early

**✅ DO**: Check preconditions at the start

```rust
pub fn new(in_features: usize, out_features: usize, dropout: f32) -> Result<Self> {
    if in_features == 0 {
        return Err(TorshError::InvalidArgument(
            "in_features must be positive".to_string()
        ));
    }

    if !(0.0..=1.0).contains(&dropout) {
        return Err(TorshError::InvalidArgument(
            format!("dropout must be in [0, 1], got {}", dropout)
        ));
    }

    // Continue with construction
    Ok(Self {
        // ...
    })
}
```

### 3: Use Type System for Safety

**✅ DO**: Use types to prevent errors

```rust
pub enum ActivationType {
    ReLU,
    GELU,
    Tanh,
}

pub struct LayerConfig {
    activation: ActivationType,  // Can't be invalid!
    dropout: Option<f32>,         // Explicit optionality
}
```

**❌ DON'T**: Use strings for enums

```rust
pub struct LayerConfig {
    activation: String,  // Could be anything!
    dropout: f32,        // -1 for "no dropout"?
}
```

---

## Performance Optimization

### 1: Batch Operations

**✅ DO**: Process in batches

```rust
// Process entire batch at once
let batch_output = model.forward(&batch_input)?;  // [32, 128]
```

**❌ DON'T**: Process one at a time

```rust
// Process samples individually
let mut outputs = Vec::new();
for sample in batch_input.iter() {
    outputs.push(model.forward(&sample)?);  // Slow!
}
```

### 2: Use SciRS2 Optimizations

**✅ DO**: Leverage SciRS2 features

```rust
// SciRS2 handles SIMD and parallelization automatically
use scirs2_core::ndarray::*;
use scirs2_core::random::*;

// Efficient array operations
let result = array1.dot(&array2);  // Optimized BLAS
```

### 3: Avoid Redundant Computations

**✅ DO**: Cache expensive computations

```rust
pub struct AttentionLayer {
    cached_keys: RefCell<Option<Tensor>>,
    cached_values: RefCell<Option<Tensor>>,
}

impl AttentionLayer {
    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        // Cache keys and values if they don't change
        let k = if let Some(cached) = self.cached_keys.borrow().as_ref() {
            cached.clone()
        } else {
            let k = self.project_keys(key)?;
            *self.cached_keys.borrow_mut() = Some(k.clone());
            k
        };

        // ... use cached key
        Ok(output)
    }
}
```

### 4: Profile Before Optimizing

**✅ DO**: Measure performance first

```rust
use std::time::Instant;

let start = Instant::now();
let output = model.forward(&input)?;
println!("Forward pass took: {:?}", start.elapsed());

// Use proper profiling tools
#[cfg(feature = "profiling")]
use torsh_nn::profiling::Profiler;

#[cfg(feature = "profiling")]
{
    let mut profiler = Profiler::new();
    profiler.start("forward_pass");
    let output = model.forward(&input)?;
    profiler.stop("forward_pass");
    profiler.report();
}
```

---

## Testing Strategies

### 1: Test Layer Creation

**✅ DO**: Test basic construction

```rust
#[test]
fn test_layer_creation() {
    let layer = MyLayer::new(128, 64, true);
    assert!(layer.is_ok());

    let layer = layer.unwrap();
    assert_eq!(layer.in_features(), 128);
    assert_eq!(layer.out_features(), 64);
}

#[test]
fn test_invalid_dimensions() {
    let layer = MyLayer::new(0, 64, true);
    assert!(layer.is_err());  // Should reject invalid input
}
```

### 2: Test Forward Pass Shapes

**✅ DO**: Verify output dimensions

```rust
#[test]
fn test_forward_shapes() {
    let layer = MyLayer::new(128, 64, true).unwrap();

    // Test 2D input
    let input_2d = randn::<f32>(&[32, 128]).unwrap();
    let output = layer.forward(&input_2d).unwrap();
    assert_eq!(output.shape().dims(), &[32, 64]);

    // Test 3D input
    let input_3d = randn::<f32>(&[16, 10, 128]).unwrap();
    let output = layer.forward(&input_3d).unwrap();
    assert_eq!(output.shape().dims(), &[16, 10, 64]);
}
```

### 3: Test Parameter Count

**✅ DO**: Verify parameters are registered

```rust
#[test]
fn test_parameter_count() {
    let layer = MyLayer::new(128, 64, true).unwrap();
    let params = layer.parameters();

    let total: usize = params.iter()
        .map(|(_, p)| p.tensor().read().numel())
        .sum();

    // weight: 128 * 64 = 8192
    // bias: 64
    // total: 8256
    assert_eq!(total, 8256);
}
```

### 4: Test Training/Eval Modes

**✅ DO**: Verify mode switching

```rust
#[test]
fn test_training_mode() {
    let mut layer = MyLayer::new(128, 64, true).unwrap();

    assert!(layer.is_training());

    layer.eval();
    assert!(!layer.is_training());

    layer.train();
    assert!(layer.is_training());
}
```

### 5: Test Numerical Correctness

**✅ DO**: Compare with reference implementation

```rust
#[test]
fn test_numerical_correctness() {
    let layer = Linear::new(10, 5, false).unwrap();

    // Set known weights
    let weight_data = vec![1.0; 50];
    let weight = Tensor::from_vec(weight_data, &[5, 10]).unwrap();
    // ... set layer weight

    let input = ones(&[2, 10]).unwrap();
    let output = layer.forward(&input).unwrap();

    // Expected: sum of weights for each output neuron
    let expected = vec![10.0; 5];
    let output_data = output.to_vec().unwrap();

    for (got, expected) in output_data.iter().zip(expected.iter()) {
        assert!((got - expected).abs() < 1e-5);
    }
}
```

### 6: Test Gradient Flow

**✅ DO**: Verify gradients are computed

```rust
#[test]
fn test_gradient_flow() {
    let layer = Linear::new(10, 5, true).unwrap();
    let input = randn::<f32>(&[2, 10]).unwrap();

    let output = layer.forward(&input).unwrap();
    let loss = output.sum().unwrap();

    // Verify parameters require gradients
    for (_, param) in layer.parameters().iter() {
        assert!(param.requires_grad());
    }
}
```

---

## API Design

### 1: Follow Rust Conventions

**✅ DO**: Use idiomatic Rust naming

```rust
// Use snake_case for functions and variables
pub fn linear_layer(input: &Tensor) -> Result<Tensor> { ... }

// Use CamelCase for types
pub struct LinearLayer { ... }

// Use SCREAMING_SNAKE_CASE for constants
const DEFAULT_DROPOUT_RATE: f32 = 0.5;
```

### 2: Builder Pattern for Complex Construction

**✅ DO**: Use builders for many options

```rust
pub struct LayerBuilder {
    in_features: usize,
    out_features: usize,
    bias: bool,
    activation: Option<ActivationType>,
    dropout: Option<f32>,
    init_method: InitMethod,
}

impl LayerBuilder {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
            activation: None,
            dropout: None,
            init_method: InitMethod::kaiming_normal(),
        }
    }

    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn activation(mut self, activation: ActivationType) -> Self {
        self.activation = Some(activation);
        self
    }

    pub fn build(self) -> Result<MyLayer> {
        // Construct layer with all options
        MyLayer::from_builder(self)
    }
}

// Usage
let layer = LayerBuilder::new(128, 64)
    .bias(false)
    .activation(ActivationType::GELU)
    .dropout(0.1)
    .build()?;
```

### 3: Provide Convenience Constructors

**✅ DO**: Offer multiple construction methods

```rust
impl Linear {
    /// Create with defaults
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self::with_init(in_features, out_features, bias, InitMethod::kaiming_uniform())
    }

    /// Create with custom initialization
    pub fn with_init(
        in_features: usize,
        out_features: usize,
        bias: bool,
        init: InitMethod,
    ) -> Self {
        // ... implementation
    }

    /// Create without bias (common pattern)
    pub fn no_bias(in_features: usize, out_features: usize) -> Self {
        Self::new(in_features, out_features, false)
    }
}
```

### 4: Use Type States for Safety

**✅ DO**: Use phantom types to enforce states

```rust
pub struct Untrained;
pub struct Trained;

pub struct Model<State> {
    layers: Vec<Box<dyn Module>>,
    _state: PhantomData<State>,
}

impl Model<Untrained> {
    pub fn new() -> Self {
        // ... create model
    }

    pub fn train(self, data: &Dataset) -> Model<Trained> {
        // ... training logic
        Model {
            layers: self.layers,
            _state: PhantomData,
        }
    }
}

impl Model<Trained> {
    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        // Only trained models can predict
        self.forward(input)
    }
}
```

---

## Documentation

### 1: Document Public APIs

**✅ DO**: Provide comprehensive documentation

```rust
/// Multi-layer perceptron for classification
///
/// This module implements a fully-connected neural network with configurable
/// hidden layers, activation functions, and dropout regularization.
///
/// # Arguments
///
/// * `input_dim` - Dimension of input features
/// * `hidden_dims` - Dimensions of hidden layers
/// * `output_dim` - Number of output classes
/// * `dropout` - Dropout probability (0.0 to 1.0)
///
/// # Examples
///
/// ```
/// use torsh_nn::MLP;
///
/// let model = MLP::new(784, vec![256, 128], 10, 0.5)?;
/// let input = randn::<f32>(&[32, 784])?;
/// let output = model.forward(&input)?;
/// assert_eq!(output.shape().dims(), &[32, 10]);
/// ```
///
/// # Panics
///
/// Panics if `dropout` is not in the range [0.0, 1.0].
pub struct MLP {
    // ...
}
```

### 2: Document Implementation Details

**✅ DO**: Explain non-obvious behavior

```rust
impl BatchNorm2d {
    /// Forward pass through batch normalization
    ///
    /// During training, normalizes using batch statistics and updates
    /// running statistics. During evaluation, uses running statistics.
    ///
    /// # Implementation Notes
    ///
    /// - Running statistics use exponential moving average
    /// - Momentum parameter controls update rate (default: 0.1)
    /// - Epsilon prevents division by zero (default: 1e-5)
    ///
    /// # Shape
    ///
    /// - Input: `(batch_size, num_features, height, width)`
    /// - Output: `(batch_size, num_features, height, width)`
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // ...
    }
}
```

### 3: Provide Examples

**✅ DO**: Include working examples

```rust
/// Apply attention mechanism to input
///
/// # Example
///
/// ```
/// use torsh_nn::Attention;
///
/// let attn = Attention::new(512, 8, 0.1)?;
/// let query = randn::<f32>(&[32, 10, 512])?;  // (batch, seq, dim)
/// let key = randn::<f32>(&[32, 20, 512])?;
/// let value = randn::<f32>(&[32, 20, 512])?;
///
/// let output = attn.forward_qkv(&query, &key, &value, None)?;
/// assert_eq!(output.shape().dims(), &[32, 10, 512]);
/// ```
pub fn forward_qkv(/* ... */) -> Result<Tensor> {
    // ...
}
```

---

## Common Anti-Patterns

### Anti-Pattern 1: Ignoring Errors

**❌ DON'T**: Use unwrap() liberally

```rust
let output = model.forward(&input).unwrap();  // Could panic!
let value = tensor.item().unwrap();  // Could panic!
```

**✅ DO**: Handle errors properly

```rust
let output = model.forward(&input)?;  // Propagate error
let value = tensor.item().map_err(|e| {
    TorshError::RuntimeError(format!("Failed to get item: {}", e))
})?;
```

### Anti-Pattern 2: Mutation Through Shared References

**❌ DON'T**: Mutate through shared references unsafely

```rust
impl Module for BadLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // ❌ Trying to mutate through &self
        self.cache = Some(input.clone());  // Won't compile!
        // ...
    }
}
```

**✅ DO**: Use interior mutability correctly

```rust
pub struct GoodLayer {
    cache: RefCell<Option<Tensor>>,
}

impl Module for GoodLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        *self.cache.borrow_mut() = Some(input.clone());
        // ...
    }
}
```

### Anti-Pattern 3: Overly Generic APIs

**❌ DON'T**: Make everything generic without reason

```rust
pub struct OverGeneric<T, U, V, W> {
    layer1: T,
    layer2: U,
    activation: V,
    dropout: W,
}
// Hard to use, hard to understand
```

**✅ DO**: Use generics judiciously

```rust
pub struct Practical {
    layer1: Linear,
    layer2: Linear,
    activation: ActivationType,  // Enum is fine
    dropout: Dropout,
}
// Easy to use, clear intent
```

### Anti-Pattern 4: Premature Optimization

**❌ DON'T**: Optimize without measuring

```rust
// Complex caching that may not help
pub struct OverOptimized {
    cache1: HashMap<String, Tensor>,
    cache2: Vec<Option<Tensor>>,
    cache3: RefCell<BTreeMap<usize, Tensor>>,
    // ... more caches
}
```

**✅ DO**: Start simple, profile, then optimize

```rust
// Simple and clear
pub struct Simple {
    layer: Linear,
}

// Add caching only if profiling shows benefit
```

---

## Checklist

Before submitting code, ensure:

- [ ] All public APIs have documentation
- [ ] Examples compile and run
- [ ] Tests cover main functionality
- [ ] Error messages are helpful
- [ ] No unnecessary clones or allocations
- [ ] Parameters are registered correctly
- [ ] Training/eval modes handled properly
- [ ] Input validation is thorough
- [ ] Code follows Rust naming conventions
- [ ] Complex logic has explanatory comments

---

## Additional Resources

- **Rust API Guidelines**: https://rust-lang.github.io/api-guidelines/
- **Layer Implementation Guide**: See `LAYER_IMPLEMENTATION_GUIDE.md`
- **Custom Module Tutorial**: See `CUSTOM_MODULE_TUTORIAL.md`
- **Performance Tuning Guide**: See `PERFORMANCE_TUNING.md`

For questions or contributions, visit: https://github.com/cool-japan/torsh
