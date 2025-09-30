# ToRSh API Reference

## Table of Contents

1. [Core Types](#core-types)
2. [Tensor API](#tensor-api)
3. [Autograd API](#autograd-api)
4. [Neural Network API](#neural-network-api)
5. [Optimization API](#optimization-api)
6. [Data Loading API](#data-loading-api)
7. [Functional API](#functional-api)
8. [Advanced APIs](#advanced-apis)
9. [Utility APIs](#utility-apis)

## Core Types

### Tensor

The fundamental data structure in ToRSh.

```rust
pub struct Tensor {
    // Internal fields...
}
```

#### Construction

```rust
impl Tensor {
    /// Create a tensor from raw data
    pub fn from_data<T: TensorElement>(data: &[T], shape: &[usize]) -> Self;
    
    /// Create a tensor with specified shape and default values
    pub fn zeros(shape: &[usize]) -> Self;
    pub fn ones(shape: &[usize]) -> Self;
    pub fn full(shape: &[usize], value: f64) -> Self;
    
    /// Create random tensors
    pub fn rand(shape: &[usize]) -> Self;
    pub fn randn(shape: &[usize]) -> Self;
    pub fn randint(low: i64, high: i64, shape: &[usize]) -> Self;
    
    /// Create special tensors
    pub fn eye(n: usize) -> Self;
    pub fn arange(start: f64, end: f64, step: f64) -> Self;
    pub fn linspace(start: f64, end: f64, steps: usize) -> Self;
}
```

**Examples:**

```rust
use torsh::prelude::*;

// From data
let tensor = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

// Zeros and ones
let zeros = Tensor::zeros(&[3, 4]);
let ones = Tensor::ones(&[3, 4]);

// Random tensors
let rand = Tensor::rand(&[2, 3]);
let randn = Tensor::randn(&[2, 3]);

// Special tensors
let eye = Tensor::eye(3);
let range = Tensor::arange(0.0, 10.0, 1.0);
```

#### Properties

```rust
impl Tensor {
    /// Get the shape of the tensor
    pub fn shape(&self) -> &Shape;
    
    /// Get the data type of the tensor
    pub fn dtype(&self) -> DType;
    
    /// Get the device of the tensor
    pub fn device(&self) -> &Device;
    
    /// Get the number of elements
    pub fn numel(&self) -> usize;
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize;
    
    /// Check if tensor requires gradients
    pub fn requires_grad(&self) -> bool;
    
    /// Check if tensor has gradients
    pub fn has_grad(&self) -> bool;
    
    /// Get tensor gradients
    pub fn grad(&self) -> Option<&Tensor>;
}
```

**Examples:**

```rust
let tensor = randn(&[2, 3, 4]);

println!("Shape: {:?}", tensor.shape());           // Shape: [2, 3, 4]
println!("DType: {:?}", tensor.dtype());           // DType: F32
println!("Device: {:?}", tensor.device());         // Device: CPU
println!("Elements: {}", tensor.numel());          // Elements: 24
println!("Dimensions: {}", tensor.ndim());         // Dimensions: 3
```

#### Operations

```rust
impl Tensor {
    // Arithmetic operations
    pub fn add(&self, other: &Tensor) -> Result<Tensor>;
    pub fn sub(&self, other: &Tensor) -> Result<Tensor>;
    pub fn mul(&self, other: &Tensor) -> Result<Tensor>;
    pub fn div(&self, other: &Tensor) -> Result<Tensor>;
    pub fn pow(&self, exponent: f64) -> Result<Tensor>;
    
    // In-place operations
    pub fn add_(&mut self, other: &Tensor) -> Result<()>;
    pub fn sub_(&mut self, other: &Tensor) -> Result<()>;
    pub fn mul_(&mut self, other: &Tensor) -> Result<()>;
    pub fn div_(&mut self, other: &Tensor) -> Result<()>;
    
    // Matrix operations
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor>;
    pub fn dot(&self, other: &Tensor) -> Result<Tensor>;
    pub fn mm(&self, other: &Tensor) -> Result<Tensor>;
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor>;
    
    // Shape operations
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor>;
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor>;
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor>;
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Tensor>;
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor>;
    
    // Indexing and slicing
    pub fn index(&self, indices: &[TensorIndex]) -> Result<Tensor>;
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Tensor>;
    pub fn gather(&self, dim: usize, indices: &Tensor) -> Result<Tensor>;
    pub fn scatter(&self, dim: usize, indices: &Tensor, src: &Tensor) -> Result<Tensor>;
    
    // Reduction operations
    pub fn sum(&self) -> Result<Tensor>;
    pub fn mean(&self) -> Result<Tensor>;
    pub fn std(&self) -> Result<Tensor>;
    pub fn var(&self) -> Result<Tensor>;
    pub fn min(&self) -> Result<Tensor>;
    pub fn max(&self) -> Result<Tensor>;
    pub fn argmin(&self) -> Result<Tensor>;
    pub fn argmax(&self) -> Result<Tensor>;
    
    // Reduction with dimension
    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor>;
    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor>;
    pub fn std_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor>;
    pub fn var_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor>;
    pub fn min_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor>;
    pub fn max_dim(&self, dim: usize, keepdim: bool) -> Result<Tensor>;
    
    // Comparison operations
    pub fn eq(&self, other: &Tensor) -> Result<Tensor>;
    pub fn ne(&self, other: &Tensor) -> Result<Tensor>;
    pub fn lt(&self, other: &Tensor) -> Result<Tensor>;
    pub fn le(&self, other: &Tensor) -> Result<Tensor>;
    pub fn gt(&self, other: &Tensor) -> Result<Tensor>;
    pub fn ge(&self, other: &Tensor) -> Result<Tensor>;
    
    // Logical operations
    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor>;
    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor>;
    pub fn logical_not(&self) -> Result<Tensor>;
    
    // Device operations
    pub fn to_device(&self, device: &Device) -> Result<Tensor>;
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor>;
    
    // Gradient operations
    pub fn requires_grad_(self, requires_grad: bool) -> Self;
    pub fn detach(&self) -> Tensor;
    pub fn backward(&self) -> Result<()>;
    
    // Utility operations
    pub fn clone(&self) -> Tensor;
    pub fn item<T: TensorElement>(&self) -> T;
    pub fn to_vec<T: TensorElement>(&self) -> Vec<T>;
}
```

**Examples:**

```rust
use torsh::prelude::*;

let a = randn(&[2, 3]);
let b = randn(&[2, 3]);

// Arithmetic operations
let c = a.add(&b)?;
let d = a.mul(&b)?;
let e = a.pow(2.0)?;

// Matrix operations
let f = a.matmul(&b.transpose(0, 1)?)?;

// Shape operations
let g = a.reshape(&[3, 2])?;
let h = a.unsqueeze(0)?;

// Reductions
let sum = a.sum()?;
let mean = a.mean_dim(1, false)?;
let max_val = a.max()?;

// Comparisons
let mask = a.gt(&tensor![0.5])?;
let selected = a.index(&[mask])?;
```

### Shape

Represents tensor dimensions.

```rust
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self;
    
    /// Get dimensions
    pub fn dims(&self) -> &[usize];
    
    /// Get number of dimensions
    pub fn ndim(&self) -> usize;
    
    /// Get total number of elements
    pub fn numel(&self) -> usize;
    
    /// Check if shapes are compatible for broadcasting
    pub fn can_broadcast_to(&self, other: &Shape) -> bool;
    
    /// Compute broadcast shape
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape>;
    
    /// Check if shape is scalar
    pub fn is_scalar(&self) -> bool;
    
    /// Check if shape is empty
    pub fn is_empty(&self) -> bool;
}
```

**Examples:**

```rust
let shape = Shape::new(vec![2, 3, 4]);
println!("Dimensions: {:?}", shape.dims());      // [2, 3, 4]
println!("Number of dims: {}", shape.ndim());   // 3
println!("Total elements: {}", shape.numel());  // 24

// Broadcasting
let shape1 = Shape::new(vec![2, 1, 4]);
let shape2 = Shape::new(vec![3, 1]);
let broadcast_shape = shape1.broadcast_with(&shape2)?;
println!("Broadcast shape: {:?}", broadcast_shape.dims()); // [2, 3, 4]
```

### Device

Represents compute devices.

```rust
pub enum DeviceType {
    CPU,
    CUDA,
    Metal,
    WebGPU,
}

pub struct Device {
    device_type: DeviceType,
    index: Option<usize>,
}

impl Device {
    /// Create a CPU device
    pub fn cpu() -> Self;
    
    /// Create a CUDA device
    pub fn cuda(index: usize) -> Self;
    
    /// Create a Metal device
    pub fn metal() -> Self;
    
    /// Create a WebGPU device
    pub fn webgpu() -> Self;
    
    /// Parse device from string
    pub fn from_str(s: &str) -> Result<Self>;
    
    /// Get device type
    pub fn device_type(&self) -> DeviceType;
    
    /// Get device index
    pub fn index(&self) -> Option<usize>;
    
    /// Check if device is available
    pub fn is_available(&self) -> bool;
    
    /// Get device properties
    pub fn properties(&self) -> DeviceProperties;
}
```

**Examples:**

```rust
use torsh::prelude::*;

// Create devices
let cpu = Device::cpu();
let cuda = Device::cuda(0);
let metal = Device::metal();

// From string
let device = Device::from_str("cuda:1")?;
let device = device!("cuda:1"); // Using macro

// Check availability
if cuda.is_available() {
    println!("CUDA is available");
}

// Move tensors between devices
let cpu_tensor = randn(&[2, 3]);
let gpu_tensor = cpu_tensor.to_device(&cuda)?;
```

### DType

Represents data types.

```rust
pub enum DType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl DType {
    /// Get size in bytes
    pub fn size(&self) -> usize;
    
    /// Check if floating point
    pub fn is_float(&self) -> bool;
    
    /// Check if integer
    pub fn is_int(&self) -> bool;
    
    /// Check if unsigned
    pub fn is_unsigned(&self) -> bool;
    
    /// Get default value
    pub fn default_value(&self) -> f64;
}
```

**Examples:**

```rust
let dtype = DType::F32;
println!("Size: {} bytes", dtype.size());        // 4 bytes
println!("Is float: {}", dtype.is_float());      // true

// Convert tensor dtype
let tensor = randn(&[2, 3]).to_dtype(DType::F16)?;
```

## Tensor API

### Creation Functions

```rust
/// Create tensor with zeros
pub fn zeros(shape: &[usize]) -> Tensor;

/// Create tensor with ones
pub fn ones(shape: &[usize]) -> Tensor;

/// Create tensor with specific value
pub fn full(shape: &[usize], value: f64) -> Tensor;

/// Create random tensor (uniform distribution)
pub fn rand(shape: &[usize]) -> Tensor;

/// Create random tensor (normal distribution)
pub fn randn(shape: &[usize]) -> Tensor;

/// Create random integer tensor
pub fn randint(low: i64, high: i64, shape: &[usize]) -> Tensor;

/// Create identity matrix
pub fn eye(n: usize) -> Tensor;

/// Create range tensor
pub fn arange(start: f64, end: f64, step: f64) -> Tensor;

/// Create linearly spaced tensor
pub fn linspace(start: f64, end: f64, steps: usize) -> Tensor;

/// Create tensor like another tensor
pub fn zeros_like(tensor: &Tensor) -> Tensor;
pub fn ones_like(tensor: &Tensor) -> Tensor;
pub fn rand_like(tensor: &Tensor) -> Tensor;
pub fn randn_like(tensor: &Tensor) -> Tensor;
```

**Examples:**

```rust
use torsh::prelude::*;

// Basic creation
let zeros = zeros(&[2, 3]);
let ones = ones(&[2, 3]);
let full = full(&[2, 3], 5.0);

// Random tensors
let rand = rand(&[2, 3]);
let randn = randn(&[2, 3]);
let randint = randint(0, 10, &[2, 3]);

// Special tensors
let eye = eye(3);
let range = arange(0.0, 10.0, 1.0);
let linspace = linspace(0.0, 1.0, 100);

// Like functions
let template = randn(&[2, 3]);
let zeros_like = zeros_like(&template);
let ones_like = ones_like(&template);
```

### Tensor Operations

#### Arithmetic Operations

```rust
// Element-wise operations
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor>;
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor>;
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor>;
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor>;
pub fn pow(a: &Tensor, exponent: f64) -> Result<Tensor>;

// Scalar operations
pub fn add_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor>;
pub fn sub_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor>;
pub fn mul_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor>;
pub fn div_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor>;

// Mathematical functions
pub fn sqrt(tensor: &Tensor) -> Result<Tensor>;
pub fn exp(tensor: &Tensor) -> Result<Tensor>;
pub fn log(tensor: &Tensor) -> Result<Tensor>;
pub fn sin(tensor: &Tensor) -> Result<Tensor>;
pub fn cos(tensor: &Tensor) -> Result<Tensor>;
pub fn tan(tensor: &Tensor) -> Result<Tensor>;
pub fn abs(tensor: &Tensor) -> Result<Tensor>;
pub fn neg(tensor: &Tensor) -> Result<Tensor>;
```

**Examples:**

```rust
let a = randn(&[2, 3]);
let b = randn(&[2, 3]);

// Element-wise operations
let sum = add(&a, &b)?;
let diff = sub(&a, &b)?;
let prod = mul(&a, &b)?;
let quot = div(&a, &b)?;

// Scalar operations
let scaled = mul_scalar(&a, 2.0)?;
let shifted = add_scalar(&a, 1.0)?;

// Mathematical functions
let sqrt_a = sqrt(&a)?;
let exp_a = exp(&a)?;
let log_a = log(&a.abs()?)?;
```

#### Linear Algebra Operations

```rust
// Matrix multiplication
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor>;
pub fn mm(a: &Tensor, b: &Tensor) -> Result<Tensor>;
pub fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor>;

// Dot product
pub fn dot(a: &Tensor, b: &Tensor) -> Result<Tensor>;

// Vector operations
pub fn cross(a: &Tensor, b: &Tensor) -> Result<Tensor>;
pub fn norm(tensor: &Tensor, p: Option<f64>) -> Result<Tensor>;

// Matrix operations
pub fn transpose(tensor: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor>;
pub fn det(tensor: &Tensor) -> Result<Tensor>;
pub fn inverse(tensor: &Tensor) -> Result<Tensor>;
pub fn solve(a: &Tensor, b: &Tensor) -> Result<Tensor>;

// Decompositions
pub fn svd(tensor: &Tensor) -> Result<(Tensor, Tensor, Tensor)>;
pub fn eig(tensor: &Tensor) -> Result<(Tensor, Tensor)>;
pub fn cholesky(tensor: &Tensor) -> Result<Tensor>;
```

**Examples:**

```rust
let a = randn(&[3, 4]);
let b = randn(&[4, 5]);

// Matrix multiplication
let c = matmul(&a, &b)?;

// Batch matrix multiplication
let batch_a = randn(&[10, 3, 4]);
let batch_b = randn(&[10, 4, 5]);
let batch_c = bmm(&batch_a, &batch_b)?;

// Vector operations
let v1 = randn(&[3]);
let v2 = randn(&[3]);
let dot_product = dot(&v1, &v2)?;

// Matrix operations
let matrix = randn(&[4, 4]);
let transposed = transpose(&matrix, 0, 1)?;
let determinant = det(&matrix)?;
```

## Autograd API

### Gradient Computation

```rust
/// Enable gradient computation
pub fn enable_grad();

/// Disable gradient computation
pub fn disable_grad();

/// Check if gradient computation is enabled
pub fn is_grad_enabled() -> bool;

/// Compute gradients
pub fn grad(
    outputs: &[Tensor],
    inputs: &[Tensor],
    grad_outputs: Option<&[Tensor]>,
    retain_graph: bool,
    create_graph: bool,
) -> Result<Vec<Tensor>>;

/// Backward pass
pub fn backward(
    tensors: &[Tensor],
    grad_tensors: Option<&[Tensor]>,
    retain_graph: bool,
    create_graph: bool,
) -> Result<()>;
```

**Examples:**

```rust
use torsh::prelude::*;

// Enable gradients
let x = tensor![2.0].requires_grad_(true);
let y = tensor![3.0].requires_grad_(true);

// Forward pass
let z = x.pow(2.0)? + y.pow(3.0)?;

// Backward pass
z.backward()?;

// Access gradients
let dx = x.grad().unwrap(); // 2 * x = 4.0
let dy = y.grad().unwrap(); // 3 * y^2 = 27.0

// Gradient context
let result = no_grad(|| {
    x.add(&y)
})?;
```

### Gradient Contexts

```rust
/// Execute closure without gradient computation
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R;

/// Execute closure with gradient computation enabled
pub fn enable_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R;

/// Execute closure with gradient computation state
pub fn set_grad_enabled<F, R>(enabled: bool, f: F) -> R
where
    F: FnOnce() -> R;
```

**Examples:**

```rust
let x = tensor![2.0].requires_grad_(true);

// Disable gradients temporarily
let result = no_grad(|| {
    x.pow(2.0)
})?;

// Enable gradients explicitly
let result = enable_grad(|| {
    x.pow(2.0)
})?;

// Set gradient state
let result = set_grad_enabled(false, || {
    x.pow(2.0)
})?;
```

### Automatic Differentiation Functions

```rust
/// Gradient checking
pub fn gradcheck<F>(
    func: F,
    inputs: &[Tensor],
    eps: f64,
    atol: f64,
    rtol: f64,
) -> Result<bool>
where
    F: Fn(&[Tensor]) -> Result<Tensor>;

/// Jacobian computation
pub fn jacobian<F>(
    func: F,
    inputs: &[Tensor],
) -> Result<Vec<Vec<Tensor>>>
where
    F: Fn(&[Tensor]) -> Result<Vec<Tensor>>;

/// Hessian computation
pub fn hessian<F>(
    func: F,
    inputs: &[Tensor],
) -> Result<Vec<Vec<Tensor>>>
where
    F: Fn(&[Tensor]) -> Result<Tensor>;
```

**Examples:**

```rust
// Gradient checking
let input = randn(&[10]).requires_grad_(true);
let is_correct = gradcheck(
    |inputs| {
        let x = &inputs[0];
        x.pow(2.0)?.sum()
    },
    &[input],
    1e-6,
    1e-5,
    1e-3,
)?;

// Jacobian computation
let inputs = vec![randn(&[5]).requires_grad_(true)];
let jac = jacobian(
    |inputs| {
        let x = &inputs[0];
        vec![x.pow(2.0)?, x.pow(3.0)?]
    },
    &inputs,
)?;
```

## Neural Network API

### Module Trait

```rust
pub trait Module {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    
    /// Get module parameters
    fn parameters(&self) -> Vec<&Tensor>;
    
    /// Get mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    
    /// Set training mode
    fn train(&mut self, mode: bool);
    
    /// Set evaluation mode
    fn eval(&mut self) {
        self.train(false);
    }
    
    /// Check if in training mode
    fn is_training(&self) -> bool;
    
    /// Move module to device
    fn to_device(&mut self, device: &Device) -> Result<()>;
    
    /// Convert module to dtype
    fn to_dtype(&mut self, dtype: DType) -> Result<()>;
    
    /// Zero gradients
    fn zero_grad(&mut self);
    
    /// Apply function to all parameters
    fn apply<F>(&mut self, f: F)
    where
        F: Fn(&mut Tensor);
}
```

### Core Layers

#### Linear Layer

```rust
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// Create new linear layer
    pub fn new(in_features: usize, out_features: usize) -> Self;
    
    /// Create linear layer without bias
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self;
    
    /// Get input features
    pub fn in_features(&self) -> usize;
    
    /// Get output features
    pub fn out_features(&self) -> usize;
    
    /// Get weight tensor
    pub fn weight(&self) -> &Tensor;
    
    /// Get bias tensor
    pub fn bias(&self) -> Option<&Tensor>;
}
```

**Examples:**

```rust
use torsh::prelude::*;

// Create linear layer
let linear = Linear::new(784, 128);

// Forward pass
let input = randn(&[32, 784]);
let output = linear.forward(&input)?;
assert_eq!(output.shape().dims(), &[32, 128]);

// Without bias
let linear_no_bias = Linear::new_no_bias(784, 128);
let output = linear_no_bias.forward(&input)?;
```

#### Convolutional Layers

```rust
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
}

impl Conv2d {
    /// Create new 2D convolution layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self>;
    
    /// Create with detailed parameters
    pub fn new_detailed(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Self>;
}
```

**Examples:**

```rust
// Basic convolution
let conv = Conv2d::new(3, 64, 3, 1, 1)?;
let input = randn(&[32, 3, 224, 224]);
let output = conv.forward(&input)?;
assert_eq!(output.shape().dims(), &[32, 64, 224, 224]);

// Detailed convolution
let conv = Conv2d::new_detailed(
    3,                  // in_channels
    64,                 // out_channels
    (3, 3),            // kernel_size
    (2, 2),            // stride
    (1, 1),            // padding
    (1, 1),            // dilation
    1,                 // groups
)?;
```

#### Activation Functions

```rust
pub struct ReLU;
pub struct Sigmoid;
pub struct Tanh;
pub struct GELU;
pub struct SiLU;
pub struct LeakyReLU {
    negative_slope: f64,
}

impl ReLU {
    pub fn new() -> Self;
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self;
}
```

**Examples:**

```rust
// Activation functions
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();
let gelu = GELU::new();
let leaky_relu = LeakyReLU::new(0.01);

let input = randn(&[32, 128]);
let activated = relu.forward(&input)?;
```

#### Normalization Layers

```rust
pub struct BatchNorm2d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
    momentum: f64,
}

impl BatchNorm2d {
    /// Create new batch normalization layer
    pub fn new(num_features: usize) -> Result<Self>;
    
    /// Create with custom parameters
    pub fn new_with_params(
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> Result<Self>;
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    normalized_shape: Vec<usize>,
    eps: f64,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>) -> Result<Self>;
}
```

**Examples:**

```rust
// Batch normalization
let bn = BatchNorm2d::new(64)?;
let input = randn(&[32, 64, 28, 28]);
let normalized = bn.forward(&input)?;

// Layer normalization
let ln = LayerNorm::new(vec![128])?;
let input = randn(&[32, 128]);
let normalized = ln.forward(&input)?;
```

### Container Modules

#### Sequential

```rust
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Create new sequential module
    pub fn new() -> Self;
    
    /// Add module to sequence
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self;
    
    /// Add boxed module to sequence
    pub fn add_boxed(mut self, module: Box<dyn Module>) -> Self;
}
```

**Examples:**

```rust
let model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(ReLU::new())
    .add(Linear::new(128, 64))
    .add(ReLU::new())
    .add(Linear::new(64, 10));

let input = randn(&[32, 784]);
let output = model.forward(&input)?;
```

#### ModuleList

```rust
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self;
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self;
    pub fn len(&self) -> usize;
    pub fn get(&self, index: usize) -> Option<&dyn Module>;
}
```

### Loss Functions

```rust
/// Mean Squared Error Loss
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Cross Entropy Loss
pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Binary Cross Entropy Loss
pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// L1 Loss
pub fn l1_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Smooth L1 Loss
pub fn smooth_l1_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Huber Loss
pub fn huber_loss(input: &Tensor, target: &Tensor, delta: f64) -> Result<Tensor>;

/// Hinge Loss
pub fn hinge_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Cosine Similarity Loss
pub fn cosine_similarity_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;
```

**Examples:**

```rust
let predictions = randn(&[32, 10]);
let targets = randint(0, 10, &[32]);

// Classification loss
let loss = cross_entropy(&predictions, &targets)?;

// Regression loss
let regression_targets = randn(&[32, 10]);
let mse = mse_loss(&predictions, &regression_targets)?;
let l1 = l1_loss(&predictions, &regression_targets)?;
```

## Optimization API

### Optimizer Trait

```rust
pub trait Optimizer {
    /// Perform optimization step
    fn step(&mut self) -> Result<()>;
    
    /// Zero gradients
    fn zero_grad(&mut self);
    
    /// Get learning rate
    fn lr(&self) -> f64;
    
    /// Set learning rate
    fn set_lr(&mut self, lr: f64);
    
    /// Get optimizer state
    fn state_dict(&self) -> StateDict;
    
    /// Load optimizer state
    fn load_state_dict(&mut self, state_dict: StateDict) -> Result<()>;
}
```

### SGD Optimizer

```rust
pub struct SGD {
    params: Vec<Tensor>,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    dampening: f64,
    nesterov: bool,
}

impl SGD {
    /// Create new SGD optimizer
    pub fn new(params: Vec<Tensor>, lr: f64) -> Result<Self>;
    
    /// Create SGD with momentum
    pub fn new_with_momentum(
        params: Vec<Tensor>,
        lr: f64,
        momentum: f64,
    ) -> Result<Self>;
    
    /// Create SGD with all parameters
    pub fn new_with_params(
        params: Vec<Tensor>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        dampening: f64,
        nesterov: bool,
    ) -> Result<Self>;
}
```

**Examples:**

```rust
// Basic SGD
let optimizer = SGD::new(model.parameters(), 0.01)?;

// SGD with momentum
let optimizer = SGD::new_with_momentum(model.parameters(), 0.01, 0.9)?;

// Training loop
for (data, target) in dataloader {
    let output = model.forward(&data)?;
    let loss = cross_entropy(&output, &target)?;
    
    optimizer.zero_grad();
    loss.backward()?;
    optimizer.step()?;
}
```

### Adam Optimizer

```rust
pub struct Adam {
    params: Vec<Tensor>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    amsgrad: bool,
}

impl Adam {
    /// Create new Adam optimizer
    pub fn new(params: Vec<Tensor>, lr: f64) -> Result<Self>;
    
    /// Create Adam with parameters
    pub fn new_with_params(
        params: Vec<Tensor>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Result<Self>;
}
```

**Examples:**

```rust
// Basic Adam
let optimizer = Adam::new(model.parameters(), 0.001)?;

// Adam with custom parameters
let optimizer = Adam::new_with_params(
    model.parameters(),
    0.001,    // lr
    0.9,      // beta1
    0.999,    // beta2
    1e-8,     // eps
    0.0,      // weight_decay
    false,    // amsgrad
)?;
```

### Other Optimizers

```rust
// AdamW
pub struct AdamW { /* ... */ }
impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Result<Self>;
}

// RMSprop
pub struct RMSprop { /* ... */ }
impl RMSprop {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Result<Self>;
}

// Adagrad
pub struct Adagrad { /* ... */ }
impl Adagrad {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Result<Self>;
}
```

### Learning Rate Schedulers

```rust
pub trait LRScheduler {
    fn step(&mut self);
    fn get_lr(&self) -> f64;
}

pub struct StepLR {
    optimizer: Box<dyn Optimizer>,
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    pub fn new(optimizer: Box<dyn Optimizer>, step_size: usize, gamma: f64) -> Self;
}

pub struct ExponentialLR {
    optimizer: Box<dyn Optimizer>,
    gamma: f64,
}

pub struct CosineAnnealingLR {
    optimizer: Box<dyn Optimizer>,
    t_max: usize,
    eta_min: f64,
}
```

**Examples:**

```rust
let optimizer = Adam::new(model.parameters(), 0.001)?;
let scheduler = StepLR::new(Box::new(optimizer), 10, 0.1);

// Training loop with scheduler
for epoch in 0..100 {
    for (data, target) in dataloader {
        // Training step...
    }
    scheduler.step(); // Update learning rate
}
```

## Data Loading API

### Dataset Trait

```rust
pub trait Dataset {
    type Item;
    
    /// Get dataset length
    fn len(&self) -> usize;
    
    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get item at index
    fn get(&self, index: usize) -> Option<Self::Item>;
}
```

### Built-in Datasets

```rust
/// Tensor dataset
pub struct TensorDataset {
    data: Vec<Tensor>,
    targets: Vec<Tensor>,
}

impl TensorDataset {
    pub fn new(data: Vec<Tensor>, targets: Vec<Tensor>) -> Self;
    pub fn from_tensors(data: Tensor, targets: Tensor) -> Self;
}

/// Random dataset
pub struct RandomDataset {
    length: usize,
    data_shape: Vec<usize>,
    target_shape: Vec<usize>,
}

impl RandomDataset {
    pub fn new(
        length: usize,
        data_shape: Vec<usize>,
        target_shape: Vec<usize>,
    ) -> Self;
}
```

### DataLoader

```rust
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    drop_last: bool,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self;
    
    pub fn new_with_options(
        dataset: D,
        batch_size: usize,
        shuffle: bool,
        num_workers: usize,
        drop_last: bool,
    ) -> Self;
    
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = Vec<D::Item>;
    
    fn next(&mut self) -> Option<Self::Item>;
}
```

**Examples:**

```rust
// Create dataset
let data = randn(&[1000, 784]);
let targets = randint(0, 10, &[1000]);
let dataset = TensorDataset::from_tensors(data, targets);

// Create dataloader
let dataloader = DataLoader::new_with_options(
    dataset,
    32,    // batch_size
    true,  // shuffle
    4,     // num_workers
    false, // drop_last
);

// Training loop
for batch in dataloader {
    let (data, targets) = batch;
    // Training step...
}
```

### Transforms

```rust
pub trait Transform {
    type Input;
    type Output;
    
    fn transform(&self, input: Self::Input) -> Self::Output;
}

// Built-in transforms
pub struct Normalize {
    mean: Vec<f64>,
    std: Vec<f64>,
}

pub struct RandomHorizontalFlip {
    p: f64,
}

pub struct RandomCrop {
    size: (usize, usize),
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform<Input = Tensor, Output = Tensor>>>,
}
```

**Examples:**

```rust
// Image transforms
let transform = Compose::new()
    .add(RandomHorizontalFlip::new(0.5))
    .add(RandomCrop::new((224, 224)))
    .add(Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]));

// Apply transform
let image = randn(&[3, 256, 256]);
let transformed = transform.transform(image);
```

## Functional API

The functional API provides stateless operations similar to PyTorch's `torch.nn.functional`.

### Activation Functions

```rust
/// ReLU activation
pub fn relu(input: &Tensor) -> Result<Tensor>;

/// Sigmoid activation
pub fn sigmoid(input: &Tensor) -> Result<Tensor>;

/// Tanh activation
pub fn tanh(input: &Tensor) -> Result<Tensor>;

/// GELU activation
pub fn gelu(input: &Tensor) -> Result<Tensor>;

/// SiLU activation
pub fn silu(input: &Tensor) -> Result<Tensor>;

/// Softmax activation
pub fn softmax(input: &Tensor, dim: isize) -> Result<Tensor>;

/// Log softmax activation
pub fn log_softmax(input: &Tensor, dim: isize) -> Result<Tensor>;

/// LeakyReLU activation
pub fn leaky_relu(input: &Tensor, negative_slope: f64) -> Result<Tensor>;

/// ELU activation
pub fn elu(input: &Tensor, alpha: f64) -> Result<Tensor>;
```

**Examples:**

```rust
use torsh::functional as F;

let input = randn(&[32, 128]);

// Activations
let relu_out = F::relu(&input)?;
let sigmoid_out = F::sigmoid(&input)?;
let softmax_out = F::softmax(&input, -1)?;
let gelu_out = F::gelu(&input)?;
```

### Convolution Operations

```rust
/// 1D convolution
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor>;

/// 2D convolution
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Tensor>;

/// 3D convolution
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> Result<Tensor>;
```

**Examples:**

```rust
let input = randn(&[32, 3, 224, 224]);
let weight = randn(&[64, 3, 3, 3]);
let bias = randn(&[64]);

let output = F::conv2d(
    &input,
    &weight,
    Some(&bias),
    (1, 1),    // stride
    (1, 1),    // padding
    (1, 1),    // dilation
    1,         // groups
)?;
```

### Pooling Operations

```rust
/// Max pooling 2D
pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
) -> Result<Tensor>;

/// Average pooling 2D
pub fn avg_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
) -> Result<Tensor>;

/// Adaptive average pooling 2D
pub fn adaptive_avg_pool2d(
    input: &Tensor,
    output_size: (usize, usize),
) -> Result<Tensor>;

/// Global average pooling
pub fn global_avg_pool2d(input: &Tensor) -> Result<Tensor>;
```

**Examples:**

```rust
let input = randn(&[32, 64, 56, 56]);

// Max pooling
let pooled = F::max_pool2d(&input, (2, 2), None, (0, 0))?;

// Average pooling
let avg_pooled = F::avg_pool2d(&input, (2, 2), None, (0, 0))?;

// Adaptive pooling
let adaptive = F::adaptive_avg_pool2d(&input, (7, 7))?;
```

### Normalization Operations

```rust
/// Batch normalization
pub fn batch_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    training: bool,
    momentum: f64,
    eps: f64,
) -> Result<Tensor>;

/// Layer normalization
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor>;

/// Group normalization
pub fn group_norm(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor>;
```

### Loss Functions

```rust
/// Mean squared error loss
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Cross entropy loss
pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Binary cross entropy loss
pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// L1 loss
pub fn l1_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;

/// Smooth L1 loss
pub fn smooth_l1_loss(input: &Tensor, target: &Tensor) -> Result<Tensor>;
```

## Advanced APIs

### Sparse Tensors

```rust
pub struct SparseTensor {
    indices: Tensor,
    values: Tensor,
    shape: Shape,
}

impl SparseTensor {
    /// Create sparse tensor from indices and values
    pub fn new(indices: Tensor, values: Tensor, shape: Shape) -> Self;
    
    /// Convert to dense tensor
    pub fn to_dense(&self) -> Result<Tensor>;
    
    /// Get indices
    pub fn indices(&self) -> &Tensor;
    
    /// Get values
    pub fn values(&self) -> &Tensor;
    
    /// Get shape
    pub fn shape(&self) -> &Shape;
    
    /// Sparse matrix multiplication
    pub fn sparse_mm(&self, other: &Tensor) -> Result<Tensor>;
}
```

### Quantization

```rust
/// Quantize tensor to int8
pub fn quantize_int8(tensor: &Tensor, scale: f64, zero_point: i8) -> Result<Tensor>;

/// Dequantize int8 tensor
pub fn dequantize_int8(tensor: &Tensor, scale: f64, zero_point: i8) -> Result<Tensor>;

/// Quantize model
pub fn quantize_model<M: Module>(model: M, config: QuantizationConfig) -> Result<M>;
```

### Distributed Training

```rust
/// Initialize distributed training
pub fn init_process_group(
    backend: &str,
    world_size: usize,
    rank: usize,
) -> Result<()>;

/// All-reduce operation
pub fn all_reduce(tensor: &mut Tensor, op: ReduceOp) -> Result<()>;

/// All-gather operation
pub fn all_gather(tensor: &Tensor, output: &mut Tensor) -> Result<()>;

/// Broadcast operation
pub fn broadcast(tensor: &mut Tensor, src: usize) -> Result<()>;
```

## Utility APIs

### Serialization

```rust
/// Save tensor to file
pub fn save_tensor(tensor: &Tensor, path: &str) -> Result<()>;

/// Load tensor from file
pub fn load_tensor(path: &str) -> Result<Tensor>;

/// Save model state
pub fn save_model<M: Module>(model: &M, path: &str) -> Result<()>;

/// Load model state
pub fn load_model<M: Module>(model: &mut M, path: &str) -> Result<()>;
```

### Memory Management

```rust
/// Get memory usage
pub fn get_memory_usage() -> MemoryInfo;

/// Clear cache
pub fn empty_cache();

/// Set memory fraction
pub fn set_memory_fraction(fraction: f64);

/// Enable memory profiling
pub fn enable_memory_profiling();

/// Disable memory profiling
pub fn disable_memory_profiling();
```

### Debugging

```rust
/// Print tensor information
pub fn print_tensor_info(tensor: &Tensor);

/// Check for NaN values
pub fn has_nan(tensor: &Tensor) -> bool;

/// Check for infinite values
pub fn has_inf(tensor: &Tensor) -> bool;

/// Tensor statistics
pub fn tensor_stats(tensor: &Tensor) -> TensorStats;
```

---

This API reference provides comprehensive coverage of all ToRSh APIs with examples and usage patterns. For more detailed information, refer to the individual crate documentation and the comprehensive guide.