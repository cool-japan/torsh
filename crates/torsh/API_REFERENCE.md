# ToRSh API Reference

## Overview

ToRSh (Tensor Operations in Rust with Sharding) is a production-ready deep learning framework built in pure Rust. It provides a PyTorch-compatible API with superior performance, memory safety, and deployment flexibility.

## Core API

### Tensor Operations

#### Tensor Creation Functions

```rust
use torsh::prelude::*;

// Create tensors from data
let t1 = tensor![1.0, 2.0, 3.0];
let t2 = tensor_2d![[1.0, 2.0], [3.0, 4.0]];

// Convenience creation functions
let zeros = zeros(&[2, 3]);
let ones = ones(&[2, 3]);
let randn = randn(&[2, 3]);
let eye = eye(5);
let arange = arange(0.0, 10.0, 1.0);
```

#### Tensor Operations

```rust
// Arithmetic operations
let result = tensor1.add(&tensor2)?;
let result = tensor1.mul(&tensor2)?;
let result = tensor1.div(&tensor2)?;
let result = tensor1.sub(&tensor2)?;

// Matrix operations
let result = tensor1.matmul(&tensor2)?;
let result = tensor1.transpose(0, 1)?;

// Reduction operations
let sum = tensor1.sum(None)?;
let mean = tensor1.mean(None)?;
let max = tensor1.max(None)?;
let min = tensor1.min(None)?;

// Shape operations
let reshaped = tensor1.reshape(&[2, 3])?;
let squeezed = tensor1.squeeze()?;
let unsqueezed = tensor1.unsqueeze(0)?;
```

### Automatic Differentiation

```rust
use torsh::prelude::*;

// Enable gradients
let x = tensor![2.0].requires_grad_(true);
let y = x.pow(2.0)?;

// Compute gradients
y.backward()?;
println!("Gradient: {:?}", x.grad());

// Gradient contexts
let result = no_grad(|| {
    // Operations here don't track gradients
    x.mul(&x)
})?;
```

### Neural Network Modules

#### Basic Layers

```rust
use torsh::nn::*;

// Linear layer
let linear = Linear::new(784, 128);
let output = linear.forward(&input)?;

// Convolutional layers
let conv2d = Conv2d::new(3, 64, 3, ConvOptions::default());
let conv_out = conv2d.forward(&input)?;

// Normalization layers
let batch_norm = BatchNorm2d::new(64);
let norm_out = batch_norm.forward(&conv_out)?;
```

#### Activation Functions

```rust
use torsh::nn::*;

// Built-in activation modules
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();
let gelu = GELU::new();

// Functional activations
let activated = F::relu(&input)?;
let activated = F::sigmoid(&input)?;
```

#### Loss Functions

```rust
use torsh::nn::*;

// Loss functions
let mse = MSELoss::new();
let cross_entropy = CrossEntropyLoss::new();
let bce = BCELoss::new();

// Compute loss
let loss = mse.forward(&predictions, &targets)?;
```

### Optimizers

```rust
use torsh::optim::*;

// Create optimizer
let mut sgd = SGD::new(model.parameters(), 0.01);
let mut adam = Adam::new(model.parameters(), 0.001);

// Optimization step
sgd.zero_grad();
loss.backward()?;
sgd.step()?;
```

### Data Loading

```rust
use torsh::data::*;

// Dataset creation
let dataset = TensorDataset::new(inputs, targets);
let dataloader = DataLoader::new(dataset, 32, true);

// Iterate over batches
for batch in dataloader {
    let (inputs, targets) = batch?;
    // Training logic here
}
```

## Advanced API

### Sparse Tensors

```rust
use torsh::sparse::*;

// Create sparse tensor
let indices = tensor![[0, 1, 1], [2, 0, 2]];
let values = tensor![3.0, 4.0, 5.0];
let sparse = SparseTensor::new(indices, values, &[2, 3]);

// Sparse operations
let result = sparse.to_dense()?;
let spmm = sparse.sparse_mm(&dense)?;
```

### Quantization

```rust
use torsh::quantization::*;

// Dynamic quantization
let quantized = quantize_dynamic(&model, &[QConfigDynamic::default()])?;

// Static quantization
let qconfig = QConfig::default();
let quantized = quantize_static(&model, &qconfig)?;
```

### Special Functions

```rust
use torsh::special::*;

// Mathematical special functions
let gamma_result = gamma(&input)?;
let bessel_result = bessel_j0(&input)?;
let erf_result = erf(&input)?;
```

### Linear Algebra

```rust
use torsh::linalg::*;

// Matrix decompositions
let (q, r) = qr(&matrix)?;
let (u, s, v) = svd(&matrix)?;
let eigenvals = eigvals(&matrix)?;

// Solve linear systems
let solution = solve(&a, &b)?;
```

### Distributed Training

```rust
use torsh::distributed::*;

// Initialize distributed training
let world_size = 4;
let rank = 0;
init_process_group(Backend::Nccl, world_size, rank)?;

// Distributed data parallel
let ddp_model = DistributedDataParallel::new(model)?;
```

### JIT Compilation

```rust
use torsh::jit::*;

// JIT compile function
let jit_fn = jit_compile(|x: &Tensor| {
    x.mul(&x)?.add(&tensor![1.0])
})?;

// Use compiled function
let result = jit_fn(&input)?;
```

### Graph Transformations

```rust
use torsh::fx::*;

// Create graph tracer
let tracer = GraphTracer::new();
let graph = tracer.trace(&model, &sample_input)?;

// Apply transformations
let optimized = optimize_graph(&graph)?;
```

## Device Management

```rust
use torsh::prelude::*;

// Device creation
let cpu = Device::cpu();
let cuda = Device::cuda(0);

// Move tensors to device
let tensor_gpu = tensor_cpu.to_device(&cuda)?;

// Check device properties
let properties = cuda.properties()?;
println!("Device: {}", properties.name);
```

## Memory Management

```rust
use torsh::prelude::*;

// Memory info
let mem_info = Device::cuda(0).memory_info()?;
println!("Free: {}, Total: {}", mem_info.free, mem_info.total);

// Manual memory management
torch::cuda::empty_cache()?;
torch::cuda::synchronize()?;
```

## Serialization

```rust
use torsh::prelude::*;

// Save/load tensors
tensor.save("tensor.pt")?;
let loaded = Tensor::load("tensor.pt")?;

// Model state dict
let state_dict = model.state_dict();
model.load_state_dict(&state_dict)?;
```

## Error Handling

```rust
use torsh::prelude::*;

// All operations return Result<T, TorshError>
match tensor1.add(&tensor2) {
    Ok(result) => println!("Success: {:?}", result),
    Err(e) => println!("Error: {}", e),
}

// Use ? operator for error propagation
fn example() -> Result<Tensor> {
    let a = tensor![1.0, 2.0];
    let b = tensor![3.0, 4.0];
    let result = a.add(&b)?;
    Ok(result)
}
```

## Performance Optimization

### SIMD Operations

```rust
// Enable SIMD optimizations
use torsh::prelude::*;

// Operations automatically use SIMD when available
let result = tensor1.add(&tensor2)?; // Uses AVX/NEON when available
```

### Memory Layout

```rust
// Contiguous memory layout
let tensor = tensor.contiguous()?;

// Memory format specification
let nhwc = tensor.to_memory_format(MemoryFormat::ChannelsLast)?;
```

## Integration Examples

### PyTorch Compatibility

```rust
use torsh::prelude::*;

// PyTorch-like API
let x = torch::randn(&[2, 3]);
let y = torch::mm(&x, &x.t());
let z = torch::relu(&y);

// Functional API
let output = F::conv2d(&input, &weight, Some(&bias), 1, 1, 1, 1)?;
```

### NumPy Compatibility

```rust
use torsh::prelude::*;

// NumPy-like operations
let arr = np::array([[1.0, 2.0], [3.0, 4.0]]);
let result = np::dot(&arr, &arr.T());
```

## Testing Utilities

```rust
use torsh::testing::*;

// Assertion helpers
assert_tensor_eq!(tensor1, tensor2);
assert_tensor_close!(tensor1, tensor2, 1e-5);

// Testing with gradients
assert_grad_close!(tensor1, tensor2, 1e-5);
```

## Best Practices

### Memory Management

```rust
// Prefer in-place operations when possible
tensor.add_(&other)?; // In-place addition
tensor.mul_(&scalar)?; // In-place multiplication

// Use proper scoping for temporary tensors
{
    let temp = tensor.clone();
    // Use temp here
} // temp is dropped here
```

### Performance

```rust
// Use appropriate data types
let float_tensor = tensor.to_dtype(DType::F32)?;
let half_tensor = tensor.to_dtype(DType::F16)?; // For memory efficiency

// Batch operations
let batched = torch::stack(&tensors, 0)?; // Better than individual operations
```

### Error Handling

```rust
// Handle errors appropriately
fn safe_operation(tensor: &Tensor) -> Result<Tensor> {
    if tensor.numel() == 0 {
        return Err(TorshError::InvalidArgument("Empty tensor".to_string()));
    }
    tensor.square()
}
```

## Version Information

```rust
use torsh::prelude::*;

// Check version
println!("ToRSh version: {}", VERSION);

// Version compatibility
check_version(0, 1)?;

// Feature information
print_feature_info();
```

## Migration from PyTorch

### Common Patterns

```rust
// PyTorch -> ToRSh
// torch.tensor([1, 2, 3])
let tensor = tensor![1, 2, 3];

// torch.randn(2, 3)
let tensor = randn(&[2, 3]);

// torch.nn.Linear(784, 128)
let linear = Linear::new(784, 128);

// torch.optim.Adam(model.parameters(), lr=0.001)
let optimizer = Adam::new(model.parameters(), 0.001);
```

### Key Differences

1. **Error Handling**: ToRSh uses `Result<T, TorshError>` for all operations
2. **Memory Safety**: No need for manual memory management
3. **Performance**: Automatic SIMD optimizations
4. **Type Safety**: Compile-time shape checking when possible

## Contributing

For API extensions and improvements, please follow the ToRSh contribution guidelines and ensure all new APIs maintain PyTorch compatibility where appropriate.