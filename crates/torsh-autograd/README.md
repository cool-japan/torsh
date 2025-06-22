# torsh-autograd

Automatic differentiation engine for ToRSh, providing PyTorch-compatible autograd functionality powered by scirs2.

## Overview

This crate leverages scirs2's powerful automatic differentiation capabilities to provide:

- Reverse-mode automatic differentiation
- PyTorch-compatible gradient computation API
- Advanced features like Jacobian/Hessian computation
- Gradient accumulation and checkpointing
- Memory-efficient training utilities

## Features

- **Full scirs2 Integration**: Built on top of scirs2-autograd for robust AD
- **Gradient Modes**: Support for no_grad, inference_mode, and anomaly detection
- **Advanced Functions**: Jacobian, Hessian, VJP, and JVP computations
- **Custom Functions**: Define your own differentiable operations
- **Memory Efficiency**: Gradient checkpointing and accumulation
- **Performance**: Profiling and optimization utilities

## Usage

### Basic Gradient Computation

```rust
use torsh_autograd::prelude::*;
use torsh_tensor::prelude::*;

// Enable gradient computation
let x = tensor![2.0].requires_grad_(true);
let y = x.pow(2.0)?;

// Compute gradients
backward(&y, None, false)?;

// Access gradient
let grad = x.grad().unwrap();
assert_eq!(grad.item(), 4.0); // dy/dx = 2x = 4
```

### Gradient Modes

```rust
// Disable gradient computation
{
    let _guard = no_grad();
    // Operations here won't track gradients
    let z = x.mul(&y)?;
}

// Inference mode for maximum performance
{
    let _guard = inference_mode();
    // No graph building, pure computation
    let output = model.forward(&input)?;
}

// Anomaly detection for debugging
{
    let _guard = detect_anomaly();
    // Will detect NaN/Inf in gradients
    backward(&loss, None, false)?;
}
```

### Advanced Gradient Functions

```rust
// Compute Jacobian matrix
let jacobian = jacobian(|x| x.pow(2.0), &input, true)?;

// Compute Hessian matrix
let hessian = hessian(|x| x.sum(), &input, true)?;

// Vector-Jacobian product
let (output, vjp) = vjp(|x| model.forward(x), &input, &v, true)?;

// Jacobian-vector product
let (output, jvp) = jvp(|x| model.forward(x), &input, &v, true)?;
```

### Custom Autograd Functions

```rust
use torsh_autograd::function::{Function, FunctionContext, apply_function};

struct MyReLU;

impl Function for MyReLU {
    fn forward<T>(&self, ctx: &mut FunctionContext, inputs: &[&Tensor<T>]) -> Result<Vec<Tensor<T>>>
    where
        T: TensorElement,
    {
        // Save input for backward
        ctx.save_for_backward(inputs);
        
        // Compute ReLU: max(0, x)
        let output = inputs[0].clamp_min(0.0)?;
        Ok(vec![output])
    }
    
    fn backward<T>(&self, ctx: &mut FunctionContext, grad_outputs: &[&Tensor<T>]) -> Result<Vec<Option<Tensor<T>>>>
    where
        T: TensorElement,
    {
        let saved = ctx.saved_tensors::<T>()?;
        let input = &saved[0];
        
        // Gradient is 1 where input > 0, else 0
        let grad_input = grad_outputs[0].mul(&input.gt(&zeros_like(input))?)?;
        Ok(vec![Some(grad_input)])
    }
}

// Apply custom function
let output = apply_function(MyReLU, &[&input])?;
```

### Gradient Accumulation

```rust
use torsh_autograd::accumulate::GradientAccumulator;

let mut accumulator = GradientAccumulator::new();

// Accumulate gradients over multiple batches
for batch in batches {
    let loss = model.forward(&batch)?;
    backward(&loss, None, true)?;
    accumulator.accumulate();
}

// Get averaged gradients
let avg_grads = accumulator.average();
```

### Memory-Efficient Training

```rust
use torsh_autograd::checkpoint::checkpoint;

// Checkpoint a function to save memory
let outputs = checkpoint(
    |inputs| {
        // Memory-intensive computation
        let x = expensive_layer1(&inputs[0])?;
        let y = expensive_layer2(&x)?;
        Ok(vec![y])
    },
    &[input],
)?;
```

### Gradient Clipping

```rust
use torsh_autograd::grad_mode::clip::{clip_grad_norm, clip_grad_value};

// Clip gradients by global norm
let total_norm = clip_grad_norm(&mut model.parameters(), 1.0, 2.0);

// Clip gradients by value
clip_grad_value(&mut model.parameters(), 0.5);
```

## Integration with SciRS2

This crate fully leverages scirs2-autograd's capabilities:

- **Variable Environment**: Managed gradient storage
- **Computation Graphs**: Efficient graph construction and traversal
- **Tensor Operations**: All operations use scirs2's optimized implementations
- **Memory Management**: Benefit from scirs2's memory optimization

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.