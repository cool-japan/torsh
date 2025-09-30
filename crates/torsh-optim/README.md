# torsh-optim

Optimization algorithms for ToRSh with PyTorch-compatible API, powered by scirs2-optim.

## Overview

This crate provides state-of-the-art optimization algorithms including:

- **SGD**: Stochastic Gradient Descent with momentum and Nesterov acceleration
- **Adam**: Adaptive Moment Estimation optimizer
- **AdamW**: Adam with decoupled weight decay
- **Learning Rate Schedulers**: Various scheduling strategies

## Usage

### Basic Optimizer Usage

```rust
use torsh_optim::prelude::*;
use torsh_nn::prelude::*;

// Create a model
let model = create_model();
let params = model.parameters();

// SGD optimizer
let mut optimizer = SGD::new(params, 0.1, Some(0.9), None, None, false);

// Training loop
for epoch in 0..num_epochs {
    let output = model.forward(&input)?;
    let loss = compute_loss(&output, &target)?;
    
    // Backward pass
    loss.backward()?;
    
    // Optimizer step
    optimizer.step()?;
    optimizer.zero_grad();
}
```

### Adam Optimizer

```rust
use torsh_optim::adam::Adam;

// Create Adam optimizer with default parameters
let mut optimizer = Adam::new(
    params,
    Some(0.001),      // learning rate
    Some((0.9, 0.999)), // betas
    Some(1e-8),       // epsilon
    Some(0.0),        // weight decay
    false,            // amsgrad
);

// Or use the builder pattern
let mut optimizer = AdamBuilder::new()
    .lr(0.001)
    .betas(0.9, 0.999)
    .weight_decay(1e-4)
    .build(params);
```

### AdamW Optimizer

```rust
use torsh_optim::adam::AdamW;

// AdamW with decoupled weight decay
let mut optimizer = AdamW::new(
    params,
    Some(0.001),      // learning rate
    Some((0.9, 0.999)), // betas
    Some(1e-8),       // epsilon
    Some(0.01),       // weight decay (decoupled)
    false,            // amsgrad
);
```

### SGD with Momentum

```rust
use torsh_optim::sgd::SGD;

// SGD with momentum
let mut optimizer = SGD::new(
    params,
    0.1,              // learning rate
    Some(0.9),        // momentum
    Some(0.0),        // dampening
    Some(1e-4),       // weight decay
    false,            // nesterov
);

// SGD with Nesterov momentum
let mut optimizer = SGDBuilder::new(0.1)
    .momentum(0.9)
    .weight_decay(5e-4)
    .nesterov(true)
    .build(params);
```

### Learning Rate Schedulers

#### Step LR
```rust
use torsh_optim::lr_scheduler::{StepLR, LRScheduler};

let optimizer = Adam::new(params, Some(0.1), None, None, None, false);
let mut scheduler = StepLR::new(optimizer, 30, 0.1);

for epoch in 0..num_epochs {
    train_epoch()?;
    scheduler.step();
    println!("LR: {:?}", scheduler.get_last_lr());
}
```

#### Exponential LR
```rust
use torsh_optim::lr_scheduler::ExponentialLR;

let mut scheduler = ExponentialLR::new(optimizer, 0.95);
```

#### Cosine Annealing
```rust
use torsh_optim::lr_scheduler::CosineAnnealingLR;

let mut scheduler = CosineAnnealingLR::new(
    optimizer,
    100,    // T_max
    1e-6,   // eta_min
);
```

#### Reduce LR on Plateau
```rust
use torsh_optim::lr_scheduler::ReduceLROnPlateau;

let mut scheduler = ReduceLROnPlateau::new(
    optimizer,
    "min",     // mode
    0.1,       // factor
    10,        // patience
    0.0001,    // threshold
    "rel",     // threshold_mode
    0,         // cooldown
    1e-6,      // min_lr
    1e-8,      // eps
)?;

// In training loop
let val_loss = validate()?;
scheduler.step(val_loss);
```

#### One Cycle LR
```rust
use torsh_optim::lr_scheduler::OneCycleLR;

let mut scheduler = OneCycleLR::new(
    optimizer,
    vec![0.1],         // max_lr
    total_steps,       // total_steps
    Some(0.3),         // pct_start
    Some("cos"),       // anneal_strategy
    Some(true),        // cycle_momentum
    Some(0.85),        // base_momentum
    Some(0.95),        // max_momentum
    Some(25.0),        // div_factor
    Some(10000.0),     // final_div_factor
);
```

### Parameter Groups

```rust
// Different learning rates for different layers
let mut optimizer = Adam::new(base_params, Some(1e-3), None, None, None, false);

// Add another parameter group with different settings
let mut options = HashMap::new();
options.insert("lr".to_string(), 1e-4);
options.insert("weight_decay".to_string(), 0.0);
optimizer.add_param_group(head_params, options);
```

### State Dict (Checkpointing)

```rust
// Save optimizer state
let state_dict = optimizer.state_dict();
save_checkpoint(&state_dict)?;

// Load optimizer state
let loaded_state = load_checkpoint()?;
optimizer.load_state_dict(loaded_state)?;
```

### Advanced Usage

#### Gradient Clipping
```rust
use torsh_autograd::grad_mode::clip;

// Clip gradients before optimizer step
let total_norm = clip::clip_grad_norm(&mut params, 1.0, 2.0);

// Then step
optimizer.step()?;
```

#### Custom Optimizer Options
```rust
let mut param_groups = vec![];

// Encoder parameters with lower learning rate
param_groups.push(ParamGroup::new(encoder_params, 1e-4)
    .with_options(hashmap!{
        "weight_decay".to_string() => 0.01,
    }));

// Decoder parameters with higher learning rate
param_groups.push(ParamGroup::new(decoder_params, 1e-3)
    .with_options(hashmap!{
        "weight_decay".to_string() => 0.0,
    }));
```

## Integration with SciRS2

This crate leverages scirs2-optim for:

- Optimized parameter updates
- Efficient state management
- Hardware acceleration support
- Advanced optimization algorithms

All optimizers are designed to work seamlessly with ToRSh's autograd system while benefiting from scirs2's performance optimizations.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.