# torsh-quantization

Quantization toolkit for ToRSh, enabling efficient model deployment with reduced precision.

## Overview

This crate provides comprehensive quantization support for deep learning models:

- **Post-Training Quantization**: Quantize trained models without retraining
- **Quantization-Aware Training**: Train models with simulated quantization
- **Dynamic Quantization**: Runtime quantization for specific operations
- **Backends**: Support for multiple quantization backends (FBGEMM, QNNPACK)
- **Formats**: INT8, INT4, and custom quantization schemes

## Usage

### Post-Training Quantization

```rust
use torsh_quantization::prelude::*;

// Static quantization (requires calibration)
let model = load_model()?;

// Prepare model for calibration
let prepared = prepare_static(
    model,
    QuantConfig::default()
        .backend(Backend::FBGEMM)
        .observer(MinMaxObserver::default()),
)?;

// Calibrate with representative data
for batch in calibration_loader {
    prepared.forward(&batch)?;
}

// Convert to quantized model
let quantized = convert(prepared)?;

// Dynamic quantization (no calibration needed)
let dynamic_quantized = quantize_dynamic(
    model,
    qconfig_spec={
        Linear: default_dynamic_qconfig(),
        LSTM: default_dynamic_qconfig(),
    },
    dtype=qint8,
)?;
```

### Quantization-Aware Training (QAT)

```rust
use torsh_quantization::qat::*;

// Prepare model for QAT
let model = create_model();
let qat_model = prepare_qat(
    model,
    QuantConfig::default()
        .backend(Backend::QNNPACK)
        .activation(FakeQuantize::default())
        .weight(FakeQuantize::default()),
)?;

// Train with fake quantization
for epoch in 0..num_epochs {
    for batch in train_loader {
        let output = qat_model.forward(&batch.input)?;
        let loss = criterion(&output, &batch.target)?;
        
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step();
    }
}

// Convert to actual quantized model
let quantized = convert(qat_model)?;
```

### Custom Quantization Configuration

```rust
use torsh_quantization::qconfig::*;

// Per-layer configuration
let qconfig_dict = QConfigDict::new()
    .set_global(get_default_qconfig())
    .set_module_name("features.0", QConfig {
        activation: HistogramObserver::with_args(bins=1024),
        weight: PerChannelMinMaxObserver::with_args(ch_axis=0),
    })
    .set_module_type::<Conv2d>(QConfig {
        activation: MovingAverageMinMaxObserver::default(),
        weight: default_weight_observer(),
    });

let quantized = quantize_fx(
    model,
    qconfig_dict,
    backend_config,
)?;
```

### Quantization Schemes

```rust
// Symmetric vs Asymmetric quantization
let symmetric_qconfig = QConfig::new()
    .activation(MinMaxObserver::symmetric())
    .weight(MinMaxObserver::symmetric());

let asymmetric_qconfig = QConfig::new()
    .activation(MinMaxObserver::asymmetric())
    .weight(PerChannelMinMaxObserver::asymmetric());

// Custom bit widths
let int4_qconfig = QConfig::new()
    .activation(MinMaxObserver::with_bits(4))
    .weight(MinMaxObserver::with_bits(4));

// Mixed precision
let mixed_qconfig = QConfigDict::new()
    .set_module_type::<Linear>(int8_qconfig())
    .set_module_type::<Conv2d>(int4_qconfig())
    .set_module_name("classifier", fp16_qconfig());
```

### Model Analysis

```rust
use torsh_quantization::analysis::*;

// Compare quantized vs original
let comparison = compare_models(
    original_model,
    quantized_model,
    test_data,
    metrics=vec!["accuracy", "latency", "model_size"],
)?;

println!("Accuracy drop: {:.2}%", comparison.accuracy_drop);
println!("Speedup: {:.2}x", comparison.speedup);
println!("Compression: {:.2}x", comparison.compression_ratio);

// Sensitivity analysis
let sensitivity = sensitivity_analysis(
    model,
    calibration_data,
    test_data,
)?;

// Find layers sensitive to quantization
for (layer_name, metrics) in sensitivity {
    if metrics.accuracy_drop > 0.01 {
        println!("Sensitive layer: {} (drop: {:.2}%)", 
                 layer_name, metrics.accuracy_drop * 100.0);
    }
}
```

### Export and Deployment

```rust
// Export for mobile (QNNPACK backend)
let mobile_model = optimize_for_mobile(quantized_model)?;
mobile_model.save("model_mobile.pt")?;

// Export to ONNX with quantization
let onnx_model = export_quantized_onnx(
    quantized_model,
    example_input,
    opset_version=13,
)?;

// TensorRT export
let trt_model = export_tensorrt(
    quantized_model,
    precision="INT8",
    calibration_cache="calibration.cache",
)?;
```

### Debugging and Visualization

```rust
use torsh_quantization::debug::*;

// Visualize quantization ranges
let observer_dict = get_observer_dict(prepared_model)?;
for (name, observer) in observer_dict {
    let (min_val, max_val) = observer.calculate_qparams();
    println!("{}: range [{:.3}, {:.3}]", name, min_val, max_val);
}

// Debug quantization errors
let debugger = QuantizationDebugger(
    model,
    quantized_model,
    test_loader,
);

let layer_errors = debugger.calculate_layer_errors()?;
debugger.plot_error_heatmap("quantization_errors.png")?;
```

### Advanced Features

```rust
// Learnable quantization parameters
let learnable_fake_quant = LearnableFakeQuantize::new(
    observer=MovingAverageMinMaxObserver::default(),
    quant_min=-128,
    quant_max=127,
    scale_lr=0.01,
    zero_point_lr=0.01,
);

// Stochastic quantization
let stochastic_quant = StochasticQuantize::new(
    bit_width=8,
    temperature=1.0,
);

// Channel-wise quantization for Conv/Linear
let per_channel_qconfig = QConfig::new()
    .weight(PerChannelMinMaxObserver::new(
        ch_axis=0,
        qscheme=per_channel_symmetric,
    ));

// Group-wise quantization
let group_wise_qconfig = GroupWiseQuantConfig::new(
    groups=32,
    bits=4,
);
```

### Quantization Backends

```rust
// FBGEMM (x86 optimized)
#[cfg(target_arch = "x86_64")]
let fbgemm_config = QuantConfig::default()
    .backend(Backend::FBGEMM);

// QNNPACK (mobile optimized)
#[cfg(target_arch = "arm")]
let qnnpack_config = QuantConfig::default()
    .backend(Backend::QNNPACK);

// Custom backend
let custom_backend = CustomBackend::new()
    .supported_ops(vec!["quantized::linear", "quantized::conv2d"])
    .kernel_library(my_kernel_lib);
```

## Supported Operations

- **Linear layers**: Linear, Bilinear
- **Convolutional**: Conv1d, Conv2d, Conv3d, ConvTranspose
- **Recurrent**: LSTM, GRU (dynamic quantization)
- **Activations**: ReLU, ReLU6, Hardswish, ELU
- **Pooling**: MaxPool, AvgPool, AdaptiveAvgPool
- **Normalization**: BatchNorm (fused with Conv/Linear)

## Best Practices

1. Use representative calibration data
2. Start with INT8 before trying lower bit widths
3. Use per-channel quantization for Conv/Linear layers
4. Keep sensitive layers in higher precision
5. Profile on target hardware

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.