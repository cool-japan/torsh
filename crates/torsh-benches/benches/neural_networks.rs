//! Neural network operations benchmarks

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use torsh_tensor::{Tensor, creation::*};
use torsh_nn::modules::*;
use torsh_nn::Module;
use torsh_core::dtype::DType;
use torsh_core::device::DeviceType;

fn bench_linear_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_layers");
    
    // Test different layer sizes
    let layer_configs = [
        (128, 64),   // Small layer
        (512, 256),  // Medium layer
        (1024, 512), // Large layer
        (2048, 1024), // Very large layer
    ];
    
    for (input_size, output_size) in layer_configs.iter() {
        let batch_size = 32;
        let flops = batch_size * input_size * output_size * 2; // Forward pass FLOPS
        group.throughput(Throughput::Elements(flops as u64));
        
        let mut linear = Linear::new(*input_size, *output_size, true);
        let input = rand::<f32>(&[batch_size, *input_size]);
        
        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}", input_size, output_size)),
            &(input_size, output_size),
            |bench, _| {
                bench.iter(|| {
                    linear.forward(&input).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");
    
    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        let input = rand::<f32>(&[*size]);
        
        group.bench_with_input(
            BenchmarkId::new("relu", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    input.relu().unwrap()
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("sigmoid", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    input.sigmoid().unwrap()
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("tanh", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    input.tanh().unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_loss_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_functions");
    
    for batch_size in [32, 64, 128, 256].iter() {
        let num_classes = 1000;
        group.throughput(Throughput::Elements((batch_size * num_classes) as u64));
        
        let predictions = rand::<f32>(&[*batch_size, num_classes]);
        // Create i64 targets directly for cross_entropy compatibility 
        let target_values: Vec<i64> = (0..*batch_size).map(|_| {
            (rand::random::<f32>() * num_classes as f32) as i64
        }).collect();
        let targets = Tensor::from_data(target_values, vec![*batch_size], DeviceType::Cpu);
        
        group.bench_with_input(
            BenchmarkId::new("cross_entropy", batch_size),
            batch_size,
            |bench, _| {
                bench.iter(|| {
                    predictions.cross_entropy(&targets).unwrap()
                });
            },
        );
        
        let binary_predictions = rand::<f32>(&[*batch_size, 1]);
        let binary_targets = rand::<f32>(&[*batch_size, 1]);
        
        group.bench_with_input(
            BenchmarkId::new("mse_loss", batch_size),
            batch_size,
            |bench, _| {
                bench.iter(|| {
                    binary_predictions.mse_loss(&binary_targets).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_conv_layers(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_layers");
    
    // Different convolution configurations
    let conv_configs = [
        (3, 16, 3),   // 3->16 channels, 3x3 kernel
        (16, 32, 3),  // 16->32 channels, 3x3 kernel
        (32, 64, 3),  // 32->64 channels, 3x3 kernel
    ];
    
    for (in_channels, out_channels, kernel_size) in conv_configs.iter() {
        let batch_size = 16;
        let input_height = 32;
        let input_width = 32;
        
        // Approximate FLOPS for convolution
        let output_height = input_height - kernel_size + 1;
        let output_width = input_width - kernel_size + 1;
        let flops = batch_size * out_channels * output_height * output_width * 
                   in_channels * kernel_size * kernel_size;
        group.throughput(Throughput::Elements(flops as u64));
        
        let mut conv = Conv2d::new(*in_channels, *out_channels, (*kernel_size, *kernel_size), None, None, None, None, true);
        let input = rand::<f32>(&[batch_size, *in_channels, input_height, input_width]);
        
        group.bench_with_input(
            BenchmarkId::new("conv2d_forward", 
                           format!("{}x{}_k{}", in_channels, out_channels, kernel_size)),
            &(in_channels, out_channels, kernel_size),
            |bench, _| {
                bench.iter(|| {
                    conv.forward(&input).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_norm");
    
    for num_features in [64, 128, 256, 512].iter() {
        let batch_size = 32;
        let height = 32;
        let width = 32;
        let elements = batch_size * num_features * height * width;
        group.throughput(Throughput::Elements(elements as u64));
        
        let mut batch_norm = BatchNorm2d::new(*num_features, None, None, true, true);
        let input = rand::<f32>(&[batch_size, *num_features, height, width]);
        
        group.bench_with_input(
            BenchmarkId::new("batch_norm_2d", num_features),
            num_features,
            |bench, _| {
                bench.iter(|| {
                    batch_norm.forward(&input).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn bench_optimizer_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_steps");
    
    for param_count in [1000, 10000, 100000, 1000000].iter() {
        group.throughput(Throughput::Elements(*param_count as u64));
        
        // Create dummy parameters and gradients
        let params = rand::<f32>(&[*param_count]);
        let grads = rand::<f32>(&[*param_count]);
        
        // SGD update benchmark
        group.bench_with_input(
            BenchmarkId::new("sgd_step", param_count),
            param_count,
            |bench, _| {
                bench.iter(|| {
                    // Simulate SGD step: param = param - lr * grad
                    let lr = 0.01;
                    let _updated = params.add_scalar(-lr).unwrap().add(&grads.mul_scalar(-lr).unwrap()).unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn bench_full_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_forward_pass");
    
    // Simulate a small CNN forward pass
    let batch_size = 16;
    let input_channels = 3;
    let input_height = 224;
    let input_width = 224;
    
    // Create layers
    let mut conv1 = Conv2d::new(input_channels, 64, (7, 7), None, None, None, None, true); // 7x7 conv
    let mut conv2 = Conv2d::new(64, 128, (3, 3), None, None, None, None, true); // 3x3 conv
    let mut linear1 = Linear::new(128 * 54 * 54, 1000, true); // Approximate size after convolutions
    
    let input = rand::<f32>(&[batch_size, input_channels, input_height, input_width]);
    
    group.bench_function("small_cnn", |bench| {
        bench.iter(|| {
            // Forward pass through the network
            let x = conv1.forward(&input).unwrap();
            let x = x.relu().unwrap();
            let x = conv2.forward(&x).unwrap();
            let x = x.relu().unwrap();
            // Flatten for linear layer (simplified)
            let x_flat = x.view(&[-1, 128 * 54 * 54]).unwrap(); // Approximate flattened size
            let _output = linear1.forward(&x_flat).unwrap();
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_linear_layers,
    bench_activation_functions,
    bench_loss_functions,
    bench_conv_layers,
    bench_batch_norm,
    bench_optimizer_steps,
    bench_full_forward_pass
);

criterion_main!(benches);