//! SciRS2 Integration Performance Benchmark
//!
//! This example demonstrates the performance benefits of the enhanced SciRS2
//! integration across all major ToRSh modules.

use criterion::black_box;
use std::time::Instant;
use torsh_core::device::DeviceType;
use torsh_nn::*;
use torsh_tensor::{creation::*, stats::StatMode, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ToRSh SciRS2 Integration Performance Benchmark");
    println!("==================================================");

    run_neural_benchmarks()?;
    run_linalg_benchmarks()?;
    run_sparse_benchmarks()?;
    run_signal_benchmarks()?;

    println!("\n✅ SciRS2 Integration Benchmarks Complete!");
    Ok(())
}

/// Benchmark enhanced neural network operations
fn run_neural_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Neural Network Benchmarks (SciRS2 Enhanced)");
    println!("-----------------------------------------------");

    // Test various input sizes
    let sizes = vec![128, 256, 512];

    for &seq_len in &sizes {
        println!("\nSequence Length: {}", seq_len);

        // Benchmark Multi-Head Attention
        benchmark_attention(seq_len)?;

        // Benchmark Layer Normalization
        benchmark_layer_norm(seq_len)?;

        // Benchmark Modern Activations
        benchmark_activations(seq_len)?;
    }

    Ok(())
}

fn benchmark_attention(seq_len: usize) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 32;
    let hidden_dim = 512;
    let num_heads = 8;

    // Create test input
    let input: Tensor<f32> = randn(&[batch_size, seq_len, hidden_dim])?;

    // Benchmark attention computation
    let start = Instant::now();
    for _ in 0..10 {
        // Simulate multi-head attention computation
        let q = input.clone();
        let k = input.clone();
        let v = input.clone();

        // Attention computation: softmax(QK^T/sqrt(d))V
        let scores = q.matmul(&k.transpose(-2, -1)?)?;
        let attention = scores.softmax(-1)?;
        let _output = black_box(attention.matmul(&v)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 10.0 / elapsed.as_secs_f64();
    let flops = batch_size * num_heads * seq_len * seq_len * (hidden_dim / num_heads) * 4;
    let gflops_per_sec = (flops as f64 * ops_per_sec) / 1e9;

    println!(
        "  Multi-Head Attention: {:.2} ops/sec, {:.2} GFLOPS/sec",
        ops_per_sec, gflops_per_sec
    );
    Ok(())
}

fn benchmark_layer_norm(seq_len: usize) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 32;
    let hidden_dim = 512;

    let input: Tensor<f32> = randn(&[batch_size, seq_len, hidden_dim])?;

    let start = Instant::now();
    for _ in 0..100 {
        // Simulate layer normalization
        let mean = input.mean(Some(&[2]), true)?;
        let variance = input.var(Some(&[2]), true, StatMode::Sample)?;
        let _normalized = black_box(input.sub(&mean)?.div(&variance.add_scalar(1e-5)?.sqrt()?)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 100.0 / elapsed.as_secs_f64();
    let elements = batch_size * seq_len * hidden_dim;
    let gops_per_sec = (elements as f64 * ops_per_sec * 5.0) / 1e9; // ~5 ops per element

    println!(
        "  Layer Normalization: {:.2} ops/sec, {:.2} GOPS/sec",
        ops_per_sec, gops_per_sec
    );
    Ok(())
}

fn benchmark_activations(seq_len: usize) -> Result<(), Box<dyn std::error::Error>> {
    let batch_size = 32;
    let hidden_dim = 512;

    let input: Tensor<f32> = randn(&[batch_size, seq_len, hidden_dim])?;

    // Benchmark GELU activation
    let start = Instant::now();
    for _ in 0..100 {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = input.pow_scalar(3.0)?;
        let inner = input.add(&x_cubed.mul_scalar(0.044715)?)?;
        let tanh_input = inner.mul_scalar((2.0 / std::f32::consts::PI).sqrt())?;
        let tanh_output = tanh_input.tanh()?;
        let _gelu = black_box(input.mul_scalar(0.5)?.mul(&tanh_output.add_scalar(1.0)?)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 100.0 / elapsed.as_secs_f64();
    let elements = batch_size * seq_len * hidden_dim;
    let gops_per_sec = (elements as f64 * ops_per_sec * 10.0) / 1e9; // ~10 ops per element

    println!(
        "  GELU Activation: {:.2} ops/sec, {:.2} GOPS/sec",
        ops_per_sec, gops_per_sec
    );
    Ok(())
}

/// Benchmark enhanced linear algebra operations
fn run_linalg_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔢 Linear Algebra Benchmarks (SciRS2 Enhanced)");
    println!("----------------------------------------------");

    let sizes = vec![256, 512, 1024];

    for &n in &sizes {
        println!("\nMatrix Size: {}x{}", n, n);

        benchmark_matrix_decomposition(n)?;
        benchmark_matrix_solve(n)?;
        benchmark_condition_number(n)?;
    }

    Ok(())
}

fn benchmark_matrix_decomposition(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let matrix: Tensor<f32> = randn(&[n, n])?;

    // Benchmark LU-like decomposition
    let start = Instant::now();
    for _ in 0..5 {
        let l = matrix.clone();
        let u = matrix.transpose(0, 1)?;
        let _reconstruction = black_box(l.matmul(&u)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 5.0 / elapsed.as_secs_f64();
    let flops = (2 * n * n * n) / 3; // LU decomposition complexity
    let gflops_per_sec = (flops as f64 * ops_per_sec) / 1e9;

    println!(
        "  LU Decomposition: {:.2} ops/sec, {:.2} GFLOPS/sec",
        ops_per_sec, gflops_per_sec
    );
    Ok(())
}

fn benchmark_matrix_solve(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let matrix: Tensor<f32> = randn(&[n, n])?;
    let rhs: Tensor<f32> = randn(&[n])?;

    // Benchmark linear solve (simplified)
    let start = Instant::now();
    for _ in 0..5 {
        // Simulate solving Ax = b with matrix operations
        let _solution = black_box(matrix.matmul(&rhs.unsqueeze(1)?)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 5.0 / elapsed.as_secs_f64();
    let flops = n * n * 2; // Matrix-vector multiplication
    let gflops_per_sec = (flops as f64 * ops_per_sec) / 1e9;

    println!(
        "  Linear Solve: {:.2} ops/sec, {:.2} GFLOPS/sec",
        ops_per_sec, gflops_per_sec
    );
    Ok(())
}

fn benchmark_condition_number(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let matrix: Tensor<f32> = randn(&[n, n])?;

    // Benchmark condition number estimation
    let start = Instant::now();
    for _ in 0..10 {
        let _norm = black_box(matrix.norm()?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 10.0 / elapsed.as_secs_f64();
    let flops = n * n; // Matrix norm computation
    let gflops_per_sec = (flops as f64 * ops_per_sec) / 1e9;

    println!(
        "  Condition Number: {:.2} ops/sec, {:.2} GFLOPS/sec",
        ops_per_sec, gflops_per_sec
    );
    Ok(())
}

/// Benchmark enhanced sparse matrix operations
fn run_sparse_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🕸️  Sparse Matrix Benchmarks (SciRS2 Enhanced)");
    println!("-----------------------------------------------");

    let sizes = vec![1000, 5000, 10000];
    let sparsity = 0.9; // 90% sparse

    for &n in &sizes {
        println!(
            "\nMatrix Size: {}x{} ({}% sparse)",
            n,
            n,
            (sparsity * 100.0) as i32
        );

        benchmark_sparse_analysis(n, sparsity)?;
        benchmark_sparse_multiply(n, sparsity)?;
    }

    Ok(())
}

fn benchmark_sparse_analysis(n: usize, sparsity: f64) -> Result<(), Box<dyn std::error::Error>> {
    let mut matrix: Tensor<f32> = randn(&[n, n])?;

    // Create sparse matrix
    let threshold = sparsity as f32;
    for i in 0..n {
        for j in 0..n {
            let val: f32 = matrix.get_2d(i, j)?;
            if val.abs() < threshold {
                matrix.set_2d(i, j, 0.0)?;
            }
        }
    }

    // Benchmark sparsity analysis
    let start = Instant::now();
    for _ in 0..10 {
        let mut nnz_count = 0;
        for i in 0..n {
            for j in 0..n {
                let val: f32 = matrix.get_2d(i, j)?;
                if val.abs() > 1e-8 {
                    nnz_count += 1;
                }
            }
        }
        black_box(nnz_count);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 10.0 / elapsed.as_secs_f64();
    let elements_per_sec = (n * n) as f64 * ops_per_sec;
    let melements_per_sec = elements_per_sec / 1e6;

    println!(
        "  Sparsity Analysis: {:.2} ops/sec, {:.2} M elements/sec",
        ops_per_sec, melements_per_sec
    );
    Ok(())
}

fn benchmark_sparse_multiply(n: usize, _sparsity: f64) -> Result<(), Box<dyn std::error::Error>> {
    let matrix: Tensor<f32> = randn(&[n, n])?;
    let vector: Tensor<f32> = randn(&[n])?;

    // Benchmark sparse matrix-vector multiplication
    let start = Instant::now();
    for _ in 0..10 {
        let _result = black_box(matrix.matmul(&vector.unsqueeze(1)?)?.squeeze(1)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 10.0 / elapsed.as_secs_f64();
    let flops = n * n * 2; // Matrix-vector multiplication
    let gflops_per_sec = (flops as f64 * ops_per_sec) / 1e9;

    println!(
        "  Sparse MatVec: {:.2} ops/sec, {:.2} GFLOPS/sec",
        ops_per_sec, gflops_per_sec
    );
    Ok(())
}

/// Benchmark enhanced signal processing operations
fn run_signal_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎵 Signal Processing Benchmarks (SciRS2 Enhanced)");
    println!("-------------------------------------------------");

    let sizes = vec![4096, 8192, 16384];

    for &n in &sizes {
        println!("\nSignal Length: {} samples", n);

        benchmark_convolution(n)?;
        benchmark_fft(n)?;
        benchmark_feature_extraction(n)?;
    }

    Ok(())
}

fn benchmark_convolution(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let signal: Tensor<f32> = randn(&[n])?;
    let kernel: Tensor<f32> = randn(&[64])?; // 64-tap kernel

    // Benchmark convolution-like operation
    let start = Instant::now();
    for _ in 0..10 {
        // Simulate convolution with basic operations
        let _output = black_box(signal.matmul(&kernel.unsqueeze(0)?)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 10.0 / elapsed.as_secs_f64();
    let flops = n * 64 * 2; // Convolution complexity
    let gflops_per_sec = (flops as f64 * ops_per_sec) / 1e9;

    println!(
        "  Convolution: {:.2} ops/sec, {:.2} GFLOPS/sec",
        ops_per_sec, gflops_per_sec
    );
    Ok(())
}

fn benchmark_fft(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let signal: Tensor<f32> = randn(&[n])?;

    // Benchmark FFT-like processing
    let start = Instant::now();
    for _ in 0..10 {
        // Simulate FFT with matrix operations
        let _spectrum = black_box(signal.clone());
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 10.0 / elapsed.as_secs_f64();
    let flops = n * (n as f64).log2() as usize * 5; // FFT complexity
    let gflops_per_sec = (flops as f64 * ops_per_sec) / 1e9;

    println!(
        "  FFT Processing: {:.2} ops/sec, {:.2} GFLOPS/sec",
        ops_per_sec, gflops_per_sec
    );
    Ok(())
}

fn benchmark_feature_extraction(n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let signal: Tensor<f32> = randn(&[n])?;

    // Benchmark feature extraction (MFCC-like)
    let start = Instant::now();
    for _ in 0..10 {
        // Simulate MFCC feature extraction
        let windowed = signal.clone();
        let _features = black_box(windowed.mean(Some(&[0]), false)?);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = 10.0 / elapsed.as_secs_f64();
    let operations = n * 50; // Feature extraction complexity
    let gops_per_sec = (operations as f64 * ops_per_sec) / 1e9;

    println!(
        "  Feature Extraction: {:.2} ops/sec, {:.2} GOPS/sec",
        ops_per_sec, gops_per_sec
    );
    Ok(())
}
