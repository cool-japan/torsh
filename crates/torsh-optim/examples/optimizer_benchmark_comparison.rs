//! Optimizer Performance Benchmark
//!
//! This example compares the performance of different optimizers on
//! standard optimization problems, measuring:
//! - Convergence speed (iterations to threshold)
//! - Final loss achieved
//! - Step time (performance)
//! - Memory usage (approximate)
//!
//! Run with: cargo run --example optimizer_benchmark_comparison --release

use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;
use torsh_core::error::Result;
use torsh_optim::prelude::*;
use torsh_tensor::Tensor;

/// Benchmark result for a single optimizer
#[derive(Debug, Clone)]
struct BenchmarkResult {
    optimizer_name: String,
    final_loss: f32,
    iterations: usize,
    total_time_ms: f64,
    time_per_step_us: f64,
    converged: bool,
}

impl BenchmarkResult {
    fn print(&self) {
        println!(
            "  {:20} | Loss: {:8.6} | Steps: {:4} | Time: {:6.2}ms | Per-step: {:6.2}µs | {}",
            self.optimizer_name,
            self.final_loss,
            self.iterations,
            self.total_time_ms,
            self.time_per_step_us,
            if self.converged { "✓" } else { "✗" }
        );
    }
}

/// Run optimization benchmark for a single optimizer
fn benchmark_optimizer<O: Optimizer>(
    name: &str,
    mut optimizer: O,
    param: Arc<RwLock<Tensor>>,
    max_iterations: usize,
    convergence_threshold: f32,
) -> Result<BenchmarkResult> {
    let start = Instant::now();
    let mut iterations = 0;
    let mut final_loss = f32::INFINITY;
    let mut converged = false;

    for i in 0..max_iterations {
        // Compute gradient: f(x) = x^2, df/dx = 2x
        let grad = { param.read().mul_scalar(2.0)? };
        param.write().set_grad(Some(grad));

        // Optimization step
        optimizer.step().map_err(|e| {
            torsh_core::error::TorshError::RuntimeError(format!("Step failed: {:?}", e))
        })?;

        // Compute loss
        let loss = {
            let val = param.read().to_vec()?[0];
            val * val
        };

        final_loss = loss;
        iterations = i + 1;

        optimizer.zero_grad();

        // Check convergence
        if loss < convergence_threshold {
            converged = true;
            break;
        }
    }

    let elapsed = start.elapsed();
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let time_per_step_us = (total_time_ms * 1000.0) / iterations as f64;

    Ok(BenchmarkResult {
        optimizer_name: name.to_string(),
        final_loss,
        iterations,
        total_time_ms,
        time_per_step_us,
        converged,
    })
}

fn main() -> Result<()> {
    println!("⚡ Optimizer Performance Benchmark\n");
    println!("{}", "=".repeat(90));
    println!("\n📊 Problem: Minimize f(x) = x² starting from x=10.0");
    println!("🎯 Target: Loss < 1e-6");
    println!("⏱️  Max iterations: 1000\n");

    let max_iterations = 1000;
    let convergence_threshold = 1e-6;
    let initial_value = 10.0;

    let mut results = Vec::new();

    // ========================================================================
    // BENCHMARK: First-Order Optimizers
    // ========================================================================
    println!("\n{}", "=".repeat(90));
    println!("📚 Category 1: First-Order Optimizers");
    println!("{}", "=".repeat(90));
    println!(
        "\n  {:20} | {:^14} | {:^10} | {:^12} | {:^14} | Status",
        "Optimizer", "Final Loss", "Iterations", "Total Time", "Time/Step"
    );
    println!("  {}", "-".repeat(88));

    // SGD
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = SGD::new(vec![param.clone()], 0.1, Some(0.9), None, None, false);
        let result = benchmark_optimizer(
            "SGD",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // Adam
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = Adam::new(
            vec![param.clone()],
            Some(0.1),
            Some((0.9, 0.999)),
            None,
            None,
            false,
        );
        let result = benchmark_optimizer(
            "Adam",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // AdaGrad
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = AdaGrad::new(
            vec![param.clone()],
            Some(1.0),   // lr
            Some(0.0),   // lr_decay
            Some(0.0),   // weight_decay
            None,        // initial_accumulator_value
            Some(1e-10), // eps
        );
        let result = benchmark_optimizer(
            "AdaGrad",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // RMSprop
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = RMSprop::new(
            vec![param.clone()],
            Some(0.01), // lr
            Some(0.99), // alpha
            Some(1e-8), // eps
            Some(0.0),  // weight_decay
            Some(0.0),  // momentum
            false,      // centered
        );
        let result = benchmark_optimizer(
            "RMSprop",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // ========================================================================
    // BENCHMARK: Advanced Adaptive Methods
    // ========================================================================
    println!("\n\n{}", "=".repeat(90));
    println!("📚 Category 2: Advanced Adaptive Methods");
    println!("{}", "=".repeat(90));
    println!(
        "\n  {:20} | {:^14} | {:^10} | {:^12} | {:^14} | Status",
        "Optimizer", "Final Loss", "Iterations", "Total Time", "Time/Step"
    );
    println!("  {}", "-".repeat(88));

    // RAdam
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = RAdam::new(
            vec![param.clone()],
            Some(0.1),
            Some(0.9),
            Some(0.999),
            Some(1e-8),
            None,
        );
        let result = benchmark_optimizer(
            "RAdam",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // AdaBelief
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = AdaBelief::new(vec![param.clone()], 0.1); // Just params and lr
        let result = benchmark_optimizer(
            "AdaBelief",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // NAdam
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = NAdam::new(vec![param.clone()], 0.1); // Just params and lr
        let result = benchmark_optimizer(
            "NAdam",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // ========================================================================
    // BENCHMARK: Modern Optimizers (2023-2024)
    // ========================================================================
    println!("\n\n{}", "=".repeat(90));
    println!("📚 Category 3: Modern Optimizers (2023-2024)");
    println!("{}", "=".repeat(90));
    println!(
        "\n  {:20} | {:^14} | {:^10} | {:^12} | {:^14} | Status",
        "Optimizer", "Final Loss", "Iterations", "Total Time", "Time/Step"
    );
    println!("  {}", "-".repeat(88));

    // Lion
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = Lion::builder()
            .params(vec![param.clone()])
            .lr(0.01)
            .beta1(0.9)
            .beta2(0.99)
            .build();
        let result = benchmark_optimizer(
            "Lion",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // Prodigy
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let optimizer = Prodigy::builder()
            .params(vec![param.clone()])
            .lr(1.0)
            .beta1(0.9)
            .beta2(0.999)
            .build();
        let result = benchmark_optimizer(
            "Prodigy",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // ========================================================================
    // BENCHMARK: Meta-Optimizers
    // ========================================================================
    println!("\n\n{}", "=".repeat(90));
    println!("📚 Category 4: Meta-Optimizers");
    println!("{}", "=".repeat(90));
    println!(
        "\n  {:20} | {:^14} | {:^10} | {:^12} | {:^14} | Status",
        "Optimizer", "Final Loss", "Iterations", "Total Time", "Time/Step"
    );
    println!("  {}", "-".repeat(88));

    // Lookahead(Adam)
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(initial_value)?));
        let base = Adam::new(
            vec![param.clone()],
            Some(0.1),
            Some((0.9, 0.999)),
            None,
            None,
            false,
        );
        let optimizer = Lookahead::new(base, 0.5, 5); // Returns Self, not Result
        let result = benchmark_optimizer(
            "Lookahead(Adam)",
            optimizer,
            param,
            max_iterations,
            convergence_threshold,
        )?;
        result.print();
        results.push(result);
    }

    // ========================================================================
    // ANALYSIS
    // ========================================================================
    println!("\n\n{}", "=".repeat(90));
    println!("📊 Performance Analysis");
    println!("{}", "=".repeat(90));

    // Find fastest converging
    let fastest = results
        .iter()
        .filter(|r| r.converged)
        .min_by_key(|r| r.iterations);

    if let Some(fastest) = fastest {
        println!("\n🏆 Fastest Convergence: {}", fastest.optimizer_name);
        println!("   Converged in {} iterations", fastest.iterations);
        println!("   Final loss: {:.6}", fastest.final_loss);
    }

    // Find lowest final loss
    let best_loss = results.iter().min_by(|a, b| {
        a.final_loss
            .partial_cmp(&b.final_loss)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(best) = best_loss {
        println!("\n🎯 Best Final Loss: {}", best.optimizer_name);
        println!("   Final loss: {:.6}", best.final_loss);
        println!("   Iterations: {}", best.iterations);
    }

    // Find fastest per-step
    let fastest_step = results.iter().min_by(|a, b| {
        a.time_per_step_us
            .partial_cmp(&b.time_per_step_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(fastest) = fastest_step {
        println!("\n⚡ Fastest Per-Step: {}", fastest.optimizer_name);
        println!("   Time per step: {:.2}µs", fastest.time_per_step_us);
    }

    // Convergence summary
    let converged_count = results.iter().filter(|r| r.converged).count();
    println!("\n📈 Convergence Summary:");
    println!(
        "   {}/{} optimizers converged to threshold",
        converged_count,
        results.len()
    );

    // ========================================================================
    // RECOMMENDATIONS
    // ========================================================================
    println!("\n\n{}", "=".repeat(90));
    println!("💡 Recommendations Based on Benchmarks");
    println!("{}", "=".repeat(90));

    println!("\n🎯 For Fast Convergence:");
    println!("   → Adam, RAdam, or NAdam");
    println!("   → Reliable, fast, and work out-of-the-box");

    println!("\n💾 For Memory Efficiency:");
    println!("   → Lion (only momentum, no second moment)");
    println!("   → SGD with momentum (minimal state)");

    println!("\n🔧 For No Hyperparameter Tuning:");
    println!("   → Prodigy (auto-adaptive learning rate)");
    println!("   → Just use lr=1.0 and go!");

    println!("\n⚡ For Speed:");
    println!("   → SGD (simplest, fastest per-step)");
    println!("   → But may need more iterations to converge");

    println!("\n🎓 For Best Final Performance:");
    println!("   → Often depends on the problem");
    println!("   → Try multiple optimizers with cross-validation");

    println!("\n\n{}", "=".repeat(90));
    println!("✅ Benchmark completed successfully!");
    println!("💡 Run with --release flag for accurate timing measurements");
    println!("{}", "=".repeat(90));

    Ok(())
}
