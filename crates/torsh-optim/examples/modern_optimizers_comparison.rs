//! Comparison of Modern Optimizers (2023-2024)
//!
//! This example demonstrates the usage of cutting-edge optimizers:
//! - Lion: Memory-efficient evolved sign momentum
//! - Sophia: Second-order optimization for LLMs
//! - Schedule-Free AdamW: No LR schedule tuning
//! - Prodigy: Automatic LR adaptation
//!
//! Run with: cargo run --example modern_optimizers_comparison

use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_optim::prelude::{Lion, Optimizer, Prodigy, ScheduleFreeAdamW, Sophia};
use torsh_tensor::Tensor;

/// Simulate a simple optimization problem for demonstration
fn optimize_with_optimizer<O: Optimizer>(
    mut optimizer: O,
    param: Arc<RwLock<Tensor>>,
    name: &str,
    steps: usize,
) -> Result<f32> {
    println!("\n=== Optimizing with {} ===", name);

    for step in 0..steps {
        // Simulate gradient computation: gradient = 2 * param (quadratic function)
        let grad = {
            let p = param.read();
            p.mul_scalar(2.0)?
        };

        param.write().set_grad(Some(grad));

        // Optimization step
        optimizer.step().expect("Optimization step failed");

        // Print progress every 20 steps
        if step % 20 == 0 {
            let loss = {
                let p = param.read();
                let p_val = p.to_vec()?[0];
                p_val * p_val // Loss = x^2
            };
            println!("Step {}: Loss = {:.6}", step, loss);
        }

        optimizer.zero_grad();
    }

    // Final loss
    let final_loss = {
        let p = param.read();
        let p_val = p.to_vec()?[0];
        p_val * p_val
    };

    println!("Final Loss: {:.6}", final_loss);
    Ok(final_loss)
}

fn main() -> Result<()> {
    println!("=== Modern Optimizers Comparison ===");
    println!("Solving a simple quadratic minimization problem: f(x) = x^2");
    println!("Starting point: x = 10.0, Target: x = 0.0\n");

    let steps = 100;

    // 1. Lion Optimizer (Google Research, 2023)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🦁 LION OPTIMIZER");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Features:");
    println!("  • Memory-efficient (only stores momentum)");
    println!("  • Sign-based updates");
    println!("  • Typical LR: 1e-4 (10x smaller than Adam)");
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(10.0)?));
        let params = vec![param.clone()];
        let optimizer = Lion::new(params, 1e-2, 0.9, 0.99, 0.0);
        optimize_with_optimizer(optimizer, param, "Lion", steps)?;
    }

    // 2. Sophia Optimizer (2023)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎓 SOPHIA OPTIMIZER");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Features:");
    println!("  • Second-order with Hessian diagonal");
    println!("  • 2-3x speedup for LLM training");
    println!("  • Clipped updates for stability");
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(10.0)?));
        let params = vec![param.clone()];
        let optimizer = Sophia::new(params, 5e-2, 0.96, 0.99, 1.0, 10, 0.0);
        optimize_with_optimizer(optimizer, param, "Sophia", steps)?;
    }

    // 3. Schedule-Free AdamW (2024)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📅 SCHEDULE-FREE AdamW");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Features:");
    println!("  • NO learning rate schedule needed!");
    println!("  • Fast/slow parameter sequences");
    println!("  • Train/eval mode switching");
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(10.0)?));
        let params = vec![param.clone()];
        let mut optimizer = ScheduleFreeAdamW::new(params, 1e-1, 0.9, 0.999, 0.05, 0.0);
        optimizer.train(); // Ensure in training mode
        optimize_with_optimizer(optimizer, param, "Schedule-Free AdamW", steps)?;
    }

    // 4. Prodigy Optimizer (2024)
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔮 PRODIGY OPTIMIZER");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Features:");
    println!("  • ZERO learning rate tuning!");
    println!("  • Just use lr=1.0 for everything");
    println!("  • Automatic adaptation");
    {
        let param = Arc::new(RwLock::new(Tensor::scalar(10.0)?));
        let params = vec![param.clone()];
        let optimizer = Prodigy::new(params, 1.0, 0.9, 0.999, 0.0);
        println!("Initial learning rate scale (d): {:.2e}", optimizer.get_d());
        optimize_with_optimizer(optimizer, param, "Prodigy", steps)?;
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✨ COMPARISON SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\nWhen to use each optimizer:");
    println!("\n🦁 Lion:");
    println!("  • When memory is constrained");
    println!("  • For large vision/language models");
    println!("  • When Adam/AdamW is working but you want better efficiency");
    println!("\n🎓 Sophia:");
    println!("  • For large language model pre-training");
    println!("  • When training transformers at scale");
    println!("  • When you can afford periodic Hessian updates");
    println!("\n📅 Schedule-Free AdamW:");
    println!("  • When you don't want to tune LR schedules");
    println!("  • For general-purpose deep learning");
    println!("  • When you want simplicity without sacrificing performance");
    println!("\n🔮 Prodigy:");
    println!("  • When you're unsure about learning rate");
    println!("  • For rapid prototyping and experiments");
    println!("  • When you want zero hyperparameter tuning");

    Ok(())
}
