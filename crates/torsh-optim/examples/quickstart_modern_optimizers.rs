//! Quick Start Guide for Modern Optimizers
//!
//! This example shows the simplest way to use each modern optimizer.
//!
//! Run with: cargo run --example quickstart_modern_optimizers

use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_optim::prelude::{Lion, Optimizer, Prodigy, ScheduleFreeAdamW, Sophia};
use torsh_tensor::creation::randn;

fn main() -> Result<()> {
    println!("=== Quick Start: Modern Optimizers ===\n");

    // Create example parameters
    let param1 = Arc::new(RwLock::new(randn::<f32>(&[128, 256])?));
    let param2 = Arc::new(RwLock::new(randn::<f32>(&[256, 10])?));
    let params = vec![param1.clone(), param2.clone()];

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 1. Lion - Simple and Memory Efficient
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("1️⃣  Lion Optimizer (Memory-Efficient)");
    println!("   Usage: Use lr that's 10x smaller than Adam");
    println!();

    // Basic usage
    let mut lion = Lion::new(params.clone(), 1e-4, 0.9, 0.99, 0.01);

    // Or use builder pattern
    let mut lion_builder = Lion::builder()
        .params(params.clone())
        .lr(1e-4)
        .beta1(0.9)
        .beta2(0.99)
        .weight_decay(0.01)
        .build();

    println!("   ✓ Created Lion optimizer");
    println!("   ✓ Learning rate: {}", lion.get_lr()[0]);
    println!();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 2. Sophia - For LLM Training
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("2️⃣  Sophia Optimizer (LLM-Optimized)");
    println!("   Usage: 2-3x speedup for transformer training");
    println!();

    let _sophia = Sophia::builder()
        .params(params.clone())
        .lr(5e-4) // Typical for transformers
        .beta1(0.96)
        .beta2(0.99)
        .gamma(1.0) // Clipping threshold
        .hessian_update_interval(10) // Update Hessian every 10 steps
        .weight_decay(0.1)
        .build();

    println!("   ✓ Created Sophia optimizer");
    println!("   ✓ Hessian updates every 10 steps");
    println!();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 3. Schedule-Free AdamW - No Schedule Needed!
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("3️⃣  Schedule-Free AdamW (No Schedule!)");
    println!("   Usage: Set constant LR, no warmup/decay needed");
    println!();

    let mut schedule_free = ScheduleFreeAdamW::builder()
        .params(params.clone())
        .lr(1e-3) // Constant learning rate!
        .beta1(0.9)
        .beta2(0.999)
        .c(0.05) // Averaging coefficient
        .weight_decay(0.01)
        .build();

    // Important: Switch between train/eval modes
    schedule_free.train(); // For training
    println!("   ✓ Created Schedule-Free optimizer");
    println!("   ✓ In training mode: {}", schedule_free.is_training());
    println!("   ℹ️  Use .eval() during evaluation");
    println!();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 4. Prodigy - Zero Hyperparameter Tuning!
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("4️⃣  Prodigy Optimizer (Zero Tuning!)");
    println!("   Usage: Just use lr=1.0, it adapts automatically");
    println!();

    let prodigy = Prodigy::builder()
        .params(params.clone())
        .lr(1.0) // Yes, really! Just use 1.0
        .beta1(0.9)
        .beta2(0.999)
        .weight_decay(0.0)
        .build();

    println!("   ✓ Created Prodigy optimizer");
    println!(
        "   ✓ Learning rate: {} (will adapt automatically!)",
        prodigy.get_lr()[0]
    );
    println!("   ✓ Initial d scale: {:.2e}", prodigy.get_d());
    println!();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Typical Training Loop Pattern
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📚 Typical Training Loop:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("for epoch in 0..num_epochs {{");
    println!("    for batch in dataloader {{");
    println!("        // 1. Forward pass");
    println!("        let output = model.forward(&batch.data);");
    println!("        let loss = criterion(&output, &batch.labels);");
    println!();
    println!("        // 2. Backward pass (computes gradients)");
    println!("        loss.backward();");
    println!();
    println!("        // 3. Optimizer step");
    println!("        optimizer.step()?;");
    println!();
    println!("        // 4. Zero gradients");
    println!("        optimizer.zero_grad();");
    println!("    }}");
    println!("}}");
    println!();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Special Features
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✨ Special Features:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("🔄 State Dict Save/Load:");
    let state = lion.state_dict()?;
    println!("   Saved optimizer state: {:?}", state.optimizer_type);
    lion_builder.load_state_dict(state)?;
    println!("   ✓ Loaded state successfully");
    println!();

    println!("⚙️  Dynamic Learning Rate:");
    lion.set_lr(2e-4);
    println!("   ✓ Changed learning rate to: {}", lion.get_lr()[0]);
    println!();

    println!("📊 Prodigy Adaptation Info:");
    println!(
        "   Current effective LR: {:.6e}",
        prodigy.get_effective_lr()
    );
    println!("   D scale factor: {:.6e}", prodigy.get_d());
    println!();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Recommendations
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("💡 Quick Recommendations:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("🚀 Starting a new project?");
    println!("   → Try Prodigy first (lr=1.0, zero tuning)");
    println!();
    println!("🏃 Need something fast and simple?");
    println!("   → Use Lion (lr=1e-4, memory efficient)");
    println!();
    println!("🤖 Training large language models?");
    println!("   → Use Sophia (lr=5e-4, 2-3x speedup)");
    println!();
    println!("😌 Don't want to tune LR schedules?");
    println!("   → Use Schedule-Free AdamW (lr=1e-3)");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Quick Start Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}
