//! Practical Optimizer Selection Guide
//!
//! This example demonstrates the most commonly used optimizers and when to use them.
//! It focuses on practical, production-ready optimizers with simple APIs.
//!
//! Run with: cargo run --example practical_optimizer_guide

use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_optim::prelude::*;
use torsh_tensor::Tensor;

fn main() -> Result<()> {
    println!("📚 ToRSh Practical Optimizer Guide\n");
    println!("{}", "=".repeat(70));

    // ========================================================================
    // 1. SGD - THE CLASSIC
    // ========================================================================
    println!("\n✅ 1. SGD (Stochastic Gradient Descent)");
    println!("{}", "-".repeat(70));
    println!("   Best for: When you need simplicity and reliability");
    println!("   Memory: Very low (just momentum buffer)");
    println!("   Speed: Fast per-step");
    println!("   Pros: Stable, well-understood, works everywhere");
    println!("   Cons: Needs careful LR tuning");

    {
        let param = Arc::new(RwLock::new(Tensor::scalar(5.0)?));
        let mut optimizer = SGD::new(
            vec![param.clone()],
            0.1,       // lr
            Some(0.9), // momentum (highly recommended!)
            None,      // dampening
            None,      // weight_decay
            false,     // nesterov
        );

        println!("\n   Example: Minimizing f(x) = x^2");
        for step in 0..10 {
            let grad = { param.read().mul_scalar(2.0)? };
            param.write().set_grad(Some(grad));
            optimizer.step().expect("Optimizer step failed");
            let val = param.read().to_vec()?[0];
            if step % 3 == 0 {
                println!("   Step {}: x = {:.4}, loss = {:.4}", step, val, val * val);
            }
            optimizer.zero_grad();
        }
        let final_val = param.read().to_vec()?[0];
        println!(
            "   Final: x = {:.4}, loss = {:.4}",
            final_val,
            final_val * final_val
        );
    }

    // ========================================================================
    // 2. ADAM - THE DEFAULT CHOICE
    // ========================================================================
    println!("\n\n✅ 2. Adam (Adaptive Moment Estimation)");
    println!("{}", "-".repeat(70));
    println!("   Best for: Almost everything! Default choice for most projects");
    println!("   Memory: Medium (momentum + squared gradients)");
    println!("   Speed: Medium");
    println!("   Pros: Works out-of-the-box, adaptive LR, minimal tuning");
    println!("   Cons: Can generalize slightly worse than SGD");

    {
        let param = Arc::new(RwLock::new(Tensor::scalar(5.0)?));
        let mut optimizer = Adam::new(
            vec![param.clone()],
            Some(0.01),         // lr (typical: 0.001-0.01)
            Some((0.9, 0.999)), // betas (usually keep default)
            None,               // eps
            None,               // weight_decay
            false,              // amsgrad
        );

        println!("\n   Example: Same f(x) = x^2");
        for step in 0..10 {
            let grad = { param.read().mul_scalar(2.0)? };
            param.write().set_grad(Some(grad));
            optimizer.step().expect("Optimizer step failed");
            let val = param.read().to_vec()?[0];
            if step % 3 == 0 {
                println!("   Step {}: x = {:.4}, loss = {:.4}", step, val, val * val);
            }
            optimizer.zero_grad();
        }
        let final_val = param.read().to_vec()?[0];
        println!(
            "   Final: x = {:.4}, loss = {:.4}",
            final_val,
            final_val * final_val
        );
    }

    // ========================================================================
    // 3. LION - MODERN & MEMORY-EFFICIENT (2023)
    // ========================================================================
    println!("\n\n✅ 3. Lion (Google, 2023)");
    println!("{}", "-".repeat(70));
    println!("   Best for: Large models, memory constraints");
    println!("   Memory: Low (only momentum, NO squared gradients!)");
    println!("   Speed: Fast");
    println!("   Pros: 10x less memory than Adam, excellent performance");
    println!("   Cons: Use 10x smaller LR than Adam");

    {
        let param = Arc::new(RwLock::new(Tensor::scalar(5.0)?));
        let mut optimizer = Lion::builder()
            .params(vec![param.clone()])
            .lr(0.001) // Note: 10x smaller than Adam!
            .beta1(0.9)
            .beta2(0.99)
            .build();

        println!("\n   Example: Same f(x) = x^2");
        for step in 0..10 {
            let grad = { param.read().mul_scalar(2.0)? };
            param.write().set_grad(Some(grad));
            optimizer.step().expect("Optimizer step failed");
            let val = param.read().to_vec()?[0];
            if step % 3 == 0 {
                println!("   Step {}: x = {:.4}, loss = {:.4}", step, val, val * val);
            }
            optimizer.zero_grad();
        }
        let final_val = param.read().to_vec()?[0];
        println!(
            "   Final: x = {:.4}, loss = {:.4}",
            final_val,
            final_val * final_val
        );
    }

    // ========================================================================
    // 4. PRODIGY - NO LR TUNING! (2024)
    // ========================================================================
    println!("\n\n✅ 4. Prodigy (2024) - Auto-Adaptive LR");
    println!("{}", "-".repeat(70));
    println!("   Best for: Research, prototyping, when you don't know the LR");
    println!("   Memory: Medium (like Adam)");
    println!("   Speed: Medium");
    println!("   Pros: Just use lr=1.0, it auto-adapts! No tuning needed!");
    println!("   Cons: Newer, less tested in production");

    {
        let param = Arc::new(RwLock::new(Tensor::scalar(5.0)?));
        let mut optimizer = Prodigy::builder()
            .params(vec![param.clone()])
            .lr(1.0) // Just use 1.0 for most problems!
            .beta1(0.9)
            .beta2(0.999)
            .build();

        println!("\n   Example: Same f(x) = x^2");
        for step in 0..10 {
            let grad = { param.read().mul_scalar(2.0)? };
            param.write().set_grad(Some(grad));
            optimizer.step().expect("Optimizer step failed");
            let val = param.read().to_vec()?[0];
            if step % 3 == 0 {
                println!("   Step {}: x = {:.4}, loss = {:.4}", step, val, val * val);
            }
            optimizer.zero_grad();
        }
        let final_val = param.read().to_vec()?[0];
        println!(
            "   Final: x = {:.4}, loss = {:.4}",
            final_val,
            final_val * final_val
        );
    }

    // ========================================================================
    // QUICK REFERENCE GUIDE
    // ========================================================================
    println!("\n\n{}", "=".repeat(70));
    println!("📝 QUICK SELECTION GUIDE");
    println!("{}", "=".repeat(70));

    println!("\n🎯 Use SGD when:");
    println!("   • You need maximum control and stability");
    println!("   • Training CNNs with known good hyperparameters");
    println!("   • You have time to tune learning rate");
    println!("   • Memory is extremely limited");

    println!("\n🎯 Use Adam when:");
    println!("   • Starting a new project (default choice)");
    println!("   • Training transformers, RNNs, or most architectures");
    println!("   • You want something that \"just works\"");
    println!("   • You don't have time for extensive hyperparameter tuning");

    println!("\n🎯 Use Lion when:");
    println!("   • Training very large models (LLMs, large vision models)");
    println!("   • Memory is a concern");
    println!("   • You want Adam-like performance with less memory");
    println!("   • Your model is well-suited to sign-based updates");

    println!("\n🎯 Use Prodigy when:");
    println!("   • Doing research or rapid prototyping");
    println!("   • You don't know what learning rate to use");
    println!("   • You want to skip hyperparameter tuning");
    println!("   • You're okay with slightly experimental approaches");

    println!("\n\n{}", "=".repeat(70));
    println!("💡 PRO TIPS");
    println!("{}", "=".repeat(70));

    println!("\n1. Learning Rate Guidelines:");
    println!("   • SGD: 0.1 (with momentum), 0.01 (without)");
    println!("   • Adam: 0.001 (standard), 0.0001 (fine-tuning)");
    println!("   • Lion: 0.0001 (10x smaller than Adam!)");
    println!("   • Prodigy: 1.0 (yes, really!)");

    println!("\n2. When to use momentum (for SGD):");
    println!("   • Almost always! Use 0.9 as default");
    println!("   • Helps escape local minima and speeds convergence");

    println!("\n3. Weight decay:");
    println!("   • Computer Vision: 1e-4 to 1e-5");
    println!("   • NLP: 0.01 to 0.1");
    println!("   • Small datasets: higher values (more regularization)");

    println!("\n4. Combining with LR Schedulers:");
    println!("   • SGD: Use OneCycleLR or CosineAnnealingLR");
    println!("   • Adam: Use CosineAnnealingLR or ReduceLROnPlateau");
    println!("   • Lion: Use linear warmup + cosine decay");
    println!("   • Prodigy: NO scheduler needed!");

    println!("\n\n{}", "=".repeat(70));
    println!("✅ Guide complete! Check other examples for more optimizers.");
    println!("📚 See: modern_optimizers_comparison.rs for Sophia & Schedule-Free");
    println!("{}", "=".repeat(70));

    Ok(())
}
