//! Auto-Configuration System Demo
//!
//! This example demonstrates the ML-powered auto-configuration system
//! that automatically selects optimal quantization configurations based
//! on tensor characteristics.
//!
//! Run with: cargo run --example auto_config_demo

use torsh_quantization::algorithms::quantize_per_tensor_affine;
use torsh_quantization::auto_config::{AutoConfigurator, ConfigConstraints, ConfigObjective};
use torsh_tensor::creation::tensor_1d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 ToRSh Auto-Configuration System Demo\n");
    println!("========================================\n");

    // Example 1: Automatic configuration for different objectives
    demo_objectives()?;

    // Example 2: Ranked recommendations
    demo_ranked_recommendations()?;

    // Example 3: Constraint-based configuration
    demo_constraints()?;

    // Example 4: Adaptive learning
    demo_adaptive_learning()?;

    // Example 5: Tensor profile analysis
    demo_tensor_profiling()?;

    println!("\n✅ All demonstrations completed successfully!");
    Ok(())
}

fn demo_objectives() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Automatic Configuration for Different Objectives");
    println!("----------------------------------------------------------\n");

    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let tensor = tensor_1d(&test_data)?;

    let objectives = vec![
        (ConfigObjective::MaximumCompression, "Maximum Compression"),
        (ConfigObjective::MaximumAccuracy, "Maximum Accuracy"),
        (ConfigObjective::BalancedQuality, "Balanced Quality"),
        (ConfigObjective::MaximumSpeed, "Maximum Speed"),
        (ConfigObjective::MinimumMemory, "Minimum Memory"),
        (ConfigObjective::EdgeOptimized, "Edge Optimized"),
    ];

    for (objective, name) in objectives {
        let configurator = AutoConfigurator::new(objective);
        let config = configurator.recommend(&tensor, None)?;

        println!("  {} Objective:", name);
        println!("    Scheme: {:?}", config.scheme);
        println!("    Observer: {:?}", config.observer_type);
        println!("    Backend: {:?}", config.backend);
        println!();
    }

    Ok(())
}

fn demo_ranked_recommendations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏆 Demo 2: Top-K Ranked Recommendations");
    println!("--------------------------------------\n");

    let test_data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let tensor = tensor_1d(&test_data)?;

    let configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);
    let ranked = configurator.recommend_ranked(&tensor, 5, None)?;

    println!("  Top 5 recommended configurations:\n");
    for (i, (config, score)) in ranked.iter().enumerate() {
        println!("    {}. Score: {:.2}", i + 1, score);
        println!("       Scheme: {:?}", config.scheme);
        println!("       Observer: {:?}", config.observer_type);
        println!();
    }

    Ok(())
}

fn demo_constraints() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚙️  Demo 3: Constraint-Based Configuration");
    println!("----------------------------------------\n");

    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let tensor = tensor_1d(&test_data)?;

    // Example: Mobile deployment constraints
    let mobile_constraints = ConfigConstraints::new()
        .with_backend(torsh_quantization::QuantBackend::Qnnpack)
        .with_min_bits(8);

    let configurator = AutoConfigurator::new(ConfigObjective::EdgeOptimized);
    let config = configurator.recommend(&tensor, Some(mobile_constraints))?;

    println!("  Mobile Deployment Configuration:");
    println!("    Scheme: {:?}", config.scheme);
    println!("    Backend: {:?} (enforced: QNNPACK)", config.backend);
    println!("    Observer: {:?}", config.observer_type);
    println!();

    Ok(())
}

fn demo_adaptive_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 Demo 4: Adaptive Learning");
    println!("---------------------------\n");

    let mut configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);

    // Simulate multiple quantization runs with feedback
    let test_cases = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![10.0, 20.0, 30.0, 40.0],
        vec![0.1, 0.2, 0.3, 0.4],
    ];

    println!("  Training configurator with feedback:\n");

    for (i, data) in test_cases.iter().enumerate() {
        let tensor = tensor_1d(data)?;
        let config = configurator.recommend(&tensor, None)?;

        // Simulate quantization and calculate error
        let (_quantized, _scale, _zp) = quantize_per_tensor_affine(&tensor, 0.1, 0)?;
        let error = 0.05 * (i as f32 + 1.0); // Simulated error
        let compression = 4.0;
        let speedup = Some(1.5);

        configurator.update_performance(&config, &tensor, error, compression, speedup)?;

        println!(
            "    Run {}: Updated with error={:.3}, compression={:.1}x",
            i + 1,
            error,
            compression
        );
    }

    println!(
        "\n  ✅ Configurator learned from {} examples",
        test_cases.len()
    );
    println!();

    Ok(())
}

fn demo_tensor_profiling() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Demo 5: Tensor Profile Analysis");
    println!("---------------------------------\n");

    // Different tensor distributions
    let test_cases = vec![
        (vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0], "Normal-like"),
        (vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "Sparse"),
        (vec![1.0, 2.0, 3.0, 100.0, 2.0, 3.0], "With Outliers"),
    ];

    for (data, description) in test_cases {
        let tensor = tensor_1d(&data)?;
        let configurator = AutoConfigurator::new(ConfigObjective::BalancedQuality);
        let config = configurator.recommend(&tensor, None)?;

        println!("  {} Distribution:", description);
        println!("    Data: {:?}", &data[..data.len().min(5)]);
        println!("    Recommended Scheme: {:?}", config.scheme);
        println!("    Observer: {:?}", config.observer_type);
        println!();
    }

    Ok(())
}
