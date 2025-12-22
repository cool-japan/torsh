//! Comprehensive demonstration of uncertainty quantification in ToRSh
//!
//! This example shows how to use various uncertainty estimation methods:
//! - MC Dropout uncertainty
//! - Deep Ensemble uncertainty
//! - Bayesian uncertainty
//! - Epistemic vs Aleatoric decomposition
//!
//! Run with: cargo run --example uncertainty_quantification_demo

use torsh_core::device::DeviceType;
use torsh_metrics::{
    BayesianUncertainty, EnsembleUncertainty, MCDropoutUncertainty, UncertaintyDecomposition,
};
use torsh_tensor::creation::from_vec;

fn main() {
    println!("==============================================");
    println!("ToRSh Uncertainty Quantification Demo");
    println!("==============================================\n");

    // Demonstrate MC Dropout uncertainty
    mc_dropout_demo();

    println!("\n");

    // Demonstrate Ensemble uncertainty
    ensemble_uncertainty_demo();

    println!("\n");

    // Demonstrate Bayesian uncertainty
    bayesian_uncertainty_demo();

    println!("\n");

    // Demonstrate uncertainty decomposition comparison
    uncertainty_comparison_demo();

    println!("\n==============================================");
    println!("Demo completed successfully!");
    println!("==============================================");
}

/// Demonstrate MC Dropout uncertainty estimation
fn mc_dropout_demo() {
    println!("📊 MC Dropout Uncertainty Estimation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let n_samples = 10;
    let mc_dropout = MCDropoutUncertainty::new(n_samples);

    // Simulate MC dropout predictions (10 forward passes with dropout enabled)
    // In practice, these would come from your model with dropout enabled at test time
    let mut mc_predictions = Vec::new();

    println!("Generating {} MC dropout samples...", n_samples);

    for i in 0..n_samples {
        // Simulate varying predictions due to dropout
        let noise = (i as f32) * 0.03;
        let pred = from_vec(
            vec![
                0.7 + noise,
                0.2 - noise / 2.0,
                0.1 - noise / 2.0,
                0.6 - noise,
                0.3 + noise / 2.0,
                0.1 + noise / 2.0,
            ],
            &[2, 3], // 2 samples, 3 classes
            DeviceType::Cpu,
        )
        .unwrap();
        mc_predictions.push(pred);
    }

    // Compute uncertainty decomposition
    if let Some(uncertainty) = mc_dropout.compute_uncertainty(&mc_predictions) {
        println!("\n{}", uncertainty.format());
    }

    // Compute predictive statistics
    if let Some((means, variances)) = mc_dropout.predictive_statistics(&mc_predictions) {
        println!("Predictive Statistics:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for (i, (mean, var)) in means.iter().zip(variances.iter()).enumerate() {
            println!("Sample {}", i + 1);
            println!("  Mean probabilities: {:?}", mean);
            println!("  Variance: {:.6}", var);
        }
    }
}

/// Demonstrate Deep Ensemble uncertainty estimation
fn ensemble_uncertainty_demo() {
    println!("📊 Deep Ensemble Uncertainty Estimation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let n_models = 5;
    let ensemble = EnsembleUncertainty::new(n_models);

    // Simulate predictions from 5 independent models
    let mut ensemble_predictions = Vec::new();

    println!(
        "Generating predictions from {} ensemble members...",
        n_models
    );

    // Model 1: Confident about class 0
    ensemble_predictions.push(
        from_vec(
            vec![0.85, 0.10, 0.05, 0.75, 0.15, 0.10],
            &[2, 3],
            DeviceType::Cpu,
        )
        .unwrap(),
    );

    // Model 2: Also confident about class 0 but slightly different
    ensemble_predictions.push(
        from_vec(
            vec![0.80, 0.12, 0.08, 0.70, 0.20, 0.10],
            &[2, 3],
            DeviceType::Cpu,
        )
        .unwrap(),
    );

    // Model 3: Very confident about class 0
    ensemble_predictions.push(
        from_vec(
            vec![0.90, 0.07, 0.03, 0.80, 0.12, 0.08],
            &[2, 3],
            DeviceType::Cpu,
        )
        .unwrap(),
    );

    // Model 4: Slightly less confident
    ensemble_predictions.push(
        from_vec(
            vec![0.75, 0.15, 0.10, 0.65, 0.25, 0.10],
            &[2, 3],
            DeviceType::Cpu,
        )
        .unwrap(),
    );

    // Model 5: Similar to others
    ensemble_predictions.push(
        from_vec(
            vec![0.82, 0.11, 0.07, 0.72, 0.18, 0.10],
            &[2, 3],
            DeviceType::Cpu,
        )
        .unwrap(),
    );

    // Compute uncertainty
    if let Some(uncertainty) = ensemble.compute_uncertainty(&ensemble_predictions) {
        println!("\n{}", uncertainty.format());
    }

    // Compute ensemble agreement and diversity
    let agreement = ensemble.ensemble_agreement(&ensemble_predictions);
    let diversity = ensemble.predictive_diversity(&ensemble_predictions);

    println!("Ensemble Metrics:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  Agreement:  {:.2}% (higher = more consensus)",
        agreement * 100.0
    );
    println!(
        "  Diversity:  {:.2}% (higher = more diverse predictions)",
        diversity * 100.0
    );

    if agreement > 0.8 {
        println!("  ✓ High agreement - Ensemble is confident");
    } else if agreement > 0.6 {
        println!("  ⚠ Moderate agreement - Some disagreement present");
    } else {
        println!("  ⚠ Low agreement - High uncertainty, consider more data");
    }
}

/// Demonstrate Bayesian uncertainty estimation
fn bayesian_uncertainty_demo() {
    println!("📊 Bayesian Uncertainty Estimation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let n_samples = 20;
    let bayesian = BayesianUncertainty::new(n_samples);

    // Simulate posterior samples from a Bayesian neural network
    let mut posterior_samples = Vec::new();

    println!("Generating {} posterior samples...", n_samples);

    for i in 0..n_samples {
        // Simulate posterior distribution samples
        let noise = ((i as f32) / n_samples as f32 - 0.5) * 0.15;
        let pred = from_vec(
            vec![
                0.7 + noise,
                0.3 - noise,
                0.6 + noise / 2.0,
                0.4 - noise / 2.0,
            ],
            &[2, 2], // 2 samples, 2 classes
            DeviceType::Cpu,
        )
        .unwrap();
        posterior_samples.push(pred);
    }

    // Compute uncertainty
    if let Some(uncertainty) = bayesian.compute_uncertainty(&posterior_samples) {
        println!("\n{}", uncertainty.format());
    }

    // Compute credible intervals at different confidence levels
    println!("Credible Intervals:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for confidence in &[0.90, 0.95, 0.99] {
        if let Some((lower, upper)) = bayesian.credible_interval(&posterior_samples, *confidence) {
            println!("{}% Credible Interval:", (confidence * 100.0) as u32);
            for (i, (l, u)) in lower.iter().zip(upper.iter()).enumerate() {
                println!(
                    "  Sample {}: [{:.4}, {:.4}] (width: {:.4})",
                    i + 1,
                    l,
                    u,
                    u - l
                );
            }
        }
    }
}

/// Compare different uncertainty estimation methods
fn uncertainty_comparison_demo() {
    println!("📊 Uncertainty Method Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create test data representing different scenarios
    println!("\n🔍 Scenario 1: High Epistemic Uncertainty");
    println!("   (Models disagree - need more data)");
    println!("   ─────────────────────────────────────────");

    let high_epistemic = vec![
        from_vec(vec![0.9, 0.1, 0.1, 0.9], &[2, 2], DeviceType::Cpu).unwrap(),
        from_vec(vec![0.1, 0.9, 0.9, 0.1], &[2, 2], DeviceType::Cpu).unwrap(),
        from_vec(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap(),
    ];

    if let Some(decomp) = UncertaintyDecomposition::from_mc_predictions(&high_epistemic) {
        println!(
            "   Epistemic: {:.4} ({:.1}%)",
            decomp.epistemic_uncertainty,
            decomp.epistemic_ratio * 100.0
        );
        println!(
            "   Aleatoric: {:.4} ({:.1}%)",
            decomp.aleatoric_uncertainty,
            decomp.aleatoric_ratio * 100.0
        );
        println!("   → {}", decomp.diagnostic());
    }

    println!("\n🔍 Scenario 2: High Aleatoric Uncertainty");
    println!("   (Inherent noise in data - irreducible)");
    println!("   ─────────────────────────────────────────");

    let high_aleatoric = vec![
        from_vec(vec![0.33, 0.33, 0.34, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap(),
        from_vec(vec![0.34, 0.33, 0.33, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap(),
        from_vec(vec![0.33, 0.34, 0.33, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap(),
    ];

    if let Some(decomp) = UncertaintyDecomposition::from_mc_predictions(&high_aleatoric) {
        println!(
            "   Epistemic: {:.4} ({:.1}%)",
            decomp.epistemic_uncertainty,
            decomp.epistemic_ratio * 100.0
        );
        println!(
            "   Aleatoric: {:.4} ({:.1}%)",
            decomp.aleatoric_uncertainty,
            decomp.aleatoric_ratio * 100.0
        );
        println!("   → {}", decomp.diagnostic());
    }

    println!("\n🔍 Scenario 3: Low Uncertainty");
    println!("   (Models agree with high confidence)");
    println!("   ─────────────────────────────────────────");

    let low_uncertainty = vec![
        from_vec(vec![0.95, 0.05, 0.92, 0.08], &[2, 2], DeviceType::Cpu).unwrap(),
        from_vec(vec![0.94, 0.06, 0.93, 0.07], &[2, 2], DeviceType::Cpu).unwrap(),
        from_vec(vec![0.96, 0.04, 0.91, 0.09], &[2, 2], DeviceType::Cpu).unwrap(),
    ];

    if let Some(decomp) = UncertaintyDecomposition::from_mc_predictions(&low_uncertainty) {
        println!(
            "   Epistemic: {:.4} ({:.1}%)",
            decomp.epistemic_uncertainty,
            decomp.epistemic_ratio * 100.0
        );
        println!(
            "   Aleatoric: {:.4} ({:.1}%)",
            decomp.aleatoric_uncertainty,
            decomp.aleatoric_ratio * 100.0
        );
        println!("   → {}", decomp.diagnostic());
    }

    println!("\n💡 Key Insights:");
    println!("   • High epistemic → Collect more training data");
    println!("   • High aleatoric → Improve feature engineering");
    println!("   • Low total → Model is well-calibrated and confident");
}
