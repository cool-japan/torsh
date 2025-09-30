//! Comprehensive Loss Functions Library Demo
//!
//! This example demonstrates the complete loss functions library in ToRSh including:
//! - Focal Loss for handling class imbalance
//! - Label Smoothing for regularization
//! - Contrastive Loss for embedding learning
//! - And all other existing loss functions with best practices

use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_functional::loss::{
    binary_cross_entropy, contrastive_loss, cosine_embedding_loss, cross_entropy,
    cross_entropy_with_label_smoothing, focal_loss, gaussian_nll_loss, hinge_embedding_loss,
    kl_div, l1_loss, margin_ranking_loss, mse_loss, multi_margin_loss, nll_loss, poisson_nll_loss,
    smooth_l1_loss, triplet_margin_loss,
};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;

/// Configuration for loss function demonstrations
#[derive(Debug, Clone)]
pub struct LossConfig {
    pub batch_size: usize,
    pub num_classes: usize,
    pub embedding_dim: usize,
    pub device: DeviceType,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            num_classes: 10,
            embedding_dim: 128,
            device: DeviceType::Cpu,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Comprehensive Loss Functions Library Demo");
    println!("=============================================\n");

    let config = LossConfig::default();

    // Demonstrate new advanced loss functions
    demonstrate_focal_loss(&config)?;
    demonstrate_label_smoothing(&config)?;
    demonstrate_contrastive_loss(&config)?;

    // Demonstrate existing classification losses
    demonstrate_classification_losses(&config)?;

    // Demonstrate regression losses
    demonstrate_regression_losses(&config)?;

    // Demonstrate ranking and embedding losses
    demonstrate_ranking_losses(&config)?;

    // Demonstrate probabilistic losses
    demonstrate_probabilistic_losses(&config)?;

    // Best practices and recommendations
    demonstrate_best_practices()?;

    println!("\n✅ Comprehensive loss functions demonstration completed!");
    Ok(())
}

/// Demonstrate Focal Loss for handling class imbalance
fn demonstrate_focal_loss(config: &LossConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Focal Loss Demo");
    println!("==================\n");

    // Create imbalanced classification scenario
    let logits = randn(&[config.batch_size, config.num_classes]);
    let targets = tensor_1d(&[0.0, 0.0, 1.0, 9.0]); // Imbalanced: mostly class 0, one class 1, one class 9

    println!("📊 Testing Focal Loss with different hyperparameters:");

    let focal_configs = vec![
        ("Standard (α=1.0, γ=2.0)", 1.0, 2.0),
        ("High Focus (α=1.0, γ=5.0)", 1.0, 5.0),
        ("Weighted (α=0.25, γ=2.0)", 0.25, 2.0),
        ("Low Focus (α=1.0, γ=1.0)", 1.0, 1.0),
    ];

    for (name, alpha, gamma) in focal_configs {
        let loss = focal_loss(&logits, &targets, alpha, gamma, "mean")?;
        let loss_value = loss.to_vec()[0];
        println!("   {} Loss: {:.4}", name, loss_value);
    }

    // Compare with standard cross-entropy
    let ce_loss = cross_entropy(&logits, &targets, None, "mean", None, 0.0)?;
    let ce_value = ce_loss.to_vec()[0];
    println!("   Standard Cross-Entropy: {:.4}", ce_value);

    println!("\n💡 Focal Loss Benefits:");
    println!("   • Reduces loss contribution from well-classified examples");
    println!("   • Focuses learning on hard examples");
    println!("   • Particularly effective for object detection and imbalanced datasets");
    println!("   • α controls class weighting, γ controls focusing strength\n");

    Ok(())
}

/// Demonstrate Label Smoothing for regularization
fn demonstrate_label_smoothing(config: &LossConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏷️ Label Smoothing Demo");
    println!("=======================\n");

    let logits = randn(&[config.batch_size, config.num_classes]);
    let targets = tensor_1d(&[2.0, 5.0, 1.0, 8.0]);

    println!("📊 Testing Label Smoothing with different smoothing values:");

    let smoothing_values = vec![0.0, 0.05, 0.1, 0.2, 0.3];

    for smoothing in smoothing_values {
        let loss = if smoothing > 0.0 {
            cross_entropy_with_label_smoothing(&logits, &targets, smoothing, None, "mean", None)?
        } else {
            cross_entropy(&logits, &targets, None, "mean", None, 0.0)?
        };
        let loss_value = loss.to_vec()[0];
        println!("   Smoothing {:.2}: Loss = {:.4}", smoothing, loss_value);
    }

    // Test integrated cross_entropy function
    println!("\n🔄 Testing integrated cross_entropy with label smoothing:");
    let integrated_loss = cross_entropy(&logits, &targets, None, "mean", None, 0.1)?;
    let integrated_value = integrated_loss.to_vec()[0];
    println!(
        "   Integrated CE with smoothing=0.1: {:.4}",
        integrated_value
    );

    println!("\n💡 Label Smoothing Benefits:");
    println!("   • Prevents overconfident predictions");
    println!("   • Improves model calibration");
    println!("   • Acts as regularization technique");
    println!("   • Particularly useful for large models and clean datasets");
    println!("   • Typical values: 0.1 for ImageNet, 0.1-0.2 for language models\n");

    Ok(())
}

/// Demonstrate Contrastive Loss for embedding learning
fn demonstrate_contrastive_loss(config: &LossConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Contrastive Loss Demo");
    println!("========================\n");

    // Create embedding pairs
    let embedding1 = randn(&[config.batch_size, config.embedding_dim]);
    let embedding2 = randn(&[config.batch_size, config.embedding_dim]);

    // Create similarity labels (1 = similar, 0 = dissimilar)
    let labels = tensor_1d(&[1.0, 0.0, 1.0, 0.0]); // Alternating similar/dissimilar pairs

    println!("📊 Testing Contrastive Loss with different margins:");

    let margins = vec![0.5, 1.0, 2.0, 5.0];

    for margin in margins {
        let loss = contrastive_loss(&embedding1, &embedding2, &labels, margin, "mean")?;
        let loss_value = loss.to_vec()[0];
        println!("   Margin {:.1}: Loss = {:.4}", margin, loss_value);
    }

    // Demonstrate per-sample losses
    let loss_per_sample = contrastive_loss(&embedding1, &embedding2, &labels, 1.0, "none")?;
    let sample_losses = loss_per_sample.to_vec();
    println!("\n📋 Per-sample losses (margin=1.0):");
    for (i, loss_val) in sample_losses.iter().enumerate() {
        let label = labels.get_1d(i)?;
        let pair_type = if label > 0.5 { "Similar" } else { "Dissimilar" };
        println!(
            "   Sample {}: {} pair, Loss = {:.4}",
            i, pair_type, loss_val
        );
    }

    println!("\n💡 Contrastive Loss Benefits:");
    println!("   • Learns discriminative embeddings");
    println!("   • Pulls similar pairs together, pushes dissimilar pairs apart");
    println!("   • Foundation for many self-supervised learning methods");
    println!("   • Margin controls the minimum distance for dissimilar pairs");
    println!("   • Common in face recognition, image retrieval, and representation learning\n");

    Ok(())
}

/// Demonstrate classification loss functions
fn demonstrate_classification_losses(
    config: &LossConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Classification Losses Demo");
    println!("=============================\n");

    let logits = randn(&[config.batch_size, config.num_classes]);
    let targets = tensor_1d(&[2.0, 5.0, 1.0, 8.0]);
    let probs = logits.softmax((logits.ndim() - 1) as i32)?;

    let mut results = Vec::new();

    // Cross Entropy
    let ce_loss = cross_entropy(&logits, &targets, None, "mean", None, 0.0)?;
    results.push(("Cross Entropy", ce_loss.to_vec()[0]));

    // NLL Loss (with log probabilities)
    let log_probs = logits.log_softmax((logits.ndim() - 1) as i32)?;
    let nll_loss_val = nll_loss(&log_probs, &targets, None, "mean", None)?;
    results.push(("NLL Loss", nll_loss_val.to_vec()[0]));

    // Multi-Margin Loss
    let mm_loss = multi_margin_loss(&logits, &targets, 1, 1.0, None, "mean")?;
    results.push(("Multi-Margin (p=1)", mm_loss.to_vec()[0]));

    let mm_loss2 = multi_margin_loss(&logits, &targets, 2, 1.0, None, "mean")?;
    results.push(("Multi-Margin (p=2)", mm_loss2.to_vec()[0]));

    println!("📈 Classification Loss Comparison:");
    for (name, loss_value) in results {
        println!("   {:<20}: {:.4}", name, loss_value);
    }

    println!("\n💡 Classification Loss Guidelines:");
    println!("   • Cross Entropy: Standard choice for multi-class classification");
    println!("   • NLL Loss: When you already have log probabilities");
    println!("   • Multi-Margin: SVM-style margin-based loss");
    println!("   • Focal Loss: For imbalanced datasets");
    println!("   • Label Smoothing: For regularization and calibration\n");

    Ok(())
}

/// Demonstrate regression loss functions
fn demonstrate_regression_losses(config: &LossConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("📉 Regression Losses Demo");
    println!("=========================\n");

    let predictions = randn(&[config.batch_size, 1]);
    let targets = randn(&[config.batch_size, 1]);

    let mut results = Vec::new();

    // MSE Loss
    let mse = mse_loss(&predictions, &targets, "mean")?;
    results.push(("MSE Loss", mse.to_vec()[0]));

    // L1 Loss (MAE)
    let l1 = l1_loss(&predictions, &targets, "mean")?;
    results.push(("L1 Loss (MAE)", l1.to_vec()[0]));

    // Smooth L1 Loss (Huber)
    let smooth_l1 = smooth_l1_loss(&predictions, &targets, "mean", 1.0)?;
    results.push(("Smooth L1 (β=1.0)", smooth_l1.to_vec()[0]));

    let smooth_l1_small = smooth_l1_loss(&predictions, &targets, "mean", 0.1)?;
    results.push(("Smooth L1 (β=0.1)", smooth_l1_small.to_vec()[0]));

    println!("📊 Regression Loss Comparison:");
    for (name, loss_value) in results {
        println!("   {:<18}: {:.4}", name, loss_value);
    }

    println!("\n💡 Regression Loss Guidelines:");
    println!("   • MSE: Penalizes large errors more, sensitive to outliers");
    println!("   • L1 (MAE): More robust to outliers, less smooth");
    println!("   • Smooth L1 (Huber): Combines benefits of both, common in object detection");
    println!("   • β parameter controls transition point between L1 and L2 behavior\n");

    Ok(())
}

/// Demonstrate ranking and embedding losses
fn demonstrate_ranking_losses(config: &LossConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🏆 Ranking and Embedding Losses Demo");
    println!("====================================\n");

    let embed_dim = 64;
    let anchor = randn(&[config.batch_size, embed_dim]);
    let positive = randn(&[config.batch_size, embed_dim]);
    let negative = randn(&[config.batch_size, embed_dim]);
    let input1 = randn(&[config.batch_size, embed_dim]);
    let input2 = randn(&[config.batch_size, embed_dim]);

    let ranking_targets = tensor_1d(&[1.0, -1.0, 1.0, -1.0]);
    let similarity_targets = tensor_1d(&[1.0, -1.0, 1.0, -1.0]);

    let mut results = Vec::new();

    // Triplet Margin Loss
    let triplet_loss =
        triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, 1e-6, false, "mean")?;
    results.push(("Triplet Margin", triplet_loss.to_vec()[0]));

    let triplet_loss_swap =
        triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, 1e-6, true, "mean")?;
    results.push(("Triplet Margin (swap)", triplet_loss_swap.to_vec()[0]));

    // Margin Ranking Loss
    let margin_ranking = margin_ranking_loss(&input1, &input2, &ranking_targets, 1.0, "mean")?;
    results.push(("Margin Ranking", margin_ranking.to_vec()[0]));

    // Cosine Embedding Loss
    let cosine_embed = cosine_embedding_loss(&input1, &input2, &similarity_targets, 0.5, "mean")?;
    results.push(("Cosine Embedding", cosine_embed.to_vec()[0]));

    // Hinge Embedding Loss
    let hinge_embed = hinge_embedding_loss(&input1, &similarity_targets, 1.0, "mean")?;
    results.push(("Hinge Embedding", hinge_embed.to_vec()[0]));

    println!("🎯 Ranking and Embedding Loss Comparison:");
    for (name, loss_value) in results {
        println!("   {:<20}: {:.4}", name, loss_value);
    }

    println!("\n💡 Ranking and Embedding Loss Guidelines:");
    println!(
        "   • Triplet Loss: Learn embeddings where anchor is closer to positive than negative"
    );
    println!("   • Margin Ranking: Learn relative ordering between pairs");
    println!("   • Cosine Embedding: Use cosine similarity for measuring relationships");
    println!("   • Hinge Embedding: SVM-style loss for binary similarity");
    println!("   • Contrastive Loss: Binary similarity with distance-based penalties\n");

    Ok(())
}

/// Demonstrate probabilistic loss functions
fn demonstrate_probabilistic_losses(config: &LossConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎲 Probabilistic Losses Demo");
    println!("============================\n");

    let predictions = randn(&[config.batch_size, 1]).abs()?; // Ensure positive for some losses
    let targets = randn(&[config.batch_size, 1]).abs()?;
    let log_predictions = predictions.log()?;
    let probabilities = randn(&[config.batch_size, 5]).abs()?;
    let log_probabilities = probabilities.log()?;
    let variance = ones(&[config.batch_size, 1]);

    let mut results = Vec::new();

    // Binary Cross Entropy - use simple binary data
    let binary_probs = tensor_1d(&[0.8, 0.2, 0.7, 0.3]); // Already probabilities
    let binary_targets = tensor_1d(&[1.0, 0.0, 1.0, 0.0]);
    let bce = binary_cross_entropy(&binary_probs, &binary_targets, None, "mean")?;
    results.push(("Binary Cross Entropy", bce.to_vec()[0]));

    // KL Divergence
    let kl_div_loss = kl_div(&log_probabilities, &probabilities, "mean", false)?;
    results.push(("KL Divergence", kl_div_loss.to_vec()[0]));

    // Poisson NLL Loss
    let poisson_loss = poisson_nll_loss(&predictions, &targets, false, false, 1e-8, "mean")?;
    results.push(("Poisson NLL", poisson_loss.to_vec()[0]));

    // Gaussian NLL Loss
    let gaussian_loss = gaussian_nll_loss(&predictions, &targets, &variance, false, 1e-6, "mean")?;
    results.push(("Gaussian NLL", gaussian_loss.to_vec()[0]));

    println!("📊 Probabilistic Loss Comparison:");
    for (name, loss_value) in results {
        println!("   {:<20}: {:.4}", name, loss_value);
    }

    println!("\n💡 Probabilistic Loss Guidelines:");
    println!("   • Binary Cross Entropy: Binary classification with sigmoid outputs");
    println!("   • KL Divergence: Measure difference between probability distributions");
    println!("   • Poisson NLL: Count data and rate prediction");
    println!("   • Gaussian NLL: Regression with uncertainty estimation");
    println!("   • Use when modeling specific probability distributions\n");

    Ok(())
}

/// Demonstrate best practices for loss function selection
fn demonstrate_best_practices() -> Result<(), Box<dyn std::error::Error>> {
    println!("💡 Loss Function Selection Best Practices");
    println!("==========================================\n");

    println!("🎯 Task-Specific Recommendations:");
    println!("   Computer Vision:");
    println!("     • Image Classification: Cross Entropy, Focal Loss (imbalanced)");
    println!("     • Object Detection: Focal Loss, Smooth L1 Loss");
    println!("     • Semantic Segmentation: Cross Entropy, Dice Loss");
    println!("     • Face Recognition: Triplet Loss, Contrastive Loss");
    println!();

    println!("   Natural Language Processing:");
    println!("     • Text Classification: Cross Entropy with Label Smoothing");
    println!("     • Language Modeling: Cross Entropy, KL Divergence");
    println!("     • Machine Translation: Cross Entropy with Label Smoothing");
    println!("     • Sentence Embeddings: Contrastive Loss, Triplet Loss");
    println!();

    println!("   Recommendation Systems:");
    println!("     • Ranking: Margin Ranking Loss, Hinge Loss");
    println!("     • Binary Prediction: Binary Cross Entropy");
    println!("     • Collaborative Filtering: MSE, Cosine Embedding Loss");
    println!();

    println!("⚙️ Hyperparameter Guidelines:");
    println!("   Focal Loss:");
    println!("     • α ∈ [0.25, 1.0]: Controls class weighting");
    println!("     • γ ∈ [2.0, 5.0]: Controls focusing strength");
    println!();

    println!("   Label Smoothing:");
    println!("     • ε ∈ [0.05, 0.2]: Smoothing factor");
    println!("     • Start with 0.1 for most tasks");
    println!();

    println!("   Contrastive Loss:");
    println!("     • margin ∈ [0.5, 2.0]: Minimum distance for dissimilar pairs");
    println!("     • Larger margins for higher-dimensional embeddings");
    println!();

    println!("   Triplet Loss:");
    println!("     • margin ∈ [0.2, 1.0]: Minimum separation");
    println!("     • Use hard negative mining for better results");
    println!();

    println!("🔧 Implementation Tips:");
    println!("   • Always validate loss implementation with known examples");
    println!("   • Monitor training stability with different reduction methods");
    println!("   • Use appropriate numerical stability techniques (eps, clamping)");
    println!("   • Consider gradient scaling for very large or small losses");
    println!("   • Implement custom losses when domain-specific requirements exist");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_functions_comprehensive() {
        let config = LossConfig {
            batch_size: 2,
            num_classes: 3,
            embedding_dim: 4,
            device: DeviceType::Cpu,
        };

        // Test that all demonstrations can run without panicking
        assert!(demonstrate_focal_loss(&config).is_ok());
        assert!(demonstrate_label_smoothing(&config).is_ok());
        assert!(demonstrate_contrastive_loss(&config).is_ok());
        assert!(demonstrate_classification_losses(&config).is_ok());
        assert!(demonstrate_regression_losses(&config).is_ok());
        assert!(demonstrate_ranking_losses(&config).is_ok());
        assert!(demonstrate_probabilistic_losses(&config).is_ok());
        assert!(demonstrate_best_practices().is_ok());
    }

    #[test]
    fn test_focal_loss_properties() {
        let logits = tensor_2d(&[&[1.0, 2.0, 0.5], &[0.1, 0.2, 3.0]]);
        let targets = tensor_1d(&[1.0, 2.0]);

        // Test different gamma values
        let loss_gamma_0 = focal_loss(&logits, &targets, 1.0, 0.0, "mean").unwrap();
        let loss_gamma_2 = focal_loss(&logits, &targets, 1.0, 2.0, "mean").unwrap();

        // With gamma=0, focal loss should be similar to cross entropy
        // With gamma=2, it should focus more on hard examples
        assert!(loss_gamma_0.to_vec()[0] != loss_gamma_2.to_vec()[0]);
    }

    #[test]
    fn test_contrastive_loss_properties() {
        // Create perfectly similar embeddings
        let emb1 = tensor_2d(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let emb2 = tensor_2d(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let similar_labels = tensor_1d(&[1.0, 1.0]);

        let loss_similar = contrastive_loss(&emb1, &emb2, &similar_labels, 1.0, "mean").unwrap();

        // Loss for identical embeddings with similar labels should be very small
        assert!(loss_similar.to_vec()[0] < 0.1);

        // Create dissimilar embeddings
        let emb3 = tensor_2d(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let emb4 = tensor_2d(&[&[-1.0, 0.0], &[0.0, -1.0]]);
        let dissimilar_labels = tensor_1d(&[0.0, 0.0]);

        let loss_dissimilar =
            contrastive_loss(&emb3, &emb4, &dissimilar_labels, 1.0, "mean").unwrap();

        // Loss should be reasonable for dissimilar pairs
        assert!(loss_dissimilar.to_vec()[0] >= 0.0);
    }
}
