//! Advanced Neural Network Operations
//!
//! This module provides sophisticated neural network operations organized into focused sub-modules:
//!
//! - [`normalization`]: Advanced normalization techniques (spectral norm, weight standardization)
//! - [`augmentation`]: Data augmentation methods (mixup, cutmix, differentiable augmentation)
//! - [`nas`]: Neural Architecture Search operations (DARTS, encoding, evolutionary search)
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_functional::advanced_nn::{
//!     normalization::{spectral_norm, weight_standardization},
//!     augmentation::{mixup, cutmix},
//!     nas::{encode_architecture, darts_operation},
//! };
//! use torsh_tensor::creation::randn;
//!
//! // Spectral normalization for stable training
//! let weight = randn(&[64, 128]).unwrap();
//! let normalized = spectral_norm(&weight, 5, 1e-12).unwrap();
//!
//! // Data augmentation for improved generalization
//! let x1 = randn(&[32, 3, 224, 224]).unwrap();
//! let x2 = randn(&[32, 3, 224, 224]).unwrap();
//! let y1 = randn(&[32, 1000]).unwrap();
//! let y2 = randn(&[32, 1000]).unwrap();
//! let (mixed_x, mixed_y) = mixup(&x1, &x2, &y1, &y2, 0.5).unwrap();
//! ```

// Module declarations
pub mod augmentation;
pub mod nas;
pub mod normalization;

// Re-export key functions for convenience
pub use augmentation::{cutmix, differentiable_augment, mixup};
pub use nas::{
    darts_operation, decode_architecture, encode_architecture, mutate_architecture,
    predict_architecture_performance,
};
pub use normalization::{spectral_norm, weight_standardization};

// Additional functions that don't fit into the main categories
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Label smoothing regularization
///
/// Applies label smoothing to one-hot encoded labels to prevent overconfident
/// predictions and improve model calibration.
///
/// ## Mathematical Definition
///
/// For a one-hot label y and smoothing parameter ε, label smoothing produces:
/// ```text
/// y_smooth = (1 - ε) * y + ε / K
/// ```
/// where K is the number of classes.
///
/// ## Benefits
///
/// 1. **Better calibration**: Reduces overconfident predictions
/// 2. **Improved generalization**: Acts as regularization technique
/// 3. **Knowledge distillation**: Useful for teacher-student training
/// 4. **Adversarial robustness**: Provides some robustness benefits
///
/// # Arguments
/// * `targets` - One-hot encoded target labels [batch_size, num_classes]
/// * `smoothing` - Label smoothing factor ε ∈ [0, 1]
///
/// # Returns
/// Smoothed labels tensor
pub fn label_smoothing(targets: &Tensor, smoothing: f32) -> TorshResult<Tensor> {
    let shape_binding = targets.shape();
    let shape = shape_binding.dims();

    if shape.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Targets must be 2D [batch_size, num_classes]",
            "label_smoothing",
        ));
    }

    let num_classes = shape[1] as f32;
    let uniform_prob = smoothing / num_classes;

    // Apply label smoothing: (1 - smoothing) * targets + uniform_prob
    let smoothed = targets
        .mul_scalar(1.0 - smoothing)?
        .add_scalar(uniform_prob)?;

    Ok(smoothed)
}

/// Temperature scaling for prediction calibration
///
/// Applies temperature scaling to logits to improve prediction calibration
/// without changing the model's accuracy.
///
/// ## Mathematical Definition
///
/// For logits z and temperature T, temperature scaling produces:
/// ```text
/// p_i = softmax(z_i / T) = exp(z_i / T) / Σⱼ exp(zⱼ / T)
/// ```
///
/// ## Temperature Effects
///
/// - **T = 1**: No change (standard softmax)
/// - **T > 1**: Softer probabilities (less confident)
/// - **T < 1**: Sharper probabilities (more confident)
/// - **T → ∞**: Uniform distribution
/// - **T → 0**: One-hot distribution
///
/// ## Calibration
///
/// Temperature T is typically learned on a validation set to minimize:
/// ```text
/// L = -Σᵢ log(σ(zᵢ / T)_yᵢ)
/// ```
/// where σ is softmax and yᵢ is the true label.
///
/// # Arguments
/// * `logits` - Raw model logits [batch_size, num_classes]
/// * `temperature` - Temperature parameter T > 0
///
/// # Returns
/// Temperature-scaled logits
pub fn temperature_scale(logits: &Tensor, temperature: f32) -> TorshResult<Tensor> {
    if temperature <= 0.0 {
        return Err(TorshError::invalid_argument_with_context(
            "Temperature must be positive",
            "temperature_scale",
        ));
    }

    logits.div_scalar(temperature)
}

/// Knowledge distillation loss
///
/// Computes the knowledge distillation loss between teacher and student models
/// using temperature-scaled softmax distributions.
///
/// ## Mathematical Definition
///
/// The knowledge distillation loss combines:
/// ```text
/// L_KD = α * L_CE(y, σ(z_s)) + (1-α) * T² * L_CE(σ(z_t/T), σ(z_s/T))
/// ```
/// where:
/// - z_s, z_t are student and teacher logits
/// - T is the distillation temperature
/// - α balances hard and soft targets
/// - σ is softmax
///
/// ## Benefits
///
/// 1. **Model compression**: Train smaller student models
/// 2. **Knowledge transfer**: Transfer learned representations
/// 3. **Improved performance**: Often outperforms training from scratch
/// 4. **Ensemble distillation**: Combine multiple teacher models
///
/// # Arguments
/// * `student_logits` - Student model logits [batch_size, num_classes]
/// * `teacher_logits` - Teacher model logits [batch_size, num_classes]
/// * `temperature` - Distillation temperature T
/// * `alpha` - Balance between hard and soft targets
/// * `hard_targets` - Optional hard target labels
///
/// # Returns
/// Knowledge distillation loss
pub fn knowledge_distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f32,
    alpha: f32,
    hard_targets: Option<&Tensor>,
) -> TorshResult<Tensor> {
    // Temperature-scaled distributions
    let student_soft = temperature_scale(student_logits, temperature)?.softmax(-1)?;
    let teacher_soft = temperature_scale(teacher_logits, temperature)?.softmax(-1)?;

    // Soft target loss (KL divergence)
    let soft_loss = teacher_soft
        .mul(&student_soft.log()?)?
        .sum()?
        .mul_scalar(-1.0)?;
    let weighted_soft_loss = soft_loss.mul_scalar((1.0 - alpha) * temperature * temperature)?;

    // Hard target loss (if provided)
    let total_loss = if let Some(targets) = hard_targets {
        let student_probs = student_logits.softmax(-1)?;
        let hard_loss = targets
            .mul(&student_probs.log()?)?
            .sum()?
            .mul_scalar(-1.0)?;
        let weighted_hard_loss = hard_loss.mul_scalar(alpha)?;
        weighted_soft_loss.add(&weighted_hard_loss)?
    } else {
        weighted_soft_loss
    };

    Ok(total_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{ones, randn};

    #[test]
    fn test_advanced_nn_integration() -> TorshResult<()> {
        // Test that all modules work together

        // Normalization
        let weight = randn(&[16, 32])?;
        let normalized = spectral_norm(&weight, 3, 1e-12)?;
        assert_eq!(weight.shape().dims(), normalized.shape().dims());

        // Augmentation
        let x1 = randn(&[4, 3, 8, 8])?;
        let x2 = randn(&[4, 3, 8, 8])?;
        let y1 = randn(&[4, 10])?;
        let y2 = randn(&[4, 10])?;
        let (mixed_x, _mixed_y) = mixup(&x1, &x2, &y1, &y2, 0.3)?;
        assert_eq!(x1.shape().dims(), mixed_x.shape().dims());

        // NAS
        let operations = vec![0, 1, 2];
        let connections = ones(&[3, 3])?;
        let encoding = encode_architecture(&operations, &connections, 4)?;
        assert!(!encoding.data()?.is_empty());

        // Additional functions
        let targets = ones(&[4, 10])?;
        let smoothed = label_smoothing(&targets, 0.1)?;
        assert_eq!(targets.shape().dims(), smoothed.shape().dims());

        Ok(())
    }

    #[test]
    fn test_label_smoothing() -> TorshResult<()> {
        let targets = ones(&[2, 5])?;
        let smoothed = label_smoothing(&targets, 0.1)?;

        assert_eq!(targets.shape().dims(), smoothed.shape().dims());

        // Values should be less than 1 due to smoothing
        let smoothed_data = smoothed.data()?;
        for &val in smoothed_data.iter() {
            assert!(val < 1.0);
            assert!(val > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_temperature_scaling() -> TorshResult<()> {
        let logits = randn(&[3, 4])?;
        let scaled = temperature_scale(&logits, 2.0)?;

        assert_eq!(logits.shape().dims(), scaled.shape().dims());

        // Scaled logits should have smaller magnitude
        let original_data = logits.data()?;
        let scaled_data = scaled.data()?;

        for (orig, scaled_val) in original_data.iter().zip(scaled_data.iter()) {
            assert!((scaled_val * 2.0 - orig).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_knowledge_distillation_loss() -> TorshResult<()> {
        let student_logits = randn(&[2, 3])?;
        let teacher_logits = randn(&[2, 3])?;
        let hard_targets = ones(&[2, 3])?;

        let loss = knowledge_distillation_loss(
            &student_logits,
            &teacher_logits,
            3.0,
            0.5,
            Some(&hard_targets),
        )?;

        // Loss should be a scalar
        assert_eq!(loss.shape().dims(), &[] as &[usize]);

        Ok(())
    }
}
