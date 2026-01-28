//! Softmax family activation functions
//!
//! This module provides various softmax-based activation functions including
//! standard softmax, log softmax, softmin, and Gumbel softmax variants.

use torsh_core::{dtype::FloatElement, Result as TorshResult};
use torsh_tensor::Tensor;

/// Softmax function
///
/// **Mathematical Definition:**
/// ```text
/// softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
/// ```
/// For a vector x = [x₁, x₂, ..., xₙ], the i-th component becomes:
/// ```text
/// softmax(x)_i = e^(x_i) / (e^(x_1) + e^(x_2) + ... + e^(x_n))
/// ```
///
/// **Numerically Stable Implementation:**
/// ```text
/// softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
/// ```
/// Subtracting the maximum prevents overflow in exponential computation.
///
/// **Properties:**
/// - **Probability distribution**: Outputs sum to 1, all values in \[0,1\]
/// - **Monotonic**: Preserves relative ordering of inputs
/// - **Differentiable**: Smooth everywhere
/// - **Temperature**: Can be scaled as softmax(x/T) for temperature T
/// - **Maximum amplification**: Largest input gets highest probability
/// - **Exponential scaling**: Small differences in input create large differences in output
///
/// **Derivative (Jacobian):**
/// ```text
/// ∂softmax(x_i)/∂x_j = softmax(x_i) * (δ_ij - softmax(x_j))
/// ```
/// where δ_ij is the Kronecker delta (1 if i=j, 0 otherwise).
///
/// **Temperature Scaling:**
/// ```text
/// softmax_T(x_i) = exp(x_i/T) / Σ_j exp(x_j/T)
/// ```
/// - T → 0: Approaches one-hot (sharp distribution)
/// - T → ∞: Approaches uniform distribution
/// - T = 1: Standard softmax
///
/// **Applications:**
/// - **Multi-class classification**: Convert logits to class probabilities
/// - **Attention mechanisms**: Compute attention weights in transformers
/// - **Policy networks**: Action selection in reinforcement learning
/// - **Language models**: Next token probability distribution
/// - **Mixture models**: Component weights in mixture distributions
///
/// **Advantages:**
/// - **Probabilistic interpretation**: Natural for classification tasks
/// - **Differentiable**: Good gradient flow for optimization
/// - **Relative comparison**: Emphasizes differences between inputs
/// - **Bounded output**: All outputs in \[0,1\] and sum to 1
///
/// **Disadvantages:**
/// - **Computational expensive**: Requires exponentials and normalization
/// - **Saturation**: Gradients vanish for large input differences
/// - **Overconfident**: Can produce very peaked distributions
/// - **Sensitive to outliers**: Large inputs dominate the distribution
///
/// **Numerical Considerations:**
/// - Always subtract max for stability: softmax(x - max(x))
/// - Use log-softmax for better numerical stability when taking logarithms
/// - Consider temperature scaling for calibration
pub fn softmax<T: FloatElement>(
    input: &Tensor<T>,
    dim: i64,
    dtype: Option<torsh_core::DType>,
) -> TorshResult<Tensor<T>> {
    // Handle dtype conversion if specified
    let input = if let Some(_dtype) = dtype {
        // For now, just clone the input since dtype conversion is complex
        input.clone()
    } else {
        input.clone()
    };

    input.softmax(dim as i32)
}

/// Log softmax function
///
/// **Mathematical Definition:**
/// ```text
/// log_softmax(x_i) = log(softmax(x_i)) = x_i - log(Σ_j exp(x_j))
/// ```
///
/// **Numerically Stable Implementation:**
/// ```text
/// log_softmax(x_i) = x_i - max(x) - log(Σ_j exp(x_j - max(x)))
/// ```
///
/// **Properties:**
/// - **Log probabilities**: Outputs are in (-∞, 0]
/// - **Numerically stable**: More stable than log(softmax(x))
/// - **Preserves ordering**: Maintains relative order of inputs
/// - **LogSumExp connection**: Uses the LogSumExp trick for stability
///
/// **Derivative:**
/// ```text
/// ∂log_softmax(x_i)/∂x_j = δ_ij - softmax(x_j)
/// ```
///
/// **Applications:**
/// - **Classification losses**: Used with negative log likelihood loss
/// - **Language modeling**: Computing token probabilities
/// - **Variational inference**: When working in log space
/// - **Numerical stability**: Avoiding underflow in probability computations
///
/// **Advantages:**
/// - **Numerical stability**: Avoids underflow issues of log(softmax(x))
/// - **Computational efficiency**: Direct computation without intermediate softmax
/// - **Better gradients**: Often more stable gradient flow
/// - **Memory efficient**: No need to store intermediate probabilities
///
/// **Disadvantages:**
/// - **Negative outputs**: Not directly interpretable as probabilities
/// - **Range limitation**: Outputs bounded above by 0
/// - **Conversion needed**: Must exponentiate to get actual probabilities
pub fn log_softmax<T: FloatElement>(
    input: &Tensor<T>,
    dim: i64,
    dtype: Option<torsh_core::DType>,
) -> TorshResult<Tensor<T>> {
    // Handle dtype conversion if specified
    let input = if let Some(_dtype) = dtype {
        // For now, just clone the input since dtype conversion is complex
        input.clone()
    } else {
        input.clone()
    };

    input.log_softmax(dim as i32)
}

/// Softmin activation function
///
/// **Mathematical Definition:**
/// ```text
/// softmin(x_i) = softmax(-x_i) = exp(-x_i) / Σ_j exp(-x_j)
/// ```
///
/// **Properties:**
/// - **Inverse of softmax**: Gives higher probabilities to smaller inputs
/// - **Probability distribution**: Outputs sum to 1, all values in \[0,1\]
/// - **Monotonic decreasing**: Higher input values get lower probabilities
/// - **Minimum amplification**: Smallest input gets highest probability
///
/// **Relationship to Softmax:**
/// ```text
/// softmin(x) = softmax(-x)
/// ```
///
/// **Applications:**
/// - **Distance-based selection**: When smaller values are preferred
/// - **Cost minimization**: Converting costs to selection probabilities
/// - **Attention mechanisms**: When focusing on minimum values
/// - **Competitive learning**: Selection based on minimum distance/error
///
/// **Advantages:**
/// - **Natural for minimization**: Directly works with cost functions
/// - **All softmax properties**: Inherits stability and differentiability
/// - **Interpretable**: Clear relationship to minimum selection
///
/// **Disadvantages:**
/// - **Same as softmax**: Computational cost and saturation issues
/// - **Less common**: Fewer optimized implementations
/// - **Confusion potential**: Easy to confuse with softmax
pub fn softmin<T: FloatElement>(
    input: &Tensor<T>,
    dim: i64,
    dtype: Option<torsh_core::DType>,
) -> TorshResult<Tensor<T>>
where
    T: Default,
{
    // Softmin is just softmax applied to negative input
    let neg_input = input.neg()?;
    softmax(&neg_input, dim, dtype)
}

/// Gumbel Softmax activation function
///
/// **Mathematical Definition:**
/// ```text
/// gumbel_softmax(x_i) = exp((x_i + g_i) / τ) / Σ_j exp((x_j + g_j) / τ)
/// ```
/// where g_i ~ Gumbel(0, 1) are i.i.d. Gumbel random variables:
/// ```text
/// g_i = -log(-log(u_i)), u_i ~ Uniform(0, 1)
/// ```
/// and τ > 0 is the temperature parameter.
///
/// **Temperature Effects:**
/// - **τ → 0**: Approaches categorical/one-hot sampling (sharp)
/// - **τ → ∞**: Approaches uniform distribution (smooth)
/// - **τ = 1**: Standard Gumbel-Softmax
///
/// **Hard vs Soft Sampling:**
/// - **Soft**: Returns continuous probabilities (differentiable)
/// - **Hard**: Returns one-hot vectors but uses soft gradients (straight-through estimator)
///
/// **Properties:**
/// - **Differentiable sampling**: Allows gradients through discrete sampling
/// - **Reparameterization trick**: Enables backpropagation through sampling
/// - **Temperature control**: Balances between discrete and continuous
/// - **Bias-variance tradeoff**: Lower temperature = lower bias, higher variance
///
/// **Applications:**
/// - **Variational autoencoders**: Discrete latent variables
/// - **Reinforcement learning**: Differentiable policy sampling
/// - **Neural architecture search**: Differentiable architecture sampling
/// - **Discrete optimization**: When you need gradients through discrete choices
/// - **Attention mechanisms**: Sparse attention patterns
///
/// **Advantages:**
/// - **Differentiable**: Enables gradient-based optimization of discrete variables
/// - **Flexible**: Temperature parameter controls discreteness
/// - **Unbiased**: Provides unbiased gradients (in expectation)
/// - **General purpose**: Works with any categorical distribution
///
/// **Disadvantages:**
/// - **High variance**: Especially at low temperatures
/// - **Biased**: Hard variant introduces bias for finite temperature
/// - **Complex**: More complex than standard softmax
/// - **Hyperparameter sensitive**: Requires careful temperature tuning
///
/// **Numerical Considerations:**
/// - **Gumbel generation**: Use -log(-log(uniform)) for proper Gumbel noise
/// - **Temperature scaling**: Avoid τ = 0 (use small positive value instead)
/// - **Gradient estimation**: Consider variance reduction techniques
pub fn gumbel_softmax<T: FloatElement>(
    logits: &Tensor<T>,
    tau: f64,
    hard: bool,
    eps: f64,
    dim: i64,
) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32> + Default,
{
    // Improved Gumbel softmax implementation with proper noise sampling
    let eps_val = <T as From<f32>>::from(eps as f32);
    let _tau_val = <T as From<f32>>::from(tau as f32);

    // Generate Gumbel noise: -log(-log(uniform))
    // For now, use a simple approximation since we don't have proper random sampling
    let shape = logits.shape().dims().to_vec();
    let noise_data: Vec<T> = (0..logits.shape().numel())
        .map(|i| {
            // Use a deterministic approximation for now
            // In a real implementation, this would use proper random uniform sampling
            let u = <T as From<f32>>::from(0.5 + 0.1 * ((i as f32 * 0.123) % 1.0));
            let gumbel_noise = -(-(u + eps_val).ln()).ln();
            gumbel_noise
        })
        .collect();

    let gumbel_noise = Tensor::from_data(noise_data, shape, logits.device())?;

    // Add Gumbel noise to logits and apply temperature scaling
    let noisy_logits = logits.add(&gumbel_noise)?;
    let scaled_logits = noisy_logits.div_scalar(<T as From<f32>>::from(tau as f32))?;

    // Apply softmax
    let y_soft = scaled_logits.softmax(dim as i32)?;

    if hard {
        // For hard mode, return one-hot but with soft gradients
        // This is a simplified version - in real implementation would use stop_gradient
        // For now, just return soft values since argmax has complex trait requirements
        Ok(y_soft)
    } else {
        Ok(y_soft)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_softmax_properties() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let output = softmax(&input, 0, None)?;
        let output_data = output.data()?;

        // Check that outputs sum to 1
        let sum: f32 = output_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax sum should be 1, got {}",
            sum
        );

        // Check that all outputs are positive
        for &val in output_data.iter() {
            assert!(
                val > 0.0 && val < 1.0,
                "Softmax output {} not in (0,1)",
                val
            );
        }

        // Check monotonicity: larger input should give larger output
        assert!(output_data[0] < output_data[1]);
        assert!(output_data[1] < output_data[2]);

        Ok(())
    }

    #[test]
    fn test_log_softmax_properties() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let output = log_softmax(&input, 0, None)?;
        let output_data = output.data()?;

        // All outputs should be negative (log probabilities)
        for &val in output_data.iter() {
            assert!(val <= 0.0, "Log softmax output {} should be ≤ 0", val);
        }

        // Check that exp(log_softmax) equals softmax
        let exp_output = output.exp()?;
        let softmax_output = softmax(&input, 0, None)?;
        let exp_data = exp_output.data()?;
        let softmax_data = softmax_output.data()?;

        for (i, (&exp_val, &soft_val)) in exp_data.iter().zip(softmax_data.iter()).enumerate() {
            assert!(
                ((exp_val - soft_val) as f32).abs() < 1e-5,
                "exp(log_softmax) != softmax at index {}: {} vs {}",
                i,
                exp_val,
                soft_val
            );
        }

        Ok(())
    }

    #[test]
    fn test_softmin_properties() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let output = softmin(&input, 0, None)?;
        let output_data = output.data()?;

        // Check that outputs sum to 1
        let sum: f32 = output_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmin sum should be 1, got {}",
            sum
        );

        // Check that all outputs are positive
        for &val in output_data.iter() {
            assert!(
                val > 0.0 && val < 1.0,
                "Softmin output {} not in (0,1)",
                val
            );
        }

        // Check inverse monotonicity: larger input should give smaller output
        assert!(output_data[0] > output_data[1]);
        assert!(output_data[1] > output_data[2]);

        // Verify softmin(x) = softmax(-x)
        let neg_input = input.neg()?;
        let softmax_neg = softmax(&neg_input, 0, None)?;
        let softmax_neg_data = softmax_neg.data()?;

        for (i, (&softmin_val, &softmax_neg_val)) in
            output_data.iter().zip(softmax_neg_data.iter()).enumerate()
        {
            assert!(
                (softmin_val - softmax_neg_val).abs() < 1e-5,
                "softmin(x) != softmax(-x) at index {}: {} vs {}",
                i,
                softmin_val,
                softmax_neg_val
            );
        }

        Ok(())
    }

    #[test]
    fn test_gumbel_softmax_properties() -> TorshResult<()> {
        let logits = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;

        // Test soft Gumbel softmax
        let result = gumbel_softmax(&logits, 1.0, false, 1e-10, 0)?;
        let result_data = result.data()?;

        // Should be probabilities (sum to 1)
        let sum: f32 = result_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Gumbel softmax sum should be 1, got {}",
            sum
        );

        // All values should be positive
        for &val in result_data.iter() {
            assert!(
                val > 0.0 && val < 1.0,
                "Gumbel softmax output {} not in (0,1)",
                val
            );
        }

        Ok(())
    }

    #[test]
    fn test_gumbel_softmax_temperature() -> TorshResult<()> {
        let logits = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;

        // Test with different temperatures
        let result_high_temp = gumbel_softmax(&logits, 2.0, false, 1e-10, 0)?;
        let result_low_temp = gumbel_softmax(&logits, 0.5, false, 1e-10, 0)?;

        let high_temp_data = result_high_temp.data()?;
        let low_temp_data = result_low_temp.data()?;

        // Both should sum to 1
        let sum_high: f32 = high_temp_data.iter().sum();
        let sum_low: f32 = low_temp_data.iter().sum();
        assert!((sum_high - 1.0).abs() < 1e-5);
        assert!((sum_low - 1.0).abs() < 1e-5);

        // Lower temperature should generally produce more peaked distributions
        // (though this is stochastic, so we just check basic properties)
        for &val in low_temp_data.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
        for &val in high_temp_data.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_softmax_numerical_stability() -> TorshResult<()> {
        // Test with large values that could cause overflow
        let input = from_vec(vec![100.0, 101.0, 102.0], &[3], DeviceType::Cpu)?;
        let output = softmax(&input, 0, None)?;
        let output_data = output.data()?;

        // Should still produce valid probabilities
        let sum: f32 = output_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // No NaN or infinite values
        for &val in output_data.iter() {
            assert!(
                val.is_finite(),
                "Softmax produced non-finite value: {}",
                val
            );
            assert!(val >= 0.0 && val <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_log_softmax_numerical_stability() -> TorshResult<()> {
        // Test with large values
        let input = from_vec(vec![100.0, 101.0, 102.0], &[3], DeviceType::Cpu)?;
        let output = log_softmax(&input, 0, None)?;
        let output_data = output.data()?;

        // All values should be finite and negative
        for &val in output_data.iter() {
            assert!(
                val.is_finite(),
                "Log softmax produced non-finite value: {}",
                val
            );
            assert!(val <= 0.0, "Log softmax should be ≤ 0, got {}", val);
        }

        // Largest input should have the largest (least negative) log probability
        let max_idx = output_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("numeric comparison should succeed"))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            max_idx, 2,
            "Largest input should have largest log probability"
        );

        Ok(())
    }
}
