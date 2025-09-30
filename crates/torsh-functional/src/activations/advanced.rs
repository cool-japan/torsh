//! Advanced activation functions
//!
//! This module provides advanced activation functions including GELU, GLU,
//! scaled dot-product attention, and local response normalization.

use torsh_core::{dtype::FloatElement, Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Gaussian Error Linear Unit (GELU) activation function
///
/// **Mathematical Definition (Exact):**
/// ```text
/// GELU(x) = x * Φ(x) = x * (1/2)[1 + erf(x/√2)]
/// ```
/// where Φ(x) is the CDF of standard normal distribution, and erf is the error function.
///
/// **Approximation (Commonly Used):**
/// ```text
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// **Alternative Sigmoid Approximation:**
/// ```text
/// GELU(x) ≈ x * σ(1.702 * x), where σ is sigmoid
/// ```
///
/// **Intuition:**
/// GELU applies a "soft" version of ReLU by weighting inputs by their percentile in a standard
/// normal distribution. Inputs that are "typical" (around 0) get partial activation, while
/// extreme positive values get nearly full activation and extreme negative values get nearly zero.
///
/// **Properties:**
/// - **Smooth**: Differentiable everywhere (unlike ReLU)
/// - **Non-monotonic**: Has a small negative region for negative inputs
/// - **Probabilistic**: Based on cumulative distribution function
/// - **Self-gating**: Multiplies input by its activation probability
/// - **Near-zero derivative**: At x = 0, derivative ≈ 0.5
///
/// **Derivative:**
/// ```text
/// d/dx GELU(x) = Φ(x) + x * φ(x)
/// ```
/// where φ(x) = (1/√(2π)) * exp(-x²/2) is the PDF of standard normal distribution.
///
/// **Applications:**
/// - **Transformers**: Used in BERT, GPT, and other transformer models
/// - **Modern CNNs**: Increasingly popular replacement for ReLU
/// - **NLP models**: Particularly effective in language models
/// - **Computer Vision**: Good performance in vision transformers
/// - **Research models**: Default choice in many recent architectures
///
/// **Advantages:**
/// - **Smooth gradients**: No dead neurons like ReLU
/// - **Non-zero negative**: Small gradient for negative inputs prevents dying neurons
/// - **Empirically strong**: Often outperforms ReLU in practice
/// - **Principled**: Theoretically motivated by dropout and ReLU
/// - **Context-dependent**: Activation depends on input distribution
///
/// **Disadvantages:**
/// - **Computational cost**: More expensive than ReLU (requires erf or tanh)
/// - **Approximation errors**: Common approximations introduce small errors
/// - **Less interpretable**: More complex than simple thresholding
///
/// **Comparison to Other Activations:**
/// - vs ReLU: Smooth, no dying neurons, but more expensive
/// - vs ELU: Similar smoothness but different mathematical foundation
/// - vs Swish: Very similar performance, GELU often preferred in transformers
pub fn gelu<T: FloatElement>(input: &Tensor<T>) -> TorshResult<Tensor<T>> {
    input.gelu()
}

/// GLU (Gated Linear Unit) activation function
///
/// **Mathematical Definition:**
/// ```text
/// GLU(x) = (x_a) ⊙ σ(x_b)
/// ```
/// where x is split into two halves: x_a (first half) and x_b (second half),
/// and ⊙ denotes element-wise multiplication.
///
/// **Detailed Form:**
/// For input tensor x with shape (..., 2d), GLU produces output with shape (..., d):
/// ```text
/// GLU(x)[..., i] = x[..., i] * σ(x[..., i + d])
/// ```
/// where σ is the sigmoid function.
///
/// **Properties:**
/// - **Gating mechanism**: Second half controls flow of first half
/// - **Dimension reduction**: Output has half the channels of input
/// - **Learnable gating**: Gate values learned during training
/// - **Multiplicative**: Uses element-wise multiplication for gating
/// - **Sigmoid-based**: Uses sigmoid for gate activation
///
/// **Variants:**
/// - **GLU**: Original with sigmoid gates: GLU(x) = x_a * σ(x_b)
/// - **Swish GLU**: Uses Swish instead of sigmoid: x_a * swish(x_b)
/// - **GELU GLU**: Uses GELU instead of sigmoid: x_a * GELU(x_b)
/// - **ReGLU**: Uses ReLU instead of sigmoid: x_a * ReLU(x_b)
///
/// **Applications:**
/// - **Language models**: Particularly effective in transformer feed-forward layers
/// - **Computer vision**: Used in some CNN architectures
/// - **Speech processing**: Helps with temporal modeling
/// - **Sequence modeling**: General purpose gating for sequences
/// - **Feed-forward networks**: Alternative to standard linear layers
///
/// **Advantages:**
/// - **Selective information flow**: Can learn to gate irrelevant information
/// - **Non-linear gating**: More flexible than linear transformations
/// - **Empirically effective**: Often improves model performance
/// - **Gradient flow**: Can help with gradient flow in deep networks
///
/// **Disadvantages:**
/// - **Parameter overhead**: Requires double the input channels
/// - **Computational cost**: Additional sigmoid and multiplication operations
/// - **Dimension constraint**: Input channels must be even
/// - **Complexity**: More complex than simple activations
///
/// **Usage Notes:**
/// - Input tensor must have even size in the specified dimension
/// - Commonly used in transformer feed-forward networks
/// - Often combined with layer normalization and residual connections
pub fn glu<T: FloatElement>(input: &Tensor<T>, dim: i64) -> TorshResult<Tensor<T>>
where
    T: Default,
{
    let input_shape = input.shape();
    let dims = input_shape.dims();
    let dim_size = dims[dim as usize];

    if dim_size % 2 != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "GLU input dimension {} must be even, got {}",
            dim, dim_size
        )));
    }

    let half_size = dim_size / 2;

    // Split the input tensor in half along the specified dimension
    let first_half = input.narrow(dim as i32, 0, half_size)?;
    let second_half = input.narrow(dim as i32, half_size as i64, half_size)?;

    // Apply sigmoid to second half and multiply with first half
    let sigmoid_second = second_half.sigmoid()?;
    first_half.mul_op(&sigmoid_second)
}

/// Scaled Dot-Product Attention activation
///
/// **Mathematical Definition:**
/// ```text
/// Attention(Q, K, V) = softmax(QK^T / √d_k)V
/// ```
/// where:
/// - Q: Query matrix (batch_size, seq_len, d_k)
/// - K: Key matrix (batch_size, seq_len, d_k)
/// - V: Value matrix (batch_size, seq_len, d_v)
/// - d_k: Dimension of key vectors (used for scaling)
///
/// **With Optional Components:**
/// ```text
/// Attention(Q, K, V) = softmax((QK^T + mask) / √d_k)V
/// ```
/// where mask can be:
/// - Attention mask: Prevents attention to certain positions
/// - Causal mask: Prevents attention to future positions
///
/// **Scaling Factor:**
/// The scaling factor 1/√d_k prevents the dot products from growing too large,
/// which would push the softmax into saturation regions with small gradients.
///
/// **Properties:**
/// - **Permutation equivariant**: Order of inputs affects output consistently
/// - **Context mixing**: Each output is a weighted combination of all values
/// - **Learned attention**: Attention weights learned during training
/// - **Differentiable**: All operations are differentiable
/// - **Parallelizable**: Can be computed efficiently in parallel
///
/// **Components:**
/// 1. **Query-Key similarity**: QK^T computes pairwise similarities
/// 2. **Scaling**: Division by √d_k for numerical stability
/// 3. **Masking**: Optional masking for causal or padding tokens
/// 4. **Normalization**: Softmax to create probability distribution
/// 5. **Value aggregation**: Weighted sum of values
///
/// **Applications:**
/// - **Transformers**: Core component of transformer architectures
/// - **BERT/GPT**: Foundation of modern language models
/// - **Vision Transformers**: Applied to image patches
/// - **Multi-modal models**: Attention across different modalities
/// - **Sequence-to-sequence**: Translation, summarization, etc.
///
/// **Advantages:**
/// - **Long-range dependencies**: Can attend to any position
/// - **Parallel computation**: Unlike RNNs, can be computed in parallel
/// - **Interpretable**: Attention weights provide interpretability
/// - **Flexible**: Can handle variable-length sequences
///
/// **Disadvantages:**
/// - **Quadratic complexity**: O(n²) memory and computation w.r.t. sequence length
/// - **Attention collapse**: May attend to only a few positions
/// - **Computational cost**: Expensive for very long sequences
/// - **Memory requirements**: Stores full attention matrix
///
/// **Optimizations:**
/// - **Flash Attention**: Memory-efficient implementation
/// - **Sparse attention**: Only attend to subset of positions
/// - **Linear attention**: Approximate attention with linear complexity
/// - **Gradient checkpointing**: Trade computation for memory
pub fn scaled_dot_product_attention<T: FloatElement>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    attn_mask: Option<&Tensor<T>>,
    dropout_p: f64,
    is_causal: bool,
) -> TorshResult<Tensor<T>>
where
    T: Copy + PartialOrd + From<f32> + Default + std::iter::Sum,
{
    // Get dimensions
    let key_shape = key.shape();
    let d_k = key_shape.dims().last().unwrap_or(&1);
    let scale = <T as From<f32>>::from(1.0 / (*d_k as f32).sqrt());

    // Compute Q * K^T
    let key_t = key.transpose(-2, -1)?;
    let scores = query.matmul(&key_t)?;

    // Scale by 1/sqrt(d_k)
    let scaled_scores = scores.mul_scalar(scale)?;

    // Apply causal mask if requested
    let masked_scores = if is_causal {
        // Create causal mask (simplified implementation)
        let seq_len = scaled_scores.shape().dims()[scaled_scores.shape().dims().len() - 1];
        let causal_mask_data: Vec<T> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len)
                    .map(|j| {
                        if j <= i {
                            <T as From<f32>>::from(0.0)
                        } else {
                            <T as From<f32>>::from(-f32::INFINITY)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let causal_mask = Tensor::from_data(
            causal_mask_data,
            vec![seq_len, seq_len],
            scaled_scores.device(),
        )?;
        scaled_scores.add(&causal_mask)?
    } else {
        scaled_scores
    };

    // Apply attention mask if provided
    let final_scores = if let Some(mask) = attn_mask {
        masked_scores.add(mask)?
    } else {
        masked_scores
    };

    // Apply softmax
    println!("final_scores shape: {:?}", final_scores.shape().dims());
    let attn_weights = final_scores.softmax(-1)?;
    println!("attn_weights shape: {:?}", attn_weights.shape().dims());

    // Apply dropout (simplified - just return weights for now)
    let final_weights = if dropout_p > 0.0 {
        // In a real implementation, this would apply dropout
        attn_weights
    } else {
        attn_weights
    };
    println!("final_weights shape: {:?}", final_weights.shape().dims());

    // Apply attention weights to values: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    let result = final_weights.matmul(value)?;
    println!("final result shape: {:?}", result.shape().dims());
    Ok(result)
}

/// Local Response Normalization (LRN) activation function
///
/// **Mathematical Definition:**
/// ```text
/// LRN(x_i) = x_i / (k + α * Σ_{j=max(0,i-n/2)}^{min(N-1,i+n/2)} x_j²)^β
/// ```
/// where:
/// - x_i: Input at position i
/// - N: Total number of elements
/// - n: Local size (number of adjacent elements to consider)
/// - k: Bias parameter (typically 1.0 or 2.0)
/// - α: Scale parameter (typically 1e-4)
/// - β: Exponential parameter (typically 0.75)
///
/// **Simplified Form (used here):**
/// For computational simplicity, we normalize across channels:
/// ```text
/// LRN(x) = x / (k + (α/size) * sum(x²))^β
/// ```
///
/// **Properties:**
/// - **Local normalization**: Each element normalized by its local neighborhood
/// - **Lateral inhibition**: Strong activations suppress nearby activations
/// - **Contrast enhancement**: Increases contrast between strong and weak activations
/// - **Non-linear**: Non-linear normalization function
/// - **Parameter dependent**: Behavior controlled by k, α, β parameters
///
/// **Historical Context:**
/// LRN was popularized by AlexNet (2012) and was commonly used in early CNNs.
/// It has largely been replaced by batch normalization in modern architectures.
///
/// **Applications:**
/// - **Legacy CNNs**: AlexNet, early convolutional networks
/// - **Object detection**: Some early detection frameworks
/// - **Contrast enhancement**: When local contrast is important
/// - **Biological inspiration**: Models lateral inhibition in neuroscience
///
/// **Advantages:**
/// - **Local contrast**: Enhances local feature contrast
/// - **Normalization**: Prevents activation magnitudes from growing too large
/// - **Biological plausibility**: Inspired by biological neural networks
/// - **Simple**: Relatively simple to implement and understand
///
/// **Disadvantages:**
/// - **Outdated**: Largely superseded by batch normalization
/// - **Computational cost**: More expensive than batch norm
/// - **Less effective**: Generally less effective than modern normalization methods
/// - **Hyperparameter sensitive**: Requires careful tuning of k, α, β
///
/// **Modern Alternatives:**
/// - **Batch Normalization**: Normalizes across batch dimension
/// - **Layer Normalization**: Normalizes across feature dimension
/// - **Group Normalization**: Normalizes across groups of channels
/// - **Instance Normalization**: Normalizes per sample and channel
///
/// **Usage Notes:**
/// - Mainly of historical interest in modern deep learning
/// - Consider batch normalization or layer normalization for new models
/// - May still be useful in specific applications requiring local contrast
pub fn local_response_norm<T: FloatElement>(
    input: &Tensor<T>,
    size: i64,
    alpha: f64,
    beta: f64,
    k: f64,
) -> TorshResult<Tensor<T>>
where
    T: Copy
        + From<f32>
        + Into<f32>
        + Default
        + std::ops::AddAssign
        + std::ops::MulAssign
        + num_traits::FromPrimitive,
{
    let alpha_val = <T as From<f32>>::from(alpha as f32);
    let beta_val = <T as From<f32>>::from(beta as f32);
    let k_val = <T as From<f32>>::from(k as f32);
    let size_val = <T as From<f32>>::from(size as f32);

    // Compute input^2
    let input_squared = input.mul_op(input)?;

    // For simplification on 2D inputs, use the sum of all elements
    // In a full implementation, this would properly handle the local window across channels
    let input_shape = input.shape();
    let input_dims = input_shape.dims();
    let sum_squared = if input_dims.len() >= 3 {
        // For 3D+ tensors, sum across the last dimension (assumed to be channels)
        let last_dim = (input_dims.len() - 1) as i32;
        input_squared.sum_dim(&[last_dim], true)?
    } else {
        // For 2D tensors, create a simple element-wise normalization
        // Use the mean of the squared input as normalization base
        let mean_squared = input_squared.mean(None, true)?;
        mean_squared.expand(input_dims)?
    };

    // Compute normalization factor: k + (alpha/size) * sum
    let norm_factor = sum_squared
        .mul_scalar(alpha_val / size_val)?
        .add_scalar(k_val)?
        .pow_scalar(beta_val)?;

    // Normalize: input / norm_factor
    input.div(&norm_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_gelu_properties() -> TorshResult<()> {
        let input = from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5], DeviceType::Cpu)?;
        let output = gelu(&input)?;
        let output_data = output.data()?;

        // GELU(0) should be approximately 0
        assert!((output_data[2] - 0.0_f32).abs() < 1e-6);

        // For large positive values, GELU should be approximately x
        assert!((output_data[4] - 2.0_f32).abs() < 0.1);

        // For negative values, GELU should be small but non-zero
        assert!(output_data[0] < 0.0 && output_data[0] > -0.5);
        assert!(output_data[1] < 0.0 && output_data[1] > -0.5);

        Ok(())
    }

    #[test]
    fn test_glu_dimension_requirements() -> TorshResult<()> {
        // Test with even dimension (should work)
        let input_even = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu)?;
        let result = glu(&input_even, 0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[2]); // Half the input size

        // Test with odd dimension (should fail)
        let input_odd = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let result = glu(&input_odd, 0);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_glu_computation() -> TorshResult<()> {
        // Test with simple values where we can verify computation
        let input = from_vec(vec![1.0, 0.0, 2.0, 0.0], &[4], DeviceType::Cpu)?; // [a1, a2, b1, b2]
        let output = glu(&input, 0)?;
        let output_data = output.data()?;

        // GLU([1, 0, 2, 0]) = [1 * sigmoid(2), 0 * sigmoid(0)]
        //                   = [1 * sigmoid(2), 0 * 0.5]
        //                   = [1 * ~0.88, 0]
        assert!(output_data[0] > 0.8 && output_data[0] < 1.0); // 1 * sigmoid(2)
        assert!((output_data[1] - 0.0_f32).abs() < 1e-6); // 0 * sigmoid(0)

        Ok(())
    }

    #[test]
    fn test_scaled_dot_product_attention_basic() -> TorshResult<()> {
        // Simple test with small tensors
        let query = from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], DeviceType::Cpu)?; // 2x2
        let key = from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], DeviceType::Cpu)?; // 2x2
        let value = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu)?; // 2x2

        let output = scaled_dot_product_attention(&query, &key, &value, None, 0.0, false)?;

        // Should produce a 2x2 output
        assert_eq!(output.shape().dims(), &[2, 2]);

        let output_data = output.data()?;
        // All values should be finite
        for &val in output_data.iter() {
            assert!(val.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_local_response_norm_basic() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu)?;
        let output = local_response_norm(&input, 5, 1e-4, 0.75, 2.0)?;

        // Should have same shape as input
        assert_eq!(output.shape().dims(), input.shape().dims());

        let output_data = output.data()?;
        // All values should be finite and positive
        for &val in output_data.iter() {
            assert!(val.is_finite() && val > 0.0);
        }

        // Output values should be smaller than input (normalization effect)
        let input_data = input.data()?;
        for (i, (&out, &inp)) in output_data.iter().zip(input_data.iter()).enumerate() {
            assert!(
                out <= inp,
                "Output {} should be <= input {} at index {}",
                out,
                inp,
                i
            );
        }

        Ok(())
    }

    #[test]
    fn test_gelu_vs_relu_smoothness() -> TorshResult<()> {
        // Test that GELU is smooth around 0, unlike ReLU
        let input = from_vec(vec![-0.1, 0.0, 0.1], &[3], DeviceType::Cpu)?;
        let gelu_output = gelu(&input)?;
        let gelu_data = gelu_output.data()?;

        // GELU should be smooth and non-zero for small negative values
        assert!(gelu_data[0] < 0.0 && gelu_data[0] > -0.1); // Small negative
        assert!((gelu_data[1] - 0.0_f32).abs() < 1e-6); // Zero at zero
        assert!(gelu_data[2] > 0.0 && gelu_data[2] < 0.1); // Small positive

        // Values should be close to each other (smoothness)
        assert!((gelu_data[1] - gelu_data[0]).abs() < 0.1_f32);
        assert!((gelu_data[2] - gelu_data[1]).abs() < 0.1_f32);

        Ok(())
    }
}
