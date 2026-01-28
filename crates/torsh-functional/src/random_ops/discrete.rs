//! Discrete probability distributions
//!
//! This module provides discrete probability distributions including multinomial
//! and Bernoulli distributions. These distributions are essential for categorical
//! sampling, binary outcomes, and discrete choice modeling in machine learning.

use scirs2_core::random::{Random, Rng};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Draw samples from a multinomial distribution
///
/// ## Mathematical Background
///
/// The multinomial distribution is a generalization of the binomial distribution
/// to multiple categories. For k categories with probabilities p₁, p₂, ..., pₖ:
///
/// ```text
/// P(X₁ = x₁, ..., Xₖ = xₖ) = n! / (x₁! × ... × xₖ!) × p₁^x₁ × ... × pₖ^xₖ
/// ```
///
/// where n = x₁ + ... + xₖ is the total number of trials.
///
/// ## Sampling Algorithm
///
/// ### With Replacement (replacement=true)
/// Uses inverse transform sampling with cumulative distribution:
/// ```text
/// 1. Compute CDF: F(i) = Σⱼ₌₀ⁱ pⱼ
/// 2. Generate U ~ Uniform(0,1)
/// 3. Return i where F(i-1) < U ≤ F(i)
/// ```
///
/// ### Without Replacement (replacement=false)
/// Uses weighted sampling without replacement:
/// ```
/// 1. Sample according to current probabilities
/// 2. Remove sampled category and renormalize
/// 3. Repeat until num_samples reached
/// ```
///
/// ## Parameters
/// * `input` - Tensor containing probabilities (must be non-negative, will be normalized)
/// * `num_samples` - Number of samples to draw
/// * `replacement` - Whether to draw with replacement
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor of indices where each index is sampled according to the multinomial
///   probability distribution located in the corresponding row of input
///
/// ## Applications
/// - **Language modeling**: Sample next token from vocabulary distribution
/// - **Reinforcement learning**: Sample actions from policy distribution
/// - **Categorical data**: Generate synthetic categorical variables
/// - **Monte Carlo methods**: Sample from discrete probability distributions
///
/// ## Example
/// ```rust
/// # use torsh_functional::random_ops::multinomial;
/// # use torsh_tensor::creation::tensor;
/// let probs = tensor(&[0.1, 0.3, 0.6])?; // Category probabilities
/// let samples = multinomial(&probs, 5, true, Some(42))?; // 5 samples with replacement
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn multinomial(
    input: &Tensor,
    num_samples: usize,
    replacement: bool,
    generator: Option<u64>,
) -> TorshResult<Tensor> {
    // Input must be 1D or 2D
    if input.ndim() > 2 {
        return Err(TorshError::dimension_error(
            "input must be 1D or 2D",
            "multinomial",
        ));
    }

    let is_1d = input.ndim() == 1;
    let input_2d = if is_1d {
        input.view(&[1, -1])?
    } else {
        input.clone()
    };

    let (num_rows, num_cols) = (input_2d.shape().dims()[0], input_2d.shape().dims()[1]);

    // Check if we can sample without replacement
    if !replacement && num_samples > num_cols {
        return Err(TorshError::InvalidArgument(format!(
            "Cannot sample {} samples without replacement from {} categories",
            num_samples, num_cols
        )));
    }

    // Create RNG
    let mut rng = if let Some(seed) = generator {
        Random::seed(seed)
    } else {
        Random::seed(42) // Default seed for reproducible behavior
    };

    // Sample for each row
    let mut output_data = Vec::with_capacity(num_rows * num_samples);

    for row in 0..num_rows {
        // Get probabilities for this row
        let row_start = row * num_cols;
        let row_end = row_start + num_cols;
        let data = input_2d.data()?;
        let row_probs: Vec<f32> = data[row_start..row_end].to_vec();

        // Normalize probabilities
        let sum: f32 = row_probs.iter().sum();
        if sum <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "multinomial: all probabilities are zero".to_string(),
            ));
        }

        let normalized_probs: Vec<f32> = row_probs.iter().map(|&p| p / sum).collect();

        if replacement {
            // Sample with replacement using weighted distribution
            // Implement weighted sampling using cumulative distribution
            let mut cumulative: Vec<f32> = Vec::with_capacity(normalized_probs.len());
            let mut sum = 0.0;
            for &prob in &normalized_probs {
                sum += prob;
                cumulative.push(sum);
            }

            for _ in 0..num_samples {
                let r: f32 = rng.gen_range(0.0..1.0);
                let idx = cumulative
                    .iter()
                    .position(|&cum_prob| r <= cum_prob)
                    .unwrap_or(cumulative.len() - 1);
                output_data.push(idx as f32);
            }
        } else {
            // Sample without replacement
            let mut indices: Vec<usize> = (0..num_cols).collect();
            let mut remaining_probs = normalized_probs.clone();

            for _ in 0..num_samples {
                // Normalize remaining probabilities
                let sum: f32 = remaining_probs.iter().sum();
                if sum <= 0.0 {
                    return Err(TorshError::InvalidArgument(
                        "multinomial: all remaining probabilities are zero".to_string(),
                    ));
                }

                let normalized: Vec<f32> = remaining_probs.iter().map(|&p| p / sum).collect();

                // Sample from remaining indices
                // Implement weighted sampling using cumulative distribution
                let mut cumulative: Vec<f32> = Vec::with_capacity(normalized.len());
                let mut sum = 0.0;
                for &prob in &normalized {
                    sum += prob;
                    cumulative.push(sum);
                }
                // Sample using cumulative distribution
                let rand_val: f32 = rng.gen_range(0.0..1.0);
                let idx = cumulative
                    .iter()
                    .position(|&x| x >= rand_val)
                    .unwrap_or(cumulative.len() - 1);

                // Add sampled index and remove it from future sampling
                output_data.push(indices[idx] as f32);
                indices.remove(idx);
                remaining_probs.remove(idx);
            }
        }
    }

    // Create output tensor
    let output_shape = if is_1d {
        vec![num_samples]
    } else {
        vec![num_rows, num_samples]
    };

    Tensor::from_vec(output_data, &output_shape)
}

/// Generate Bernoulli distributed random values with fixed probability
///
/// ## Mathematical Background
///
/// The Bernoulli distribution models a single binary trial with success probability p:
///
/// ```
/// P(X = 1) = p
/// P(X = 0) = 1 - p
/// ```
///
/// Properties:
/// - **Mean**: μ = p
/// - **Variance**: σ² = p(1-p)
/// - **Support**: {0, 1}
/// - **Maximum variance**: p = 0.5 gives σ² = 0.25
///
/// ## Inverse Transform Sampling
///
/// Uses the inverse transform method:
/// ```
/// 1. Generate U ~ Uniform(0,1)
/// 2. Return 1 if U < p, else 0
/// ```
///
/// This method is optimal for Bernoulli distribution due to its simplicity.
///
/// ## Parameters
/// * `shape` - Shape of the output tensor
/// * `p` - Success probability (must be in [0, 1])
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor filled with Bernoulli distributed values (0s and 1s)
///
/// ## Applications
/// - **Binary classification**: Generate synthetic binary labels
/// - **Dropout**: Random neuron deactivation in neural networks
/// - **A/B testing**: Random assignment to treatment groups
/// - **Monte Carlo**: Binary outcome simulation
///
/// ## Errors
/// * Returns error if p is not in [0, 1]
pub fn bernoulli_(shape: &[usize], p: f32, generator: Option<u64>) -> TorshResult<Tensor> {
    if !(0.0..=1.0).contains(&p) {
        return Err(TorshError::InvalidArgument(
            "bernoulli_: p must be between 0 and 1".to_string(),
        ));
    }

    // Create RNG
    let mut rng = if let Some(seed) = generator {
        Random::seed(seed)
    } else {
        Random::seed(42) // Default seed for reproducible behavior
    };

    // Generate Bernoulli distributed values
    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);

    for _ in 0..size {
        let val: f32 = if rng.random::<f32>() < p { 1.0 } else { 0.0 };
        values.push(val);
    }

    Tensor::from_vec(values, shape)
}

/// Generate Bernoulli distributed random values based on input probabilities
///
/// ## Element-wise Bernoulli Distribution
///
/// For a tensor of probabilities P = [p₁, p₂, ..., pₙ], generates a tensor
/// X = [x₁, x₂, ..., xₙ] where each xᵢ ~ Bernoulli(pᵢ) independently.
///
/// This allows for:
/// - **Spatially varying probabilities**: Different success rates across tensor elements
/// - **Conditional sampling**: Probabilities computed from other tensors
/// - **Structured dropout**: Non-uniform dropout patterns
///
/// ## Mathematical Properties
///
/// For independent Bernoulli variables:
/// - **Joint probability**: P(X₁=x₁, ..., Xₙ=xₙ) = ∏ᵢ pᵢˣⁱ(1-pᵢ)¹⁻ˣⁱ
/// - **Expected count**: E[∑ᵢ Xᵢ] = ∑ᵢ pᵢ
/// - **Variance of count**: Var[∑ᵢ Xᵢ] = ∑ᵢ pᵢ(1-pᵢ)
///
/// ## Parameters
/// * `input` - Tensor containing probabilities (must be between 0 and 1)
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor with same shape as input, filled with Bernoulli distributed values
///
/// ## Applications
/// - **Variational dropout**: Learned dropout probabilities
/// - **Stochastic networks**: Random connectivity patterns
/// - **Attention mechanisms**: Sparse attention masks
/// - **Data augmentation**: Random feature masking
///
/// ## Example
/// ```rust
/// # use torsh_functional::random_ops::bernoulli;
/// # use torsh_tensor::creation::tensor;
/// let probs = tensor(&[[0.1, 0.9], [0.5, 0.3]])?; // Different probabilities per element
/// let samples = bernoulli(&probs, Some(42))?; // Element-wise Bernoulli sampling
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn bernoulli(input: &Tensor, generator: Option<u64>) -> TorshResult<Tensor> {
    // Create RNG
    let mut rng = if let Some(seed) = generator {
        Random::seed(seed)
    } else {
        Random::seed(42) // Default seed for reproducible behavior
    };

    let data = input.data()?;
    let values: Vec<f32> = data
        .iter()
        .map(|&p| {
            if !(0.0..=1.0).contains(&p) {
                panic!("bernoulli: all values in input must be between 0 and 1");
            }
            if rng.random::<f32>() < p {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    Tensor::from_vec(values, &input.shape().dims().to_vec())
}
