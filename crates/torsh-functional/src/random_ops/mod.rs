//! Random sampling operations module
//!
//! This module provides comprehensive random number generation and sampling functions
//! organized into logical categories for better maintainability and discoverability.
//!
//! # Module Organization
//!
//! - [`basic`]: Basic random operations (rand, randn, uniform, randint, randperm)
//! - [`discrete`]: Discrete distributions (multinomial, Bernoulli)
//! - [`continuous`]: Continuous distributions (gamma, beta, exponential, etc.)
//!
//! # SciRS2 Integration
//!
//! All random operations leverage the SciRS2 ecosystem for:
//! - **High-quality RNG**: Cryptographically secure and statistically robust generators
//! - **Distribution accuracy**: Mathematically precise sampling algorithms
//! - **Performance**: Optimized implementations with SIMD acceleration where applicable
//! - **Reproducibility**: Seeded generation for deterministic results
//!
//! # Mathematical Foundation
//!
//! This module implements state-of-the-art sampling algorithms:
//! - **Inverse transform sampling**: For distributions with known inverse CDF
//! - **Box-Muller transform**: For normal distribution generation
//! - **Acceptance-rejection**: For complex distributions
//! - **Ziggurat algorithm**: Fast normal and exponential sampling
//! - **Marsaglia's method**: Efficient sampling techniques
//!
//! # API Compatibility
//!
//! All functions follow PyTorch's random generation API for seamless migration:
//! - Function names match PyTorch conventions
//! - Parameter order and defaults align with torch.* functions
//! - Optional generator seeding for reproducibility
//! - Tensor shape specification consistent with PyTorch
//!
//! # Performance Considerations
//!
//! - **Vectorized operations**: Batch generation for efficiency
//! - **Memory efficiency**: Pre-allocated vectors for large tensors
//! - **Cache-friendly**: Sequential memory access patterns
//! - **SIMD optimization**: Hardware acceleration where available
//!
//! # Examples
//!
//! ```rust
//! use torsh_functional::random_ops::{rand, randn, multinomial, gamma};
//! use torsh_tensor::Tensor;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Basic random operations
//!     let uniform = rand(&[3, 4], None, None, Some(42))?; // Uniform [0,1)
//!     let normal = randn(&[100], Some(0.0), Some(1.0), Some(42))?; // Standard normal
//!
//!     // Discrete distributions
//!     let probs = Tensor::from_vec(vec![0.2, 0.3, 0.5], &[3])?;
//!     let samples = multinomial(&probs, 10, true, Some(123))?;
//!
//!     // Continuous distributions
//!     let gamma_samples = gamma(&[1000], 2.0, Some(1.5), Some(42))?;
//!     Ok(())
//! }
//! ```

pub mod basic;
pub mod continuous;
pub mod continuous_simplified;
pub mod discrete;

// Re-export basic operations
pub use basic::{normal_, rand, randint, randint_, randn, randperm, uniform_};

// Re-export discrete distributions
pub use discrete::{bernoulli, bernoulli_, multinomial};

// Re-export continuous distributions
pub use continuous::{
    beta, cauchy, chi_squared, exponential_, f_distribution, gamma, log_normal, student_t, weibull,
};

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_basic_random_operations() -> torsh_core::Result<()> {
        // Test rand function
        let tensor = rand(&[3, 4], None, None, Some(42))?;
        assert_eq!(tensor.shape().dims(), &[3, 4]);

        let data = tensor.data()?;
        for &val in data.iter() {
            assert!(val >= 0.0 && val < 1.0, "Values should be in [0, 1)");
        }

        // Test uniform with custom range
        let uniform = uniform_(&[10], -2.0, 2.0, Some(42))?;
        let uniform_data = uniform.data()?;
        for &val in uniform_data.iter() {
            assert!(val >= -2.0 && val < 2.0, "Values should be in [-2, 2)");
        }

        // Test randn function
        let normal = randn(&[100], Some(0.0), Some(1.0), Some(42))?;
        assert_eq!(normal.shape().dims(), &[100]);

        Ok(())
    }

    #[test]
    fn test_discrete_distributions() -> torsh_core::Result<()> {
        // Test multinomial
        let probs = ones::<f32>(&[3])?; // Uniform probabilities
        let samples = multinomial(&probs, 5, true, Some(42))?;
        assert_eq!(samples.shape().dims(), &[5]);

        let sample_data = samples.data()?;
        for &val in sample_data.iter() {
            assert!(val >= 0.0 && val < 3.0, "Samples should be valid indices");
        }

        // Test Bernoulli
        let bernoulli_tensor = bernoulli_(&[20], 0.5, Some(42))?;
        let bernoulli_data = bernoulli_tensor.data()?;
        for &val in bernoulli_data.iter() {
            assert!(
                val == 0.0 || val == 1.0,
                "Bernoulli values should be 0 or 1"
            );
        }

        Ok(())
    }

    #[test]
    fn test_continuous_distributions() -> torsh_core::Result<()> {
        // Test exponential distribution
        let exp_tensor = exponential_(&[50], 2.0, Some(42))?;
        let exp_data = exp_tensor.data()?;
        for &val in exp_data.iter() {
            assert!(val > 0.0, "Exponential values should be positive");
        }

        // Test gamma distribution
        let gamma_tensor = gamma(&[30], 2.0, Some(1.0), Some(42))?;
        let gamma_data = gamma_tensor.data()?;
        for &val in gamma_data.iter() {
            assert!(val > 0.0, "Gamma values should be positive");
        }

        // Test beta distribution
        let beta_tensor = beta(&[25], 2.0, 2.0, Some(42))?;
        let beta_data = beta_tensor.data()?;
        for &val in beta_data.iter() {
            assert!(val >= 0.0 && val <= 1.0, "Beta values should be in [0,1]");
        }

        Ok(())
    }

    #[test]
    fn test_random_integers() -> torsh_core::Result<()> {
        // Test randint
        let int_tensor = randint(&[20], 10, Some(42))?;
        let int_data = int_tensor.data()?;
        for &val in int_data.iter() {
            assert!(
                val >= 0.0 && val < 10.0,
                "Random integers should be in [0, 10)"
            );
        }

        // Test randint_ with custom range
        let int_range = randint_(&[15], -5, 5, Some(42))?;
        let range_data = int_range.data()?;
        for &val in range_data.iter() {
            assert!(
                val >= -5.0 && val < 5.0,
                "Random integers should be in [-5, 5)"
            );
        }

        Ok(())
    }

    #[test]
    fn test_random_permutation() -> torsh_core::Result<()> {
        let perm = randperm(10, Some(42))?;
        assert_eq!(perm.shape().dims(), &[10]);

        let perm_data = perm.data()?;
        let mut values: Vec<usize> = perm_data.iter().map(|&x| x as usize).collect();
        values.sort();

        // Should contain exactly the values 0..9
        for (i, &val) in values.iter().enumerate() {
            assert_eq!(
                val, i,
                "Permutation should contain all values 0..n-1 exactly once"
            );
        }

        Ok(())
    }

    #[test]
    fn test_seeded_reproducibility() -> torsh_core::Result<()> {
        // Same seed should produce same results
        let tensor1 = rand(&[5], None, None, Some(123))?;
        let tensor2 = rand(&[5], None, None, Some(123))?;

        let data1 = tensor1.data()?;
        let data2 = tensor2.data()?;

        for (val1, val2) in data1.iter().zip(data2.iter()) {
            assert_eq!(val1, val2, "Same seed should produce identical results");
        }

        // Different seeds should produce different results (with high probability)
        let tensor3 = rand(&[100], None, None, Some(456))?;
        let data3 = tensor3.data()?;

        let mut same_count = 0;
        for (val1, val3) in data1.iter().zip(data3.iter()) {
            if (val1 - val3).abs() < 1e-6 {
                same_count += 1;
            }
        }

        // Should be very unlikely that many values are identical
        assert!(
            same_count < 3,
            "Different seeds should produce different results"
        );

        Ok(())
    }

    #[test]
    fn test_parameter_validation() -> torsh_core::Result<()> {
        // Test invalid parameters
        assert!(uniform_(&[5], 2.0, 1.0, None).is_err()); // low >= high
        assert!(normal_(&[5], 0.0, -1.0, None).is_err()); // negative std
        assert!(randint_(&[5], 5, 5, None).is_err()); // low >= high
        assert!(exponential_(&[5], -1.0, None).is_err()); // negative rate
        assert!(gamma(&[5], -1.0, None, None).is_err()); // negative alpha
        assert!(beta(&[5], -1.0, 1.0, None).is_err()); // negative alpha
        assert!(chi_squared(&[5], -1.0, None).is_err()); // negative df

        Ok(())
    }

    #[test]
    fn test_statistical_properties() -> torsh_core::Result<()> {
        // Test that large samples have approximately correct statistical properties
        let n = 10000;

        // Test uniform distribution mean
        let uniform = uniform_(&[n], 0.0, 1.0, Some(42))?;
        let uniform_data = uniform.data()?;
        let mean: f32 = uniform_data.iter().sum::<f32>() / n as f32;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Uniform mean should be approximately 0.5"
        );

        // Test normal distribution
        let normal = normal_(&[n], 0.0, 1.0, Some(42))?;
        let normal_data = normal.data()?;
        let normal_mean: f32 = normal_data.iter().sum::<f32>() / n as f32;
        assert!(
            normal_mean.abs() < 0.1,
            "Normal mean should be approximately 0"
        );

        // Test exponential distribution mean
        let exp = exponential_(&[n], 2.0, Some(42))?;
        let exp_data = exp.data()?;
        let exp_mean: f32 = exp_data.iter().sum::<f32>() / n as f32;
        let expected_mean = 1.0 / 2.0; // 1/lambda
        assert!(
            (exp_mean - expected_mean).abs() < 0.1,
            "Exponential mean should be approximately 1/lambda"
        );

        Ok(())
    }
}
