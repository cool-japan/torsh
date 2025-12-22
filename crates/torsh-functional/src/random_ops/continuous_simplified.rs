//! Complete continuous probability distributions using SciRS2 unified random module
//!
//! Full statistical distributions using proper SciRS2 primitives with mathematical accuracy.

use scirs2_core::random::prelude::*;
use scirs2_core::random::{FisherF, RandBeta};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Generate tensor with exponential distribution using proper SciRS2 implementation
pub fn exponential_(shape: &[usize], lambd: f32, _generator: Option<u64>) -> TorshResult<Tensor> {
    if lambd <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "exponential: lambda must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper exponential distribution from SciRS2
    let exp_dist = Exponential::new(lambd).map_err(|e| {
        TorshError::InvalidArgument(format!("exponential: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&exp_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with gamma distribution using proper SciRS2 implementation
pub fn gamma(
    shape: &[usize],
    concentration: f32,
    rate: Option<f32>,
    _generator: Option<u64>,
) -> TorshResult<Tensor> {
    if concentration <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "gamma: concentration must be greater than 0".to_string(),
        ));
    }

    let actual_rate = rate.unwrap_or(1.0);
    if actual_rate <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "gamma: rate must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper gamma distribution from SciRS2
    let gamma_dist = Gamma::new(concentration, actual_rate).map_err(|e| {
        TorshError::InvalidArgument(format!("gamma: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&gamma_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with beta distribution using proper SciRS2 implementation
pub fn beta(
    shape: &[usize],
    alpha: f32,
    beta: f32,
    _generator: Option<u64>,
) -> TorshResult<Tensor> {
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "beta: alpha and beta must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper beta distribution from SciRS2
    let beta_dist = RandBeta::new(alpha, beta).map_err(|e| {
        TorshError::InvalidArgument(format!("beta: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&beta_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with chi-squared distribution using proper SciRS2 implementation
pub fn chi_squared(shape: &[usize], df: f32, _generator: Option<u64>) -> TorshResult<Tensor> {
    if df <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "chi_squared: degrees of freedom must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper chi-squared distribution from SciRS2
    let chi2_dist = ChiSquared::new(df).map_err(|e| {
        TorshError::InvalidArgument(format!("chi_squared: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&chi2_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with Student's t distribution using proper SciRS2 implementation
pub fn student_t(shape: &[usize], df: f32, _generator: Option<u64>) -> TorshResult<Tensor> {
    if df <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "student_t: degrees of freedom must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper Student's t distribution from SciRS2
    let t_dist = StudentT::new(df).map_err(|e| {
        TorshError::InvalidArgument(format!("student_t: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&t_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with F distribution using proper SciRS2 implementation
pub fn f_distribution(
    shape: &[usize],
    dfnum: f32,
    dfden: f32,
    _generator: Option<u64>,
) -> TorshResult<Tensor> {
    if dfnum <= 0.0 || dfden <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "f_distribution: degrees of freedom must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper Fisher F distribution from SciRS2
    let f_dist = FisherF::new(dfnum, dfden).map_err(|e| {
        TorshError::InvalidArgument(format!(
            "f_distribution: failed to create distribution: {}",
            e
        ))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&f_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with log-normal distribution using proper SciRS2 implementation
pub fn log_normal(
    shape: &[usize],
    loc: f32,
    scale: f32,
    _generator: Option<u64>,
) -> TorshResult<Tensor> {
    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "log_normal: scale must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper log-normal distribution from SciRS2
    let lognorm_dist = LogNormal::new(loc, scale).map_err(|e| {
        TorshError::InvalidArgument(format!("log_normal: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&lognorm_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with Weibull distribution using proper SciRS2 implementation
pub fn weibull(
    shape_param: f32,
    shape: &[usize],
    scale: f32,
    _generator: Option<u64>,
) -> TorshResult<Tensor> {
    if shape_param <= 0.0 || scale <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "weibull: shape and scale must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper Weibull distribution from SciRS2
    let weibull_dist = Weibull::new(scale, shape_param).map_err(|e| {
        TorshError::InvalidArgument(format!("weibull: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&weibull_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with Cauchy distribution using proper SciRS2 implementation
pub fn cauchy(
    shape: &[usize],
    median: f32,
    sigma: f32,
    _generator: Option<u64>,
) -> TorshResult<Tensor> {
    if sigma <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "cauchy: sigma must be greater than 0".to_string(),
        ));
    }

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);
    let mut rng = thread_rng();

    // Use proper Cauchy distribution from SciRS2
    let cauchy_dist = Cauchy::new(median, sigma).map_err(|e| {
        TorshError::InvalidArgument(format!("cauchy: failed to create distribution: {}", e))
    })?;

    for _ in 0..size {
        let sample = rng.sample(&cauchy_dist);
        values.push(sample);
    }

    Tensor::from_vec(values, shape)
}

/// Sample from a Dirichlet distribution using proper SciRS2 implementation
///
/// The Dirichlet distribution is a multivariate generalization of the Beta distribution.
/// It is commonly used as a prior distribution in Bayesian statistics and for generating
/// probability distributions that sum to 1.
///
/// # Arguments
/// * `alpha` - Concentration parameters (must be positive)
/// * `num_samples` - Number of samples to generate
/// * `generator` - Optional seed for reproducibility
///
/// # Returns
/// A tensor of shape [num_samples, alpha.len()] where each row is a valid probability
/// distribution (sums to 1.0) sampled from the Dirichlet distribution.
///
/// # Mathematical Details
/// The Dirichlet distribution is sampled using the Gamma-based method:
/// 1. Sample Y_i ~ Gamma(alpha_i, 1) for each dimension i
/// 2. Normalize: X_i = Y_i / sum(Y_j) for all j
///
/// # Examples
/// ```ignore
/// use torsh_functional::random_ops::continuous_simplified::dirichlet;
///
/// // Sample from symmetric Dirichlet (uniform over simplexes)
/// let samples = dirichlet(&[1.0, 1.0, 1.0], 1000, None)?;
/// assert_eq!(samples.shape().dims(), &[1000, 3]);
///
/// // High concentration parameters create peaked distributions
/// let peaked = dirichlet(&[100.0, 100.0, 100.0], 100, Some(42))?;
/// ```
pub fn dirichlet(alpha: &[f32], num_samples: usize, generator: Option<u64>) -> TorshResult<Tensor> {
    // Validate inputs
    if alpha.is_empty() {
        return Err(TorshError::InvalidArgument(
            "dirichlet: alpha must have at least one element".to_string(),
        ));
    }

    for &a in alpha {
        if a <= 0.0 {
            return Err(TorshError::InvalidArgument(format!(
                "dirichlet: all alpha values must be positive, got {}",
                a
            )));
        }
    }

    if num_samples == 0 {
        return Err(TorshError::InvalidArgument(
            "dirichlet: num_samples must be greater than 0".to_string(),
        ));
    }

    // Create RNG
    let mut rng = thread_rng();

    // If seed is provided, use seeded RNG
    if let Some(seed) = generator {
        let mut seeded = scirs2_core::random::seeded_rng(seed);

        let k = alpha.len();
        let mut samples = Vec::with_capacity(num_samples * k);

        // Sample from Dirichlet using the Gamma-based method
        for _ in 0..num_samples {
            // Step 1: Sample from Gamma distributions
            let mut gamma_samples = Vec::with_capacity(k);
            let mut sum = 0.0f32;

            for &alpha_i in alpha {
                // Gamma(alpha, 1.0) - shape = alpha, scale = 1.0
                let gamma_dist = Gamma::new(alpha_i, 1.0).map_err(|e| {
                    TorshError::InvalidArgument(format!(
                        "dirichlet: failed to create Gamma distribution: {}",
                        e
                    ))
                })?;

                let sample = seeded.sample(&gamma_dist);
                gamma_samples.push(sample);
                sum += sample;
            }

            // Step 2: Normalize to sum to 1 (project onto probability simplex)
            for gamma_sample in gamma_samples {
                samples.push(gamma_sample / sum);
            }
        }

        // Create tensor with shape [num_samples, k]
        return Tensor::from_vec(samples, &[num_samples, k]);
    }

    // Non-seeded path
    let k = alpha.len();
    let mut samples = Vec::with_capacity(num_samples * k);

    // Sample from Dirichlet using the Gamma-based method
    for _ in 0..num_samples {
        // Step 1: Sample from Gamma distributions
        let mut gamma_samples = Vec::with_capacity(k);
        let mut sum = 0.0f32;

        for &alpha_i in alpha {
            // Gamma(alpha, 1.0) - shape = alpha, scale = 1.0
            let gamma_dist = Gamma::new(alpha_i, 1.0).map_err(|e| {
                TorshError::InvalidArgument(format!(
                    "dirichlet: failed to create Gamma distribution: {}",
                    e
                ))
            })?;

            let sample = rng.sample(&gamma_dist);
            gamma_samples.push(sample);
            sum += sample;
        }

        // Step 2: Normalize to sum to 1 (project onto probability simplex)
        for gamma_sample in gamma_samples {
            samples.push(gamma_sample / sum);
        }
    }

    // Create tensor with shape [num_samples, k]
    Tensor::from_vec(samples, &[num_samples, k])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_basic() -> TorshResult<()> {
        // Test basic Dirichlet sampling
        let alpha = vec![1.0, 1.0, 1.0]; // Symmetric Dirichlet (uniform distribution)
        let num_samples = 1000;

        let samples = dirichlet(&alpha, num_samples, Some(42))?;

        // Check shape
        assert_eq!(samples.shape().dims(), &[num_samples, alpha.len()]);

        // Check that each row sums to approximately 1.0
        let data = samples.to_vec()?;
        for i in 0..num_samples {
            let row_sum: f32 = (0..alpha.len()).map(|j| data[i * alpha.len() + j]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Row {} sum is {}, expected 1.0",
                i,
                row_sum
            );
        }

        Ok(())
    }

    #[test]
    fn test_dirichlet_validation() {
        // Test empty alpha
        assert!(dirichlet(&[], 10, None).is_err());

        // Test negative alpha
        assert!(dirichlet(&[-1.0, 1.0], 10, None).is_err());

        // Test zero alpha
        assert!(dirichlet(&[0.0, 1.0], 10, None).is_err());

        // Test zero samples
        assert!(dirichlet(&[1.0, 1.0], 0, None).is_err());
    }

    #[test]
    fn test_dirichlet_reproducibility() -> TorshResult<()> {
        // Test that same seed produces same results
        let alpha = vec![2.0, 3.0, 5.0];
        let num_samples = 100;
        let seed = Some(12345);

        let samples1 = dirichlet(&alpha, num_samples, seed)?;
        let samples2 = dirichlet(&alpha, num_samples, seed)?;

        let data1 = samples1.to_vec()?;
        let data2 = samples2.to_vec()?;

        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert!(
                (v1 - v2).abs() < 1e-6,
                "Reproducibility failed: {} vs {}",
                v1,
                v2
            );
        }

        Ok(())
    }

    #[test]
    fn test_dirichlet_concentration() -> TorshResult<()> {
        // Test that concentration parameters affect the distribution
        // High alpha should concentrate around the mean (1/k for symmetric)
        let high_alpha = vec![100.0, 100.0, 100.0];
        let samples = dirichlet(&high_alpha, 100, Some(42))?;

        let data = samples.to_vec()?;
        let k = high_alpha.len();
        let expected_mean = 1.0 / k as f32;

        // Calculate mean of first dimension
        let mean: f32 = (0..100).map(|i| data[i * k]).sum::<f32>() / 100.0;

        // With high concentration, mean should be very close to 1/3
        assert!(
            (mean - expected_mean).abs() < 0.05,
            "Mean {} too far from expected {}",
            mean,
            expected_mean
        );

        Ok(())
    }

    #[test]
    fn test_dirichlet_asymmetric() -> TorshResult<()> {
        // Test asymmetric Dirichlet
        let alpha = vec![10.0, 1.0, 1.0]; // First dimension should dominate
        let samples = dirichlet(&alpha, 100, Some(42))?;

        let data = samples.to_vec()?;
        let k = alpha.len();

        // Calculate mean of each dimension
        let mean0: f32 = (0..100).map(|i| data[i * k]).sum::<f32>() / 100.0;
        let mean1: f32 = (0..100).map(|i| data[i * k + 1]).sum::<f32>() / 100.0;

        // First dimension should have higher mean due to higher alpha
        assert!(
            mean0 > mean1,
            "First dimension mean {} should be greater than second {}",
            mean0,
            mean1
        );

        Ok(())
    }
}
