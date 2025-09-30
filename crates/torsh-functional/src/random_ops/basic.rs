//! Basic random number generation operations
//!
//! This module provides fundamental random number generation functions including
//! uniform distributions, normal distributions, and basic random sampling operations.
//! All operations follow PyTorch's random generation API for compatibility.

use scirs2_core::random::{Random, Rng};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Generate tensor with values drawn from uniform distribution [0, 1)
///
/// ## Mathematical Background
///
/// The uniform distribution U(a,b) has probability density function:
/// ```
/// f(x) = 1/(b-a)  for a ≤ x ≤ b
///      = 0        otherwise
/// ```
///
/// Properties:
/// - **Mean**: μ = (a+b)/2
/// - **Variance**: σ² = (b-a)²/12
/// - **Support**: [a, b) (a ≤ x < b)
///
/// ## Parameters
/// * `shape` - Shape of the output tensor
/// * `low` - Lower bound (inclusive, default 0.0)
/// * `high` - Upper bound (exclusive, default 1.0)
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor filled with uniformly distributed random values
///
/// ## Example
/// ```rust
/// # use torsh_functional::random_ops::rand;
/// let tensor = rand(&[3, 4], None, None, Some(42))?; // [3, 4] tensor with values in [0, 1)
/// let custom = rand(&[2, 2], Some(-1.0), Some(1.0), None)?; // Values in [-1, 1)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn rand(
    shape: &[usize],
    low: Option<f32>,
    high: Option<f32>,
    generator: Option<u64>,
) -> TorshResult<Tensor> {
    let low = low.unwrap_or(0.0);
    let high = high.unwrap_or(1.0);

    uniform_(shape, low, high, generator)
}

/// Fill tensor with values drawn from uniform distribution
///
/// ## Mathematical Implementation
///
/// Uses the linear congruential generator method:
/// ```
/// X_n+1 = (a × X_n + c) mod m
/// U = X_n / m  ∈ [0, 1)
/// Y = low + U × (high - low)  ∈ [low, high)
/// ```
///
/// ## Parameters
/// * `shape` - Shape of the tensor
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor filled with uniformly distributed values in [low, high)
///
/// ## Errors
/// * Returns error if low >= high
pub fn uniform_(
    shape: &[usize],
    low: f32,
    high: f32,
    generator: Option<u64>,
) -> TorshResult<Tensor> {
    if low >= high {
        return Err(TorshError::InvalidArgument(
            "uniform_: low must be less than high".to_string(),
        ));
    }

    // Create RNG using SciRS2
    let mut rng = if let Some(seed) = generator {
        Random::seed(seed)
    } else {
        Random::seed(42) // Default seed for reproducible behavior
    };

    // Generate uniform distributed values using SciRS2
    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);

    for _ in 0..size {
        // Use SciRS2's uniform generation
        values.push(rng.gen::<f32>() * (high - low) + low);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with values drawn from normal distribution
///
/// ## Mathematical Background
///
/// The normal (Gaussian) distribution N(μ, σ²) has probability density function:
/// ```
/// f(x) = (1/(σ√(2π))) × exp(-½((x-μ)/σ)²)
/// ```
///
/// Properties:
/// - **Mean**: μ
/// - **Variance**: σ²
/// - **Standard deviation**: σ
/// - **Support**: (-∞, ∞)
/// - **68-95-99.7 rule**: ~68% within μ±σ, ~95% within μ±2σ, ~99.7% within μ±3σ
///
/// ## Box-Muller Transformation
///
/// Converts uniform random variables to normal:
/// ```
/// U₁, U₂ ~ Uniform(0,1)
/// Z₀ = √(-2 ln U₁) × cos(2π U₂)
/// Z₁ = √(-2 ln U₁) × sin(2π U₂)
/// Z₀, Z₁ ~ N(0,1)
/// X = μ + σZ  ~ N(μ, σ²)
/// ```
///
/// ## Parameters
/// * `shape` - Shape of the tensor
/// * `mean` - Mean of the normal distribution
/// * `std` - Standard deviation of the normal distribution
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor filled with normally distributed values N(mean, std²)
pub fn normal_(
    shape: &[usize],
    mean: f32,
    std: f32,
    generator: Option<u64>,
) -> TorshResult<Tensor> {
    if std < 0.0 {
        return Err(TorshError::InvalidArgument(
            "normal_: std must be non-negative".to_string(),
        ));
    }

    // Create RNG
    let mut rng = if let Some(seed) = generator {
        Random::seed(seed)
    } else {
        Random::seed(42) // Default seed for reproducible behavior
    };

    // Generate normal distributed values
    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);

    for _ in 0..size {
        // Box-Muller transform for normal distribution
        let u1: f32 = rng.gen_range(0.0..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let z0 = (-2.0f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        values.push(mean + std * z0);
    }

    Tensor::from_vec(values, shape)
}

/// Generate tensor with values drawn from standard normal distribution
///
/// Equivalent to `normal_(shape, 0.0, 1.0, generator)`.
/// This is a convenience function matching PyTorch's `randn()` API.
///
/// ## Standard Normal Distribution
///
/// The standard normal distribution N(0,1) is the foundation for all normal distributions:
/// - **Mean**: 0
/// - **Variance**: 1
/// - **Standard deviation**: 1
///
/// Any normal distribution can be derived from standard normal:
/// ```
/// X ~ N(μ, σ²)  ⟺  X = μ + σZ where Z ~ N(0,1)
/// ```
///
/// ## Parameters
/// * `shape` - Shape of the output tensor
/// * `mean` - Mean of the normal distribution (default 0.0)
/// * `std` - Standard deviation of the normal distribution (default 1.0)
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor filled with standard normally distributed values N(0,1)
pub fn randn(
    shape: &[usize],
    mean: Option<f32>,
    std: Option<f32>,
    generator: Option<u64>,
) -> TorshResult<Tensor> {
    let mean = mean.unwrap_or(0.0);
    let std = std.unwrap_or(1.0);

    normal_(shape, mean, std, generator)
}

/// Generate random integers in the range [low, high)
///
/// ## Discrete Uniform Distribution
///
/// For integers in range [a, b), each value has equal probability:
/// ```
/// P(X = k) = 1/(b-a)  for k ∈ {a, a+1, ..., b-1}
///          = 0        otherwise
/// ```
///
/// Properties:
/// - **Mean**: μ = (a+b-1)/2
/// - **Variance**: σ² = ((b-a)²-1)/12
/// - **Support**: {a, a+1, ..., b-1}
///
/// ## Parameters
/// * `shape` - Shape of the tensor
/// * `low` - Lower bound (inclusive)
/// * `high` - Upper bound (exclusive)
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor filled with random integers in [low, high)
///
/// ## Errors
/// * Returns error if low >= high
pub fn randint_(
    shape: &[usize],
    low: i32,
    high: i32,
    generator: Option<u64>,
) -> TorshResult<Tensor> {
    if low >= high {
        return Err(TorshError::InvalidArgument(
            "randint_: low must be less than high".to_string(),
        ));
    }

    // Create RNG
    let mut rng = if let Some(seed) = generator {
        Random::seed(seed)
    } else {
        Random::seed(42) // Default seed for reproducible behavior
    };

    let size: usize = shape.iter().product();
    let mut values = Vec::with_capacity(size);

    for _ in 0..size {
        let val = rng.gen_range(low..high);
        values.push(val as f32);
    }

    Tensor::from_vec(values, shape)
}

/// Generate random integers in the range [0, high)
///
/// Convenience function equivalent to `randint_(shape, 0, high, generator)`.
///
/// ## Parameters
/// * `shape` - Shape of the tensor
/// * `high` - Upper bound (exclusive)
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * Tensor filled with random integers in [0, high)
pub fn randint(shape: &[usize], high: i32, generator: Option<u64>) -> TorshResult<Tensor> {
    randint_(shape, 0, high, generator)
}

/// Generate random permutation of integers from 0 to n-1
///
/// ## Mathematical Background
///
/// A random permutation is a bijective mapping π: {0,1,...,n-1} → {0,1,...,n-1}
/// where each of the n! possible permutations has equal probability 1/n!.
///
/// ## Fisher-Yates Shuffle Algorithm
///
/// Modern implementation uses the Fisher-Yates shuffle for O(n) efficiency:
/// ```
/// for i = n-1 down to 1:
///     j = random(0, i+1)
///     swap(array[i], array[j])
/// ```
///
/// This algorithm ensures:
/// - **Unbiased**: Each permutation has probability 1/n!
/// - **Efficient**: O(n) time complexity
/// - **In-place**: O(1) additional space
///
/// ## Parameters
/// * `n` - Number of elements to permute (generates 0, 1, ..., n-1)
/// * `generator` - Optional random number generator seed
///
/// ## Returns
/// * 1D tensor containing a random permutation of [0, 1, ..., n-1]
///
/// ## Applications
/// - **Data shuffling**: Randomize dataset order for training
/// - **Sampling**: Create random subsets without replacement
/// - **Bootstrap methods**: Generate random resampling indices
/// - **Monte Carlo**: Random ordering for statistical simulations
pub fn randperm(n: usize, generator: Option<u64>) -> TorshResult<Tensor> {
    if n == 0 {
        return Tensor::from_vec(vec![], &[0]);
    }

    // Create initial sequence [0, 1, 2, ..., n-1]
    let mut values: Vec<f32> = (0..n).map(|i| i as f32).collect();

    // Create RNG
    let mut rng = if let Some(seed) = generator {
        Random::seed(seed)
    } else {
        Random::seed(42) // Default seed for reproducible behavior
    };

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        values.swap(i, j);
    }

    Tensor::from_vec(values, &[n])
}
