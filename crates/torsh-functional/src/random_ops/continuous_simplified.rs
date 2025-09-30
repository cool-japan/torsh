//! Complete continuous probability distributions using SciRS2 unified random module
//!
//! Full statistical distributions using proper SciRS2 primitives with mathematical accuracy.

use scirs2_core::random::prelude::*;
use scirs2_core::random::{FisherF, RandBeta};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Generate tensor with exponential distribution using proper SciRS2 implementation
pub fn exponential_(shape: &[usize], lambd: f32, generator: Option<u64>) -> TorshResult<Tensor> {
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
    generator: Option<u64>,
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
pub fn beta(shape: &[usize], alpha: f32, beta: f32, generator: Option<u64>) -> TorshResult<Tensor> {
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
pub fn chi_squared(shape: &[usize], df: f32, generator: Option<u64>) -> TorshResult<Tensor> {
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
pub fn student_t(shape: &[usize], df: f32, generator: Option<u64>) -> TorshResult<Tensor> {
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
    generator: Option<u64>,
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
    generator: Option<u64>,
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
    generator: Option<u64>,
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
    generator: Option<u64>,
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
