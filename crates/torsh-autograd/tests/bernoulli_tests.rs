use torsh_autograd::stochastic_graphs::{
    BernoulliDistribution, StochasticConfig, StochasticOperation,
};
use torsh_tensor::creation;

#[test]
fn test_bernoulli_sample_is_binary() -> Result<(), Box<dyn std::error::Error>> {
    let probs = creation::full(&[100], 0.5f32)?;
    let dist = BernoulliDistribution;
    let config = StochasticConfig::default();
    let sample = dist.sample(&[&probs], &config)?;
    let values = sample.to_vec()?;
    for v in values {
        assert!(
            v == 0.0 || v == 1.0,
            "Bernoulli sample must be 0.0 or 1.0, got {v}"
        );
    }
    Ok(())
}

#[test]
fn test_bernoulli_probs_all_ones() -> Result<(), Box<dyn std::error::Error>> {
    let probs = creation::ones::<f32>(&[10])?;
    let dist = BernoulliDistribution;
    let config = StochasticConfig::default();
    let sample = dist.sample(&[&probs], &config)?;
    let values = sample.to_vec()?;
    for v in values {
        assert_eq!(v, 1.0, "With probs=1.0, every sample must be 1.0");
    }
    Ok(())
}

#[test]
fn test_bernoulli_probs_all_zeros() -> Result<(), Box<dyn std::error::Error>> {
    let probs = creation::zeros::<f32>(&[10])?;
    let dist = BernoulliDistribution;
    let config = StochasticConfig::default();
    let sample = dist.sample(&[&probs], &config)?;
    let values = sample.to_vec()?;
    for v in values {
        assert_eq!(v, 0.0, "With probs=0.0, every sample must be 0.0");
    }
    Ok(())
}

#[test]
fn test_bernoulli_mean_approx_half() -> Result<(), Box<dyn std::error::Error>> {
    let probs = creation::full(&[1000], 0.5f32)?;
    let dist = BernoulliDistribution;
    let config = StochasticConfig::default();
    let sample = dist.sample(&[&probs], &config)?;
    let values = sample.to_vec()?;
    let count = values.len() as f32;
    let sum: f32 = values.iter().sum();
    let mean = sum / count;
    assert!(
        (0.45..=0.55).contains(&mean),
        "Mean of Bernoulli(0.5) sample should be near 0.5, got {mean}"
    );
    Ok(())
}
