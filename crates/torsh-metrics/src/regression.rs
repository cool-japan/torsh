//! Regression metrics

use crate::Metric;
use scirs2_core::ndarray_ext::{manipulation, stats};
use torsh_core::{device::DeviceType, error::TorshError};
use torsh_tensor::{creation::from_vec, Tensor};

/// Mean Squared Error
pub struct MSE;

impl Metric for MSE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_mse(predictions, targets)
    }

    fn name(&self) -> &str {
        "mse"
    }
}

/// Root Mean Squared Error
pub struct RMSE;

impl Metric for RMSE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        MSE.compute(predictions, targets).sqrt()
    }

    fn name(&self) -> &str {
        "rmse"
    }
}

/// Mean Absolute Error
pub struct MAE;

impl Metric for MAE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_mae(predictions, targets)
    }

    fn name(&self) -> &str {
        "mae"
    }
}

/// Mean Absolute Percentage Error
pub struct MAPE {
    epsilon: f64,
}

impl MAPE {
    /// Create a new MAPE metric
    pub fn new() -> Self {
        Self { epsilon: 1e-7 }
    }

    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Metric for MAPE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_mape(predictions, targets, self.epsilon)
    }

    fn name(&self) -> &str {
        "mape"
    }
}

/// R-squared (Coefficient of Determination)
pub struct R2Score {
    multioutput: MultiOutputMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum MultiOutputMethod {
    RawValues,
    UniformAverage,
    VarianceWeighted,
}

impl R2Score {
    /// Create a new R² score metric
    pub fn new() -> Self {
        Self {
            multioutput: MultiOutputMethod::UniformAverage,
        }
    }

    /// Set multi-output method
    pub fn with_multioutput(mut self, method: MultiOutputMethod) -> Self {
        self.multioutput = method;
        self
    }
}

impl Metric for R2Score {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_r2_score(predictions, targets, &self.multioutput)
    }

    fn name(&self) -> &str {
        "r2_score"
    }
}

/// Explained Variance Score
pub struct ExplainedVariance {
    multioutput: MultiOutputMethod,
}

impl ExplainedVariance {
    /// Create a new explained variance metric
    pub fn new() -> Self {
        Self {
            multioutput: MultiOutputMethod::UniformAverage,
        }
    }
}

impl Metric for ExplainedVariance {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_explained_variance(predictions, targets, &self.multioutput)
    }

    fn name(&self) -> &str {
        "explained_variance"
    }
}

/// Huber Loss (Smooth MAE)
pub struct HuberLoss {
    delta: f64,
}

impl HuberLoss {
    /// Create a new Huber loss metric
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl Metric for HuberLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_huber_loss(predictions, targets, self.delta)
    }

    fn name(&self) -> &str {
        "huber_loss"
    }
}

/// Log Cosh Loss
pub struct LogCoshLoss;

impl Metric for LogCoshLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_log_cosh_loss(predictions, targets)
    }

    fn name(&self) -> &str {
        "log_cosh_loss"
    }
}

/// Quantile Loss
pub struct QuantileLoss {
    quantile: f64,
}

impl QuantileLoss {
    /// Create a new quantile loss metric
    pub fn new(quantile: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&quantile),
            "Quantile must be in [0, 1]"
        );
        Self { quantile }
    }
}

impl Metric for QuantileLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_quantile_loss(predictions, targets, self.quantile)
    }

    fn name(&self) -> &str {
        "quantile_loss"
    }
}

/// Symmetric Mean Absolute Percentage Error
pub struct SMAPE {
    epsilon: f64,
}

impl SMAPE {
    /// Create a new SMAPE metric
    pub fn new() -> Self {
        Self { epsilon: 1e-7 }
    }
}

impl Metric for SMAPE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        compute_smape(predictions, targets, self.epsilon)
    }

    fn name(&self) -> &str {
        "smape"
    }
}

/// Implementation functions for regression metrics

fn compute_mse(predictions: &Tensor, targets: &Tensor) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let mut sum_squared_error = 0.0;
            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let diff = pred - target;
                sum_squared_error += (diff * diff) as f64;
            }

            sum_squared_error / pred_vec.len() as f64
        }
        _ => 0.0,
    }
}

fn compute_mae(predictions: &Tensor, targets: &Tensor) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let mut sum_abs_error = 0.0;
            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                sum_abs_error += (pred - target).abs() as f64;
            }

            sum_abs_error / pred_vec.len() as f64
        }
        _ => 0.0,
    }
}

fn compute_mape(predictions: &Tensor, targets: &Tensor, epsilon: f64) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let mut sum_percentage_error = 0.0;
            let mut count = 0;

            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let target_val = *target as f64;
                let pred_val = *pred as f64;

                // Skip if target is near zero to avoid division by zero
                if target_val.abs() > epsilon {
                    let percentage_error = ((pred_val - target_val) / target_val).abs();
                    sum_percentage_error += percentage_error;
                    count += 1;
                }
            }

            if count > 0 {
                (sum_percentage_error / count as f64) * 100.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn compute_r2_score(
    predictions: &Tensor,
    targets: &Tensor,
    _multioutput: &MultiOutputMethod,
) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            // Calculate mean of targets
            let target_mean: f64 =
                target_vec.iter().map(|&x| x as f64).sum::<f64>() / target_vec.len() as f64;

            // Calculate sum of squares total (TSS) and residual (RSS)
            let mut tss = 0.0;
            let mut rss = 0.0;

            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let target_val = *target as f64;
                let pred_val = *pred as f64;

                tss += (target_val - target_mean).powi(2);
                rss += (target_val - pred_val).powi(2);
            }

            // R² = 1 - (RSS / TSS)
            if tss > 0.0 {
                1.0 - (rss / tss)
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn compute_explained_variance(
    predictions: &Tensor,
    targets: &Tensor,
    _multioutput: &MultiOutputMethod,
) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let n = pred_vec.len() as f64;

            // Calculate means
            let target_mean: f64 = target_vec.iter().map(|&x| x as f64).sum::<f64>() / n;
            let pred_mean: f64 = pred_vec.iter().map(|&x| x as f64).sum::<f64>() / n;

            // Calculate variances and residual variance
            let mut target_var = 0.0;
            let mut residual_var = 0.0;

            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let target_val = *target as f64;
                let pred_val = *pred as f64;

                target_var += (target_val - target_mean).powi(2);
                residual_var += (target_val - pred_val - target_mean + pred_mean).powi(2);
            }

            target_var /= n;
            residual_var /= n;

            // Explained variance = 1 - (residual_var / target_var)
            if target_var > 0.0 {
                1.0 - (residual_var / target_var)
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn compute_huber_loss(predictions: &Tensor, targets: &Tensor, delta: f64) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let mut sum_loss = 0.0;

            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let diff = (*pred as f64 - *target as f64).abs();

                if diff <= delta {
                    // Quadratic for small errors
                    sum_loss += 0.5 * diff.powi(2);
                } else {
                    // Linear for large errors
                    sum_loss += delta * diff - 0.5 * delta.powi(2);
                }
            }

            sum_loss / pred_vec.len() as f64
        }
        _ => 0.0,
    }
}

fn compute_log_cosh_loss(predictions: &Tensor, targets: &Tensor) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let mut sum_loss = 0.0;

            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let diff = *pred as f64 - *target as f64;
                sum_loss += diff.cosh().ln();
            }

            sum_loss / pred_vec.len() as f64
        }
        _ => 0.0,
    }
}

fn compute_quantile_loss(predictions: &Tensor, targets: &Tensor, quantile: f64) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let mut sum_loss = 0.0;

            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let error = *target as f64 - *pred as f64;
                if error > 0.0 {
                    sum_loss += quantile * error;
                } else {
                    sum_loss += (quantile - 1.0) * error;
                }
            }

            sum_loss / pred_vec.len() as f64
        }
        _ => 0.0,
    }
}

fn compute_smape(predictions: &Tensor, targets: &Tensor, epsilon: f64) -> f64 {
    match (predictions.to_vec(), targets.to_vec()) {
        (Ok(pred_vec), Ok(target_vec)) => {
            if pred_vec.len() != target_vec.len() || pred_vec.is_empty() {
                return 0.0;
            }

            let mut sum_percentage_error = 0.0;
            let mut count = 0;

            for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
                let target_val = *target as f64;
                let pred_val = *pred as f64;

                let denominator = (target_val.abs() + pred_val.abs()) / 2.0;

                if denominator > epsilon {
                    let error = (pred_val - target_val).abs() / denominator;
                    sum_percentage_error += error;
                    count += 1;
                }
            }

            if count > 0 {
                (sum_percentage_error / count as f64) * 100.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}
