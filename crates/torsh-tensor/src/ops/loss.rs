//! Loss functions for tensors
//!
//! This module provides comprehensive loss functions commonly used in machine learning:
//! - Regression losses (MSE, L1, Huber)
//! - Classification losses (Cross Entropy, Binary Cross Entropy, NLL)
//! - Support for reduction modes (mean, sum, none)

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

/// Reduction modes for loss functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction, return loss for each element
    None,
    /// Return mean of losses
    Mean,
    /// Return sum of losses
    Sum,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean
    }
}

/// Loss functions for float tensors
impl<T: FloatElement> Tensor<T> {
    /// Mean Squared Error (MSE) loss
    /// L(x, y) = (x - y)^2
    pub fn mse_loss(&self, target: &Self) -> Result<Self> {
        self.mse_loss_with_reduction(target, Reduction::Mean)
    }

    /// MSE loss with custom reduction
    pub fn mse_loss_with_reduction(&self, target: &Self, reduction: Reduction) -> Result<Self> {
        if self.shape() != target.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: target.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let target_data = target.data()?;

        let squared_errors: Vec<T> = self_data
            .iter()
            .zip(target_data.iter())
            .map(|(&pred, &targ)| {
                let diff = pred - targ;
                diff * diff
            })
            .collect();

        let loss_tensor = Self::from_data(
            squared_errors,
            self.shape().dims().to_vec(),
            self.device,
        )?;

        apply_reduction(&loss_tensor, reduction)
    }

    /// L1 Loss (Mean Absolute Error)
    /// L(x, y) = |x - y|
    pub fn l1_loss(&self, target: &Self) -> Result<Self> {
        self.l1_loss_with_reduction(target, Reduction::Mean)
    }

    /// L1 loss with custom reduction
    pub fn l1_loss_with_reduction(&self, target: &Self, reduction: Reduction) -> Result<Self> {
        if self.shape() != target.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: target.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let target_data = target.data()?;

        let abs_errors: Vec<T> = self_data
            .iter()
            .zip(target_data.iter())
            .map(|(&pred, &targ)| {
                let diff = pred - targ;
                if diff >= <T as TensorElement>::zero() { diff } else { -diff }
            })
            .collect();

        let loss_tensor = Self::from_data(
            abs_errors,
            self.shape().dims().to_vec(),
            self.device,
        )?;

        apply_reduction(&loss_tensor, reduction)
    }

    /// Huber Loss (Smooth L1 Loss)
    /// L(x, y) = 0.5 * (x - y)^2 if |x - y| < delta
    ///         = delta * (|x - y| - 0.5 * delta) otherwise
    pub fn huber_loss(&self, target: &Self, delta: f64) -> Result<Self> {
        self.huber_loss_with_reduction(target, delta, Reduction::Mean)
    }

    /// Huber loss with custom reduction
    pub fn huber_loss_with_reduction(&self, target: &Self, delta: f64, reduction: Reduction) -> Result<Self> {
        if self.shape() != target.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: target.shape().dims().to_vec(),
            });
        }

        let delta_t = T::from_f64(delta).unwrap_or_else(|| <T as TensorElement>::one());
        let half = T::from_f64(0.5).unwrap_or_else(|| <T as TensorElement>::zero());

        let self_data = self.data()?;
        let target_data = target.data()?;

        let huber_losses: Vec<T> = self_data
            .iter()
            .zip(target_data.iter())
            .map(|(&pred, &targ)| {
                let diff = pred - targ;
                let abs_diff = if diff >= <T as TensorElement>::zero() { diff } else { -diff };

                if abs_diff < delta_t {
                    half * diff * diff
                } else {
                    delta_t * (abs_diff - half * delta_t)
                }
            })
            .collect();

        let loss_tensor = Self::from_data(
            huber_losses,
            self.shape().dims().to_vec(),
            self.device,
        )?;

        apply_reduction(&loss_tensor, reduction)
    }

    /// Binary Cross Entropy (BCE) loss
    /// L(x, y) = -[y * log(x) + (1 - y) * log(1 - x)]
    pub fn bce_loss(&self, target: &Self) -> Result<Self> {
        self.bce_loss_with_reduction(target, Reduction::Mean)
    }

    /// BCE loss with custom reduction
    pub fn bce_loss_with_reduction(&self, target: &Self, reduction: Reduction) -> Result<Self> {
        if self.shape() != target.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: target.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let target_data = target.data()?;

        let one = <T as TensorElement>::one();
        let eps = T::from_f64(1e-8).unwrap_or_else(|| T::from_f64(1e-7).unwrap());

        let bce_losses: Vec<T> = self_data
            .iter()
            .zip(target_data.iter())
            .map(|(&pred, &targ)| {
                // Clamp predictions to avoid log(0)
                let pred_clamped = if pred < eps {
                    eps
                } else if pred > one - eps {
                    one - eps
                } else {
                    pred
                };

                let log_pred = pred_clamped.ln();
                let log_one_minus_pred = (one - pred_clamped).ln();

                -(targ * log_pred + (one - targ) * log_one_minus_pred)
            })
            .collect();

        let loss_tensor = Self::from_data(
            bce_losses,
            self.shape().dims().to_vec(),
            self.device,
        )?;

        apply_reduction(&loss_tensor, reduction)
    }

    /// Negative Log Likelihood (NLL) loss
    /// Expects log-probabilities as input and target class indices
    pub fn nll_loss(&self, target: &Tensor<i64>) -> Result<Self> {
        self.nll_loss_with_reduction(target, Reduction::Mean)
    }

    /// NLL loss with custom reduction
    pub fn nll_loss_with_reduction(&self, target: &Tensor<i64>, reduction: Reduction) -> Result<Self> {
        let self_shape = self.shape();
        let target_shape = target.shape();

        // Input should be (N, C) and target should be (N,) for classification
        if self_shape.ndim() != 2 {
            return Err(TorshError::InvalidShape(
                "NLL loss expects 2D input tensor (N, C)".to_string()
            ));
        }

        if target_shape.ndim() != 1 {
            return Err(TorshError::InvalidShape(
                "NLL loss expects 1D target tensor (N,)".to_string()
            ));
        }

        let batch_size = self_shape.dims()[0];
        let num_classes = self_shape.dims()[1];

        if target_shape.dims()[0] != batch_size {
            return Err(TorshError::ShapeMismatch {
                expected: vec![batch_size],
                got: target_shape.dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let target_data = target.data()?;

        let mut losses = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let target_class = target_data[i];

            if target_class < 0 || target_class as usize >= num_classes {
                return Err(TorshError::InvalidArgument(
                    format!("Target class {} out of range [0, {})", target_class, num_classes)
                ));
            }

            let log_prob_idx = i * num_classes + target_class as usize;
            let log_prob = self_data[log_prob_idx];
            losses.push(-log_prob);
        }

        let loss_tensor = Self::from_data(
            losses,
            vec![batch_size],
            self.device,
        )?;

        apply_reduction(&loss_tensor, reduction)
    }

    /// Cross Entropy loss (combines log_softmax and nll_loss)
    /// More numerically stable than applying softmax + log + nll separately
    pub fn cross_entropy(&self, target: &Tensor<i64>) -> Result<Self> {
        self.cross_entropy_with_reduction(target, Reduction::Mean)
    }

    /// Cross entropy loss with custom reduction
    pub fn cross_entropy_with_reduction(&self, target: &Tensor<i64>, reduction: Reduction) -> Result<Self> {
        // Apply log_softmax then nll_loss for numerical stability
        let log_probs = self.log_softmax(-1)?;
        log_probs.nll_loss_with_reduction(target, reduction)
    }
}

/// Apply reduction to loss tensor
fn apply_reduction<T: FloatElement>(loss_tensor: &Tensor<T>, reduction: Reduction) -> Result<Tensor<T>> {
    match reduction {
        Reduction::None => Ok(loss_tensor.clone()),
        Reduction::Mean => {
            let data = loss_tensor.data()?;
            let sum: T = data.iter().fold(<T as TensorElement>::zero(), |acc, &x| acc + x);
            let count = T::from_f64(data.len() as f64).unwrap_or_else(|| <T as TensorElement>::one());
            let mean = sum / count;

            Tensor::from_data(vec![mean], vec![1], loss_tensor.device)
        },
        Reduction::Sum => {
            let data = loss_tensor.data()?;
            let sum: T = data.iter().fold(<T as TensorElement>::zero(), |acc, &x| acc + x);

            Tensor::from_data(vec![sum], vec![1], loss_tensor.device)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_mse_loss() {
        let predictions = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let targets = Tensor::from_data(vec![1.5f32, 2.5, 2.5], vec![3], DeviceType::Cpu).unwrap();

        let loss = predictions.mse_loss(&targets).unwrap();
        let loss_data = loss.data().unwrap();

        // MSE = mean((1.0-1.5)^2, (2.0-2.5)^2, (3.0-2.5)^2) = mean(0.25, 0.25, 0.25) = 0.25
        assert!((loss_data[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_l1_loss() {
        let predictions = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let targets = Tensor::from_data(vec![1.5f32, 2.5, 2.5], vec![3], DeviceType::Cpu).unwrap();

        let loss = predictions.l1_loss(&targets).unwrap();
        let loss_data = loss.data().unwrap();

        // L1 = mean(|1.0-1.5|, |2.0-2.5|, |3.0-2.5|) = mean(0.5, 0.5, 0.5) = 0.5
        assert!((loss_data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_huber_loss() {
        let predictions = Tensor::from_data(vec![1.0f32, 2.0, 5.0], vec![3], DeviceType::Cpu).unwrap();
        let targets = Tensor::from_data(vec![1.5f32, 2.5, 2.0], vec![3], DeviceType::Cpu).unwrap();

        let loss = predictions.huber_loss(&targets, 1.0).unwrap();
        let loss_data = loss.data().unwrap();

        // For delta=1.0:
        // |1.0-1.5| = 0.5 < 1.0, so L = 0.5 * 0.5^2 = 0.125
        // |2.0-2.5| = 0.5 < 1.0, so L = 0.5 * 0.5^2 = 0.125
        // |5.0-2.0| = 3.0 > 1.0, so L = 1.0 * (3.0 - 0.5) = 2.5
        // Mean = (0.125 + 0.125 + 2.5) / 3 = 0.916...
        assert!((loss_data[0] - 0.91666667).abs() < 1e-6);
    }

    #[test]
    fn test_bce_loss() {
        let predictions = Tensor::from_data(vec![0.8f32, 0.2, 0.9], vec![3], DeviceType::Cpu).unwrap();
        let targets = Tensor::from_data(vec![1.0f32, 0.0, 1.0], vec![3], DeviceType::Cpu).unwrap();

        let loss = predictions.bce_loss(&targets).unwrap();
        let loss_data = loss.data().unwrap();

        // BCE loss should be positive and finite
        assert!(loss_data[0] > 0.0);
        assert!(loss_data[0].is_finite());
    }

    #[test]
    fn test_nll_loss() {
        // Log probabilities for 2 samples, 3 classes each
        let log_probs = Tensor::from_data(
            vec![-0.5f32, -1.0, -2.0, -1.5, -0.3, -3.0],
            vec![2, 3],
            DeviceType::Cpu
        ).unwrap();

        // Target classes
        let targets = Tensor::from_data(vec![0i64, 1], vec![2], DeviceType::Cpu).unwrap();

        let loss = log_probs.nll_loss(&targets).unwrap();
        let loss_data = loss.data().unwrap();

        // NLL = -mean(log_probs[0,0], log_probs[1,1]) = -mean(-0.5, -0.3) = 0.4
        assert!((loss_data[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy() {
        // Raw logits for 2 samples, 3 classes each
        let logits = Tensor::from_data(
            vec![1.0f32, 2.0, 0.5, 0.8, 3.0, 0.2],
            vec![2, 3],
            DeviceType::Cpu
        ).unwrap();

        // Target classes
        let targets = Tensor::from_data(vec![1i64, 1], vec![2], DeviceType::Cpu).unwrap();

        let loss = logits.cross_entropy(&targets).unwrap();
        let loss_data = loss.data().unwrap();

        // Cross entropy should be positive
        assert!(loss_data[0] > 0.0);
        assert!(loss_data[0].is_finite());
    }

    #[test]
    fn test_reduction_modes() {
        let predictions = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let targets = Tensor::from_data(vec![1.5f32, 2.5, 2.5], vec![3], DeviceType::Cpu).unwrap();

        // Test None reduction
        let loss_none = predictions.mse_loss_with_reduction(&targets, Reduction::None).unwrap();
        let loss_none_data = loss_none.data().unwrap();
        assert_eq!(loss_none_data.len(), 3); // Should have 3 elements

        // Test Sum reduction
        let loss_sum = predictions.mse_loss_with_reduction(&targets, Reduction::Sum).unwrap();
        let loss_sum_data = loss_sum.data().unwrap();
        assert_eq!(loss_sum_data.len(), 1); // Should be scalar

        // Test Mean reduction (default)
        let loss_mean = predictions.mse_loss(&targets).unwrap();
        let loss_mean_data = loss_mean.data().unwrap();
        assert_eq!(loss_mean_data.len(), 1); // Should be scalar

        // Mean should be Sum / count
        assert!((loss_mean_data[0] * 3.0 - loss_sum_data[0]).abs() < 1e-6);
    }
}