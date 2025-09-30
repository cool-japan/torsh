//! Comprehensive tensor statistical operations
//!
//! This module provides a wide range of statistical functions for tensors,
//! including descriptive statistics, percentiles, histograms, correlations,
//! and probability distributions.

use crate::{FloatElement, Tensor, TensorElement};
use num_traits::{FromPrimitive, One, Zero};
use torsh_core::error::{Result, TorshError};

/// Statistical computation modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatMode {
    /// Population statistics (divide by N)
    Population,
    /// Sample statistics (divide by N-1)
    Sample,
}

/// Histogram configuration
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Number of bins
    pub bins: usize,
    /// Minimum value (auto-computed if None)
    pub min_val: Option<f64>,
    /// Maximum value (auto-computed if None)
    pub max_val: Option<f64>,
    /// Include values outside range in first/last bins
    pub include_outliers: bool,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            bins: 50,
            min_val: None,
            max_val: None,
            include_outliers: true,
        }
    }
}

/// Histogram result
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bin counts
    pub counts: Vec<usize>,
    /// Bin edges (length = counts.len() + 1)
    pub edges: Vec<f64>,
    /// Total number of values
    pub total_count: usize,
}

/// Correlation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationMethod {
    /// Pearson correlation coefficient
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Kendall tau correlation
    Kendall,
}

/// Statistical summary
#[derive(Debug, Clone)]
pub struct StatSummary {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub q25: f64, // 25th percentile
    pub q50: f64, // 50th percentile (median)
    pub q75: f64, // 75th percentile
}

/// Statistical operations for tensors
impl<
        T: TensorElement
            + FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::AddAssign
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::MulAssign
            + std::ops::Div<Output = T>
            + PartialOrd
            + num_traits::FromPrimitive
            + std::iter::Sum,
    > Tensor<T>
{
    /// Compute mean along specified dimensions (legacy stats implementation)
    pub fn mean_stats(&self, dims: Option<&[usize]>, keepdim: bool) -> Result<Self> {
        let sum = if let Some(dims) = dims {
            self.sum_dim(&dims.iter().map(|&d| d as i32).collect::<Vec<_>>(), keepdim)?
        } else {
            self.sum()?
        };
        let count = if let Some(dims) = dims {
            dims.iter()
                .map(|&d| self.shape().dims()[d])
                .product::<usize>() as f64
        } else {
            self.numel() as f64
        };

        sum.div_scalar(
            <T as num_traits::FromPrimitive>::from_f64(count)
                .unwrap_or_else(|| <T as num_traits::One>::one()),
        )
    }

    /// Compute variance along specified dimensions
    pub fn var(&self, dims: Option<&[usize]>, keepdim: bool, mode: StatMode) -> Result<Self> {
        let mean = self.mean(dims, false)?; // Always get scalar mean for broadcasting
        let mean_value = mean.item()?; // Extract scalar value
        let diff = self.sub_scalar(mean_value)?;
        let squared_diff = diff.mul_op(&diff)?;
        let sum_sq = if let Some(dims) = dims {
            squared_diff.sum_dim(&dims.iter().map(|&d| d as i32).collect::<Vec<_>>(), keepdim)?
        } else {
            squared_diff.sum()?
        };

        let count = if let Some(dims) = dims {
            dims.iter()
                .map(|&d| self.shape().dims()[d])
                .product::<usize>()
        } else {
            self.numel()
        };

        let divisor = match mode {
            StatMode::Population => count,
            StatMode::Sample => count - 1,
        };

        if divisor == 0 {
            return Err(TorshError::InvalidArgument(
                "Cannot compute variance with zero degrees of freedom".to_string(),
            ));
        }

        sum_sq.div_scalar(
            <T as num_traits::FromPrimitive>::from_usize(divisor)
                .unwrap_or_else(|| <T as num_traits::One>::one()),
        )
    }

    /// Compute standard deviation along specified dimensions
    pub fn std(&self, dims: Option<&[usize]>, keepdim: bool, mode: StatMode) -> Result<Self> {
        let variance = self.var(dims, keepdim, mode)?;
        variance.sqrt()
    }

    /// Compute percentile along the last dimension
    pub fn percentile(&self, q: f64, dim: Option<usize>, _keepdim: bool) -> Result<Self> {
        if !(0.0..=100.0).contains(&q) {
            return Err(TorshError::InvalidArgument(format!(
                "Percentile must be between 0 and 100, got {q}"
            )));
        }

        let dim = dim.unwrap_or(self.shape().ndim() - 1);
        if dim >= self.shape().ndim() {
            return Err(TorshError::dimension_error(
                &format!(
                    "Dimension {} out of bounds for tensor with {} dimensions",
                    dim,
                    self.shape().ndim()
                ),
                "tensor statistics operation",
            ));
        }

        // For now, implement a simple linear interpolation method
        // In a full implementation, this would be optimized
        let (sorted, _indices) = self.sort(Some(dim as i32), false)?; // Sort in ascending order
        let size = self.shape().dims()[dim];

        // Calculate the position in the sorted array
        let pos = q / 100.0 * (size - 1) as f64;
        let lower_idx = pos.floor() as usize;
        let upper_idx = (pos.ceil() as usize).min(size - 1);
        let weight = pos - pos.floor();

        if lower_idx == upper_idx {
            // Exact index
            sorted.select(dim as i32, lower_idx as i64)
        } else {
            // Interpolate between two values
            let lower = sorted.select(dim as i32, lower_idx as i64)?;
            let upper = sorted.select(dim as i32, upper_idx as i64)?;
            let diff = upper.sub(&lower)?;
            let weight_scalar = <T as TensorElement>::from_f64(weight)
                .unwrap_or_else(|| <T as TensorElement>::from_f64(0.0).unwrap_or_default());
            let weighted_diff = diff.mul_scalar(weight_scalar)?;
            lower.add_op(&weighted_diff)
        }
    }

    /// Compute median (50th percentile)
    pub fn median(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        self.percentile(50.0, dim, keepdim)
    }

    /// Compute quantiles at specified levels
    pub fn quantile(&self, q: &[f64], dim: Option<usize>, keepdim: bool) -> Result<Vec<Self>> {
        let mut results = Vec::new();
        for &quantile in q {
            results.push(self.percentile(quantile * 100.0, dim, keepdim)?);
        }
        Ok(results)
    }

    /// Create histogram of tensor values
    pub fn histogram(&self, config: &HistogramConfig) -> Result<Histogram> {
        let data = self.to_vec()?;

        if data.is_empty() {
            return Ok(Histogram {
                counts: vec![0; config.bins],
                edges: (0..=config.bins).map(|i| i as f64).collect(),
                total_count: 0,
            });
        }

        // Compute min and max if not provided
        let min_val = config.min_val.unwrap_or_else(|| {
            data.iter()
                .map(|&x| TensorElement::to_f64(&x).unwrap())
                .fold(f64::INFINITY, f64::min)
        });
        let max_val = config.max_val.unwrap_or_else(|| {
            data.iter()
                .map(|&x| TensorElement::to_f64(&x).unwrap())
                .fold(f64::NEG_INFINITY, f64::max)
        });

        if min_val >= max_val {
            return Err(TorshError::InvalidArgument(
                "Minimum value must be less than maximum value".to_string(),
            ));
        }

        // Create bin edges
        let bin_width = (max_val - min_val) / config.bins as f64;
        let edges: Vec<f64> = (0..=config.bins)
            .map(|i| min_val + i as f64 * bin_width)
            .collect();

        // Count values in each bin
        let mut counts = vec![0; config.bins];
        for &value in data.iter() {
            let val = TensorElement::to_f64(&value).unwrap();

            let bin_idx = if val <= min_val {
                if config.include_outliers {
                    0
                } else {
                    continue;
                }
            } else if val >= max_val {
                if config.include_outliers {
                    config.bins - 1
                } else {
                    continue;
                }
            } else {
                ((val - min_val) / bin_width).floor() as usize
            };

            let bin_idx = bin_idx.min(config.bins - 1);
            counts[bin_idx] += 1;
        }

        Ok(Histogram {
            counts,
            edges,
            total_count: data.len(),
        })
    }

    /// Compute correlation coefficient with another tensor
    pub fn correlation(&self, other: &Self, method: CorrelationMethod) -> Result<T> {
        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        match method {
            CorrelationMethod::Pearson => self.pearson_correlation(other),
            CorrelationMethod::Spearman => self.spearman_correlation(other),
            CorrelationMethod::Kendall => self.kendall_correlation(other),
        }
    }

    /// Pearson correlation coefficient
    fn pearson_correlation(&self, other: &Self) -> Result<T> {
        let n = self.numel() as f64;
        if n < 2.0 {
            return Err(TorshError::InvalidArgument(
                "Need at least 2 values for correlation".to_string(),
            ));
        }

        // Compute means
        let mean_x = self.mean(None, false)?;
        let mean_y = other.mean(None, false)?;

        let mean_x_data = mean_x.to_vec()?;
        let mean_y_data = mean_y.to_vec()?;
        let mean_x_val = mean_x_data[0];
        let mean_y_val = mean_y_data[0];

        // Compute deviations and products
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for (&x, &y) in self_data.iter().zip(other_data.iter()) {
            let dx =
                TensorElement::to_f64(&x).unwrap() - TensorElement::to_f64(&mean_x_val).unwrap();
            let dy =
                TensorElement::to_f64(&y).unwrap() - TensorElement::to_f64(&mean_y_val).unwrap();

            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        let denominator = (sum_xx * sum_yy).sqrt();
        if denominator.abs() < f64::EPSILON {
            return Err(TorshError::InvalidArgument(
                "Cannot compute correlation: one variable has zero variance".to_string(),
            ));
        }

        let correlation = sum_xy / denominator;
        Ok(<T as TensorElement>::from_f64(correlation).unwrap())
    }

    /// Spearman rank correlation coefficient
    fn spearman_correlation(&self, other: &Self) -> Result<T> {
        // Convert to ranks and compute Pearson correlation of ranks
        let self_ranks = self.rank()?;
        let other_ranks = other.rank()?;
        self_ranks.pearson_correlation(&other_ranks)
    }

    /// Kendall tau correlation coefficient
    fn kendall_correlation(&self, other: &Self) -> Result<T> {
        let n = self.numel();
        if n < 2 {
            return Err(TorshError::InvalidArgument(
                "Need at least 2 values for Kendall correlation".to_string(),
            ));
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let mut concordant = 0;
        let mut discordant = 0;
        let mut tied_x = 0;
        let mut tied_y = 0;
        let mut tied_xy = 0;

        for i in 0..n {
            for j in i + 1..n {
                let x1 = TensorElement::to_f64(&self_data[i]).unwrap();
                let x2 = TensorElement::to_f64(&self_data[j]).unwrap();
                let y1 = TensorElement::to_f64(&other_data[i]).unwrap();
                let y2 = TensorElement::to_f64(&other_data[j]).unwrap();

                let dx = x2 - x1;
                let dy = y2 - y1;

                if dx.abs() < f64::EPSILON && dy.abs() < f64::EPSILON {
                    tied_xy += 1;
                } else if dx.abs() < f64::EPSILON {
                    tied_x += 1;
                } else if dy.abs() < f64::EPSILON {
                    tied_y += 1;
                } else if dx * dy > 0.0 {
                    concordant += 1;
                } else {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        let effective_pairs = total_pairs - tied_x - tied_y - tied_xy;

        if effective_pairs == 0 {
            return Ok(<T as TensorElement>::from_f64(0.0).unwrap());
        }

        let tau = (concordant as f64 - discordant as f64) / effective_pairs as f64;
        Ok(<T as TensorElement>::from_f64(tau).unwrap())
    }

    /// Compute ranks of tensor elements
    fn rank(&self) -> Result<Self> {
        let data = self.to_vec()?;
        let n = data.len();

        // Create index-value pairs and sort by value
        let mut indexed_values: Vec<(usize, T)> =
            data.iter().enumerate().map(|(i, &val)| (i, val)).collect();

        indexed_values.sort_by(|a, b| {
            TensorElement::to_f64(&a.1)
                .unwrap()
                .partial_cmp(&TensorElement::to_f64(&b.1).unwrap())
                .unwrap()
        });

        // Assign ranks (1-based, handle ties with average rank)
        let mut ranks = vec![T::default(); n];
        let mut i = 0;

        while i < n {
            let mut j = i;
            while j < n
                && TensorElement::to_f64(&indexed_values[j].1).unwrap()
                    == TensorElement::to_f64(&indexed_values[i].1).unwrap()
            {
                j += 1;
            }

            // Average rank for tied values
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[indexed_values[k].0] = <T as TensorElement>::from_f64(avg_rank).unwrap();
            }
            i = j;
        }

        Self::from_data(ranks, self.shape().dims().to_vec(), self.device)
    }

    /// Generate comprehensive statistical summary
    pub fn describe(&self) -> Result<StatSummary> {
        let data = self.to_vec()?;
        if data.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot compute statistics on empty tensor".to_string(),
            ));
        }

        let count = data.len();
        let values: Vec<f64> = data
            .iter()
            .map(|&x| TensorElement::to_f64(&x).unwrap())
            .collect();

        // Basic statistics
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (count - 1).max(1) as f64;
        let std = variance.sqrt();

        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Percentiles
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q25 = percentile_sorted(&sorted_values, 25.0);
        let q50 = percentile_sorted(&sorted_values, 50.0);
        let q75 = percentile_sorted(&sorted_values, 75.0);

        Ok(StatSummary {
            count,
            mean,
            std,
            min,
            max,
            q25,
            q50,
            q75,
        })
    }

    /// Compute covariance matrix for 2D tensor (each column is a variable)
    pub fn cov(&self, mode: StatMode) -> Result<Self> {
        let shape = self.shape();
        if shape.ndim() != 2 {
            return Err(TorshError::dimension_error(
                "Covariance matrix requires 2D tensor",
                "covariance computation",
            ));
        }

        let (n_samples, _n_features) = (shape.dims()[0], shape.dims()[1]);
        if n_samples < 2 {
            return Err(TorshError::InvalidArgument(
                "Need at least 2 samples for covariance".to_string(),
            ));
        }

        // Center the data (subtract mean from each column)
        // For now, implement a simple version that computes global mean and subtracts it
        // TODO: Implement proper column-wise mean subtraction with broadcasting
        let global_mean = self.mean(None, false)?;
        let mean_value = global_mean.item()?;
        let centered = self.sub_scalar(mean_value)?;

        // Compute covariance matrix: (X^T * X) / (n - 1)
        let centered_t = centered.transpose(1, 0)?;
        let cov_unnormalized = centered_t.matmul(&centered)?;

        let divisor = match mode {
            StatMode::Population => n_samples,
            StatMode::Sample => n_samples - 1,
        };

        cov_unnormalized.div_scalar(
            <T as num_traits::FromPrimitive>::from_usize(divisor)
                .unwrap_or_else(|| <T as num_traits::One>::one()),
        )
    }

    /// Compute correlation matrix for 2D tensor
    pub fn corrcoef(&self) -> Result<Self> {
        let cov_matrix = self.cov(StatMode::Sample)?;
        let cov_data = cov_matrix.to_vec()?;
        let n_features = cov_matrix.shape().dims()[0];

        // Extract diagonal elements (variances)
        let mut std_devs = Vec::with_capacity(n_features);
        for i in 0..n_features {
            let variance = TensorElement::to_f64(&cov_data[i * n_features + i]).unwrap();
            std_devs.push(variance.sqrt());
        }

        // Normalize covariance matrix to get correlation matrix
        let mut corr_data = Vec::with_capacity(cov_data.len());
        for i in 0..n_features {
            for j in 0..n_features {
                let cov_val = TensorElement::to_f64(&cov_data[i * n_features + j]).unwrap();
                let corr_val = if std_devs[i] > f64::EPSILON && std_devs[j] > f64::EPSILON {
                    cov_val / (std_devs[i] * std_devs[j])
                } else {
                    0.0
                };
                corr_data.push(<T as TensorElement>::from_f64(corr_val).unwrap());
            }
        }

        Self::from_data(corr_data, vec![n_features, n_features], self.device)
    }
}

/// Helper function to compute percentile from sorted array
fn percentile_sorted(sorted_values: &[f64], q: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let pos = q / 100.0 * (sorted_values.len() - 1) as f64;
    let lower_idx = pos.floor() as usize;
    let upper_idx = (pos.ceil() as usize).min(sorted_values.len() - 1);

    if lower_idx == upper_idx {
        sorted_values[lower_idx]
    } else {
        let weight = pos - pos.floor();
        sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_basic_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_data(data, vec![5], DeviceType::Cpu).unwrap();

        let mean = tensor.mean(None, false).unwrap();
        assert!((mean.to_vec().unwrap()[0] - 3.0_f32).abs() < 1e-6_f32);

        let var_sample = tensor.var(None, false, StatMode::Sample).unwrap();
        assert!((var_sample.to_vec().unwrap()[0] - 2.5_f32).abs() < 1e-6_f32);

        let std_sample = tensor.std(None, false, StatMode::Sample).unwrap();
        assert!((std_sample.to_vec().unwrap()[0] - 2.5_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_percentiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tensor = Tensor::from_data(data, vec![10], DeviceType::Cpu).unwrap();

        let median = tensor.median(None, false).unwrap();
        assert!((median.to_vec().unwrap()[0] - 5.5_f32).abs() < 1e-6_f32);

        let q25 = tensor.percentile(25.0, None, false).unwrap();
        assert!((q25.to_vec().unwrap()[0] - 3.25_f32).abs() < 1e-6_f32);

        let q75 = tensor.percentile(75.0, None, false).unwrap();
        assert!((q75.to_vec().unwrap()[0] - 7.75_f32).abs() < 1e-6_f32);
    }

    #[test]
    fn test_histogram() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let tensor = Tensor::from_data(data, vec![9], DeviceType::Cpu).unwrap();

        let config = HistogramConfig {
            bins: 5,
            min_val: Some(1.0),
            max_val: Some(5.0),
            include_outliers: true,
        };

        let hist = tensor.histogram(&config).unwrap();
        assert_eq!(hist.total_count, 9);
        assert_eq!(hist.counts.len(), 5);
        assert_eq!(hist.edges.len(), 6);
    }

    #[test]
    fn test_correlation() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let x = Tensor::from_data(x_data, vec![5], DeviceType::Cpu).unwrap();
        let y = Tensor::from_data(y_data, vec![5], DeviceType::Cpu).unwrap();

        let corr = x.correlation(&y, CorrelationMethod::Pearson).unwrap();
        assert!((corr - 1.0_f32).abs() < 1e-6_f32); // Should be 1.0 for perfect positive correlation
    }

    #[test]
    fn test_statistical_summary() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tensor = Tensor::from_data(data, vec![10], DeviceType::Cpu).unwrap();

        let summary = tensor.describe().unwrap();
        assert_eq!(summary.count, 10);
        assert!((summary.mean - 5.5).abs() < 1e-6);
        assert!((summary.q50 - 5.5).abs() < 1e-6); // Median
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 10.0);
    }

    #[test]
    fn test_covariance_matrix() {
        // Create a 2D tensor (samples x features)
        let data = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        let tensor = Tensor::from_data(data, vec![4, 2], DeviceType::Cpu).unwrap();

        let cov_matrix = tensor.cov(StatMode::Sample).unwrap();
        assert_eq!(cov_matrix.shape().dims(), &[2, 2]);

        // Check that it's symmetric (off-diagonal elements should be equal)
        let cov_data = cov_matrix.to_vec().unwrap();
        // For a 2x2 matrix [[a, b], [c, d]] stored as [a, b, c, d]:
        // Symmetry means b == c (cov_data[1] == cov_data[2])
        assert!((cov_data[1] as f64 - cov_data[2] as f64).abs() < 1e-6);
    }

    #[test]
    fn test_ranks() {
        let data = vec![3.0, 1.0, 4.0, 2.0, 2.0]; // Contains ties
        let tensor = Tensor::from_data(data, vec![5], DeviceType::Cpu).unwrap();

        let ranks = tensor.rank().unwrap();
        let rank_data = ranks.to_vec().unwrap();

        // Check that ranks are in expected range
        for &rank in rank_data.iter() {
            assert!((1.0_f32..=5.0_f32).contains(&rank));
        }
    }
}
