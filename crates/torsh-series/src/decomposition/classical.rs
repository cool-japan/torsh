//! Classical time series decomposition methods

use crate::TimeSeries;
use torsh_tensor::creation::{ones, zeros};

// Re-export STLResult for X11 compatibility
pub use super::stl::STLResult;

/// Compute centered moving average
///
/// For even window sizes, uses a weighted average at the center.
/// At the boundaries, pads with the original values.
fn centered_moving_average(data: &[f32], window: usize) -> Vec<f32> {
    let n = data.len();
    let mut result = vec![0.0; n];

    if window == 0 || window > n {
        return data.to_vec();
    }

    let half_window = window / 2;

    for i in 0..n {
        // Determine the range for averaging
        let start = if i >= half_window { i - half_window } else { 0 };
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Compute average
        let sum: f32 = data[start..end].iter().sum();
        let count = (end - start) as f32;
        result[i] = sum / count;

        // For boundaries, use original value if window doesn't fit well
        if i < half_window || i >= n - half_window {
            // Use a blend: weighted average between original and MA
            let weight = if i < half_window {
                i as f32 / half_window as f32
            } else {
                (n - 1 - i) as f32 / half_window as f32
            };
            result[i] = weight * result[i] + (1.0 - weight) * data[i];
        }
    }

    result
}

/// Apply seasonal filter to smooth seasonal component
///
/// Uses convolution with the provided filter weights.
/// The filter is applied with wrapping (circular) to handle period boundaries.
fn apply_seasonal_filter(seasonal: &[f32], filter: &[f64]) -> Vec<f32> {
    let n = seasonal.len();
    let filter_len = filter.len();

    if filter_len == 0 || filter_len > n {
        return seasonal.to_vec();
    }

    let mut result = vec![0.0; n];
    let half_filter = filter_len / 2;

    for i in 0..n {
        let mut sum = 0.0;
        for (j, &weight) in filter.iter().enumerate() {
            // Circular indexing for seasonal pattern
            let idx = if i + j >= half_filter {
                (i + j - half_filter) % n
            } else {
                (n + i + j - half_filter) % n
            };
            sum += seasonal[idx] * weight as f32;
        }
        result[i] = sum;
    }

    result
}

/// X11 decomposition
pub struct X11Decomposition {
    period: usize,
    seasonal_filter: Option<Vec<f64>>,
}

impl X11Decomposition {
    /// Create a new X11 decomposition
    pub fn new(period: usize) -> Self {
        Self {
            period,
            seasonal_filter: None,
        }
    }

    /// Set custom seasonal filter weights
    pub fn with_seasonal_filter(mut self, filter: Vec<f64>) -> Self {
        self.seasonal_filter = Some(filter);
        self
    }

    /// Apply X11 decomposition
    ///
    /// # Algorithm
    /// Simplified X11 implementation:
    /// 1. Extract trend using centered moving average
    /// 2. Apply seasonal filtering (custom filter if provided, otherwise simple averaging)
    /// 3. Compute seasonal component with smoothing
    /// 4. Compute residuals
    ///
    /// Note: This is a simplified version. The full X11 algorithm includes
    /// multiple iterations with outlier detection and sophisticated filters.
    pub fn fit(&self, series: &TimeSeries) -> STLResult {
        let data = series.values.to_vec().unwrap_or_default();
        let n = data.len();

        if n < self.period * 2 {
            // Not enough data for decomposition
            return STLResult {
                trend: series.values.clone(),
                seasonal: zeros(&[n]).unwrap(),
                residual: zeros(&[n]).unwrap(),
            };
        }

        // Step 1: Extract trend using centered moving average
        let trend_data = centered_moving_average(&data, self.period);

        // Step 2: Detrend the series
        let mut detrended = vec![0.0; n];
        for i in 0..n {
            detrended[i] = data[i] - trend_data[i];
        }

        // Step 3: Extract seasonal component with optional custom filtering
        let mut seasonal_pattern = vec![0.0; self.period];
        let mut counts = vec![0; self.period];

        for (i, &val) in detrended.iter().enumerate() {
            let season_idx = i % self.period;
            // Use values from the middle section where trend is well-defined
            if i >= self.period / 2 && i < n - self.period / 2 {
                seasonal_pattern[season_idx] += val;
                counts[season_idx] += 1;
            }
        }

        // Average seasonal pattern
        for i in 0..self.period {
            if counts[i] > 0 {
                seasonal_pattern[i] /= counts[i] as f32;
            }
        }

        // Apply custom seasonal filter if provided
        if let Some(ref filter) = self.seasonal_filter {
            seasonal_pattern = apply_seasonal_filter(&seasonal_pattern, filter);
        }

        // Normalize seasonal component to have mean 0
        let seasonal_mean: f32 = seasonal_pattern.iter().sum::<f32>() / self.period as f32;
        for val in seasonal_pattern.iter_mut() {
            *val -= seasonal_mean;
        }

        // Replicate seasonal pattern for full series length
        let seasonal_data: Vec<f32> = (0..n).map(|i| seasonal_pattern[i % self.period]).collect();

        // Step 4: Compute residuals
        let residual_data: Vec<f32> = (0..n)
            .map(|i| data[i] - trend_data[i] - seasonal_data[i])
            .collect();

        STLResult {
            trend: torsh_tensor::Tensor::from_vec(trend_data, &[n]).unwrap(),
            seasonal: torsh_tensor::Tensor::from_vec(seasonal_data, &[n]).unwrap(),
            residual: torsh_tensor::Tensor::from_vec(residual_data, &[n]).unwrap(),
        }
    }
}

/// Classical additive decomposition
pub struct AdditiveDecomposition {
    period: usize,
}

impl AdditiveDecomposition {
    /// Create a new additive decomposition
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Apply additive decomposition: Y(t) = Trend(t) + Seasonal(t) + Residual(t)
    ///
    /// # Algorithm
    /// 1. Extract trend using centered moving average of window = period
    /// 2. Detrend: detrended(t) = Y(t) - Trend(t)
    /// 3. Extract seasonal component by averaging each seasonal position
    /// 4. Residual(t) = Y(t) - Trend(t) - Seasonal(t)
    pub fn fit(&self, series: &TimeSeries) -> STLResult {
        let data = series.values.to_vec().unwrap_or_default();
        let n = data.len();

        if n < self.period * 2 {
            // Not enough data for decomposition
            return STLResult {
                trend: series.values.clone(),
                seasonal: zeros(&[n]).unwrap(),
                residual: zeros(&[n]).unwrap(),
            };
        }

        // Step 1: Extract trend using centered moving average
        let trend_data = centered_moving_average(&data, self.period);

        // Step 2: Detrend the series
        let mut detrended = vec![0.0; n];
        for i in 0..n {
            detrended[i] = data[i] - trend_data[i];
        }

        // Step 3: Extract seasonal component
        // Average the detrended values at each seasonal position
        let mut seasonal_pattern = vec![0.0; self.period];
        let mut counts = vec![0; self.period];

        for (i, &val) in detrended.iter().enumerate() {
            let season_idx = i % self.period;
            // Only use valid detrended values (where trend was computed)
            if trend_data[i] != data[i] || i >= self.period / 2 && i < n - self.period / 2 {
                seasonal_pattern[season_idx] += val;
                counts[season_idx] += 1;
            }
        }

        // Average and normalize seasonal pattern (mean = 0)
        for i in 0..self.period {
            if counts[i] > 0 {
                seasonal_pattern[i] /= counts[i] as f32;
            }
        }

        // Normalize seasonal component to have mean 0
        let seasonal_mean: f32 = seasonal_pattern.iter().sum::<f32>() / self.period as f32;
        for val in seasonal_pattern.iter_mut() {
            *val -= seasonal_mean;
        }

        // Replicate seasonal pattern for full series length
        let seasonal_data: Vec<f32> = (0..n).map(|i| seasonal_pattern[i % self.period]).collect();

        // Step 4: Compute residuals
        let residual_data: Vec<f32> = (0..n)
            .map(|i| data[i] - trend_data[i] - seasonal_data[i])
            .collect();

        STLResult {
            trend: torsh_tensor::Tensor::from_vec(trend_data, &[n]).unwrap(),
            seasonal: torsh_tensor::Tensor::from_vec(seasonal_data, &[n]).unwrap(),
            residual: torsh_tensor::Tensor::from_vec(residual_data, &[n]).unwrap(),
        }
    }
}

/// Classical multiplicative decomposition
pub struct MultiplicativeDecomposition {
    period: usize,
}

impl MultiplicativeDecomposition {
    /// Create a new multiplicative decomposition
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Apply multiplicative decomposition: Y(t) = Trend(t) * Seasonal(t) * Residual(t)
    ///
    /// # Algorithm
    /// 1. Extract trend using centered moving average of window = period
    /// 2. Detrend: detrended(t) = Y(t) / Trend(t)
    /// 3. Extract seasonal component by averaging each seasonal position
    /// 4. Residual(t) = Y(t) / (Trend(t) * Seasonal(t))
    ///
    /// # Note
    /// Requires all values to be positive (> 0). If negative or zero values
    /// are present, they are treated as small positive values (epsilon = 1e-8).
    pub fn fit(&self, series: &TimeSeries) -> STLResult {
        let data = series.values.to_vec().unwrap_or_default();
        let n = data.len();

        if n < self.period * 2 {
            // Not enough data for decomposition
            return STLResult {
                trend: series.values.clone(),
                seasonal: ones(&[n]).unwrap(), // All ones for multiplicative
                residual: ones(&[n]).unwrap(),
            };
        }

        // Ensure all data is positive (multiplicative decomposition requirement)
        let epsilon = 1e-8;
        let positive_data: Vec<f32> = data.iter().map(|&x| x.max(epsilon)).collect();

        // Step 1: Extract trend using centered moving average
        let trend_data = centered_moving_average(&positive_data, self.period);

        // Ensure trend is positive
        let trend_data: Vec<f32> = trend_data.iter().map(|&x| x.max(epsilon)).collect();

        // Step 2: Detrend the series (divide by trend)
        let mut detrended = vec![1.0; n];
        for i in 0..n {
            detrended[i] = positive_data[i] / trend_data[i];
        }

        // Step 3: Extract seasonal component
        // Average the detrended values at each seasonal position
        let mut seasonal_pattern = vec![0.0; self.period];
        let mut counts = vec![0; self.period];

        for (i, &val) in detrended.iter().enumerate() {
            let season_idx = i % self.period;
            // Only use valid detrended values (where trend was computed reasonably)
            if i >= self.period / 2 && i < n - self.period / 2 {
                seasonal_pattern[season_idx] += val;
                counts[season_idx] += 1;
            }
        }

        // Average seasonal pattern
        for i in 0..self.period {
            if counts[i] > 0 {
                seasonal_pattern[i] /= counts[i] as f32;
            } else {
                seasonal_pattern[i] = 1.0; // Neutral element for multiplication
            }
        }

        // Normalize seasonal component to have mean 1 (multiplicative neutral)
        let seasonal_mean: f32 = seasonal_pattern.iter().sum::<f32>() / self.period as f32;
        if seasonal_mean > epsilon {
            for val in seasonal_pattern.iter_mut() {
                *val /= seasonal_mean;
            }
        }

        // Replicate seasonal pattern for full series length
        let seasonal_data: Vec<f32> = (0..n).map(|i| seasonal_pattern[i % self.period]).collect();

        // Step 4: Compute residuals (multiplicative: divide)
        let residual_data: Vec<f32> = (0..n)
            .map(|i| positive_data[i] / (trend_data[i] * seasonal_data[i]))
            .collect();

        STLResult {
            trend: torsh_tensor::Tensor::from_vec(trend_data, &[n]).unwrap(),
            seasonal: torsh_tensor::Tensor::from_vec(seasonal_data, &[n]).unwrap(),
            residual: torsh_tensor::Tensor::from_vec(residual_data, &[n]).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;
    use torsh_tensor::Tensor;

    fn create_test_series() -> TimeSeries {
        // Create synthetic time series with trend and seasonality
        let mut data = Vec::new();
        for i in 0..50 {
            let trend = i as f32 * 0.1;
            let seasonal = (i as f32 * 2.0 * std::f32::consts::PI / 12.0).sin() * 2.0;
            let noise = 0.1;
            data.push(trend + seasonal + noise);
        }
        let tensor = Tensor::from_vec(data, &[50]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_x11_decomposition() {
        let series = create_test_series();
        let x11 = X11Decomposition::new(12);
        let result = x11.fit(&series);

        assert_eq!(result.trend.shape().dims()[0], series.len());
        assert_eq!(result.seasonal.shape().dims()[0], series.len());
        assert_eq!(result.residual.shape().dims()[0], series.len());
    }

    #[test]
    fn test_x11_with_filter() {
        let filter = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let x11 = X11Decomposition::new(12).with_seasonal_filter(filter);
        assert!(x11.seasonal_filter.is_some());
    }

    #[test]
    fn test_additive_decomposition() {
        let series = create_test_series();
        let decomp = AdditiveDecomposition::new(12);
        let result = decomp.fit(&series);

        assert_eq!(result.trend.shape().dims()[0], series.len());
    }

    #[test]
    fn test_multiplicative_decomposition() {
        let series = create_test_series();
        let decomp = MultiplicativeDecomposition::new(12);
        let result = decomp.fit(&series);

        assert_eq!(result.trend.shape().dims()[0], series.len());
    }
}
