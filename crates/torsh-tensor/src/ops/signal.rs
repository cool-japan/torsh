//! Signal processing operations for tensors
//!
//! This module provides comprehensive signal processing operations:
//! - Cross-correlation (correlate1d)
//! - Digital filters (moving average, Gaussian, median)
//! - Signal smoothing and noise reduction
//! - 1D signal analysis and processing
//!
//! All operations support 1D tensors and include proper boundary handling and validation.

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

/// Signal processing operations for tensors with comprehensive type constraints
impl<T> Tensor<T>
where
    T: TensorElement
        + FloatElement
        + Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + PartialOrd,
{
    /// Cross-correlation operation
    ///
    /// Computes the cross-correlation between two 1D tensors.
    /// Cross-correlation measures the similarity between two signals as one is shifted relative to the other.
    ///
    /// # Arguments
    /// * `other` - The other tensor to correlate with (kernel/filter)
    /// * `mode` - The correlation mode:
    ///   - "full": Output size is input_len + other_len - 1 (default)
    ///   - "valid": Output size is input_len - other_len + 1 (only where they fully overlap)
    ///   - "same": Output size is input_len (centered correlation)
    ///
    /// # Returns
    /// * `Result<Self>` - The cross-correlation result
    ///
    /// # Example
    /// ```rust
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let signal = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
    /// let kernel = Tensor::from_data(vec![1.0f32, 0.5], vec![2], DeviceType::Cpu).unwrap();
    /// let correlation = signal.correlate1d(&kernel, "same").unwrap();
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn correlate1d(&self, other: &Self, mode: &str) -> Result<Self> {
        let input_shape = self.shape();
        let other_shape = other.shape();

        // Validate dimensions
        if input_shape.dims().len() != 1 || other_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensors, got {}D and {}D",
                input_shape.dims().len(),
                other_shape.dims().len()
            )));
        }

        let input_data = self.data()?;
        let other_data = other.data()?;

        let input_len = input_shape.dims()[0];
        let other_len = other_shape.dims()[0];

        let output_len = match mode {
            "full" => input_len + other_len - 1,
            "valid" => {
                if input_len >= other_len {
                    input_len - other_len + 1
                } else {
                    0
                }
            }
            "same" => input_len,
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Mode must be 'full', 'valid', or 'same'".to_string(),
                ))
            }
        };

        if output_len == 0 {
            return Self::from_data(vec![], vec![0], self.device);
        }

        let mut output_data = vec![T::default(); output_len];

        // Calculate the starting position for each mode
        let start_pos = match mode {
            "full" => 0,
            "valid" => other_len - 1,
            "same" => (other_len - 1) / 2,
            _ => 0,
        };

        // Perform cross-correlation
        for i in 0..output_len {
            let mut sum = T::default();
            let actual_i = i + start_pos;

            for j in 0..other_len {
                let input_idx = actual_i as isize - j as isize;
                if input_idx >= 0 && (input_idx as usize) < input_len {
                    sum = sum + input_data[input_idx as usize] * other_data[j];
                }
            }
            output_data[i] = sum;
        }

        Self::from_data(output_data, vec![output_len], self.device)
    }

    /// Simple moving average filter
    ///
    /// Applies a simple moving average filter to smooth the signal.
    /// This filter reduces noise by averaging values within a sliding window.
    ///
    /// # Arguments
    /// * `window_size` - The size of the moving average window
    ///
    /// # Returns
    /// * `Result<Self>` - The filtered tensor with length (input_len - window_size + 1)
    ///
    /// # Example
    /// ```rust
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let noisy_signal = Tensor::from_data(vec![1.0f32, 5.0, 2.0, 8.0, 3.0], vec![5], DeviceType::Cpu).unwrap();
    /// let smoothed = noisy_signal.moving_average(3).unwrap();
    /// ```
    pub fn moving_average(&self, window_size: usize) -> Result<Self> {
        let input_shape = self.shape();

        // Validate dimensions
        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D",
                input_shape.dims().len()
            )));
        }

        if window_size == 0 {
            return Err(TorshError::InvalidArgument(
                "Window size must be greater than 0".to_string(),
            ));
        }

        let input_data = self.data()?;
        let input_len = input_shape.dims()[0];

        if window_size > input_len {
            return Err(TorshError::InvalidArgument(format!(
                "Window size ({window_size}) cannot be larger than input length ({input_len})"
            )));
        }

        let output_len = input_len - window_size + 1;
        let mut output_data = vec![T::default(); output_len];

        // Calculate moving average
        for i in 0..output_len {
            let mut sum = T::default();
            for j in 0..window_size {
                sum = sum + input_data[i + j];
            }
            // Convert window_size to T for division
            let window_size_t = T::from_f64(window_size as f64).unwrap_or(T::default());
            output_data[i] = sum / window_size_t;
        }

        Self::from_data(output_data, vec![output_len], self.device)
    }

    /// Gaussian filter
    ///
    /// Applies a Gaussian filter to smooth the signal using a Gaussian kernel.
    /// This provides better frequency response than simple moving average.
    ///
    /// # Arguments
    /// * `sigma` - The standard deviation of the Gaussian kernel (controls smoothing strength)
    /// * `kernel_size` - The size of the Gaussian kernel (should be odd)
    ///
    /// # Returns
    /// * `Result<Self>` - The filtered tensor (same length as input with padding)
    ///
    /// # Example
    /// ```rust
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let signal = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
    /// let smoothed = signal.gaussian_filter(1.0, 5).unwrap();
    /// ```
    pub fn gaussian_filter(&self, sigma: f64, kernel_size: usize) -> Result<Self> {
        let input_shape = self.shape();

        // Validate dimensions
        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D",
                input_shape.dims().len()
            )));
        }

        if kernel_size == 0 || kernel_size % 2 == 0 {
            return Err(TorshError::InvalidArgument(
                "Kernel size must be odd and greater than 0".to_string(),
            ));
        }

        if sigma <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Sigma must be greater than 0".to_string(),
            ));
        }

        // Create Gaussian kernel
        let half_size = kernel_size / 2;
        let mut kernel_data = vec![T::default(); kernel_size];
        let mut kernel_sum = T::default();

        for (i, kernel_val) in kernel_data.iter_mut().enumerate() {
            let x = i as f64 - half_size as f64;
            let gauss_val = (-0.5 * x * x / (sigma * sigma)).exp();
            let kernel_val_computed = T::from_f64(gauss_val).unwrap_or(T::default());
            *kernel_val = kernel_val_computed;
            kernel_sum = kernel_sum + kernel_val_computed;
        }

        // Normalize kernel so the sum equals 1
        for kernel_val in kernel_data.iter_mut() {
            *kernel_val = *kernel_val / kernel_sum;
        }

        // Create kernel tensor
        let kernel = Self::from_data(kernel_data, vec![kernel_size], self.device)?;

        // Apply convolution with padding
        let padding = half_size;
        self.conv1d(&kernel, None, 1, padding, 1, 1)
    }

    /// Median filter
    ///
    /// Applies a median filter to remove impulse noise from the signal.
    /// This filter is effective at preserving edges while removing outliers.
    ///
    /// # Arguments
    /// * `window_size` - The size of the median filter window (should be odd)
    ///
    /// # Returns
    /// * `Result<Self>` - The filtered tensor (same length as input)
    ///
    /// # Example
    /// ```rust
    /// use torsh_tensor::Tensor;
    /// use torsh_core::device::DeviceType;
    ///
    /// let noisy_signal = Tensor::from_data(vec![1.0f32, 100.0, 2.0, 3.0, 200.0, 4.0], vec![6], DeviceType::Cpu).unwrap();
    /// let denoised = noisy_signal.median_filter(3).unwrap();
    /// ```
    pub fn median_filter(&self, window_size: usize) -> Result<Self> {
        let input_shape = self.shape();

        // Validate dimensions
        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D",
                input_shape.dims().len()
            )));
        }

        if window_size == 0 || window_size % 2 == 0 {
            return Err(TorshError::InvalidArgument(
                "Window size must be odd and greater than 0".to_string(),
            ));
        }

        let input_data = self.data()?;
        let input_len = input_shape.dims()[0];

        if window_size > input_len {
            return Err(TorshError::InvalidArgument(format!(
                "Window size ({window_size}) cannot be larger than input length ({input_len})"
            )));
        }

        let half_window = window_size / 2;
        let mut output_data = vec![T::default(); input_len];

        // Apply median filter with boundary handling
        for (i, output_val) in output_data.iter_mut().enumerate() {
            let start = i.saturating_sub(half_window);
            let end = if i + half_window < input_len {
                i + half_window + 1
            } else {
                input_len
            };

            // Extract window values
            let mut window_values: Vec<T> = input_data[start..end].to_vec();

            // Sort to find median
            window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Get median value
            let median_idx = window_values.len() / 2;
            *output_val = window_values[median_idx];
        }

        Self::from_data(output_data, vec![input_len], self.device)
    }

    /// Convolution-based filter using a custom kernel
    ///
    /// Applies a custom convolution kernel for filtering. This is a generic
    /// filtering operation that can implement various filter types.
    ///
    /// # Arguments
    /// * `kernel` - The convolution kernel/filter
    /// * `mode` - Convolution mode ("full", "valid", "same")
    ///
    /// # Returns
    /// * `Result<Self>` - The filtered signal
    pub fn filter1d(&self, kernel: &Self, mode: &str) -> Result<Self> {
        // For convolution filtering, we can reuse the correlation function
        // but with a flipped kernel (convolution = correlation with flipped kernel)
        let kernel_data = kernel.data()?;
        let flipped_kernel_data: Vec<T> = kernel_data.iter().rev().copied().collect();
        let flipped_kernel = Self::from_data(
            flipped_kernel_data,
            kernel.shape().dims().to_vec(),
            kernel.device,
        )?;

        self.correlate1d(&flipped_kernel, mode)
    }

    /// High-pass filter using simple difference
    ///
    /// Applies a simple high-pass filter by computing differences between adjacent samples.
    /// Useful for edge detection and removing low-frequency components.
    ///
    /// # Returns
    /// * `Result<Self>` - The high-pass filtered signal (length = input_len - 1)
    pub fn highpass_diff(&self) -> Result<Self> {
        let input_shape = self.shape();

        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D",
                input_shape.dims().len()
            )));
        }

        let input_data = self.data()?;
        let input_len = input_shape.dims()[0];

        if input_len < 2 {
            return Err(TorshError::InvalidArgument(
                "Input must have at least 2 elements for difference operation".to_string(),
            ));
        }

        let output_len = input_len - 1;
        let mut output_data = vec![T::default(); output_len];

        // Compute differences
        for i in 0..output_len {
            output_data[i] = input_data[i + 1] - input_data[i];
        }

        Self::from_data(output_data, vec![output_len], self.device)
    }

    /// Low-pass filter using simple averaging of adjacent pairs
    ///
    /// Applies a simple low-pass filter by averaging adjacent samples.
    /// Useful for noise reduction and smoothing.
    ///
    /// # Returns
    /// * `Result<Self>` - The low-pass filtered signal (length = input_len - 1)
    pub fn lowpass_avg(&self) -> Result<Self> {
        let input_shape = self.shape();

        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D",
                input_shape.dims().len()
            )));
        }

        let input_data = self.data()?;
        let input_len = input_shape.dims()[0];

        if input_len < 2 {
            return Err(TorshError::InvalidArgument(
                "Input must have at least 2 elements for averaging operation".to_string(),
            ));
        }

        let output_len = input_len - 1;
        let mut output_data = vec![T::default(); output_len];
        let two = T::from_f64(2.0).unwrap_or_else(|| <T as TensorElement>::one() + <T as TensorElement>::one());

        // Compute averages
        for i in 0..output_len {
            output_data[i] = (input_data[i] + input_data[i + 1]) / two;
        }

        Self::from_data(output_data, vec![output_len], self.device)
    }
}

/// Signal processing utility functions
pub mod utils {
    use super::*;

    /// Create a Gaussian kernel for filtering
    pub fn gaussian_kernel<T: FloatElement + TensorElement + Default>(
        size: usize,
        sigma: f64,
        device: torsh_core::device::DeviceType,
    ) -> Result<Tensor<T>> {
        if size == 0 || size % 2 == 0 {
            return Err(TorshError::InvalidArgument(
                "Kernel size must be odd and greater than 0".to_string(),
            ));
        }

        if sigma <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Sigma must be greater than 0".to_string(),
            ));
        }

        let half_size = size / 2;
        let mut kernel_data = vec![T::default(); size];
        let mut kernel_sum = T::default();

        for (i, kernel_val) in kernel_data.iter_mut().enumerate() {
            let x = i as f64 - half_size as f64;
            let gauss_val = (-0.5 * x * x / (sigma * sigma)).exp();
            let kernel_val_computed = T::from_f64(gauss_val).unwrap_or(T::default());
            *kernel_val = kernel_val_computed;
            kernel_sum = kernel_sum + kernel_val_computed;
        }

        // Normalize kernel
        for kernel_val in kernel_data.iter_mut() {
            *kernel_val = *kernel_val / kernel_sum;
        }

        Tensor::from_data(kernel_data, vec![size], device)
    }

    /// Create a simple box filter kernel
    pub fn box_kernel<T: FloatElement + TensorElement + Default>(
        size: usize,
        device: torsh_core::device::DeviceType,
    ) -> Result<Tensor<T>> {
        if size == 0 {
            return Err(TorshError::InvalidArgument(
                "Kernel size must be greater than 0".to_string(),
            ));
        }

        let kernel_val = T::from_f64(1.0 / size as f64).unwrap_or(T::default());
        let kernel_data = vec![kernel_val; size];

        Tensor::from_data(kernel_data, vec![size], device)
    }

    /// Create a simple derivative kernel [-1, 0, 1] for edge detection
    pub fn derivative_kernel<T: FloatElement + TensorElement + Default>(
        device: torsh_core::device::DeviceType,
    ) -> Result<Tensor<T>> {
        let kernel_data = vec![
            T::from_f64(-1.0).unwrap_or(T::default()),
            <T as TensorElement>::zero(),
            T::from_f64(1.0).unwrap_or(T::default()),
        ];

        Tensor::from_data(kernel_data, vec![3], device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_correlate1d_full() {
        let signal = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let kernel = Tensor::from_data(vec![1.0f32, 0.5], vec![2], DeviceType::Cpu).unwrap();

        let result = signal.correlate1d(&kernel, "full").unwrap();
        let data = result.data().unwrap();

        // Expected output size: 4 + 2 - 1 = 5
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn test_correlate1d_valid() {
        let signal = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let kernel = Tensor::from_data(vec![1.0f32, 0.5], vec![2], DeviceType::Cpu).unwrap();

        let result = signal.correlate1d(&kernel, "valid").unwrap();
        let data = result.data().unwrap();

        // Expected output size: 4 - 2 + 1 = 3
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn test_correlate1d_same() {
        let signal = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let kernel = Tensor::from_data(vec![1.0f32, 0.5], vec![2], DeviceType::Cpu).unwrap();

        let result = signal.correlate1d(&kernel, "same").unwrap();
        let data = result.data().unwrap();

        // Expected output size: 4 (same as input)
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn test_moving_average() {
        let signal = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let result = signal.moving_average(3).unwrap();
        let data = result.data().unwrap();

        // Expected output: [(1+2+3)/3, (2+3+4)/3, (3+4+5)/3] = [2.0, 3.0, 4.0]
        assert_eq!(data.len(), 3);
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
        assert!((data[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_filter() {
        let signal = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();
        let result = signal.gaussian_filter(1.0, 3);

        // Should not error and should produce smoothed output
        assert!(result.is_ok());
        let smoothed = result.unwrap();
        assert_eq!(smoothed.data().unwrap().len(), signal.shape().numel());
    }

    #[test]
    fn test_median_filter() {
        let signal = Tensor::from_data(vec![1.0f32, 100.0, 2.0, 3.0, 200.0, 4.0], vec![6], DeviceType::Cpu).unwrap();
        let result = signal.median_filter(3).unwrap();
        let data = result.data().unwrap();

        // Median filter should reduce the impact of outliers (100.0, 200.0)
        assert_eq!(data.len(), 6);
        // The large outliers should be replaced by median values
        assert!(data[1] < 50.0); // 100.0 should be reduced
        assert!(data[4] < 50.0); // 200.0 should be reduced
    }

    #[test]
    fn test_highpass_diff() {
        let signal = Tensor::from_data(vec![1.0f32, 3.0, 2.0, 5.0], vec![4], DeviceType::Cpu).unwrap();
        let result = signal.highpass_diff().unwrap();
        let data = result.data().unwrap();

        // Expected differences: [3-1, 2-3, 5-2] = [2.0, -1.0, 3.0]
        assert_eq!(data.len(), 3);
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - (-1.0)).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lowpass_avg() {
        let signal = Tensor::from_data(vec![1.0f32, 3.0, 2.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let result = signal.lowpass_avg().unwrap();
        let data = result.data().unwrap();

        // Expected averages: [(1+3)/2, (3+2)/2, (2+4)/2] = [2.0, 2.5, 3.0]
        assert_eq!(data.len(), 3);
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 2.5).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_signal_processing_error_cases() {
        // Test invalid dimensions
        let signal_2d = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        assert!(signal_2d.moving_average(2).is_err());
        assert!(signal_2d.median_filter(3).is_err());

        // Test invalid parameters
        let signal_1d = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        assert!(signal_1d.moving_average(0).is_err()); // Zero window size
        assert!(signal_1d.moving_average(5).is_err()); // Window larger than signal
        assert!(signal_1d.median_filter(4).is_err()); // Even window size
        assert!(signal_1d.gaussian_filter(0.0, 3).is_err()); // Zero sigma
        assert!(signal_1d.gaussian_filter(1.0, 4).is_err()); // Even kernel size
    }

    #[test]
    fn test_utils_gaussian_kernel() {
        let kernel = utils::gaussian_kernel::<f32>(5, 1.0, DeviceType::Cpu).unwrap();
        let data = kernel.data().unwrap();

        // Kernel should sum to approximately 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Kernel should be symmetric
        assert!((data[0] - data[4]).abs() < 1e-6);
        assert!((data[1] - data[3]).abs() < 1e-6);

        // Center should be the maximum value
        assert!(data[2] >= data[0]);
        assert!(data[2] >= data[1]);
    }

    #[test]
    fn test_utils_box_kernel() {
        let kernel = utils::box_kernel::<f32>(3, DeviceType::Cpu).unwrap();
        let data = kernel.data().unwrap();

        // All values should be 1/3
        for &val in data.iter() {
            assert!((val - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_utils_derivative_kernel() {
        let kernel = utils::derivative_kernel::<f32>(DeviceType::Cpu).unwrap();
        let data = kernel.data().unwrap();

        // Should be [-1, 0, 1]
        assert_eq!(data.len(), 3);
        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
    }
}