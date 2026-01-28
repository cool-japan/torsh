//! Quantization parameters and configuration
//!
//! This module provides the QuantizationParams struct and related functionality
//! for managing quantization configuration. It handles parameter calculation
//! from statistics, preset configurations for common quantization schemes,
//! and parameter validation.

use super::types::{QuantizationScheme, QuantizedDType};
use crate::BackendResult;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Quantization parameters
///
/// Contains all the parameters needed to quantize and dequantize tensors,
/// including scale factors, zero points, and metadata about the quantization
/// scheme being used.
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Quantization data type
    ///
    /// Specifies the target quantized data type (e.g., Int8, UInt8, Int4)
    pub dtype: QuantizedDType,

    /// Quantization scheme
    ///
    /// Defines how the quantization mapping is performed (linear, symmetric, etc.)
    pub scheme: QuantizationScheme,

    /// Scale factor(s)
    ///
    /// Maps quantized values back to floating-point range.
    /// For per-channel quantization, contains one scale per channel.
    /// Formula: float_val = scale * (quantized_val - zero_point)
    pub scale: Vec<f32>,

    /// Zero point(s)
    ///
    /// The quantized value that corresponds to floating-point zero.
    /// For per-channel quantization, contains one zero point per channel.
    /// For symmetric quantization, this is always 0.
    pub zero_point: Vec<i32>,

    /// Block size for block-wise quantization
    ///
    /// When using block-wise quantization, specifies the size of each block
    /// that gets its own quantization parameters. None for other schemes.
    pub block_size: Option<usize>,

    /// Minimum value observed during calibration
    ///
    /// Used for parameter calculation and validation. Set during calibration
    /// or when computing parameters from statistics.
    pub min_val: Option<f32>,

    /// Maximum value observed during calibration
    ///
    /// Used for parameter calculation and validation. Set during calibration
    /// or when computing parameters from statistics.
    pub max_val: Option<f32>,
}

impl Default for QuantizationParams {
    /// Default quantization parameters
    ///
    /// Creates parameters for UInt8 linear quantization with scale=1.0
    /// and zero_point=0, suitable for testing and initialization.
    fn default() -> Self {
        Self {
            dtype: QuantizedDType::UInt8,
            scheme: QuantizationScheme::Linear,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }
}

impl QuantizationParams {
    /// Create parameters for INT8 symmetric quantization
    ///
    /// INT8 symmetric quantization is commonly used for weights in neural networks
    /// due to its simplicity and good hardware support. The zero point is always 0,
    /// and the range is symmetric around zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizationParams;
    ///
    /// let params = QuantizationParams::int8_symmetric();
    /// assert_eq!(params.zero_point[0], 0);
    /// ```
    pub fn int8_symmetric() -> Self {
        Self {
            dtype: QuantizedDType::Int8,
            scheme: QuantizationScheme::Symmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create basic quantization parameters with custom scale and zero point
    ///
    /// This is a general-purpose constructor for creating quantization parameters
    /// with custom scale and zero point values. Useful for benchmarking and
    /// testing with specific parameter configurations.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factor for the quantization
    /// * `zero_point` - Zero point for the quantization
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizationParams;
    ///
    /// let params = QuantizationParams::new(255.0, 128);
    /// assert_eq!(params.scale[0], 255.0);
    /// assert_eq!(params.zero_point[0], 128);
    /// ```
    pub fn new(scale: f32, zero_point: i32) -> Self {
        Self {
            dtype: QuantizedDType::UInt8, // Default to UInt8 for general usage
            scheme: QuantizationScheme::Asymmetric,
            scale: vec![scale],
            zero_point: vec![zero_point],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for UINT8 asymmetric quantization
    ///
    /// UInt8 asymmetric quantization is commonly used for activations,
    /// especially after ReLU layers where values are non-negative.
    /// The zero point is typically set to 128 for balanced range utilization.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizationParams;
    ///
    /// let params = QuantizationParams::uint8_asymmetric();
    /// assert_eq!(params.zero_point[0], 128);
    /// ```
    pub fn uint8_asymmetric() -> Self {
        Self {
            dtype: QuantizedDType::UInt8,
            scheme: QuantizationScheme::Asymmetric,
            scale: vec![1.0],
            zero_point: vec![128],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for INT4 symmetric quantization
    ///
    /// INT4 quantization provides extreme compression at the cost of accuracy.
    /// Symmetric INT4 is often used for weights in models where 4-bit precision
    /// is sufficient.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizationParams;
    ///
    /// let params = QuantizationParams::int4_symmetric();
    /// assert_eq!(params.dtype.bits(), 4);
    /// ```
    pub fn int4_symmetric() -> Self {
        Self {
            dtype: QuantizedDType::Int4,
            scheme: QuantizationScheme::Symmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for channel-wise quantization
    ///
    /// Channel-wise quantization applies different quantization parameters
    /// to each channel, providing better accuracy for models with varying
    /// channel sensitivities at the cost of increased parameter storage.
    ///
    /// # Arguments
    ///
    /// * `num_channels` - Number of channels in the tensor
    /// * `dtype` - Quantization data type to use
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::{QuantizationParams, QuantizedDType};
    ///
    /// let params = QuantizationParams::channel_wise(64, QuantizedDType::Int8);
    /// assert_eq!(params.scale.len(), 64);
    /// assert_eq!(params.zero_point.len(), 64);
    /// ```
    pub fn channel_wise(num_channels: usize, dtype: QuantizedDType) -> Self {
        Self {
            dtype,
            scheme: QuantizationScheme::ChannelWise,
            scale: vec![1.0; num_channels],
            zero_point: vec![0; num_channels],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for block-wise quantization
    ///
    /// Block-wise quantization divides the tensor into blocks and applies
    /// different quantization parameters to each block. This can provide
    /// better accuracy than tensor-wise quantization while being more
    /// memory-efficient than channel-wise quantization.
    ///
    /// # Arguments
    ///
    /// * `block_size` - Size of each quantization block
    /// * `dtype` - Quantization data type to use
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::{QuantizationParams, QuantizedDType};
    ///
    /// let params = QuantizationParams::block_wise(128, QuantizedDType::Int8);
    /// assert_eq!(params.block_size, Some(128));
    /// ```
    pub fn block_wise(block_size: usize, dtype: QuantizedDType) -> Self {
        Self {
            dtype,
            scheme: QuantizationScheme::BlockWise,
            scale: vec![1.0], // Will be expanded based on tensor size
            zero_point: vec![0],
            block_size: Some(block_size),
            min_val: None,
            max_val: None,
        }
    }

    /// Calculate quantization parameters from input statistics
    ///
    /// Computes the optimal scale and zero point parameters based on the
    /// observed minimum and maximum values in the data. The calculation
    /// depends on the quantization scheme being used.
    ///
    /// # Arguments
    ///
    /// * `min_val` - Minimum value observed in the data
    /// * `max_val` - Maximum value observed in the data
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if parameters were calculated successfully,
    /// or an error if the statistics are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizationParams;
    ///
    /// let mut params = QuantizationParams::int8_symmetric();
    /// params.from_statistics(-2.0, 2.0).unwrap();
    /// // Scale will be calculated to map [-2.0, 2.0] to [-128, 127]
    /// ```
    pub fn from_statistics(&mut self, min_val: f32, max_val: f32) -> BackendResult<()> {
        // Validate input statistics
        if min_val > max_val {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "min_val must be <= max_val".to_string(),
            ));
        }

        if !min_val.is_finite() || !max_val.is_finite() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "min_val and max_val must be finite".to_string(),
            ));
        }

        self.min_val = Some(min_val);
        self.max_val = Some(max_val);

        let (qmin, qmax) = self.dtype.value_range();
        let qmin = qmin as f32;
        let qmax = qmax as f32;

        match self.scheme {
            QuantizationScheme::Symmetric => {
                self.calculate_symmetric_params(min_val, max_val, qmin, qmax)?;
            }
            QuantizationScheme::Asymmetric | QuantizationScheme::Linear => {
                self.calculate_asymmetric_params(min_val, max_val, qmin, qmax)?;
            }
            QuantizationScheme::Logarithmic => {
                self.calculate_logarithmic_params(min_val, max_val, qmin, qmax)?;
            }
            QuantizationScheme::BlockWise | QuantizationScheme::ChannelWise => {
                // For block-wise and channel-wise, use asymmetric as base
                // Individual blocks/channels will be calculated separately
                self.calculate_asymmetric_params(min_val, max_val, qmin, qmax)?;
            }
        }

        Ok(())
    }

    /// Calculate symmetric quantization parameters
    fn calculate_symmetric_params(
        &mut self,
        min_val: f32,
        max_val: f32,
        qmin: f32,
        qmax: f32,
    ) -> BackendResult<()> {
        let max_range = max_val.abs().max(min_val.abs());
        if max_range == 0.0 {
            self.scale[0] = 1.0;
        } else {
            // For symmetric quantization, we map [-max_range, max_range] to [qmin, qmax]
            self.scale[0] = (2.0 * max_range) / (qmax - qmin);
        }
        self.zero_point[0] = 0;
        Ok(())
    }

    /// Calculate asymmetric quantization parameters
    fn calculate_asymmetric_params(
        &mut self,
        min_val: f32,
        max_val: f32,
        qmin: f32,
        qmax: f32,
    ) -> BackendResult<()> {
        if max_val == min_val {
            // Degenerate case: all values are the same
            self.scale[0] = 1.0;
            self.zero_point[0] = qmin as i32;
        } else {
            // Calculate scale to map [min_val, max_val] to [qmin, qmax]
            self.scale[0] = (max_val - min_val) / (qmax - qmin);

            // Calculate zero point such that min_val maps to qmin
            let zero_point_from_min = qmin - min_val / self.scale[0];
            self.zero_point[0] = zero_point_from_min.round().clamp(qmin, qmax) as i32;
        }
        Ok(())
    }

    /// Calculate logarithmic quantization parameters
    fn calculate_logarithmic_params(
        &mut self,
        min_val: f32,
        max_val: f32,
        qmin: f32,
        qmax: f32,
    ) -> BackendResult<()> {
        // For logarithmic quantization, we need positive values
        if min_val <= 0.0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Logarithmic quantization requires positive values".to_string(),
            ));
        }

        // Use logarithmic scale mapping
        let log_min = min_val.ln();
        let log_max = max_val.ln();

        if log_max == log_min {
            self.scale[0] = 1.0;
            self.zero_point[0] = qmin as i32;
        } else {
            self.scale[0] = (log_max - log_min) / (qmax - qmin);
            self.zero_point[0] = (qmin - log_min / self.scale[0]).round() as i32;
        }
        Ok(())
    }

    /// Validate that the parameters are consistent and usable
    ///
    /// Checks that all parameter vectors have consistent lengths,
    /// scale factors are positive, and zero points are within valid ranges.
    pub fn validate(&self) -> BackendResult<()> {
        // Check that scale and zero_point vectors have consistent lengths
        if self.scale.len() != self.zero_point.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Scale and zero_point vectors must have the same length".to_string(),
            ));
        }

        // Check that all scale factors are positive and finite
        for (i, &scale) in self.scale.iter().enumerate() {
            if scale <= 0.0 || !scale.is_finite() {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Scale factor at index {} must be positive and finite, got {}",
                    i, scale
                )));
            }
        }

        // Check that zero points are within the valid range for the data type
        let (qmin, qmax) = self.dtype.value_range();
        for (i, &zero_point) in self.zero_point.iter().enumerate() {
            if (zero_point as i64) < qmin || (zero_point as i64) > qmax {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Zero point at index {} ({}) is outside valid range [{}, {}]",
                    i, zero_point, qmin, qmax
                )));
            }
        }

        // Scheme-specific validation
        match self.scheme {
            QuantizationScheme::Symmetric => {
                // Symmetric quantization should have zero_point = 0
                for (i, &zero_point) in self.zero_point.iter().enumerate() {
                    if zero_point != 0 {
                        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                            "Symmetric quantization requires zero_point[{}] = 0, got {}",
                            i, zero_point
                        )));
                    }
                }
            }
            QuantizationScheme::BlockWise => {
                // Block-wise quantization should have a block size specified
                if self.block_size.is_none() {
                    return Err(torsh_core::error::TorshError::InvalidArgument(
                        "Block-wise quantization requires block_size to be specified".to_string(),
                    ));
                }
            }
            QuantizationScheme::ChannelWise => {
                // Channel-wise should have multiple parameters
                if self.scale.len() == 1 {
                    return Err(torsh_core::error::TorshError::InvalidArgument(
                        "Channel-wise quantization requires multiple scale/zero_point values"
                            .to_string(),
                    ));
                }
            }
            _ => {} // Other schemes have no specific requirements
        }

        Ok(())
    }

    /// Get the effective number of quantization parameter sets
    ///
    /// Returns the number of independent parameter sets (scale/zero_point pairs)
    /// that this configuration represents. For tensor-wise quantization this is 1,
    /// for channel-wise it's the number of channels.
    pub fn num_parameter_sets(&self) -> usize {
        self.scale.len()
    }

    /// Check if this configuration uses per-channel parameters
    pub fn is_per_channel(&self) -> bool {
        self.scheme.is_per_channel() && self.scale.len() > 1
    }

    /// Get the quantization error bound for this configuration
    ///
    /// Returns the maximum possible quantization error (in the original
    /// floating-point scale) for this quantization configuration.
    pub fn quantization_error_bound(&self) -> f32 {
        // The maximum error is half the quantization step size
        self.scale
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(0.0)
            * 0.5
    }

    /// Calculate the compression ratio achieved by this quantization
    ///
    /// Returns the ratio of original size to quantized size.
    /// Assumes the original data was 32-bit floating point.
    pub fn compression_ratio(&self) -> f32 {
        let bits_per_value = self.dtype.bits() as f32;
        32.0 / bits_per_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = QuantizationParams::default();
        assert_eq!(params.dtype, QuantizedDType::UInt8);
        assert_eq!(params.scheme, QuantizationScheme::Linear);
        assert_eq!(params.scale, vec![1.0]);
        assert_eq!(params.zero_point, vec![0]);
    }

    #[test]
    fn test_preset_configs() {
        let int8_sym = QuantizationParams::int8_symmetric();
        assert_eq!(int8_sym.dtype, QuantizedDType::Int8);
        assert_eq!(int8_sym.scheme, QuantizationScheme::Symmetric);
        assert_eq!(int8_sym.zero_point[0], 0);

        let uint8_asym = QuantizationParams::uint8_asymmetric();
        assert_eq!(uint8_asym.dtype, QuantizedDType::UInt8);
        assert_eq!(uint8_asym.scheme, QuantizationScheme::Asymmetric);
        assert_eq!(uint8_asym.zero_point[0], 128);

        let int4_sym = QuantizationParams::int4_symmetric();
        assert_eq!(int4_sym.dtype, QuantizedDType::Int4);
        assert_eq!(int4_sym.zero_point[0], 0);
    }

    #[test]
    fn test_channel_wise_params() {
        let params = QuantizationParams::channel_wise(64, QuantizedDType::Int8);
        assert_eq!(params.scheme, QuantizationScheme::ChannelWise);
        assert_eq!(params.scale.len(), 64);
        assert_eq!(params.zero_point.len(), 64);
        assert!(params.is_per_channel());
    }

    #[test]
    fn test_block_wise_params() {
        let params = QuantizationParams::block_wise(128, QuantizedDType::Int8);
        assert_eq!(params.scheme, QuantizationScheme::BlockWise);
        assert_eq!(params.block_size, Some(128));
    }

    #[test]
    fn test_from_statistics_symmetric() {
        let mut params = QuantizationParams::int8_symmetric();
        params.from_statistics(-2.0, 2.0).unwrap();

        assert_eq!(params.zero_point[0], 0);
        assert!(params.scale[0] > 0.0);
        assert_eq!(params.min_val, Some(-2.0));
        assert_eq!(params.max_val, Some(2.0));
    }

    #[test]
    fn test_from_statistics_asymmetric() {
        let mut params = QuantizationParams::uint8_asymmetric();
        params.from_statistics(0.0, 255.0).unwrap();

        assert!(params.scale[0] > 0.0);
        assert!(params.zero_point[0] >= 0 && params.zero_point[0] <= 255);
    }

    #[test]
    fn test_from_statistics_invalid() {
        let mut params = QuantizationParams::default();

        // min > max should fail
        assert!(params.from_statistics(2.0, 1.0).is_err());

        // Non-finite values should fail
        assert!(params.from_statistics(f32::NAN, 1.0).is_err());
        assert!(params.from_statistics(1.0, f32::INFINITY).is_err());
    }

    #[test]
    fn test_validation() {
        let mut params = QuantizationParams::default();
        assert!(params.validate().is_ok());

        // Mismatched vector lengths should fail
        params.scale.push(2.0);
        assert!(params.validate().is_err());

        // Reset and test negative scale
        params.scale = vec![-1.0];
        params.zero_point = vec![0];
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validation_symmetric() {
        let mut params = QuantizationParams::int8_symmetric();
        params.zero_point[0] = 10; // Should fail for symmetric
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_compression_ratio() {
        let int8_params = QuantizationParams::int8_symmetric();
        assert_eq!(int8_params.compression_ratio(), 4.0); // 32 bits -> 8 bits

        let int4_params = QuantizationParams::int4_symmetric();
        assert_eq!(int4_params.compression_ratio(), 8.0); // 32 bits -> 4 bits
    }

    #[test]
    fn test_error_bound() {
        let mut params = QuantizationParams::int8_symmetric();
        params.scale = vec![0.1];
        assert_eq!(params.quantization_error_bound(), 0.05); // Half the scale
    }

    #[test]
    fn test_num_parameter_sets() {
        let tensor_wise = QuantizationParams::default();
        assert_eq!(tensor_wise.num_parameter_sets(), 1);

        let channel_wise = QuantizationParams::channel_wise(64, QuantizedDType::Int8);
        assert_eq!(channel_wise.num_parameter_sets(), 64);
    }
}
