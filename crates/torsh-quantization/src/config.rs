//! Quantization configuration types and builders
//!
//! This module provides the core configuration types for quantization operations,
//! including quantization schemes, backend configurations, and observer types.
//!
//! # Features
//!
//! - **Quantization Schemes**: Support for various quantization types (INT8, INT4, Binary, etc.)
//! - **Backend Configuration**: Multiple backend support (FBGEMM, QNNPACK, Native)
//! - **Observer Types**: Different calibration observers for optimal quantization parameters
//! - **Mixed Precision**: Advanced configuration for layer-specific precision
//! - **Builder Pattern**: Fluent API for configuration construction
//! - **Validation**: Comprehensive configuration validation

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, string::String};

use torsh_core::{
    dtype::DType,
    error::{Result as TorshResult, TorshError},
};

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QScheme {
    /// Per-tensor affine quantization
    PerTensorAffine,
    /// Per-channel affine quantization
    PerChannelAffine,
    /// Symmetric quantization
    PerTensorSymmetric,
    /// Per-channel symmetric
    PerChannelSymmetric,
    /// INT4 quantization (4-bit)
    Int4PerTensor,
    /// INT4 per-channel quantization
    Int4PerChannel,
    /// Mixed precision quantization
    MixedPrecision,
    /// Binary quantization (1-bit)
    Binary,
    /// Ternary quantization (2-bit with -1, 0, 1)
    Ternary,
    /// Group-wise quantization (groups channels and quantizes per-group)
    GroupWise,
}

/// Quantization backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuantBackend {
    /// FBGEMM backend (CPU optimized)
    Fbgemm,
    /// QNNPACK backend (mobile optimized)
    Qnnpack,
    /// Native backend (fallback)
    Native,
    /// XNNPACK backend
    Xnnpack,
}

/// Quantization reduction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReduceRange {
    /// No range reduction
    None,
    /// Reduce range for better accuracy
    Reduce,
}

/// Observer types for quantization parameter calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ObserverType {
    /// Min-max observer
    MinMax,
    /// Moving average min-max observer
    MovingAverage,
    /// Histogram observer
    Histogram,
    /// Percentile observer
    Percentile,
    /// KL divergence observer (for mixed precision)
    KLDivergence,
    /// Entropy-based observer
    Entropy,
}

/// Quantization configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantConfig {
    pub dtype: DType,
    pub scheme: QScheme,
    pub enable_fake_quant: bool,
    pub observer_type: ObserverType,
    pub backend: QuantBackend,
    pub reduce_range: ReduceRange,
    pub qint_min: Option<i32>,
    pub qint_max: Option<i32>,
    pub eps: f32,
    pub averaging_constant: f32,
    pub ch_axis: Option<usize>,
    /// Group size for group-wise quantization
    pub group_size: Option<usize>,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            dtype: DType::I8,
            scheme: QScheme::PerTensorAffine,
            enable_fake_quant: false,
            observer_type: ObserverType::MinMax,
            backend: QuantBackend::Native,
            reduce_range: ReduceRange::None,
            qint_min: None,
            qint_max: None,
            eps: 1e-8,
            averaging_constant: 0.01,
            ch_axis: None,
            group_size: None,
        }
    }
}

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision for different layer types
    pub layer_precision: HashMap<String, DType>,
    /// Default precision for unspecified layers
    pub default_precision: DType,
    /// Sensitivity threshold for precision selection
    pub sensitivity_threshold: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        let mut layer_precision = HashMap::new();
        layer_precision.insert("embedding".to_string(), DType::I8);
        layer_precision.insert("attention".to_string(), DType::F16);
        layer_precision.insert("output".to_string(), DType::F32);

        Self {
            layer_precision,
            default_precision: DType::I8,
            sensitivity_threshold: 0.1,
        }
    }
}

impl QuantConfig {
    /// Create a new quantization config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create config for INT8 quantization
    pub fn int8() -> Self {
        Self {
            dtype: DType::I8,
            qint_min: Some(-128),
            qint_max: Some(127),
            ..Self::default()
        }
    }

    /// Create config for INT4 quantization
    pub fn int4() -> Self {
        Self {
            dtype: DType::I8, // Store as I8 but quantize to 4-bit range
            scheme: QScheme::Int4PerTensor,
            qint_min: Some(-8),
            qint_max: Some(7),
            observer_type: ObserverType::Histogram,
            ..Self::default()
        }
    }

    /// Create config for binary quantization
    pub fn binary() -> Self {
        Self {
            dtype: DType::I8,
            scheme: QScheme::Binary,
            qint_min: Some(-1),
            qint_max: Some(1),
            observer_type: ObserverType::MinMax,
            ..Self::default()
        }
    }

    /// Create config for ternary quantization
    pub fn ternary() -> Self {
        Self {
            dtype: DType::I8,
            scheme: QScheme::Ternary,
            qint_min: Some(-1),
            qint_max: Some(1),
            observer_type: ObserverType::MinMax,
            ..Self::default()
        }
    }

    /// Create config for mixed precision
    pub fn mixed_precision() -> Self {
        Self {
            dtype: DType::I8, // Default precision
            scheme: QScheme::MixedPrecision,
            observer_type: ObserverType::KLDivergence,
            ..Self::default()
        }
    }

    /// Create config for UINT8 quantization
    pub fn uint8() -> Self {
        Self {
            dtype: DType::U8,
            qint_min: Some(0),
            qint_max: Some(255),
            ..Self::default()
        }
    }

    /// Create config for per-channel quantization
    pub fn per_channel(ch_axis: usize) -> Self {
        Self {
            scheme: QScheme::PerChannelAffine,
            ch_axis: Some(ch_axis),
            ..Self::default()
        }
    }

    /// Create config for group-wise quantization
    pub fn group_wise(ch_axis: usize, group_size: usize) -> Self {
        Self {
            scheme: QScheme::GroupWise,
            ch_axis: Some(ch_axis),
            group_size: Some(group_size),
            observer_type: ObserverType::Histogram,
            ..Self::default()
        }
    }

    /// Create config for QAT (Quantization Aware Training)
    pub fn qat() -> Self {
        Self {
            enable_fake_quant: true,
            observer_type: ObserverType::MovingAverage,
            ..Self::default()
        }
    }

    /// Create config with specific backend
    pub fn with_backend(mut self, backend: QuantBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set observer type
    pub fn with_observer(mut self, observer_type: ObserverType) -> Self {
        self.observer_type = observer_type;
        self
    }

    /// Set quantization scheme
    pub fn with_scheme(mut self, scheme: QScheme) -> Self {
        self.scheme = scheme;
        if matches!(
            scheme,
            QScheme::PerChannelAffine | QScheme::PerChannelSymmetric | QScheme::GroupWise
        ) && self.ch_axis.is_none()
        {
            self.ch_axis = Some(0); // Default channel axis
        }
        if matches!(scheme, QScheme::GroupWise) && self.group_size.is_none() {
            self.group_size = Some(32); // Default group size
        }
        self
    }

    /// Enable/disable fake quantization
    pub fn with_fake_quant(mut self, enable: bool) -> Self {
        self.enable_fake_quant = enable;
        self
    }

    /// Set reduce range option
    pub fn with_reduce_range(mut self, reduce_range: ReduceRange) -> Self {
        self.reduce_range = reduce_range;
        self
    }

    /// Set group size for group-wise quantization
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = Some(group_size);
        self
    }

    /// Get effective quantization range considering scheme and reduce_range
    pub fn get_qint_range(&self) -> (i32, i32) {
        let (base_min, base_max) = match self.scheme {
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => (-8, 7),
            QScheme::Binary => (-1, 1),
            QScheme::Ternary => (-1, 1),
            _ => match self.dtype {
                DType::I8 => (-128, 127),
                DType::U8 => (0, 255),
                _ => (self.qint_min.unwrap_or(-128), self.qint_max.unwrap_or(127)),
            },
        };

        let (qmin, qmax) = match self.reduce_range {
            ReduceRange::None => (base_min, base_max),
            ReduceRange::Reduce => {
                // Reduce range by 1 bit for better accuracy
                let range = base_max - base_min;
                let reduced_range = range / 2;
                let mid = (base_min + base_max) / 2;
                (mid - reduced_range / 2, mid + reduced_range / 2)
            }
        };

        (qmin, qmax)
    }

    /// Validate configuration
    pub fn validate(&self) -> TorshResult<()> {
        // Check if per-channel scheme has channel axis
        if matches!(
            self.scheme,
            QScheme::PerChannelAffine | QScheme::PerChannelSymmetric | QScheme::GroupWise
        ) && self.ch_axis.is_none()
        {
            return Err(TorshError::InvalidArgument(
                "Per-channel/Group-wise quantization requires channel axis".to_string(),
            ));
        }

        // Check if group-wise scheme has group size
        if matches!(self.scheme, QScheme::GroupWise) {
            if self.group_size.is_none() {
                return Err(TorshError::InvalidArgument(
                    "Group-wise quantization requires group size".to_string(),
                ));
            }
            if let Some(group_size) = self.group_size {
                if group_size == 0 {
                    return Err(TorshError::InvalidArgument(
                        "Group size must be greater than 0".to_string(),
                    ));
                }
            }
        }

        // Check if symmetric scheme is compatible with zero point
        if matches!(
            self.scheme,
            QScheme::PerTensorSymmetric | QScheme::PerChannelSymmetric
        ) {
            // Symmetric quantization should have zero_point = 0
        }

        // Check if binary/ternary schemes are valid
        if matches!(self.scheme, QScheme::Binary | QScheme::Ternary)
            && !matches!(
                self.observer_type,
                ObserverType::MinMax | ObserverType::MovingAverage
            )
        {
            return Err(TorshError::InvalidArgument(
                "Binary/ternary quantization requires MinMax or MovingAverage observer".to_string(),
            ));
        }

        // Check if mixed precision has valid configuration
        if matches!(self.scheme, QScheme::MixedPrecision)
            && !matches!(
                self.observer_type,
                ObserverType::KLDivergence | ObserverType::Entropy
            )
        {
            return Err(TorshError::InvalidArgument(
                "Mixed precision quantization requires KLDivergence or Entropy observer"
                    .to_string(),
            ));
        }

        // Validate eps
        if self.eps <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "eps must be positive".to_string(),
            ));
        }

        // Validate averaging constant
        if self.averaging_constant <= 0.0 || self.averaging_constant >= 1.0 {
            return Err(TorshError::InvalidArgument(
                "averaging_constant must be in (0, 1)".to_string(),
            ));
        }

        Ok(())
    }
}

/// Configuration builder for specific quantization backends
pub struct QuantConfigBuilder {
    config: QuantConfig,
}

impl QuantConfigBuilder {
    /// Start building a new configuration
    pub fn new() -> Self {
        Self {
            config: QuantConfig::default(),
        }
    }

    /// Set the data type
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.config.dtype = dtype;
        self
    }

    /// Set the quantization scheme
    pub fn scheme(mut self, scheme: QScheme) -> Self {
        self.config = self.config.with_scheme(scheme);
        self
    }

    /// Set the observer type
    pub fn observer(mut self, observer_type: ObserverType) -> Self {
        self.config.observer_type = observer_type;
        self
    }

    /// Set the backend
    pub fn backend(mut self, backend: QuantBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Enable/disable fake quantization
    pub fn fake_quant(mut self, enable: bool) -> Self {
        self.config.enable_fake_quant = enable;
        self
    }

    /// Set channel axis for per-channel quantization
    pub fn channel_axis(mut self, axis: usize) -> Self {
        self.config.ch_axis = Some(axis);
        self
    }

    /// Set group size for group-wise quantization
    pub fn group_size(mut self, size: usize) -> Self {
        self.config.group_size = Some(size);
        self
    }

    /// Build the final configuration
    pub fn build(self) -> TorshResult<QuantConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for QuantConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_config_defaults() {
        let config = QuantConfig::default();
        assert_eq!(config.dtype, DType::I8);
        assert_eq!(config.scheme, QScheme::PerTensorAffine);
        assert!(!config.enable_fake_quant);
        assert_eq!(config.observer_type, ObserverType::MinMax);
        assert_eq!(config.backend, QuantBackend::Native);
        assert_eq!(config.reduce_range, ReduceRange::None);
    }

    #[test]
    fn test_quant_config_presets() {
        let int8_config = QuantConfig::int8();
        assert_eq!(int8_config.dtype, DType::I8);
        assert_eq!(int8_config.qint_min, Some(-128));
        assert_eq!(int8_config.qint_max, Some(127));

        let binary_config = QuantConfig::binary();
        assert_eq!(binary_config.scheme, QScheme::Binary);
        assert_eq!(binary_config.qint_min, Some(-1));
        assert_eq!(binary_config.qint_max, Some(1));

        let int4_config = QuantConfig::int4();
        assert_eq!(int4_config.scheme, QScheme::Int4PerTensor);
        assert_eq!(int4_config.observer_type, ObserverType::Histogram);
    }

    #[test]
    fn test_quant_config_builder() {
        let config = QuantConfigBuilder::new()
            .dtype(DType::I8)
            .scheme(QScheme::PerChannelAffine)
            .observer(ObserverType::Histogram)
            .backend(QuantBackend::Fbgemm)
            .channel_axis(1)
            .build()
            .unwrap();

        assert_eq!(config.dtype, DType::I8);
        assert_eq!(config.scheme, QScheme::PerChannelAffine);
        assert_eq!(config.observer_type, ObserverType::Histogram);
        assert_eq!(config.backend, QuantBackend::Fbgemm);
        assert_eq!(config.ch_axis, Some(1));
    }

    #[test]
    fn test_config_validation() {
        // Valid configuration
        let valid_config = QuantConfig::per_channel(0);
        assert!(valid_config.validate().is_ok());

        // Invalid per-channel without axis
        let mut invalid_config = QuantConfig::default();
        invalid_config.scheme = QScheme::PerChannelAffine;
        invalid_config.ch_axis = None;
        assert!(invalid_config.validate().is_err());

        // Invalid group-wise without size
        let mut invalid_group = QuantConfig::default();
        invalid_group.scheme = QScheme::GroupWise;
        invalid_group.ch_axis = Some(0);
        invalid_group.group_size = None;
        assert!(invalid_group.validate().is_err());

        // Invalid eps
        let mut invalid_eps = QuantConfig::default();
        invalid_eps.eps = -1.0;
        assert!(invalid_eps.validate().is_err());

        // Invalid averaging constant
        let mut invalid_avg = QuantConfig::default();
        invalid_avg.averaging_constant = 1.5;
        assert!(invalid_avg.validate().is_err());
    }

    #[test]
    fn test_get_qint_range() {
        let int8_config = QuantConfig::int8();
        assert_eq!(int8_config.get_qint_range(), (-128, 127));

        let uint8_config = QuantConfig::uint8();
        assert_eq!(uint8_config.get_qint_range(), (0, 255));

        let int4_config = QuantConfig::int4();
        assert_eq!(int4_config.get_qint_range(), (-8, 7));

        let binary_config = QuantConfig::binary();
        assert_eq!(binary_config.get_qint_range(), (-1, 1));

        // Test reduced range
        let reduced_config = QuantConfig::int8().with_reduce_range(ReduceRange::Reduce);
        let (min, max) = reduced_config.get_qint_range();
        assert!(min > -128 && max < 127);
    }

    #[test]
    fn test_mixed_precision_config() {
        let mixed_config = MixedPrecisionConfig::default();
        assert_eq!(mixed_config.default_precision, DType::I8);
        assert_eq!(mixed_config.sensitivity_threshold, 0.1);
        assert!(mixed_config.layer_precision.contains_key("embedding"));
    }

    #[test]
    fn test_config_serialization() {
        let config = QuantConfig::int8().with_observer(ObserverType::Histogram);

        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: QuantConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.dtype, deserialized.dtype);
        assert_eq!(config.scheme, deserialized.scheme);
        assert_eq!(config.observer_type, deserialized.observer_type);
    }
}
