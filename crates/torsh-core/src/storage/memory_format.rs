//! Memory format management for tensor storage
//!
//! This module defines different memory layout formats for tensors and provides
//! utilities for working with various memory organization patterns.

/// Memory format for tensor storage
///
/// This enum specifies how tensor data is laid out in memory. Different formats
/// can provide performance benefits for different types of operations and hardware.
///
/// # Examples
///
/// ```
/// use torsh_core::MemoryFormat;
///
/// // Create different memory formats
/// let contiguous = MemoryFormat::Contiguous;
/// let channels_last = MemoryFormat::ChannelsLast;
/// let channels_last_3d = MemoryFormat::ChannelsLast3d;
/// let strided = MemoryFormat::Strided;
///
/// // Default format is Contiguous
/// let default_format = MemoryFormat::default();
/// assert_eq!(default_format, MemoryFormat::Contiguous);
///
/// // Memory formats can be compared
/// assert_ne!(MemoryFormat::Contiguous, MemoryFormat::ChannelsLast);
///
/// // Use in memory layout decisions
/// match contiguous {
///     MemoryFormat::Contiguous => println!("Using row-major layout"),
///     MemoryFormat::ChannelsLast => println!("Using NHWC layout"),
///     MemoryFormat::ChannelsLast3d => println!("Using NDHWC layout"),
///     MemoryFormat::Strided => println!("Using custom strided layout"),
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryFormat {
    /// Standard row-major (C-contiguous) format
    /// Best for most general operations and CPU performance
    #[default]
    Contiguous,
    /// Channel-last format for 4D tensors (NHWC)
    /// Optimized for convolutions on certain GPUs
    ChannelsLast,
    /// Channel-last format for 5D tensors (NDHWC)
    /// Optimized for 3D convolutions
    ChannelsLast3d,
    /// Custom strided format
    /// Allows arbitrary memory layouts with custom strides
    Strided,
}

impl MemoryFormat {
    /// Check if this format is contiguous
    pub fn is_contiguous(&self) -> bool {
        matches!(self, MemoryFormat::Contiguous)
    }

    /// Check if this format is channels-last
    pub fn is_channels_last(&self) -> bool {
        matches!(
            self,
            MemoryFormat::ChannelsLast | MemoryFormat::ChannelsLast3d
        )
    }

    /// Check if this format uses custom strides
    pub fn is_strided(&self) -> bool {
        matches!(self, MemoryFormat::Strided)
    }

    /// Get the expected number of dimensions for this format
    pub fn expected_dims(&self) -> Option<usize> {
        match self {
            MemoryFormat::Contiguous => None,        // Any number of dimensions
            MemoryFormat::ChannelsLast => Some(4),   // NHWC
            MemoryFormat::ChannelsLast3d => Some(5), // NDHWC
            MemoryFormat::Strided => None,           // Any number of dimensions
        }
    }

    /// Check if this format is compatible with the given number of dimensions
    pub fn is_compatible_with_dims(&self, ndim: usize) -> bool {
        match self.expected_dims() {
            Some(expected) => ndim == expected,
            None => true,
        }
    }

    /// Get the optimal format for the given operation and hardware
    pub fn optimal_for_operation(operation: OperationType, hardware: HardwareType) -> Self {
        match (operation, hardware) {
            (OperationType::Convolution, HardwareType::GPU) => MemoryFormat::ChannelsLast,
            (OperationType::Convolution3D, HardwareType::GPU) => MemoryFormat::ChannelsLast3d,
            (OperationType::Linear, _) => MemoryFormat::Contiguous,
            (OperationType::ElementWise, _) => MemoryFormat::Contiguous,
            _ => MemoryFormat::Contiguous,
        }
    }

    /// Convert between memory formats (conceptual - actual implementation would be more complex)
    pub fn conversion_cost(&self, target: MemoryFormat) -> ConversionCost {
        if self == &target {
            ConversionCost::None
        } else {
            match (self, target) {
                (MemoryFormat::Contiguous, MemoryFormat::ChannelsLast) => ConversionCost::Medium,
                (MemoryFormat::ChannelsLast, MemoryFormat::Contiguous) => ConversionCost::Medium,
                (MemoryFormat::Contiguous, MemoryFormat::ChannelsLast3d) => ConversionCost::Medium,
                (MemoryFormat::ChannelsLast3d, MemoryFormat::Contiguous) => ConversionCost::Medium,
                (_, MemoryFormat::Strided) => ConversionCost::High,
                (MemoryFormat::Strided, _) => ConversionCost::High,
                _ => ConversionCost::Low,
            }
        }
    }

    /// Get all available memory formats
    pub fn all_formats() -> &'static [MemoryFormat] {
        &[
            MemoryFormat::Contiguous,
            MemoryFormat::ChannelsLast,
            MemoryFormat::ChannelsLast3d,
            MemoryFormat::Strided,
        ]
    }

    /// Get the canonical memory format for a given tensor shape
    pub fn canonical_for_shape(shape: &[usize]) -> Self {
        match shape.len() {
            0 | 1 => MemoryFormat::Contiguous,
            2 | 3 => MemoryFormat::Contiguous,
            4 => MemoryFormat::Contiguous, // Could be ChannelsLast based on use case
            5 => MemoryFormat::Contiguous, // Could be ChannelsLast3d based on use case
            _ => MemoryFormat::Strided,
        }
    }
}

impl std::fmt::Display for MemoryFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryFormat::Contiguous => write!(f, "contiguous"),
            MemoryFormat::ChannelsLast => write!(f, "channels_last"),
            MemoryFormat::ChannelsLast3d => write!(f, "channels_last_3d"),
            MemoryFormat::Strided => write!(f, "strided"),
        }
    }
}

impl std::str::FromStr for MemoryFormat {
    type Err = crate::error::TorshError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "contiguous" | "c" => Ok(MemoryFormat::Contiguous),
            "channels_last" | "nhwc" => Ok(MemoryFormat::ChannelsLast),
            "channels_last_3d" | "ndhwc" => Ok(MemoryFormat::ChannelsLast3d),
            "strided" => Ok(MemoryFormat::Strided),
            _ => Err(crate::error::TorshError::InvalidArgument(format!(
                "Unknown memory format: {s}"
            ))),
        }
    }
}

/// Types of operations that benefit from specific memory formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Convolution operations (2D)
    Convolution,
    /// 3D convolution operations
    Convolution3D,
    /// Linear/dense layer operations
    Linear,
    /// Element-wise operations
    ElementWise,
    /// Reduction operations
    Reduction,
    /// Matrix multiplication
    MatMul,
    /// Transpose operations
    Transpose,
    /// Pooling operations
    Pooling,
    /// Normalization operations
    Normalization,
}

/// Hardware types that may prefer different memory formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareType {
    /// CPU processing
    CPU,
    /// GPU processing (CUDA, OpenCL, Metal, etc.)
    GPU,
    /// Specialized neural processing units
    NPU,
    /// Mobile/embedded processors
    Mobile,
    /// WebGPU in browser environments
    WebGPU,
}

/// Cost of converting between memory formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConversionCost {
    /// No conversion needed
    None,
    /// Low cost conversion (e.g., reshape without copy)
    Low,
    /// Medium cost conversion (e.g., data copy with reordering)
    Medium,
    /// High cost conversion (e.g., complex strided operations)
    High,
}

/// Memory format preference configuration
#[derive(Debug, Clone)]
pub struct FormatPreference {
    /// Preferred memory format
    pub preferred: MemoryFormat,
    /// Acceptable alternative formats in order of preference
    pub alternatives: Vec<MemoryFormat>,
    /// Maximum conversion cost willing to accept
    pub max_conversion_cost: ConversionCost,
    /// Whether to cache converted tensors
    pub cache_conversions: bool,
}

impl Default for FormatPreference {
    fn default() -> Self {
        Self {
            preferred: MemoryFormat::Contiguous,
            alternatives: vec![MemoryFormat::ChannelsLast, MemoryFormat::ChannelsLast3d],
            max_conversion_cost: ConversionCost::Medium,
            cache_conversions: true,
        }
    }
}

impl FormatPreference {
    /// Create new format preference
    pub fn new(preferred: MemoryFormat) -> Self {
        Self {
            preferred,
            alternatives: Vec::new(),
            max_conversion_cost: ConversionCost::High,
            cache_conversions: true,
        }
    }

    /// Add alternative format
    pub fn with_alternative(mut self, format: MemoryFormat) -> Self {
        self.alternatives.push(format);
        self
    }

    /// Set maximum conversion cost
    pub fn with_max_cost(mut self, cost: ConversionCost) -> Self {
        self.max_conversion_cost = cost;
        self
    }

    /// Enable or disable conversion caching
    pub fn with_caching(mut self, cache: bool) -> Self {
        self.cache_conversions = cache;
        self
    }

    /// Check if a format is acceptable given conversion cost
    pub fn is_acceptable(&self, format: MemoryFormat) -> bool {
        if format == self.preferred {
            return true;
        }

        if self.alternatives.contains(&format) {
            return true;
        }

        let cost = self.preferred.conversion_cost(format);
        cost <= self.max_conversion_cost
    }

    /// Get the best format from a list of available formats
    pub fn choose_best(&self, available: &[MemoryFormat]) -> Option<MemoryFormat> {
        // First, try preferred format
        if available.contains(&self.preferred) {
            return Some(self.preferred);
        }

        // Then try alternatives in order
        for &alt in &self.alternatives {
            if available.contains(&alt) {
                return Some(alt);
            }
        }

        // Finally, find the best available format with acceptable conversion cost
        available
            .iter()
            .filter(|&&fmt| self.is_acceptable(fmt))
            .min_by_key(|&&fmt| self.preferred.conversion_cost(fmt))
            .copied()
    }
}

/// Utility functions for memory format operations
pub mod utils {
    use super::*;

    /// Calculate the optimal memory format for a tensor with given shape and usage
    pub fn optimal_format_for_tensor(
        shape: &[usize],
        operation: OperationType,
        hardware: HardwareType,
    ) -> MemoryFormat {
        let base_format = MemoryFormat::optimal_for_operation(operation, hardware);

        // Check if format is compatible with tensor dimensions
        if base_format.is_compatible_with_dims(shape.len()) {
            base_format
        } else {
            MemoryFormat::canonical_for_shape(shape)
        }
    }

    /// Estimate memory bandwidth requirements for different formats
    pub fn estimate_bandwidth_efficiency(format: MemoryFormat, operation: OperationType) -> f32 {
        match (format, operation) {
            (MemoryFormat::Contiguous, OperationType::Linear) => 1.0,
            (MemoryFormat::ChannelsLast, OperationType::Convolution) => 0.9,
            (MemoryFormat::ChannelsLast3d, OperationType::Convolution3D) => 0.9,
            (MemoryFormat::Strided, _) => 0.5,
            _ => 0.8,
        }
    }

    /// Check if two memory formats can be efficiently interconverted
    pub fn can_efficiently_convert(from: MemoryFormat, to: MemoryFormat) -> bool {
        from.conversion_cost(to) <= ConversionCost::Medium
    }

    /// Get memory format recommendations for a given use case
    pub fn format_recommendations(
        shape: &[usize],
        operations: &[OperationType],
        hardware: HardwareType,
    ) -> Vec<MemoryFormat> {
        let mut recommendations = Vec::new();

        // Add format for each operation type
        for &op in operations {
            let format = optimal_format_for_tensor(shape, op, hardware);
            if !recommendations.contains(&format) {
                recommendations.push(format);
            }
        }

        // Sort by preference (contiguous first for compatibility)
        recommendations.sort_by_key(|&fmt| match fmt {
            MemoryFormat::Contiguous => 0,
            MemoryFormat::ChannelsLast => 1,
            MemoryFormat::ChannelsLast3d => 2,
            MemoryFormat::Strided => 3,
        });

        recommendations
    }
}
