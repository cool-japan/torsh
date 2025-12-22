// Data Types: Core Type Definitions and Basic Operations
//
// This module provides the fundamental DType enum and basic operations for the ToRSh
// tensor framework. It defines all supported data types and their basic properties
// such as size, classification methods, and external system integrations.

use std::fmt;

/// Supported data types for tensors
///
/// This enum represents all the fundamental data types that can be stored in ToRSh tensors.
/// Each variant corresponds to a specific numeric representation with well-defined
/// size and arithmetic properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// 8-bit unsigned integer (0 to 255)
    U8,
    /// 8-bit signed integer (-128 to 127)
    I8,
    /// 16-bit signed integer (-32,768 to 32,767)
    I16,
    /// 32-bit signed integer (-2,147,483,648 to 2,147,483,647)
    I32,
    /// 32-bit unsigned integer (0 to 4,294,967,295)
    U32,
    /// 64-bit signed integer (-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)
    I64,
    /// 64-bit unsigned integer (0 to 18,446,744,073,709,551,615)
    U64,
    /// 16-bit floating point (half precision) - IEEE 754-2008 binary16
    F16,
    /// 32-bit floating point (single precision) - IEEE 754 binary32
    F32,
    /// 64-bit floating point (double precision) - IEEE 754 binary64
    F64,
    /// Boolean (true/false)
    Bool,
    /// Brain floating point 16-bit - Google's bfloat16 format
    BF16,
    /// Complex 64-bit (32-bit real + 32-bit imaginary)
    C64,
    /// Complex 128-bit (64-bit real + 64-bit imaginary)
    C128,
    /// Quantized 8-bit signed integer with scale and zero-point
    QInt8,
    /// Quantized 8-bit unsigned integer with scale and zero-point
    QUInt8,
    /// Quantized 32-bit signed integer with scale and zero-point (higher precision)
    QInt32,
}

impl DType {
    /// Get the size of the dtype in bytes
    ///
    /// Returns the number of bytes required to store a single value of this data type.
    /// This is fundamental for memory allocation and tensor storage calculations.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// // Integer types
    /// assert_eq!(DType::U8.size(), 1);
    /// assert_eq!(DType::I8.size(), 1);
    /// assert_eq!(DType::I16.size(), 2);
    /// assert_eq!(DType::I32.size(), 4);
    /// assert_eq!(DType::U32.size(), 4);
    /// assert_eq!(DType::I64.size(), 8);
    /// assert_eq!(DType::U64.size(), 8);
    ///
    /// // Floating point types
    /// assert_eq!(DType::F16.size(), 2);
    /// assert_eq!(DType::F32.size(), 4);
    /// assert_eq!(DType::F64.size(), 8);
    /// assert_eq!(DType::BF16.size(), 2);
    ///
    /// // Complex types
    /// assert_eq!(DType::C64.size(), 8);
    /// assert_eq!(DType::C128.size(), 16);
    ///
    /// // Other types
    /// assert_eq!(DType::Bool.size(), 1);
    /// assert_eq!(DType::QInt8.size(), 1);
    /// assert_eq!(DType::QUInt8.size(), 1);
    /// ```
    pub const fn size(&self) -> usize {
        match self {
            DType::U8 | DType::I8 | DType::Bool | DType::QInt8 | DType::QUInt8 => 1,
            DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::U32 | DType::F32 | DType::QInt32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
        }
    }

    /// Alias for size() method for compatibility
    ///
    /// This provides an alternative name for the size method to maintain
    /// compatibility with different naming conventions.
    pub const fn size_bytes(&self) -> usize {
        self.size()
    }

    /// Check if the dtype is a floating point type
    ///
    /// Returns true for IEEE 754 floating point types (F16, F32, F64) and
    /// Google's bfloat16 format (BF16). Complex types are not considered
    /// floating point types by this method.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// // Floating point types return true
    /// assert!(DType::F16.is_float());
    /// assert!(DType::F32.is_float());
    /// assert!(DType::F64.is_float());
    /// assert!(DType::BF16.is_float());
    ///
    /// // Non-floating point types return false
    /// assert!(!DType::U8.is_float());
    /// assert!(!DType::I8.is_float());
    /// assert!(!DType::I16.is_float());
    /// assert!(!DType::I32.is_float());
    /// assert!(!DType::I64.is_float());
    /// assert!(!DType::Bool.is_float());
    /// assert!(!DType::C64.is_float());
    /// assert!(!DType::C128.is_float());
    /// assert!(!DType::QInt8.is_float());
    /// assert!(!DType::QUInt8.is_float());
    /// ```
    pub const fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::F32 | DType::F64 | DType::BF16)
    }

    /// Check if the dtype is a complex number type
    ///
    /// Returns true for complex number types that store both real and imaginary
    /// components. Complex types use twice the storage of their component type.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// // Complex types return true
    /// assert!(DType::C64.is_complex());
    /// assert!(DType::C128.is_complex());
    ///
    /// // Non-complex types return false
    /// assert!(!DType::U8.is_complex());
    /// assert!(!DType::I8.is_complex());
    /// assert!(!DType::I16.is_complex());
    /// assert!(!DType::I32.is_complex());
    /// assert!(!DType::I64.is_complex());
    /// assert!(!DType::F16.is_complex());
    /// assert!(!DType::F32.is_complex());
    /// assert!(!DType::F64.is_complex());
    /// assert!(!DType::BF16.is_complex());
    /// assert!(!DType::Bool.is_complex());
    /// assert!(!DType::QInt8.is_complex());
    /// assert!(!DType::QUInt8.is_complex());
    /// ```
    pub const fn is_complex(&self) -> bool {
        matches!(self, DType::C64 | DType::C128)
    }

    /// Check if the dtype is an integer type
    ///
    /// Returns true for all signed and unsigned integer types, including
    /// quantized integer types. Boolean and floating point types return false.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// // Integer types return true
    /// assert!(DType::U8.is_int());
    /// assert!(DType::I8.is_int());
    /// assert!(DType::I16.is_int());
    /// assert!(DType::I32.is_int());
    /// assert!(DType::U32.is_int());
    /// assert!(DType::I64.is_int());
    /// assert!(DType::U64.is_int());
    /// assert!(DType::QInt8.is_int());
    /// assert!(DType::QUInt8.is_int());
    ///
    /// // Non-integer types return false
    /// assert!(!DType::F16.is_int());
    /// assert!(!DType::F32.is_int());
    /// assert!(!DType::F64.is_int());
    /// assert!(!DType::BF16.is_int());
    /// assert!(!DType::Bool.is_int());
    /// assert!(!DType::C64.is_int());
    /// assert!(!DType::C128.is_int());
    /// ```
    pub const fn is_int(&self) -> bool {
        matches!(
            self,
            DType::U8
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::U32
                | DType::I64
                | DType::U64
                | DType::QInt8
                | DType::QUInt8
        )
    }

    /// Check if the dtype is a quantized type
    ///
    /// Returns true for quantized integer types that include scale and zero-point
    /// parameters for representing floating point values with reduced precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// // Quantized types return true
    /// assert!(DType::QInt8.is_quantized());
    /// assert!(DType::QUInt8.is_quantized());
    /// assert!(DType::QInt32.is_quantized());
    ///
    /// // Non-quantized types return false
    /// assert!(!DType::U8.is_quantized());
    /// assert!(!DType::I8.is_quantized());
    /// assert!(!DType::F32.is_quantized());
    /// assert!(!DType::Bool.is_quantized());
    /// ```
    pub const fn is_quantized(&self) -> bool {
        matches!(self, DType::QInt8 | DType::QUInt8 | DType::QInt32)
    }

    /// Get the name of the dtype as a string
    ///
    /// Returns a human-readable string representation of the data type.
    /// This is useful for debugging, serialization, and user interfaces.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// assert_eq!(DType::F32.name(), "f32");
    /// assert_eq!(DType::I64.name(), "i64");
    /// assert_eq!(DType::Bool.name(), "bool");
    /// assert_eq!(DType::C64.name(), "c64");
    /// ```
    pub const fn name(&self) -> &'static str {
        match self {
            DType::U8 => "u8",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::U64 => "u64",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::Bool => "bool",
            DType::BF16 => "bf16",
            DType::C64 => "c64",
            DType::C128 => "c128",
            DType::QInt8 => "qint8",
            DType::QUInt8 => "quint8",
            DType::QInt32 => "qint32",
        }
    }

    /// Check if the dtype is signed
    ///
    /// Returns true for signed integer types, floating point types, and complex types.
    /// Unsigned integers, booleans, and quantized types return false.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// // Signed types
    /// assert!(DType::I8.is_signed());
    /// assert!(DType::I16.is_signed());
    /// assert!(DType::I32.is_signed());
    /// assert!(DType::I64.is_signed());
    /// assert!(DType::F32.is_signed());
    /// assert!(DType::F64.is_signed());
    /// assert!(DType::C64.is_signed());
    ///
    /// // Unsigned types
    /// assert!(!DType::U8.is_signed());
    /// assert!(!DType::U32.is_signed());
    /// assert!(!DType::U64.is_signed());
    /// assert!(!DType::Bool.is_signed());
    /// ```
    pub const fn is_signed(&self) -> bool {
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::BF16
                | DType::C64
                | DType::C128
        )
    }

    /// Get the number of bits used by this data type
    ///
    /// Returns the total number of bits required to represent this data type.
    /// This is useful for bit-level operations and memory calculations.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// assert_eq!(DType::U8.bits(), 8);
    /// assert_eq!(DType::I16.bits(), 16);
    /// assert_eq!(DType::F32.bits(), 32);
    /// assert_eq!(DType::F64.bits(), 64);
    /// assert_eq!(DType::C64.bits(), 64); // Complex64 = 2 * 32-bit floats
    /// assert_eq!(DType::C128.bits(), 128); // Complex128 = 2 * 64-bit floats
    /// ```
    pub const fn bits(&self) -> usize {
        self.size() * 8
    }

    /// Get the component type for complex numbers
    ///
    /// For complex types, returns the data type of the real and imaginary components.
    /// For non-complex types, returns None.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// assert_eq!(DType::C64.component_type(), Some(DType::F32));
    /// assert_eq!(DType::C128.component_type(), Some(DType::F64));
    /// assert_eq!(DType::F32.component_type(), None);
    /// assert_eq!(DType::I32.component_type(), None);
    /// ```
    pub const fn component_type(&self) -> Option<DType> {
        match self {
            DType::C64 => Some(DType::F32),
            DType::C128 => Some(DType::F64),
            _ => None,
        }
    }

    /// Convert to cuDNN data type enumeration
    ///
    /// Maps ToRSh data types to cuDNN's internal data type representations
    /// for GPU acceleration. Only supports types that have cuDNN equivalents.
    ///
    /// # Panics
    ///
    /// Panics if the data type is not supported by cuDNN (e.g., complex types,
    /// quantized types, or boolean).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use torsh_core::dtype::DType;
    ///
    /// // These work with cuDNN (requires cudnn feature and library)
    /// let _ = DType::F32.to_cudnn_data_type();
    /// let _ = DType::F64.to_cudnn_data_type();
    /// let _ = DType::F16.to_cudnn_data_type();
    /// ```
    #[cfg(feature = "cudnn")]
    pub fn to_cudnn_data_type(self) -> cudnn_sys::cudnnDataType_t {
        match self {
            DType::F32 => cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            DType::F64 => cudnn_sys::cudnnDataType_t::CUDNN_DATA_DOUBLE,
            DType::F16 => cudnn_sys::cudnnDataType_t::CUDNN_DATA_HALF,
            // Integer and unsigned types not supported in this cuDNN version
            // Use float as fallback for unsupported types
            DType::I8 | DType::I32 | DType::U8 => cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            _ => panic!("Unsupported data type for cuDNN: {:?}", self),
        }
    }

    /// Convert to CUDA data type enumeration (placeholder)
    ///
    /// Maps ToRSh data types to CUDA runtime data type representations.
    /// This is a placeholder implementation that would map to actual CUDA types.
    #[cfg(feature = "cuda")]
    pub fn to_cuda_data_type(self) -> u32 {
        match self {
            DType::F32 => 0,  // CUDA_R_32F
            DType::F64 => 1,  // CUDA_R_64F
            DType::F16 => 2,  // CUDA_R_16F
            DType::I8 => 3,   // CUDA_R_8I
            DType::I32 => 4,  // CUDA_R_32I
            DType::U8 => 5,   // CUDA_R_8U
            DType::C64 => 6,  // CUDA_C_32F
            DType::C128 => 7, // CUDA_C_64F
            _ => panic!("Unsupported data type for CUDA: {:?}", self),
        }
    }

    /// Get default value for this data type
    ///
    /// Returns the zero/default value appropriate for this data type.
    /// This is useful for initialization and padding operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::DType;
    ///
    /// assert_eq!(DType::F32.default_value_f32(), 0.0);
    /// assert_eq!(DType::I32.default_value_i32(), 0);
    /// assert_eq!(DType::Bool.default_value_bool(), false);
    /// ```
    pub const fn default_value_f32(&self) -> f32 {
        0.0
    }

    pub const fn default_value_i32(&self) -> i32 {
        0
    }

    pub const fn default_value_bool(&self) -> bool {
        false
    }
}

/// Display implementation for DType
///
/// Provides a human-readable string representation using the name() method.
/// This is used when printing or formatting DType values.
impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Convert from string to DType
///
/// Parses a string representation of a data type name into the corresponding
/// DType enum variant. This is useful for configuration files, command-line
/// arguments, and serialization.
impl std::str::FromStr for DType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "u8" => Ok(DType::U8),
            "i8" => Ok(DType::I8),
            "i16" => Ok(DType::I16),
            "i32" => Ok(DType::I32),
            "u32" => Ok(DType::U32),
            "i64" => Ok(DType::I64),
            "u64" => Ok(DType::U64),
            "f16" => Ok(DType::F16),
            "f32" => Ok(DType::F32),
            "f64" => Ok(DType::F64),
            "bool" => Ok(DType::Bool),
            "bf16" => Ok(DType::BF16),
            "c64" => Ok(DType::C64),
            "c128" => Ok(DType::C128),
            "qint8" => Ok(DType::QInt8),
            "quint8" => Ok(DType::QUInt8),
            _ => Err(format!("Unknown data type: {}", s)),
        }
    }
}

/// Default implementation for DType
///
/// The default data type is F32 (32-bit floating point), which is the most
/// commonly used type for deep learning and scientific computing.
impl Default for DType {
    fn default() -> Self {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::U8.size(), 1);
        assert_eq!(DType::I8.size(), 1);
        assert_eq!(DType::I16.size(), 2);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::U32.size(), 4);
        assert_eq!(DType::I64.size(), 8);
        assert_eq!(DType::U64.size(), 8);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F64.size(), 8);
        assert_eq!(DType::Bool.size(), 1);
        assert_eq!(DType::BF16.size(), 2);
        assert_eq!(DType::C64.size(), 8);
        assert_eq!(DType::C128.size(), 16);
        assert_eq!(DType::QInt8.size(), 1);
        assert_eq!(DType::QUInt8.size(), 1);
    }

    #[test]
    fn test_dtype_classification() {
        // Test floating point classification
        assert!(DType::F16.is_float());
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::U8.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::Bool.is_float());
        assert!(!DType::C64.is_float());

        // Test complex classification
        assert!(DType::C64.is_complex());
        assert!(DType::C128.is_complex());
        assert!(!DType::F32.is_complex());
        assert!(!DType::I32.is_complex());

        // Test integer classification
        assert!(DType::U8.is_int());
        assert!(DType::I8.is_int());
        assert!(DType::I16.is_int());
        assert!(DType::I32.is_int());
        assert!(DType::U32.is_int());
        assert!(DType::I64.is_int());
        assert!(DType::U64.is_int());
        assert!(DType::QInt8.is_int());
        assert!(DType::QUInt8.is_int());
        assert!(!DType::F32.is_int());
        assert!(!DType::Bool.is_int());
        assert!(!DType::C64.is_int());

        // Test quantized classification
        assert!(DType::QInt8.is_quantized());
        assert!(DType::QUInt8.is_quantized());
        assert!(!DType::I8.is_quantized());
        assert!(!DType::U8.is_quantized());
        assert!(!DType::F32.is_quantized());
    }

    #[test]
    fn test_dtype_names() {
        assert_eq!(DType::U8.name(), "u8");
        assert_eq!(DType::I8.name(), "i8");
        assert_eq!(DType::I16.name(), "i16");
        assert_eq!(DType::I32.name(), "i32");
        assert_eq!(DType::U32.name(), "u32");
        assert_eq!(DType::I64.name(), "i64");
        assert_eq!(DType::U64.name(), "u64");
        assert_eq!(DType::F16.name(), "f16");
        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::F64.name(), "f64");
        assert_eq!(DType::Bool.name(), "bool");
        assert_eq!(DType::BF16.name(), "bf16");
        assert_eq!(DType::C64.name(), "c64");
        assert_eq!(DType::C128.name(), "c128");
        assert_eq!(DType::QInt8.name(), "qint8");
        assert_eq!(DType::QUInt8.name(), "quint8");
    }

    #[test]
    fn test_dtype_signed() {
        // Signed types
        assert!(DType::I8.is_signed());
        assert!(DType::I16.is_signed());
        assert!(DType::I32.is_signed());
        assert!(DType::I64.is_signed());
        assert!(DType::F16.is_signed());
        assert!(DType::F32.is_signed());
        assert!(DType::F64.is_signed());
        assert!(DType::BF16.is_signed());
        assert!(DType::C64.is_signed());
        assert!(DType::C128.is_signed());

        // Unsigned types
        assert!(!DType::U8.is_signed());
        assert!(!DType::U32.is_signed());
        assert!(!DType::U64.is_signed());
        assert!(!DType::Bool.is_signed());
        assert!(!DType::QInt8.is_signed()); // Quantized types are special
        assert!(!DType::QUInt8.is_signed());
    }

    #[test]
    fn test_dtype_bits() {
        assert_eq!(DType::U8.bits(), 8);
        assert_eq!(DType::I16.bits(), 16);
        assert_eq!(DType::I32.bits(), 32);
        assert_eq!(DType::F64.bits(), 64);
        assert_eq!(DType::C64.bits(), 64);
        assert_eq!(DType::C128.bits(), 128);
    }

    #[test]
    fn test_component_type() {
        assert_eq!(DType::C64.component_type(), Some(DType::F32));
        assert_eq!(DType::C128.component_type(), Some(DType::F64));
        assert_eq!(DType::F32.component_type(), None);
        assert_eq!(DType::I32.component_type(), None);
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::I64), "i64");
        assert_eq!(format!("{}", DType::Bool), "bool");
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!("f32".parse::<DType>().unwrap(), DType::F32);
        assert_eq!("i64".parse::<DType>().unwrap(), DType::I64);
        assert_eq!("bool".parse::<DType>().unwrap(), DType::Bool);
        assert_eq!("c64".parse::<DType>().unwrap(), DType::C64);

        // Test case insensitive
        assert_eq!("F32".parse::<DType>().unwrap(), DType::F32);
        assert_eq!("I64".parse::<DType>().unwrap(), DType::I64);

        // Test error case
        assert!("invalid".parse::<DType>().is_err());
    }

    #[test]
    fn test_dtype_default() {
        assert_eq!(DType::default(), DType::F32);
    }

    #[test]
    fn test_dtype_alias_methods() {
        assert_eq!(DType::F32.size(), DType::F32.size_bytes());
        assert_eq!(DType::I64.size(), DType::I64.size_bytes());
    }

    #[test]
    fn test_default_values() {
        assert_eq!(DType::F32.default_value_f32(), 0.0);
        assert_eq!(DType::I32.default_value_i32(), 0);
        assert!(!DType::Bool.default_value_bool());
    }
}
