//! Data types for tensors

use num_traits::{Float, NumCast, Zero, One};
use std::fmt;

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// 8-bit unsigned integer
    U8,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 16-bit floating point (half precision)
    F16,
    /// 32-bit floating point (single precision)
    F32,
    /// 64-bit floating point (double precision)
    F64,
    /// Boolean
    Bool,
    /// Brain floating point 16-bit
    BF16,
}

impl DType {
    /// Get the size of the dtype in bytes
    pub fn size(&self) -> usize {
        match self {
            DType::U8 | DType::I8 | DType::Bool => 1,
            DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::F32 => 4,
            DType::I64 | DType::F64 => 8,
        }
    }
    
    /// Check if the dtype is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::F32 | DType::F64 | DType::BF16)
    }
    
    /// Check if the dtype is integer
    pub fn is_int(&self) -> bool {
        matches!(self, DType::U8 | DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }
    
    /// Get the name of the dtype
    pub fn name(&self) -> &'static str {
        match self {
            DType::U8 => "uint8",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::Bool => "bool",
            DType::BF16 => "bfloat16",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for types that can be used as tensor elements
pub trait TensorElement: 
    Clone + Send + Sync + PartialEq + fmt::Debug + 'static
{
    /// Get the dtype for this element type
    fn dtype() -> DType;
    
    /// Convert from f64 (for initialization)
    fn from_f64(v: f64) -> Option<Self>;
    
    /// Convert to f64 (for display/debugging)
    fn to_f64(&self) -> Option<f64>;
    
    /// Zero value
    fn zero() -> Self;
    
    /// One value
    fn one() -> Self;
}

// Implement TensorElement for standard types
macro_rules! impl_tensor_element {
    ($ty:ty, $dtype:expr) => {
        impl TensorElement for $ty {
            fn dtype() -> DType {
                $dtype
            }
            
            fn from_f64(v: f64) -> Option<Self> {
                NumCast::from(v)
            }
            
            fn to_f64(&self) -> Option<f64> {
                NumCast::from(*self)
            }
            
            fn zero() -> Self {
                Zero::zero()
            }
            
            fn one() -> Self {
                One::one()
            }
        }
    };
}

impl_tensor_element!(u8, DType::U8);
impl_tensor_element!(i8, DType::I8);
impl_tensor_element!(i16, DType::I16);
impl_tensor_element!(i32, DType::I32);
impl_tensor_element!(i64, DType::I64);
impl_tensor_element!(f32, DType::F32);
impl_tensor_element!(f64, DType::F64);
// Custom implementation for bool (doesn't implement numeric traits)
impl TensorElement for bool {
    fn dtype() -> DType {
        DType::Bool
    }
    
    fn from_f64(v: f64) -> Option<Self> {
        Some(v != 0.0)
    }
    
    fn to_f64(&self) -> Option<f64> {
        Some(if *self { 1.0 } else { 0.0 })
    }
    
    fn zero() -> Self {
        false
    }
    
    fn one() -> Self {
        true
    }
}

/// Trait for floating point tensor elements
pub trait FloatElement: TensorElement + Float {}

impl FloatElement for f32 {}
impl FloatElement for f64 {}