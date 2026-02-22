//! Const Generic Shape Types for Compile-Time Shape Verification
//!
//! This module provides compile-time shape checking using const generics,
//! enabling type-safe tensor operations with shape verification at compile time.
//!
//! # Features
//! - Compile-time shape dimension checking
//! - Type-level shape constraints
//! - Zero-cost runtime abstractions
//! - Compatible with runtime Shape type
//!
//! # Examples
//!
//! ```ignore
//! use torsh_core::shape::{ConstShape, Rank1, Rank2};
//!
//! // 1D shape with 10 elements
//! type Vector10 = ConstShape<Rank1<10>>;
//!
//! // 2D shape: 3x4 matrix
//! type Matrix3x4 = ConstShape<Rank2<3, 4>>;
//!
//! // Compile-time dimension verification
//! let shape = Vector10::new();
//! assert_eq!(shape.ndim(), 1);
//! assert_eq!(shape.numel(), 10);
//! ```

use crate::error::{Result, TorshError};
use crate::shape::Shape;
use std::marker::PhantomData;

/// Marker trait for compile-time shape rank
pub trait ShapeRank {
    /// Number of dimensions (rank)
    const NDIM: usize;

    /// Get dimensions as array
    const DIMS: &'static [usize];

    /// Total number of elements
    const NUMEL: usize;

    /// Get runtime Shape
    fn to_runtime() -> Shape {
        Shape::new(Self::DIMS.to_vec())
    }

    /// Validate compatibility with another rank
    fn is_compatible_with<Other: ShapeRank>() -> bool {
        Self::NDIM == Other::NDIM
    }
}

/// Rank 0 (scalar)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank0;

impl ShapeRank for Rank0 {
    const NDIM: usize = 0;
    const DIMS: &'static [usize] = &[];
    const NUMEL: usize = 1;
}

/// Rank 1 (vector)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank1<const D0: usize>;

impl<const D0: usize> ShapeRank for Rank1<D0> {
    const NDIM: usize = 1;
    const DIMS: &'static [usize] = &[D0];
    const NUMEL: usize = D0;
}

/// Rank 2 (matrix)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank2<const D0: usize, const D1: usize>;

impl<const D0: usize, const D1: usize> ShapeRank for Rank2<D0, D1> {
    const NDIM: usize = 2;
    const DIMS: &'static [usize] = &[D0, D1];
    const NUMEL: usize = D0 * D1;
}

/// Rank 3 (3D tensor)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank3<const D0: usize, const D1: usize, const D2: usize>;

impl<const D0: usize, const D1: usize, const D2: usize> ShapeRank for Rank3<D0, D1, D2> {
    const NDIM: usize = 3;
    const DIMS: &'static [usize] = &[D0, D1, D2];
    const NUMEL: usize = D0 * D1 * D2;
}

/// Rank 4 (4D tensor - common for NCHW/NHWC)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank4<const D0: usize, const D1: usize, const D2: usize, const D3: usize>;

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ShapeRank
    for Rank4<D0, D1, D2, D3>
{
    const NDIM: usize = 4;
    const DIMS: &'static [usize] = &[D0, D1, D2, D3];
    const NUMEL: usize = D0 * D1 * D2 * D3;
}

/// Rank 5 (5D tensor)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank5<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
>;

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ShapeRank
    for Rank5<D0, D1, D2, D3, D4>
{
    const NDIM: usize = 5;
    const DIMS: &'static [usize] = &[D0, D1, D2, D3, D4];
    const NUMEL: usize = D0 * D1 * D2 * D3 * D4;
}

/// Compile-time shape with const generic rank
#[derive(Debug, Clone, Copy)]
pub struct ConstShape<R: ShapeRank> {
    _phantom: PhantomData<R>,
}

impl<R: ShapeRank> ConstShape<R> {
    /// Create a new const shape
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Get the number of dimensions (compile-time constant)
    pub const fn ndim() -> usize {
        R::NDIM
    }

    /// Get the dimensions (compile-time constant)
    pub const fn dims() -> &'static [usize] {
        R::DIMS
    }

    /// Get the total number of elements (compile-time constant)
    pub const fn numel() -> usize {
        R::NUMEL
    }

    /// Check if this is a scalar (rank 0)
    pub const fn is_scalar() -> bool {
        R::NDIM == 0
    }

    /// Check if this is a vector (rank 1)
    pub const fn is_vector() -> bool {
        R::NDIM == 1
    }

    /// Check if this is a matrix (rank 2)
    pub const fn is_matrix() -> bool {
        R::NDIM == 2
    }

    /// Convert to runtime Shape
    pub fn to_runtime(&self) -> Shape {
        R::to_runtime()
    }

    /// Verify shape matches at runtime
    pub fn verify_runtime(&self, shape: &Shape) -> Result<()> {
        if shape.ndim() != R::NDIM {
            return Err(TorshError::InvalidShape(format!(
                "Rank mismatch: expected {}, got {}",
                R::NDIM,
                shape.ndim()
            )));
        }

        let dims = shape.dims();
        for (i, (&expected, &actual)) in R::DIMS.iter().zip(dims.iter()).enumerate() {
            if expected != actual {
                return Err(TorshError::InvalidShape(format!(
                    "Dimension {} mismatch: expected {}, got {}",
                    i, expected, actual
                )));
            }
        }

        Ok(())
    }
}

impl<R: ShapeRank> Default for ConstShape<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for compile-time shape operations
pub trait ConstShapeOps<Other> {
    /// Result type of the operation
    type Output: ShapeRank;
}

/// Matrix multiplication shape compatibility
pub trait MatMulCompatible<Other: ShapeRank>: ShapeRank {
    /// Result shape after matrix multiplication
    type Output: ShapeRank;

    /// Check compatibility at compile time
    const COMPATIBLE: bool;
}

// Matrix x Matrix multiplication
impl<const M: usize, const N: usize, const K: usize> MatMulCompatible<Rank2<N, K>> for Rank2<M, N> {
    type Output = Rank2<M, K>;
    const COMPATIBLE: bool = true;
}

// Matrix x Vector multiplication
impl<const M: usize, const N: usize> MatMulCompatible<Rank1<N>> for Rank2<M, N> {
    type Output = Rank1<M>;
    const COMPATIBLE: bool = true;
}

/// Broadcasting compatibility trait
pub trait BroadcastCompatible<Other: ShapeRank>: ShapeRank {
    /// Result shape after broadcasting
    type Output: ShapeRank;

    /// Check if broadcasting is possible
    const COMPATIBLE: bool;
}

// Scalar broadcasts to any shape
impl<R: ShapeRank> BroadcastCompatible<R> for Rank0 {
    type Output = R;
    const COMPATIBLE: bool = true;
}

// Vector broadcasts to matrix if dimensions match
impl<const N: usize> BroadcastCompatible<Rank1<N>> for Rank1<N> {
    type Output = Rank1<N>;
    const COMPATIBLE: bool = true;
}

// Same rank matrices are always compatible
impl<const M: usize, const N: usize> BroadcastCompatible<Rank2<M, N>> for Rank2<M, N> {
    type Output = Rank2<M, N>;
    const COMPATIBLE: bool = true;
}

/// Reshape operation with compile-time verification
pub trait ReshapeInto<Target: ShapeRank>: ShapeRank {
    /// Check if reshape is valid (same number of elements)
    const VALID: bool = Self::NUMEL == Target::NUMEL;
}

impl<S: ShapeRank, T: ShapeRank> ReshapeInto<T> for S {}

/// Transpose operations
pub trait TransposeOps: ShapeRank {
    /// Transpose result type
    type Transposed: ShapeRank;
}

// 2D transpose
impl<const M: usize, const N: usize> TransposeOps for Rank2<M, N> {
    type Transposed = Rank2<N, M>;
}

// 1D transpose is identity
impl<const N: usize> TransposeOps for Rank1<N> {
    type Transposed = Rank1<N>;
}

/// Squeeze operation (remove dimensions of size 1)
///
/// Note: Due to trait coherence rules, we only implement squeezing the first dimension.
/// For squeezing other dimensions, use transpose + squeeze + transpose.
pub trait SqueezeOps: ShapeRank {
    /// Squeezed shape type
    type Squeezed: ShapeRank;
}

// Squeeze Rank2<1, N> -> Rank1<N> (first dimension is 1)
impl<const N: usize> SqueezeOps for Rank2<1, N> {
    type Squeezed = Rank1<N>;
}

/// Unsqueeze operation (add dimension of size 1)
pub trait UnsqueezeOps<const DIM: usize>: ShapeRank {
    /// Unsqueezed shape type
    type Unsqueezed: ShapeRank;
}

// Unsqueeze Rank1<N> at dim 0 -> Rank2<1, N>
impl<const N: usize> UnsqueezeOps<0> for Rank1<N> {
    type Unsqueezed = Rank2<1, N>;
}

// Unsqueeze Rank1<N> at dim 1 -> Rank2<N, 1>
impl<const N: usize> UnsqueezeOps<1> for Rank1<N> {
    type Unsqueezed = Rank2<N, 1>;
}

/// Common shape type aliases
pub mod common {
    use super::*;

    /// Scalar (0D)
    pub type Scalar = ConstShape<Rank0>;

    /// Common vector sizes
    pub type Vec2 = ConstShape<Rank1<2>>;
    pub type Vec3 = ConstShape<Rank1<3>>;
    pub type Vec4 = ConstShape<Rank1<4>>;
    pub type Vec8 = ConstShape<Rank1<8>>;
    pub type Vec16 = ConstShape<Rank1<16>>;
    pub type Vec32 = ConstShape<Rank1<32>>;
    pub type Vec64 = ConstShape<Rank1<64>>;
    pub type Vec128 = ConstShape<Rank1<128>>;
    pub type Vec256 = ConstShape<Rank1<256>>;
    pub type Vec512 = ConstShape<Rank1<512>>;
    pub type Vec1024 = ConstShape<Rank1<1024>>;

    /// Common matrix sizes
    pub type Mat2x2 = ConstShape<Rank2<2, 2>>;
    pub type Mat3x3 = ConstShape<Rank2<3, 3>>;
    pub type Mat4x4 = ConstShape<Rank2<4, 4>>;

    /// Common image shapes (NCHW format)
    pub type ImageRGB224 = ConstShape<Rank4<1, 3, 224, 224>>; // ImageNet standard
    pub type ImageRGB32 = ConstShape<Rank4<1, 3, 32, 32>>; // CIFAR-10
    pub type ImageRGB28 = ConstShape<Rank4<1, 1, 28, 28>>; // MNIST

    /// Common batch sizes (as const values, not types)
    pub const BATCH_1: usize = 1;
    pub const BATCH_8: usize = 8;
    pub const BATCH_16: usize = 16;
    pub const BATCH_32: usize = 32;
    pub const BATCH_64: usize = 64;
    pub const BATCH_128: usize = 128;
    pub const BATCH_256: usize = 256;
}

/// Utilities for working with const shapes
pub mod utils {
    use super::*;

    /// Verify that two const shapes are compatible for element-wise operations
    pub fn verify_elementwise_compatible<R1: ShapeRank, R2: ShapeRank>() -> Result<()> {
        if R1::NDIM != R2::NDIM {
            return Err(TorshError::InvalidShape(format!(
                "Rank mismatch for element-wise operation: {} vs {}",
                R1::NDIM,
                R2::NDIM
            )));
        }

        for (i, (&d1, &d2)) in R1::DIMS.iter().zip(R2::DIMS.iter()).enumerate() {
            if d1 != d2 && d1 != 1 && d2 != 1 {
                return Err(TorshError::InvalidShape(format!(
                    "Dimension {} incompatible for broadcasting: {} vs {}",
                    i, d1, d2
                )));
            }
        }

        Ok(())
    }

    /// Verify matrix multiplication compatibility
    pub fn verify_matmul_compatible<R1: ShapeRank, R2: ShapeRank>() -> Result<()> {
        if R1::NDIM < 2 || R2::NDIM < 2 {
            return Err(TorshError::InvalidShape(
                "Matrix multiplication requires at least 2D tensors".to_string(),
            ));
        }

        let inner1 = R1::DIMS[R1::NDIM - 1];
        let inner2 = R2::DIMS[R2::NDIM - 2];

        if inner1 != inner2 {
            return Err(TorshError::InvalidShape(format!(
                "Inner dimensions must match for matmul: {} vs {}",
                inner1, inner2
            )));
        }

        Ok(())
    }

    /// Create a const shape from runtime shape (with verification)
    pub fn from_runtime<R: ShapeRank>(shape: &Shape) -> Result<ConstShape<R>> {
        let const_shape = ConstShape::<R>::new();
        const_shape.verify_runtime(shape)?;
        Ok(const_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::common::*;
    use super::*;

    #[test]
    fn test_const_shape_basics() {
        let _scalar = Scalar::new();
        assert_eq!(Scalar::ndim(), 0);
        assert_eq!(Scalar::numel(), 1);
        assert!(Scalar::is_scalar());

        let _vec = Vec3::new();
        assert_eq!(Vec3::ndim(), 1);
        assert_eq!(Vec3::numel(), 3);
        assert!(Vec3::is_vector());

        let _mat = Mat3x3::new();
        assert_eq!(Mat3x3::ndim(), 2);
        assert_eq!(Mat3x3::numel(), 9);
        assert!(Mat3x3::is_matrix());
    }

    #[test]
    fn test_runtime_conversion() {
        let vec = Vec3::new();
        let runtime = vec.to_runtime();
        assert_eq!(runtime.ndim(), 1);
        assert_eq!(runtime.dims(), &[3]);
        assert_eq!(runtime.numel(), 3);
    }

    #[test]
    fn test_runtime_verification() {
        let const_vec = Vec3::new();
        let runtime_vec = Shape::new(vec![3]);
        assert!(const_vec.verify_runtime(&runtime_vec).is_ok());

        let wrong_vec = Shape::new(vec![4]);
        assert!(const_vec.verify_runtime(&wrong_vec).is_err());

        let wrong_rank = Shape::new(vec![3, 3]);
        assert!(const_vec.verify_runtime(&wrong_rank).is_err());
    }

    #[test]
    fn test_matmul_compatibility() {
        // This would be checked at compile time in real usage
        assert!(<Rank2<3, 4> as MatMulCompatible<Rank2<4, 5>>>::COMPATIBLE);

        // Result type is Mat3x5
        type ResultShape = <Rank2<3, 4> as MatMulCompatible<Rank2<4, 5>>>::Output;
        assert_eq!(ResultShape::DIMS, &[3, 5]);
    }

    #[test]
    fn test_broadcast_compatibility() {
        // Scalar broadcasts to any shape
        assert!(<Rank0 as BroadcastCompatible<Rank1<3>>>::COMPATIBLE);
        assert!(<Rank0 as BroadcastCompatible<Rank2<3, 4>>>::COMPATIBLE);

        // Same shapes are compatible
        assert!(<Rank1<3> as BroadcastCompatible<Rank1<3>>>::COMPATIBLE);
        assert!(<Rank2<3, 4> as BroadcastCompatible<Rank2<3, 4>>>::COMPATIBLE);
    }

    #[test]
    fn test_reshape_validity() {
        // Valid reshape: 12 elements -> 12 elements
        assert!(<Rank1<12> as ReshapeInto<Rank2<3, 4>>>::VALID);
        assert!(<Rank1<12> as ReshapeInto<Rank2<2, 6>>>::VALID);
        assert!(<Rank1<12> as ReshapeInto<Rank3<2, 2, 3>>>::VALID);

        // Invalid reshape: different element counts
        assert!(!<Rank1<12> as ReshapeInto<Rank2<3, 5>>>::VALID);
        assert!(!<Rank1<10> as ReshapeInto<Rank2<3, 4>>>::VALID);
    }

    #[test]
    fn test_transpose() {
        type Mat3x4 = Rank2<3, 4>;

        // Transpose of 3x4 is 4x3
        type Transposed = <Mat3x4 as TransposeOps>::Transposed;
        assert_eq!(Transposed::DIMS, &[4, 3]);

        // Double transpose is identity
        type DoubleTransposed = <Transposed as TransposeOps>::Transposed;
        assert_eq!(DoubleTransposed::DIMS, Mat3x4::DIMS);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        // Squeeze Rank2<1, N> to Rank1<N>
        type Mat1x3 = Rank2<1, 3>;
        type Squeezed = <Mat1x3 as SqueezeOps>::Squeezed;
        assert_eq!(Squeezed::DIMS, &[3]);

        // Unsqueeze Rank1<N> to Rank2<1, N>
        type Vec3Rank = Rank1<3>;
        type Unsqueezed = <Vec3Rank as UnsqueezeOps<0>>::Unsqueezed;
        assert_eq!(Unsqueezed::DIMS, &[1, 3]);
    }

    #[test]
    fn test_common_shapes() {
        assert_eq!(ImageRGB224::ndim(), 4);
        assert_eq!(ImageRGB224::dims(), &[1, 3, 224, 224]);
        assert_eq!(ImageRGB224::numel(), 1 * 3 * 224 * 224);

        assert_eq!(ImageRGB32::dims(), &[1, 3, 32, 32]);
        assert_eq!(ImageRGB28::dims(), &[1, 1, 28, 28]);
    }

    #[test]
    fn test_utils_verification() {
        assert!(utils::verify_elementwise_compatible::<Rank2<3, 4>, Rank2<3, 4>>().is_ok());
        assert!(utils::verify_elementwise_compatible::<Rank2<3, 4>, Rank2<3, 5>>().is_err());

        assert!(utils::verify_matmul_compatible::<Rank2<3, 4>, Rank2<4, 5>>().is_ok());
        assert!(utils::verify_matmul_compatible::<Rank2<3, 4>, Rank2<5, 4>>().is_err());
    }
}
