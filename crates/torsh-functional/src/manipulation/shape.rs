//! Tensor shape manipulation operations
//!
//! This module provides functions for ensuring tensors have minimum dimensionality
//! and manipulating tensor shapes while preserving data. These operations are
//! essential for broadcasting compatibility and shape consistency in tensor operations.

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Ensure tensor has at least 1 dimension
///
/// ## Mathematical Background
///
/// Converts scalar tensors (0-dimensional) to 1-dimensional tensors by adding
/// a dimension of size 1. This operation preserves all data while ensuring
/// the tensor can participate in operations requiring at least 1 dimension.
///
/// ## Shape Transformations
/// ```text
/// () → (1,)         # Scalar to 1D
/// (n,) → (n,)       # 1D unchanged
/// (m,n) → (m,n)     # Higher dimensions unchanged
/// ```text
///
/// ## Broadcasting Compatibility
///
/// Many tensor operations require inputs to have compatible shapes for broadcasting.
/// The `atleast_1d` function ensures tensors can participate in element-wise operations:
///
/// ```text
/// scalar + vector   # Requires scalar to be at least 1D
/// 0D + 1D → 1D + 1D → broadcasting possible
/// ```text
///
/// ## Parameters
/// * `tensor` - Input tensor of any dimensionality
///
/// ## Returns
/// * Tensor with at least 1 dimension, preserving all data
///
/// ## Example
/// ```rust
/// # use torsh_functional::manipulation::atleast_1d;
/// # use torsh_tensor::creation::{scalar, ones};
/// let scalar = scalar(5.0)?;          // Shape: ()
/// let vector = atleast_1d(&scalar)?;  // Shape: (1,)
///
/// let existing = ones(&[3])?;         // Shape: (3,)
/// let unchanged = atleast_1d(&existing)?; // Shape: (3,) - no change
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```text
///
/// ## Applications
/// - **Broadcasting preparation**: Ensure compatibility with vector operations
/// - **API consistency**: Standardize inputs to expect at least 1D
/// - **Neural networks**: Batch dimension handling
/// - **Linear algebra**: Vector-matrix operation preparation
pub fn atleast_1d(tensor: &Tensor) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    if shape.ndim() == 0 {
        tensor.view(&[1])
    } else {
        Ok(tensor.clone())
    }
}

/// Ensure tensor has at least 2 dimensions
///
/// ## Mathematical Background
///
/// Converts tensors with fewer than 2 dimensions to 2-dimensional tensors
/// by adding dimensions of size 1. This is essential for matrix operations
/// which require 2D inputs.
///
/// ## Shape Transformations
/// ```text
/// ()     → (1,1)     # Scalar to 2D
/// (n,)   → (n,1)     # Vector to column matrix
/// (m,n)  → (m,n)     # Matrix unchanged
/// (l,m,n) → (l,m,n)  # Higher dimensions unchanged
/// ```text
///
/// ## Matrix Operation Compatibility
///
/// Many linear algebra operations require 2D inputs:
/// - **Matrix multiplication**: A @ B requires both A and B to be 2D
/// - **Linear systems**: Solving Ax = b requires A to be 2D
/// - **Decompositions**: SVD, QR, LU require 2D matrices
///
/// ## Column Vector Convention
///
/// 1D tensors are converted to column vectors (n,1) following the convention
/// that vectors are treated as column matrices in linear algebra:
///
/// ```text
/// vector: (n,) → column matrix: (n,1)
/// ```text
///
/// ## Parameters
/// * `tensor` - Input tensor of any dimensionality
///
/// ## Returns
/// * Tensor with at least 2 dimensions, preserving all data
///
/// ## Example
/// ```rust
/// # use torsh_functional::manipulation::atleast_2d;
/// # use torsh_tensor::creation::{scalar, ones};
/// let scalar = scalar(5.0)?;          // Shape: ()
/// let matrix = atleast_2d(&scalar)?;  // Shape: (1,1)
///
/// let vector = ones(&[3])?;           // Shape: (3,)
/// let column = atleast_2d(&vector)?;  // Shape: (3,1)
///
/// let existing = ones(&[2, 4])?;      // Shape: (2,4)
/// let unchanged = atleast_2d(&existing)?; // Shape: (2,4) - no change
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```text
///
/// ## Applications
/// - **Linear algebra**: Prepare vectors for matrix operations
/// - **Neural networks**: Ensure weight matrices are 2D
/// - **Image processing**: Treat 1D signals as single-row images
/// - **Batch operations**: Prepare single samples for batch processing
pub fn atleast_2d(tensor: &Tensor) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    match shape.ndim() {
        0 => tensor.view(&[1, 1]),
        1 => tensor.view(&[shape.dims()[0] as i32, 1]),
        _ => Ok(tensor.clone()),
    }
}

/// Ensure tensor has at least 3 dimensions
///
/// ## Mathematical Background
///
/// Converts tensors with fewer than 3 dimensions to 3-dimensional tensors
/// by adding dimensions of size 1. This is essential for operations requiring
/// 3D inputs such as 3D convolutions, volume processing, or batch operations.
///
/// ## Shape Transformations
/// ```text
/// ()       → (1,1,1)   # Scalar to 3D
/// (n,)     → (n,1,1)   # Vector to 3D
/// (m,n)    → (m,n,1)   # Matrix to 3D
/// (l,m,n)  → (l,m,n)   # 3D unchanged
/// (k,l,m,n) → (k,l,m,n) # Higher dimensions unchanged
/// ```text
///
/// ## Batch Processing Convention
///
/// The added dimensions follow the convention (batch, height, width) or
/// (batch, channels, spatial) for compatibility with deep learning frameworks:
///
/// ```text
/// 1D signal: (n,) → (n,1,1)     # n samples, 1x1 spatial
/// 2D image:  (h,w) → (h,w,1)    # h×w image, 1 channel
/// ```text
///
/// ## 3D Operation Compatibility
///
/// Many operations require 3D inputs:
/// - **3D convolutions**: Conv3D requires (batch, channels, depth, height, width)
/// - **Volume processing**: Medical imaging, 3D computer vision
/// - **Sequence modeling**: (batch, sequence, features)
/// - **Batch operations**: Adding batch dimension for single samples
///
/// ## Parameters
/// * `tensor` - Input tensor of any dimensionality
///
/// ## Returns
/// * Tensor with at least 3 dimensions, preserving all data
///
/// ## Example
/// ```rust
/// # use torsh_functional::manipulation::atleast_3d;
/// # use torsh_tensor::creation::{scalar, ones};
/// let scalar = scalar(5.0)?;          // Shape: ()
/// let volume = atleast_3d(&scalar)?;  // Shape: (1,1,1)
///
/// let vector = ones(&[5])?;           // Shape: (5,)
/// let volume_vec = atleast_3d(&vector)?; // Shape: (5,1,1)
///
/// let matrix = ones(&[3, 4])?;        // Shape: (3,4)
/// let volume_mat = atleast_3d(&matrix)?; // Shape: (3,4,1)
///
/// let existing = ones(&[2, 3, 4])?;   // Shape: (2,3,4)
/// let unchanged = atleast_3d(&existing)?; // Shape: (2,3,4) - no change
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```text
///
/// ## Applications
/// - **3D computer vision**: Prepare 2D images for 3D processing
/// - **Deep learning**: Add batch/channel dimensions
/// - **Scientific computing**: Volume and tensor field operations
/// - **Signal processing**: Multi-dimensional signal analysis
/// - **Medical imaging**: 3D volume reconstruction and analysis
pub fn atleast_3d(tensor: &Tensor) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    match shape.ndim() {
        0 => tensor.view(&[1, 1, 1]),
        1 => tensor.view(&[shape.dims()[0] as i32, 1, 1]),
        2 => tensor.view(&[shape.dims()[0] as i32, shape.dims()[1] as i32, 1]),
        _ => Ok(tensor.clone()),
    }
}
