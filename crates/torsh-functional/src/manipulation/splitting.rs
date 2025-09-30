//! Tensor splitting operations
//!
//! This module provides comprehensive tensor splitting functionality for dividing tensors
//! into multiple sub-tensors along specified dimensions. These operations are essential for
//! data partitioning, parallel processing, and implementing neural network architectures
//! that require tensor decomposition.

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Split tensor into multiple sub-tensors
///
/// ## Mathematical Background
///
/// Tensor splitting partitions a tensor A ∈ ℝ^(d₁×d₂×...×dₙ) along dimension k into
/// multiple sub-tensors {B₁, B₂, ..., Bₘ} such that:
///
/// ```
/// concatenate([B₁, B₂, ..., Bₘ], dim=k) = A
/// ```
///
/// ## Splitting Modes
///
/// ### Fixed Size Splitting
/// For split size s, creates ⌈dₖ/s⌉ sub-tensors where:
/// - First ⌊dₖ/s⌋ tensors have size s along dimension k
/// - Last tensor has size dₖ mod s (if non-zero)
///
/// ### Section-Based Splitting
/// For n sections, creates n sub-tensors with sizes:
/// - Base size: b = ⌊dₖ/n⌋
/// - First r = dₖ mod n tensors: size b+1
/// - Remaining n-r tensors: size b
///
/// ### Index-Based Splitting
/// For indices [i₁, i₂, ..., iₘ], creates m+1 sub-tensors:
/// - A[:i₁], A[i₁:i₂], ..., A[iₘ:]
///
/// ## Parameters
/// * `tensor` - Input tensor to split
/// * `split_size_or_sections` - Splitting specification (size, sections, or indices)
/// * `dim` - Dimension along which to split (negative indexing supported)
///
/// ## Returns
/// * Vector of sub-tensors resulting from the split operation
///
/// ## Applications
/// - **Data batching**: Split large datasets into smaller batches
/// - **Parallel processing**: Distribute tensor chunks across workers
/// - **Memory management**: Process large tensors in smaller chunks
/// - **Model parallelism**: Split layers across multiple devices
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::split;
/// # use torsh_functional::manipulation::SplitArg;
/// # use torsh_tensor::creation::ones;
/// let tensor = ones(&[12, 4])?;
///
/// // Split into chunks of size 3
/// let chunks = split(&tensor, SplitArg::Size(3), 0)?; // 4 chunks of [3,4]
///
/// // Split into 4 equal sections
/// let sections = split(&tensor, SplitArg::Sections(4), 0)?; // 4 chunks of [3,4]
///
/// // Split at specific indices
/// let splits = split(&tensor, SplitArg::Indices(vec![3, 8]), 0)?; // [3,4], [5,4], [4,4]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn split(
    tensor: &Tensor,
    split_size_or_sections: SplitArg,
    dim: isize,
) -> TorshResult<Vec<Tensor>> {
    let shape = tensor.shape();
    let ndim = shape.ndim() as isize;

    // Normalize dimension
    let dim = if dim < 0 { ndim + dim } else { dim } as usize;

    if dim >= shape.ndim() {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            &format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                shape.ndim()
            ),
            "split",
        ));
    }

    match split_size_or_sections {
        SplitArg::Size(size) => {
            // Split into chunks of given size
            let dim_size = shape.dims()[dim];
            let num_splits = dim_size.div_ceil(size);

            let mut splits = Vec::new();
            for i in 0..num_splits {
                let start = i * size;
                let end = ((i + 1) * size).min(dim_size);

                // Create slice for this split
                let split = tensor.slice(dim as usize, start, end)?.to_tensor()?;
                splits.push(split);
            }

            Ok(splits)
        }
        SplitArg::Sections(sections) => {
            // Split into given number of sections
            let dim_size = shape.dims()[dim];
            let base_size = dim_size / sections;
            let remainder = dim_size % sections;

            let mut splits = Vec::new();
            let mut offset = 0;

            for i in 0..sections {
                let size = if i < remainder {
                    base_size + 1
                } else {
                    base_size
                };

                // Create slice for this split
                let split = tensor
                    .slice(dim as usize, offset, offset + size)?
                    .to_tensor()?;
                splits.push(split);

                offset += size;
            }

            Ok(splits)
        }
        SplitArg::Indices(indices) => {
            // Split at the specified indices
            let mut splits = Vec::new();
            let mut start = 0;

            for &index in &indices {
                let split = tensor.slice(dim as usize, start, index)?.to_tensor()?;
                splits.push(split);
                start = index;
            }

            // Add final split from last index to end
            let dim_size = shape.dims()[dim];
            if start < dim_size {
                let split = tensor.slice(dim as usize, start, dim_size)?.to_tensor()?;
                splits.push(split);
            }

            Ok(splits)
        }
    }
}

/// Split argument specification for tensor splitting operations
///
/// ## Variants
///
/// ### Size
/// Split into chunks of specified size along the dimension.
/// The last chunk may be smaller if the dimension size is not evenly divisible.
///
/// ### Sections
/// Split into the specified number of approximately equal sections.
/// If not evenly divisible, earlier sections will be one element larger.
///
/// ### Indices
/// Split at the specified indices, creating len(indices)+1 sub-tensors.
/// Indices must be sorted and within bounds of the dimension.
#[derive(Debug, Clone)]
pub enum SplitArg {
    /// Split into chunks of fixed size
    Size(usize),
    /// Split into specified number of sections
    Sections(usize),
    /// Split at specified indices
    Indices(Vec<usize>),
}

/// Split tensor into approximately equal chunks
///
/// ## Mathematical Background
///
/// Chunks a tensor into approximately equal pieces along the specified dimension.
/// For tensor with dimension size d and n chunks:
///
/// ```
/// chunk_size = ⌈d/n⌉
/// num_full_chunks = n - (n * chunk_size - d)
/// ```
///
/// The first `num_full_chunks` will have size `chunk_size`, and remaining chunks
/// will have size `chunk_size - 1`.
///
/// ## Parameters
/// * `tensor` - Input tensor to chunk
/// * `chunks` - Number of chunks to create
/// * `dim` - Dimension along which to chunk (negative indexing supported)
///
/// ## Returns
/// * Vector of approximately equal-sized tensor chunks
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::chunk;
/// # use torsh_tensor::creation::ones;
/// let tensor = ones(&[10, 3])?; // 10 rows, 3 columns
///
/// // Split into 3 chunks along first dimension
/// let chunks = chunk(&tensor, 3, 0)?; // [4,3], [3,3], [3,3]
///
/// // Split into 4 chunks along second dimension
/// let chunks = chunk(&tensor, 4, 1)?; // [10,1], [10,1], [10,1], [10,0] (empty)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn chunk(tensor: &Tensor, chunks: usize, dim: isize) -> TorshResult<Vec<Tensor>> {
    let shape = tensor.shape();
    let ndim = shape.ndim() as isize;

    // Normalize dimension
    let dim = if dim < 0 { ndim + dim } else { dim } as usize;

    if dim >= shape.ndim() {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            &format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                shape.ndim()
            ),
            "chunk",
        ));
    }

    split(tensor, SplitArg::Sections(chunks), dim as isize)
}

/// Split tensor into sections at specified indices
///
/// ## Mathematical Background
///
/// Performs tensor splitting at explicitly specified indices along a dimension.
/// For tensor A with dimension size d and indices [i₁, i₂, ..., iₘ]:
///
/// ```
/// Result = [A[..., :i₁, ...], A[..., i₁:i₂, ...], ..., A[..., iₘ:, ...]]
/// ```
///
/// Where the ellipsis represents all other dimensions.
///
/// ## Index Validation
/// - All indices must be within bounds: 0 ≤ iⱼ ≤ d
/// - Indices should be in ascending order for meaningful results
/// - Empty sections are allowed (consecutive identical indices)
///
/// ## Parameters
/// * `tensor` - Input tensor to split
/// * `indices_or_sections` - Either number of sections or explicit indices
/// * `dim` - Dimension along which to split (negative indexing supported)
///
/// ## Returns
/// * Vector of tensor sections
///
/// ## Applications
/// - **Sequence processing**: Split variable-length sequences at boundaries
/// - **Data preprocessing**: Extract regions of interest from images/signals
/// - **Batch processing**: Create non-uniform batches based on data characteristics
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::{tensor_split, TensorSplitArg};
/// # use torsh_tensor::creation::ones;
/// let tensor = ones(&[8, 4])?;
///
/// // Split into 3 sections
/// let sections = tensor_split(&tensor, TensorSplitArg::Sections(3), 0)?;
/// // Results: [3,4], [3,4], [2,4]
///
/// // Split at specific indices
/// let splits = tensor_split(&tensor, TensorSplitArg::Indices(vec![2, 5]), 0)?;
/// // Results: [2,4], [3,4], [3,4]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tensor_split(
    tensor: &Tensor,
    indices_or_sections: TensorSplitArg,
    dim: isize,
) -> TorshResult<Vec<Tensor>> {
    let shape = tensor.shape();
    let ndim = shape.ndim() as isize;

    // Normalize dimension
    let dim = if dim < 0 { ndim + dim } else { dim } as usize;

    if dim >= shape.ndim() {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            &format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                shape.ndim()
            ),
            "tensor_split",
        ));
    }

    match indices_or_sections {
        TensorSplitArg::Sections(sections) => {
            split(tensor, SplitArg::Sections(sections), dim as isize)
        }
        TensorSplitArg::Indices(indices) => {
            let dim_size = shape.dims()[dim];
            let mut splits = Vec::new();
            let mut prev_idx = 0;

            for &idx in &indices {
                if idx > dim_size {
                    return Err(torsh_core::TorshError::invalid_argument_with_context(
                        &format!(
                            "Split index {} out of range for dimension size {}",
                            idx, dim_size
                        ),
                        "tensor_split",
                    ));
                }

                if idx > prev_idx {
                    let split = tensor.slice(dim as usize, prev_idx, idx)?.to_tensor()?;
                    splits.push(split);
                }
                prev_idx = idx;
            }

            // Add final split if needed
            if prev_idx < dim_size {
                let split = tensor
                    .slice(dim as usize, prev_idx, dim_size)?
                    .to_tensor()?;
                splits.push(split);
            }

            Ok(splits)
        }
    }
}

/// Split argument specification for tensor_split operations
///
/// ## Variants
///
/// ### Sections
/// Split into the specified number of approximately equal sections.
///
/// ### Indices
/// Split at the specified indices. The indices define the boundaries
/// where the tensor should be divided.
#[derive(Debug, Clone)]
pub enum TensorSplitArg {
    /// Split into specified number of sections
    Sections(usize),
    /// Split at specified indices
    Indices(Vec<usize>),
}

/// Split tensor horizontally (along second dimension)
///
/// ## Mathematical Background
///
/// Horizontal splitting divides a tensor along its second dimension (columns for 2D matrices).
/// For tensor A ∈ ℝ^(m×n×...), hsplit creates sub-tensors along dimension 1:
///
/// ```
/// A = [A₁ | A₂ | ... | Aₖ]  (column-wise concatenation)
/// ```
///
/// ## Requirements
/// - Input tensor must have at least 2 dimensions
/// - Equivalent to `tensor_split(tensor, indices_or_sections, 1)`
///
/// ## Parameters
/// * `tensor` - Input tensor (≥2D required)
/// * `indices_or_sections` - Split specification (sections or indices)
///
/// ## Returns
/// * Vector of horizontally split tensors
///
/// ## Applications
/// - **Image processing**: Split images into vertical strips
/// - **Feature extraction**: Separate different feature groups in matrices
/// - **Data analysis**: Split datasets by column groups
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::{hsplit, TensorSplitArg};
/// # use torsh_tensor::creation::ones;
/// let image = ones(&[100, 200, 3])?; // Height × Width × Channels
///
/// // Split into 4 vertical strips
/// let strips = hsplit(&image, TensorSplitArg::Sections(4))?;
/// // Each strip: [100, 50, 3]
///
/// // Split at specific column positions
/// let splits = hsplit(&image, TensorSplitArg::Indices(vec![50, 150]))?;
/// // Results: [100,50,3], [100,100,3], [100,50,3]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn hsplit(tensor: &Tensor, indices_or_sections: TensorSplitArg) -> TorshResult<Vec<Tensor>> {
    let shape = tensor.shape();
    if shape.ndim() < 2 {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Input tensor must have at least 2 dimensions for hsplit",
            "hsplit",
        ));
    }

    tensor_split(tensor, indices_or_sections, 1)
}

/// Split tensor vertically (along first dimension)
///
/// ## Mathematical Background
///
/// Vertical splitting divides a tensor along its first dimension (rows for 2D matrices).
/// For tensor A ∈ ℝ^(m×n×...), vsplit creates sub-tensors along dimension 0:
///
/// ```
/// A = [A₁; A₂; ...; Aₖ]  (row-wise concatenation)
/// ```
///
/// ## Requirements
/// - Input tensor must have at least 2 dimensions
/// - Equivalent to `tensor_split(tensor, indices_or_sections, 0)`
///
/// ## Parameters
/// * `tensor` - Input tensor (≥2D required)
/// * `indices_or_sections` - Split specification (sections or indices)
///
/// ## Returns
/// * Vector of vertically split tensors
///
/// ## Applications
/// - **Image processing**: Split images into horizontal strips
/// - **Batch processing**: Divide batches into smaller sub-batches
/// - **Time series**: Split sequences into temporal segments
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::{vsplit, TensorSplitArg};
/// # use torsh_tensor::creation::ones;
/// let batch = ones(&[64, 784])?; // Batch size × Features
///
/// // Split into 4 mini-batches
/// let mini_batches = vsplit(&batch, TensorSplitArg::Sections(4))?;
/// // Each mini-batch: [16, 784]
///
/// // Split at specific row positions
/// let splits = vsplit(&batch, TensorSplitArg::Indices(vec![16, 48]))?;
/// // Results: [16,784], [32,784], [16,784]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn vsplit(tensor: &Tensor, indices_or_sections: TensorSplitArg) -> TorshResult<Vec<Tensor>> {
    let shape = tensor.shape();
    if shape.ndim() < 2 {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Input tensor must have at least 2 dimensions for vsplit",
            "vsplit",
        ));
    }

    tensor_split(tensor, indices_or_sections, 0)
}

/// Split tensor along depth dimension (third dimension)
///
/// ## Mathematical Background
///
/// Depth splitting divides a tensor along its third dimension (depth for 3D tensors).
/// For tensor A ∈ ℝ^(m×n×d×...), dsplit creates sub-tensors along dimension 2:
///
/// ```
/// A[:,:,k₁:k₂,:] for each split k₁:k₂
/// ```
///
/// ## Requirements
/// - Input tensor must have at least 3 dimensions
/// - Equivalent to `tensor_split(tensor, indices_or_sections, 2)`
///
/// ## Parameters
/// * `tensor` - Input tensor (≥3D required)
/// * `indices_or_sections` - Split specification (sections or indices)
///
/// ## Returns
/// * Vector of depth-wise split tensors
///
/// ## Applications
/// - **3D data processing**: Split volumetric data along depth
/// - **Video analysis**: Split video frames into temporal chunks
/// - **Multi-channel data**: Separate different channels or modalities
/// - **Neural networks**: Split feature maps along channel dimension
///
/// ## Examples
/// ```rust
/// # use torsh_functional::manipulation::{dsplit, TensorSplitArg};
/// # use torsh_tensor::creation::ones;
/// let volume = ones(&[64, 64, 32])?; // Height × Width × Depth
///
/// // Split into 4 depth sections
/// let sections = dsplit(&volume, TensorSplitArg::Sections(4))?;
/// // Each section: [64, 64, 8]
///
/// // Split at specific depth positions
/// let splits = dsplit(&volume, TensorSplitArg::Indices(vec![8, 24]))?;
/// // Results: [64,64,8], [64,64,16], [64,64,8]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn dsplit(tensor: &Tensor, indices_or_sections: TensorSplitArg) -> TorshResult<Vec<Tensor>> {
    let shape = tensor.shape();
    if shape.ndim() < 3 {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Input tensor must have at least 3 dimensions for dsplit",
            "dsplit",
        ));
    }

    tensor_split(tensor, indices_or_sections, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_ops::randn;

    #[test]
    fn test_split() -> TorshResult<()> {
        // Test equal splits
        let tensor = randn(&[6, 4], None, None, None)?;
        let result = split(&tensor, SplitArg::Sections(3), 0)?;
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(chunk.shape().dims(), &[2, 4]);
        }

        // Test split with indices
        let result = split(&tensor, SplitArg::Indices(vec![2, 4]), 0)?;
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].shape().dims(), &[2, 4]);
        assert_eq!(result[1].shape().dims(), &[2, 4]);
        assert_eq!(result[2].shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_chunk() -> TorshResult<()> {
        // Test chunking along dimension 0
        let tensor = randn(&[8, 3], None, None, None)?;
        let result = chunk(&tensor, 3, 0)?;

        // Should create 3 chunks of sizes [3, 3, 2]
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].shape().dims(), &[3, 3]);
        assert_eq!(result[1].shape().dims(), &[3, 3]);
        assert_eq!(result[2].shape().dims(), &[2, 3]);

        // Test chunking along dimension 1
        let result = chunk(&tensor, 2, 1)?;
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape().dims(), &[8, 2]); // ceil(3/2) = 2
        assert_eq!(result[1].shape().dims(), &[8, 1]); // remaining 1

        Ok(())
    }

    #[test]
    fn test_tensor_split() -> TorshResult<()> {
        // Test with sections
        let tensor = randn(&[6, 4], None, None, None)?;
        let result = tensor_split(&tensor, TensorSplitArg::Sections(3), 0)?;
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(chunk.shape().dims(), &[2, 4]);
        }

        // Test with indices
        let result = tensor_split(&tensor, TensorSplitArg::Indices(vec![2, 4]), 0)?;
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].shape().dims(), &[2, 4]);
        assert_eq!(result[1].shape().dims(), &[2, 4]);
        assert_eq!(result[2].shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_hsplit() -> TorshResult<()> {
        // Test horizontal split with sections
        let tensor = randn(&[4, 6], None, None, None)?;
        let result = hsplit(&tensor, TensorSplitArg::Sections(3))?;
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(chunk.shape().dims(), &[4, 2]);
        }

        // Test horizontal split with indices
        let result = hsplit(&tensor, TensorSplitArg::Indices(vec![2, 4]))?;
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].shape().dims(), &[4, 2]);
        assert_eq!(result[1].shape().dims(), &[4, 2]);
        assert_eq!(result[2].shape().dims(), &[4, 2]);

        Ok(())
    }

    #[test]
    fn test_vsplit() -> TorshResult<()> {
        // Test vertical split with sections
        let tensor = randn(&[6, 4], None, None, None)?;
        let result = vsplit(&tensor, TensorSplitArg::Sections(3))?;
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(chunk.shape().dims(), &[2, 4]);
        }

        // Test vertical split with indices
        let result = vsplit(&tensor, TensorSplitArg::Indices(vec![2, 4]))?;
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].shape().dims(), &[2, 4]);
        assert_eq!(result[1].shape().dims(), &[2, 4]);
        assert_eq!(result[2].shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_dsplit() -> TorshResult<()> {
        // Test depth split with 3D tensor
        let tensor = randn(&[2, 3, 6], None, None, None)?;
        let result = dsplit(&tensor, TensorSplitArg::Sections(3))?;
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(chunk.shape().dims(), &[2, 3, 2]);
        }

        // Test depth split with indices
        let result = dsplit(&tensor, TensorSplitArg::Indices(vec![2, 4]))?;
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].shape().dims(), &[2, 3, 2]);
        assert_eq!(result[1].shape().dims(), &[2, 3, 2]);
        assert_eq!(result[2].shape().dims(), &[2, 3, 2]);

        Ok(())
    }

    #[test]
    #[should_panic(expected = "Input tensor must have at least 2 dimensions for hsplit")]
    fn test_hsplit_invalid_dimensions() {
        let tensor = randn(&[5], None, None, None).unwrap(); // 1D tensor
        hsplit(&tensor, TensorSplitArg::Sections(2)).unwrap();
    }

    #[test]
    #[should_panic(expected = "Input tensor must have at least 2 dimensions for vsplit")]
    fn test_vsplit_invalid_dimensions() {
        let tensor = randn(&[5], None, None, None).unwrap(); // 1D tensor
        vsplit(&tensor, TensorSplitArg::Sections(2)).unwrap();
    }

    #[test]
    #[should_panic(expected = "Input tensor must have at least 3 dimensions for dsplit")]
    fn test_dsplit_invalid_dimensions() {
        let tensor = randn(&[3, 4], None, None, None).unwrap(); // 2D tensor
        dsplit(&tensor, TensorSplitArg::Sections(2)).unwrap();
    }

    #[test]
    fn test_split_size_mode() -> TorshResult<()> {
        let tensor = randn(&[10, 4], None, None, None)?;
        let result = split(&tensor, SplitArg::Size(3), 0)?;

        // Should create 4 chunks: [3,4], [3,4], [3,4], [1,4]
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].shape().dims(), &[3, 4]);
        assert_eq!(result[1].shape().dims(), &[3, 4]);
        assert_eq!(result[2].shape().dims(), &[3, 4]);
        assert_eq!(result[3].shape().dims(), &[1, 4]);

        Ok(())
    }

    #[test]
    fn test_negative_dimension_indexing() -> TorshResult<()> {
        let tensor = randn(&[4, 6, 8], None, None, None)?;

        // Test negative dimension indexing
        let result1 = split(&tensor, SplitArg::Sections(2), -1)?; // Last dimension
        let result2 = split(&tensor, SplitArg::Sections(2), 2)?; // Explicit last dimension

        assert_eq!(result1.len(), result2.len());
        assert_eq!(result1[0].shape().dims(), result2[0].shape().dims());

        Ok(())
    }
}
