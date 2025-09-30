//! Tensor construction operations
//!
//! This module provides functions for constructing new tensors from existing ones
//! through operations like block diagonal matrices, Cartesian products, and
//! coordinate grids. These operations are fundamental for creating structured
//! tensors and coordinate systems in scientific computing and machine learning.

use torsh_core::Result as TorshResult;
use torsh_tensor::{creation::zeros, Tensor};

use super::shape::atleast_2d;

/// Create a block diagonal matrix from input tensors
///
/// ## Mathematical Background
///
/// A block diagonal matrix is a matrix where the non-zero elements are confined
/// to square blocks along the main diagonal:
///
/// ```
/// A = [ A₁  0   0  ]
///     [ 0   A₂  0  ]
///     [ 0   0   A₃ ]
/// ```
///
/// Each block Aᵢ can be a scalar, vector, or matrix. The resulting matrix has:
/// - **Size**: (∑ mᵢ) × (∑ nᵢ) where Aᵢ is mᵢ × nᵢ
/// - **Structure**: Only diagonal blocks contain non-zero elements
/// - **Determinant**: det(A) = ∏ det(Aᵢ) (product of block determinants)
///
/// ## Matrix Properties
///
/// Block diagonal matrices preserve many algebraic properties:
/// - **Multiplication**: (A ⊕ B)(C ⊕ D) = AC ⊕ BD (when compatible)
/// - **Inverse**: (A₁ ⊕ A₂)⁻¹ = A₁⁻¹ ⊕ A₂⁻¹ (when blocks are invertible)
/// - **Eigenvalues**: λ(A₁ ⊕ A₂) = λ(A₁) ∪ λ(A₂) (union of eigenvalues)
/// - **Trace**: tr(A₁ ⊕ A₂) = tr(A₁) + tr(A₂)
///
/// ## Implementation Details
///
/// The function automatically promotes inputs to 2D:
/// - Scalars (0D) → 1×1 matrices
/// - Vectors (1D) → column vectors (n×1)
/// - Matrices (2D) → unchanged
///
/// ## Parameters
/// * `tensors` - Slice of tensors to arrange in block diagonal form
///
/// ## Returns
/// * Block diagonal matrix with input tensors along the main diagonal
///
/// ## Example
/// ```rust
/// # use torsh_functional::manipulation::block_diag;
/// # use torsh_tensor::creation::{scalar, ones, eye};
/// let scalar = scalar(3.0)?;       // Will become 1×1 block
/// let vector = ones(&[2])?;        // Will become 2×1 block
/// let matrix = eye(3)?;            // Will become 3×3 block
///
/// let result = block_diag(&[scalar, vector, matrix])?;
/// // Result shape: (6, 5) with structure:
/// // [ 3  0  0  0  0 ]
/// // [ 0  1  0  0  0 ]
/// // [ 0  1  0  0  0 ]
/// // [ 0  0  1  0  0 ]
/// // [ 0  0  0  1  0 ]
/// // [ 0  0  0  0  1 ]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Applications
/// - **Linear systems**: Decoupled system representation
/// - **Control theory**: Multi-input multi-output systems
/// - **Graph theory**: Adjacency matrices of disconnected components
/// - **Neural networks**: Architecture design with independent blocks
/// - **Optimization**: Separable problem formulations
pub fn block_diag(tensors: &[Tensor]) -> TorshResult<Tensor> {
    if tensors.is_empty() {
        return zeros(&[0, 0]);
    }

    // Ensure all tensors are 2D
    let matrices: Vec<_> = tensors
        .iter()
        .map(atleast_2d)
        .collect::<TorshResult<Vec<_>>>()?;

    // Calculate output shape
    let total_rows: usize = matrices.iter().map(|m| m.shape().dims()[0]).sum();
    let total_cols: usize = matrices.iter().map(|m| m.shape().dims()[1]).sum();

    // Create output tensor
    let result = zeros(&[total_rows, total_cols])?;

    // Fill block diagonal
    let mut row_offset = 0;
    let mut col_offset = 0;

    for matrix in &matrices {
        let rows = matrix.shape().dims()[0];
        let cols = matrix.shape().dims()[1];

        // Copy matrix to the diagonal block
        for i in 0..rows {
            for j in 0..cols {
                let value = matrix.get(&[i, j])?;
                result.set(&[row_offset + i, col_offset + j], value)?;
            }
        }

        row_offset += rows;
        col_offset += cols;
    }

    Ok(result)
}

/// Compute the Cartesian product of input tensors
///
/// ## Mathematical Background
///
/// The Cartesian product A × B × ... creates all possible combinations where
/// each element comes from a different set:
///
/// ```
/// A × B = {(a,b) : a ∈ A, b ∈ B}
/// ```
///
/// For tensors, this creates a matrix where each row represents one combination.
///
/// ## Combinatorial Properties
///
/// For sets of sizes |A₁|, |A₂|, ..., |Aₙ|:
/// - **Output size**: |A₁| × |A₂| × ... × |Aₙ| combinations
/// - **Output shape**: (∏ᵢ |Aᵢ|) × n where n is number of input tensors
/// - **Ordering**: Lexicographic ordering of combinations
///
/// ## Generation Algorithm
///
/// Uses counter-based generation:
/// 1. Initialize counters [0, 0, ..., 0]
/// 2. For each row, use current counter values as indices
/// 3. Increment counters in lexicographic order
/// 4. Repeat until all combinations generated
///
/// ## Parameters
/// * `tensors` - Slice of tensors to compute Cartesian product
///
/// ## Returns
/// * Matrix where each row is one combination from the Cartesian product
///
/// ## Example
/// ```rust
/// # use torsh_functional::manipulation::cartesian_prod;
/// # use torsh_tensor::creation::tensor;
/// let a = tensor(&[1.0, 2.0])?;     // Set A = {1, 2}
/// let b = tensor(&[10.0, 20.0])?;   // Set B = {10, 20}
///
/// let product = cartesian_prod(&[a, b])?;
/// // Result shape: (4, 2) with combinations:
/// // [ 1, 10 ]  # (1, 10)
/// // [ 1, 20 ]  # (1, 20)
/// // [ 2, 10 ]  # (2, 10)
/// // [ 2, 20 ]  # (2, 20)
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Applications
/// - **Parameter grids**: Generate all parameter combinations for experiments
/// - **Coordinate systems**: Create discrete coordinate grids
/// - **Combinatorial optimization**: Enumerate solution space
/// - **Statistical sampling**: Factorial design experiments
/// - **Machine learning**: Hyperparameter grid search
pub fn cartesian_prod(tensors: &[Tensor]) -> TorshResult<Tensor> {
    if tensors.is_empty() {
        return zeros(&[0, 0]);
    }

    // Flatten all tensors
    let flattened: Vec<_> = tensors
        .iter()
        .map(|t| t.flatten())
        .collect::<TorshResult<Vec<_>>>()?;

    // Calculate total number of combinations
    let total_combinations: usize = flattened.iter().map(|t| t.shape().dims()[0]).product();

    let num_tensors = tensors.len();
    let result = zeros(&[total_combinations, num_tensors])?;

    // Generate Cartesian product
    let mut indices = vec![0; num_tensors];

    for i in 0..total_combinations {
        // Set values for current combination
        for (j, idx) in indices.iter().enumerate() {
            let value = flattened[j].get(&[*idx])?;
            result.set(&[i, j], value)?;
        }

        // Update indices (lexicographic increment)
        for j in (0..num_tensors).rev() {
            indices[j] += 1;
            if indices[j] < flattened[j].shape().dims()[0] {
                break;
            }
            indices[j] = 0;
        }
    }

    Ok(result)
}

/// Create coordinate matrices from coordinate vectors
///
/// ## Mathematical Background
///
/// Meshgrid creates coordinate matrices for evaluating functions on grids.
/// Given coordinate vectors x, y, ..., it produces arrays X, Y, ... where
/// each array contains coordinates for one dimension at every grid point.
///
/// For 2D case with x ∈ ℝᵐ and y ∈ ℝⁿ:
/// ```
/// X[i,j] = x[j]  (x-coordinates at each point)
/// Y[i,j] = y[i]  (y-coordinates at each point)
/// ```
///
/// ## Indexing Conventions
///
/// ### Matrix Indexing ('ij')
/// - **X[i,j] = x[j]**: X varies along columns (dimension 1)
/// - **Y[i,j] = y[i]**: Y varies along rows (dimension 0)
/// - **Convention**: Matches array indexing (row, column)
///
/// ### Cartesian Indexing ('xy')
/// - **X[i,j] = x[i]**: X varies along rows (dimension 0)
/// - **Y[i,j] = y[j]**: Y varies along columns (dimension 1)
/// - **Convention**: Matches mathematical (x, y) convention
///
/// ## Grid Structure
///
/// For vectors x = [x₁, x₂, ...] and y = [y₁, y₂, ...]:
///
/// Matrix indexing ('ij'):
/// ```
/// X = [ x₁  x₂  x₃ ]    Y = [ y₁  y₁  y₁ ]
///     [ x₁  x₂  x₃ ]        [ y₂  y₂  y₂ ]
///     [ x₁  x₂  x₃ ]        [ y₃  y₃  y₃ ]
/// ```
///
/// Cartesian indexing ('xy'):
/// ```
/// X = [ x₁  x₁  x₁ ]    Y = [ y₁  y₂  y₃ ]
///     [ x₂  x₂  x₂ ]        [ y₁  y₂  y₃ ]
///     [ x₃  x₃  x₃ ]        [ y₁  y₂  y₃ ]
/// ```
///
/// ## Parameters
/// * `tensors` - Coordinate vectors (1D tensors)
/// * `indexing` - Either "ij" (matrix) or "xy" (Cartesian) indexing
///
/// ## Returns
/// * Vector of coordinate grids, one for each input dimension
///
/// ## Example
/// ```rust
/// # use torsh_functional::manipulation::meshgrid;
/// # use torsh_tensor::creation::tensor;
/// let x = tensor(&[1.0, 2.0, 3.0])?;
/// let y = tensor(&[10.0, 20.0])?;
///
/// let grids = meshgrid(&[x, y], "ij")?;
/// let (x_grid, y_grid) = (&grids[0], &grids[1]);
/// // x_grid: [[1, 2, 3],     y_grid: [[10, 10, 10],
/// //          [1, 2, 3]]              [20, 20, 20]]
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Applications
/// - **Function evaluation**: Evaluate f(x,y) on regular grids
/// - **Numerical methods**: Finite difference discretization
/// - **Visualization**: Create coordinate systems for plotting
/// - **Image processing**: Pixel coordinate generation
/// - **Scientific computing**: Physical simulation grids
/// - **Interpolation**: Define evaluation points for splines
pub fn meshgrid(tensors: &[Tensor], indexing: &str) -> TorshResult<Vec<Tensor>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    // Validate indexing parameter
    if indexing != "xy" && indexing != "ij" {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            &format!("Invalid indexing mode: {}. Use 'xy' or 'ij'", indexing),
            "meshgrid",
        ));
    }

    // Get lengths of each tensor
    let lengths: Vec<_> = tensors.iter().map(|t| t.shape().dims()[0]).collect();

    // Create output shape
    let output_shape = if indexing == "xy" && tensors.len() >= 2 {
        let mut shape = lengths.clone();
        shape.swap(0, 1);
        shape
    } else {
        lengths.clone()
    };

    // Create coordinate grids
    let mut grids = Vec::new();

    for (i, tensor) in tensors.iter().enumerate() {
        let grid = zeros(&output_shape)?;

        // Fill grid with broadcasted values
        let tensor_data = tensor.to_vec()?;

        // For each position in the output grid
        for idx in 0..output_shape.iter().product::<usize>() {
            // Convert flat index to multi-dimensional coordinates
            let mut coords = Vec::new();
            let mut remaining = idx;
            for &dim_size in output_shape.iter().rev() {
                coords.push(remaining % dim_size);
                remaining /= dim_size;
            }
            coords.reverse();

            // Get the value from the i-th input tensor
            let coord_i = if indexing == "xy" && tensors.len() >= 2 {
                if i == 0 {
                    coords[1]
                } else if i == 1 {
                    coords[0]
                } else {
                    coords[i]
                }
            } else {
                coords[i]
            };

            // Set the value in the grid
            if coord_i < tensor_data.len() {
                let value = tensor_data[coord_i];
                grid.set(&coords.to_vec(), value)?;
            }
        }

        grids.push(grid);
    }

    Ok(grids)
}
