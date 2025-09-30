/*!
 * Basic Usage Example for ToRSh-Sparse
 *
 * This example demonstrates the fundamental operations of sparse tensors,
 * including creation, format conversion, and basic arithmetic operations.
 */

use torsh_core::TorshError;
use torsh_sparse::*;
use torsh_tensor::creation::zeros;

fn main() -> Result<(), TorshError> {
    println!("ToRSh-Sparse Basic Usage Example");
    println!("================================");

    // 1. Create a sparse matrix from triplets (COO format)
    println!("1. Creating sparse matrix from triplets...");
    let triplets = vec![
        (0, 0, 1.0),
        (0, 2, 2.0),
        (1, 1, 3.0),
        (2, 0, 4.0),
        (2, 2, 5.0),
    ];

    let coo_matrix = CooTensor::from_triplets(triplets, (3, 3))?;
    println!("COO matrix created with {} non-zeros", coo_matrix.nnz());

    // 2. Convert to different formats
    println!("\n2. Converting to different formats...");
    let csr_matrix = CsrTensor::from_coo(&coo_matrix)?;
    let csc_matrix = CscTensor::from_coo(&coo_matrix)?;

    println!("CSR matrix: {} non-zeros", csr_matrix.nnz());
    println!("CSC matrix: {} non-zeros", csc_matrix.nnz());

    // 3. Element access
    println!("\n3. Element access...");
    let value = csr_matrix.get(0, 0).unwrap_or(0.0);
    println!("Element at (0, 0): {value}");

    let value = csr_matrix.get(1, 1).unwrap_or(0.0);
    println!("Element at (1, 1): {value}");

    let value = csr_matrix.get(0, 1).unwrap_or(0.0); // Zero element
    println!("Element at (0, 1): {value}");

    // 4. Matrix properties
    println!("\n4. Matrix properties...");
    println!("Shape: {:?}", csr_matrix.shape());
    println!("Non-zeros: {}", csr_matrix.nnz());
    println!("Density: {:.2}%", csr_matrix.density() * 100.0);

    // 5. Convert to dense for visualization
    println!("\n5. Dense representation:");
    let dense = csr_matrix.to_dense()?;
    println!("Dense matrix:");
    for i in 0..3 {
        for j in 0..3 {
            print!("{:.1} ", dense.get(&[i, j]).unwrap());
        }
        println!();
    }

    // 6. Matrix-vector multiplication
    println!("\n6. Matrix-vector multiplication...");
    let vector = [1.0, 2.0, 3.0];
    let vector_tensor = zeros::<f32>(&[3])?;
    for (i, &v) in vector.iter().enumerate() {
        vector_tensor.set(&[i], v)?;
    }
    let result = csr_matrix.matvec(&vector_tensor)?;
    println!("Matrix * vector = {:?}", result.to_vec()?);

    // 7. Transpose
    println!("\n7. Matrix transpose...");
    let transposed = csr_matrix.transpose()?;
    println!("Transposed matrix has {} non-zeros", transposed.nnz());

    // 8. Scalar operations
    println!("\n8. Scalar operations...");
    let scaled = csr_matrix.scale(2.0)?;
    println!("Scaled matrix (2x) has {} non-zeros", scaled.nnz());

    // 9. Matrix addition
    println!("\n9. Matrix addition...");
    let sum = csr_matrix.add(&scaled)?;
    println!("Sum matrix has {} non-zeros", sum.nnz());

    // 10. Reductions
    println!("\n10. Reduction operations...");
    let total_sum = csr_matrix.sum()?;
    println!("Sum of all elements: {total_sum}");

    let l2_norm = csr_matrix.norm(2.0)?;
    println!("L2 norm: {l2_norm:.4}");

    let diagonal = csr_matrix.diagonal()?;
    println!("Diagonal elements: {diagonal:?}");

    println!("\nBasic usage example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let result = main();
        assert!(result.is_ok());
    }

    #[test]
    fn test_matrix_properties() -> Result<(), TorshError> {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let coo = CooTensor::from_triplets(triplets, (3, 3))?;
        let csr = CsrTensor::from_coo(&coo)?;

        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.shape(), (3, 3));
        assert!((csr.density() - 3.0 / 9.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_format_conversions() -> Result<(), TorshError> {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let coo = CooTensor::from_triplets(triplets, (3, 3))?;

        let csr = CsrTensor::from_coo(&coo)?;
        let csc = CscTensor::from_coo(&coo)?;

        assert_eq!(coo.nnz(), csr.nnz());
        assert_eq!(coo.nnz(), csc.nnz());

        // Test element access consistency
        assert_eq!(csr.get(0, 0)?, 1.0);
        assert_eq!(csr.get(1, 1)?, 2.0);
        assert_eq!(csr.get(2, 2)?, 3.0);

        Ok(())
    }
}
