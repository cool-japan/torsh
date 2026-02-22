//! Broadcasting utilities

use torsh_core::{Result as TorshResult, Shape, TorshError};
use torsh_tensor::Tensor;

/// Broadcast multiple tensors to a common shape
pub fn broadcast_tensors(tensors: &[Tensor]) -> TorshResult<Vec<Tensor>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    // Get broadcast shape
    let shapes: Vec<_> = tensors.iter().map(|t| t.shape().clone()).collect();
    let broadcast_shape = broadcast_shapes(&shapes)?;

    // Broadcast each tensor
    tensors
        .iter()
        .map(|t| t.broadcast_to(broadcast_shape.dims()))
        .collect()
}

/// Compute the broadcast shape of multiple shapes
pub fn broadcast_shapes(shapes: &[Shape]) -> TorshResult<Shape> {
    if shapes.is_empty() {
        return Ok(Shape::new(vec![]));
    }

    // Find maximum number of dimensions
    let max_dims = shapes.iter().map(|s| s.ndim()).max().unwrap_or(0);

    // Initialize result shape with 1s
    let mut result = vec![1; max_dims];

    // Apply broadcasting rules
    for shape in shapes {
        let offset = max_dims - shape.ndim();

        for (i, &dim) in shape.dims().iter().enumerate() {
            let result_idx = offset + i;

            // dim is usize, so can't be negative

            if result[result_idx] == 1 {
                result[result_idx] = dim;
            } else if result[result_idx] != dim && dim != 1 {
                return Err(TorshError::invalid_argument_with_context(
                    &format!(
                        "Shapes cannot be broadcast together: dimension {} has sizes {} and {}",
                        result_idx, result[result_idx], dim
                    ),
                    "broadcast",
                ));
            }
        }
    }

    Ok(Shape::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes() {
        // Test basic broadcasting
        let shapes = vec![
            Shape::new(vec![2]),
            Shape::new(vec![3, 1]),
            Shape::new(vec![1, 1, 1]),
        ];

        let result = broadcast_shapes(&shapes).unwrap();
        assert_eq!(result.dims(), &[1, 3, 2]);

        // Test incompatible shapes
        let shapes = vec![Shape::new(vec![3, 4]), Shape::new(vec![2, 5])];

        assert!(broadcast_shapes(&shapes).is_err());
    }
}
