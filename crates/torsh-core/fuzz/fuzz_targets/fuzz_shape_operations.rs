#![no_main]
use libfuzzer_sys::fuzz_target;
use torsh_core::shape::Shape;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    
    // Parse input to create shape and operation parameters
    let dims_len = (data[0] % 8) + 1; // 1-8 dimensions
    let operation = data[1] % 4; // 4 different operations
    
    // Create dimensions from bytes
    let dims: Vec<usize> = data[2..2+dims_len as usize]
        .iter()
        .map(|&b| ((b as usize) % 20) + 1) // Limit dimension size to 1-20
        .collect();
    
    if dims.is_empty() {
        return;
    }
    
    let shape = Shape::new(dims.clone());
    
    match operation {
        0 => {
            // Test reshape operation
            let remaining_data = &data[2+dims_len as usize..];
            if remaining_data.is_empty() {
                return;
            }
            
            let new_dims_len = (remaining_data[0] % 6) + 1; // 1-6 dimensions
            let new_dims: Vec<usize> = remaining_data[1..1+new_dims_len as usize]
                .iter()
                .map(|&b| ((b as usize) % 20) + 1)
                .collect();
            
            if new_dims.is_empty() {
                return;
            }
            
            // Calculate expected element count
            let original_numel = shape.numel();
            let new_numel = new_dims.iter().product::<usize>();
            
            // Test reshape
            let reshape_result = shape.reshape(&new_dims);
            if original_numel == new_numel {
                assert!(reshape_result.is_ok(), "Reshape should succeed with same element count");
                if let Ok(reshaped) = reshape_result {
                    assert_eq!(reshaped.dims(), &new_dims, "Reshaped shape should match new dimensions");
                    assert_eq!(reshaped.numel(), original_numel, "Reshaped shape should have same element count");
                }
            } else {
                assert!(reshape_result.is_err(), "Reshape should fail with different element count");
            }
        }
        1 => {
            // Test transpose operation
            if dims.len() >= 2 {
                let transpose_result = shape.transpose();
                assert!(transpose_result.is_ok(), "Transpose should succeed for multi-dimensional shapes");
                
                if let Ok(transposed) = transpose_result {
                    assert_eq!(transposed.ndim(), shape.ndim(), "Transpose should preserve number of dimensions");
                    assert_eq!(transposed.numel(), shape.numel(), "Transpose should preserve element count");
                    
                    // Check that dimensions are correctly transposed
                    let original_dims = shape.dims();
                    let transposed_dims = transposed.dims();
                    assert_eq!(transposed_dims[0], original_dims[original_dims.len() - 1], "First dimension should be last");
                    assert_eq!(transposed_dims[transposed_dims.len() - 1], original_dims[0], "Last dimension should be first");
                }
            }
        }
        2 => {
            // Test squeeze operation
            let squeeze_result = shape.squeeze();
            assert!(squeeze_result.is_ok(), "Squeeze should always succeed");
            
            if let Ok(squeezed) = squeeze_result {
                assert_eq!(squeezed.numel(), shape.numel(), "Squeeze should preserve element count");
                
                // Check that all dimensions in squeezed shape are > 1
                for &dim in squeezed.dims() {
                    assert!(dim > 1 || squeezed.dims().is_empty(), "Squeezed shape should have no size-1 dimensions");
                }
            }
        }
        3 => {
            // Test unsqueeze operation
            let remaining_data = &data[2+dims_len as usize..];
            if remaining_data.is_empty() {
                return;
            }
            
            let dim_to_unsqueeze = remaining_data[0] as usize % (dims.len() + 1);
            let unsqueeze_result = shape.unsqueeze(dim_to_unsqueeze);
            assert!(unsqueeze_result.is_ok(), "Unsqueeze should succeed with valid dimension");
            
            if let Ok(unsqueezed) = unsqueeze_result {
                assert_eq!(unsqueezed.ndim(), shape.ndim() + 1, "Unsqueeze should add one dimension");
                assert_eq!(unsqueezed.numel(), shape.numel(), "Unsqueeze should preserve element count");
                assert_eq!(unsqueezed.dims()[dim_to_unsqueeze], 1, "Unsqueezed dimension should be 1");
            }
        }
        _ => unreachable!(),
    }
    
    // Test common operations that should always work
    let _ = shape.is_scalar();
    let _ = shape.is_empty();
    let _ = shape.is_contiguous();
    let _ = shape.clone();
    let _ = format!("{}", shape);
    let _ = format!("{:?}", shape);
    
    // Test strides calculation
    let strides = shape.strides();
    assert_eq!(strides.len(), shape.ndim(), "Strides length should match number of dimensions");
    
    // Test element count invariants
    let numel = shape.numel();
    if dims.contains(&0) {
        assert_eq!(numel, 0, "Shape with zero dimension should have 0 elements");
    } else {
        assert!(numel > 0, "Shape without zero dimensions should have positive element count");
    }
    
    // Test shape comparison
    let same_shape = Shape::new(dims.clone());
    assert_eq!(shape, same_shape, "Shapes with same dimensions should be equal");
    
    // Test broadcasting with self
    assert!(shape.broadcast_compatible(&shape), "Shape should be compatible with itself");
    let self_broadcast = shape.broadcast_shape(&shape);
    assert!(self_broadcast.is_ok(), "Broadcasting with self should succeed");
    if let Ok(result) = self_broadcast {
        assert_eq!(result.dims(), shape.dims(), "Broadcasting with self should yield same shape");
    }
});