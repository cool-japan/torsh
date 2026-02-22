#![no_main]
use libfuzzer_sys::fuzz_target;
use torsh_core::shape::Shape;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    
    // Parse input to create two shapes for broadcasting
    let split_idx = data.len() / 2;
    let shape1_data = &data[0..split_idx];
    let shape2_data = &data[split_idx..];
    
    // Create dimensions from bytes (limit to reasonable sizes)
    let shape1_dims: Vec<usize> = shape1_data
        .iter()
        .take(8) // Maximum 8 dimensions
        .map(|&b| ((b as usize) % 32) + 1) // Limit dimension size to 1-32
        .collect();
    
    let shape2_dims: Vec<usize> = shape2_data
        .iter()
        .take(8) // Maximum 8 dimensions
        .map(|&b| ((b as usize) % 32) + 1) // Limit dimension size to 1-32
        .collect();
    
    // Skip empty shapes
    if shape1_dims.is_empty() || shape2_dims.is_empty() {
        return;
    }
    
    // Create shapes
    let shape1 = Shape::new(shape1_dims);
    let shape2 = Shape::new(shape2_dims);
    
    // Test broadcasting operations
    let _ = shape1.broadcast_compatible(&shape2);
    let _ = shape1.broadcast_shape(&shape2);
    let _ = shape1.broadcast_with(shape2.dims());
    
    // Test symmetric operations
    let _ = shape2.broadcast_compatible(&shape1);
    let _ = shape2.broadcast_shape(&shape1);
    let _ = shape2.broadcast_with(shape1.dims());
    
    // Test broadcasting with scalar
    let scalar = Shape::scalar();
    let _ = shape1.broadcast_compatible(&scalar);
    let _ = shape1.broadcast_shape(&scalar);
    let _ = scalar.broadcast_compatible(&shape1);
    let _ = scalar.broadcast_shape(&shape1);
    
    // Test with ones shape
    let ones_shape = Shape::new(vec![1; shape1.ndim()]);
    let _ = shape1.broadcast_compatible(&ones_shape);
    let _ = shape1.broadcast_shape(&ones_shape);
    let _ = ones_shape.broadcast_compatible(&shape1);
    let _ = ones_shape.broadcast_shape(&shape1);
    
    // Test invariants
    // 1. Broadcasting should be symmetric for compatibility
    let compat_1_2 = shape1.broadcast_compatible(&shape2);
    let compat_2_1 = shape2.broadcast_compatible(&shape1);
    assert_eq!(compat_1_2, compat_2_1, "Broadcasting compatibility should be symmetric");
    
    // 2. If shapes are compatible, broadcast_shape should succeed
    if compat_1_2 {
        let broadcast_1_2 = shape1.broadcast_shape(&shape2);
        let broadcast_2_1 = shape2.broadcast_shape(&shape1);
        assert!(broadcast_1_2.is_ok(), "Broadcast should succeed if shapes are compatible");
        assert!(broadcast_2_1.is_ok(), "Broadcast should succeed if shapes are compatible");
        
        // Broadcast result should be the same regardless of order
        if let (Ok(result1), Ok(result2)) = (broadcast_1_2, broadcast_2_1) {
            assert_eq!(result1.dims(), result2.dims(), "Broadcast result should be symmetric");
        }
    }
    
    // 3. Any shape should be compatible with scalar
    assert!(shape1.broadcast_compatible(&scalar), "Any shape should be compatible with scalar");
    assert!(scalar.broadcast_compatible(&shape1), "Scalar should be compatible with any shape");
    
    // 4. Broadcasting with self should always succeed
    assert!(shape1.broadcast_compatible(&shape1), "Shape should be compatible with itself");
    let self_broadcast = shape1.broadcast_shape(&shape1);
    assert!(self_broadcast.is_ok(), "Broadcasting with self should succeed");
    if let Ok(result) = self_broadcast {
        assert_eq!(result.dims(), shape1.dims(), "Broadcasting with self should yield same shape");
    }
});