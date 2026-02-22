#![no_main]
use libfuzzer_sys::fuzz_target;
use torsh_core::shape::{Shape, ShapeBuilder};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    
    // Create dimensions from bytes (limit to reasonable sizes to avoid memory issues)
    let dims: Vec<usize> = data
        .iter()
        .take(10) // Maximum 10 dimensions
        .map(|&b| {
            if b == 0 {
                0 // Test zero dimensions
            } else {
                ((b as usize) % 1000) + 1 // Limit dimension size to 1-1000
            }
        })
        .collect();
    
    // Test basic shape creation
    let shape = Shape::new(dims.clone());
    
    // Test basic invariants
    assert_eq!(shape.dims(), &dims, "Shape dimensions should match input");
    assert_eq!(shape.ndim(), dims.len(), "Shape ndim should match number of dimensions");
    
    // Test element count calculation (with overflow protection)
    let numel = shape.numel();
    
    // Verify element count calculation manually (with overflow protection)
    let mut expected_numel = 1usize;
    let mut overflow = false;
    for &dim in &dims {
        if dim == 0 {
            expected_numel = 0;
            break;
        }
        match expected_numel.checked_mul(dim) {
            Some(new_numel) => expected_numel = new_numel,
            None => {
                overflow = true;
                break;
            }
        }
    }
    
    if !overflow {
        assert_eq!(numel, expected_numel, "Element count should be correct");
    }
    
    // Test strides calculation
    let strides = shape.strides();
    assert_eq!(strides.len(), dims.len(), "Strides length should match dimensions");
    
    // Test shape builder with various configurations
    if !dims.is_empty() {
        let builder_result = ShapeBuilder::new()
            .dims(dims.clone())
            .max_dims(dims.len() + 2)
            .max_elements(usize::MAX)
            .allow_empty(true)
            .build();
        
        assert!(builder_result.is_ok(), "Shape builder should succeed with valid parameters");
        
        if let Ok(built_shape) = builder_result {
            assert_eq!(built_shape.dims(), &dims, "Builder shape should match input");
        }
    }
    
    // Test with restrictive builder parameters
    if !dims.is_empty() && dims.len() > 2 {
        let restrictive_result = ShapeBuilder::new()
            .dims(dims.clone())
            .max_dims(2) // Too restrictive
            .build();
        
        assert!(restrictive_result.is_err(), "Builder should fail with restrictive max_dims");
    }
    
    // Test scalar shape
    let scalar = Shape::scalar();
    assert_eq!(scalar.dims(), &[] as &[usize], "Scalar should have no dimensions");
    assert_eq!(scalar.ndim(), 0, "Scalar should have 0 dimensions");
    assert_eq!(scalar.numel(), 1, "Scalar should have 1 element");
    
    // Test empty shape
    let empty_shape = Shape::new(vec![]);
    assert_eq!(empty_shape.dims(), &[] as &[usize], "Empty shape should have no dimensions");
    assert_eq!(empty_shape.ndim(), 0, "Empty shape should have 0 dimensions");
    assert_eq!(empty_shape.numel(), 1, "Empty shape should have 1 element");
    
    // Test shapes with zero dimensions
    if dims.contains(&0) {
        assert_eq!(shape.numel(), 0, "Shape with zero dimension should have 0 elements");
    }
    
    // Test shape equality
    let shape2 = Shape::new(dims.clone());
    assert_eq!(shape, shape2, "Shapes with same dimensions should be equal");
    
    // Test shape display
    let _ = format!("{}", shape);
    let _ = format!("{:?}", shape);
    
    // Test shape cloning
    let cloned = shape.clone();
    assert_eq!(shape, cloned, "Cloned shape should be equal to original");
});