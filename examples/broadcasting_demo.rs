//! Broadcasting Operations Demo
//!
//! This example demonstrates comprehensive tensor broadcasting operations
//! with error handling and memory optimization features.

use torsh_core::device::DeviceType;
use torsh_tensor::{
    broadcast::{BroadcastOps, BroadcastShape},
    Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Broadcasting Operations Demo");
    println!("===========================");

    // Example 1: Basic Broadcasting
    println!("\n1. Basic Broadcasting Examples");
    basic_broadcasting_examples()?;

    // Example 2: Error Handling
    println!("\n2. Broadcasting Error Handling");
    error_handling_examples()?;

    // Example 3: Memory Efficiency Analysis
    println!("\n3. Memory Efficiency Analysis");
    memory_efficiency_examples()?;

    // Example 4: Complex Broadcasting Scenarios
    println!("\n4. Complex Broadcasting Scenarios");
    complex_broadcasting_examples()?;

    // Example 5: Broadcasting Utilities
    println!("\n5. Broadcasting Utilities");
    broadcasting_utilities_examples()?;

    Ok(())
}

fn basic_broadcasting_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1a: Vector + Scalar
    let vector = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu);
    let scalar = Tensor::from_data(vec![10.0], vec![1], DeviceType::Cpu);
    let result = vector.add(&scalar)?;
    println!("Vector [1,2,3] + Scalar [10] = {:?}", result.data());

    // Example 1b: Matrix + Vector (column broadcasting)
    let matrix = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DeviceType::Cpu,
    );
    let vector = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![1, 3], DeviceType::Cpu);
    let result = matrix.add(&vector)?;
    println!("Matrix (2x3) + Vector (1x3):");
    println!("  Input Matrix: {:?}", matrix.data());
    println!("  Input Vector: {:?}", vector.data());
    println!("  Result: {:?}", result.data());

    // Example 1c: Different dimension broadcasting
    let tensor_3d = Tensor::from_data(
        vec![1.0; 24], // 2x3x4 tensor
        vec![2, 3, 4],
        DeviceType::Cpu,
    );
    let tensor_2d = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4], DeviceType::Cpu);
    let result = tensor_3d.add(&tensor_2d)?;
    println!(
        "3D Tensor (2x3x4) + 2D Tensor (1x4) = {} elements",
        result.numel()
    );

    Ok(())
}

fn error_handling_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Example 2a: Incompatible shapes
    let tensor1 = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu);
    let tensor2 = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu);

    match tensor1.add(&tensor2) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error for incompatible shapes [3] + [2]: {}", e),
    }

    // Example 2b: Validation before operation
    let shape1 = vec![3, 4];
    let shape2 = vec![2, 5];
    match BroadcastOps::validate_broadcast_operation(&shape1, &shape2, "addition") {
        Ok(_) => println!("Shapes are compatible"),
        Err(e) => println!(
            "Validation caught incompatible shapes {:?} + {:?}: {}",
            shape1, shape2, e
        ),
    }

    // Example 2c: Valid but memory-intensive broadcasting
    let shape1 = vec![1000, 1];
    let shape2 = vec![1, 1000];
    if let Ok(info) = BroadcastOps::get_broadcast_info(&shape1, &shape2) {
        println!("Broadcasting info for shapes {:?} + {:?}:", shape1, shape2);
        println!("  Resulting shape: {:?}", info.broadcast_shape);
        println!("  Expansion factor 1: {}x", info.expansion_factor1);
        println!("  Expansion factor 2: {}x", info.expansion_factor2);
        println!("  Memory efficient: {}", info.is_memory_efficient);
    }

    Ok(())
}

fn memory_efficiency_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Example 3a: Memory-efficient broadcasting
    let efficient_shape1 = vec![4, 1];
    let efficient_shape2 = vec![1, 3];
    let memory_req = BroadcastOps::estimate_broadcast_memory(
        &efficient_shape1,
        &efficient_shape2,
        std::mem::size_of::<f32>(),
    )?;
    println!(
        "Memory-efficient broadcasting {:?} + {:?}:",
        efficient_shape1, efficient_shape2
    );
    println!("  Memory required: {} bytes", memory_req);

    // Example 3b: Memory-intensive broadcasting
    let intensive_shape1 = vec![100, 1];
    let intensive_shape2 = vec![1, 100];
    let memory_req = BroadcastOps::estimate_broadcast_memory(
        &intensive_shape1,
        &intensive_shape2,
        std::mem::size_of::<f32>(),
    )?;
    println!(
        "Memory-intensive broadcasting {:?} + {:?}:",
        intensive_shape1, intensive_shape2
    );
    println!("  Memory required: {} bytes", memory_req);

    // Example 3c: Broadcasting efficiency check
    let tensor1 = Tensor::from_data(vec![1.0; 4], vec![4, 1], DeviceType::Cpu);
    let tensor2 = Tensor::from_data(vec![1.0; 3], vec![1, 3], DeviceType::Cpu);

    if tensor1.shape().is_broadcast_efficient(&tensor2.shape()) {
        println!(
            "Broadcasting {:?} + {:?} is memory efficient",
            tensor1.shape().dims(),
            tensor2.shape().dims()
        );
    } else {
        println!(
            "Broadcasting {:?} + {:?} may use significant memory",
            tensor1.shape().dims(),
            tensor2.shape().dims()
        );
    }

    Ok(())
}

fn complex_broadcasting_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Example 4a: Multi-dimensional broadcasting
    let tensor_4d = Tensor::from_data(vec![1.0; 2 * 1 * 3 * 4], vec![2, 1, 3, 4], DeviceType::Cpu);
    let tensor_3d = Tensor::from_data(vec![0.5; 5 * 1 * 4], vec![5, 1, 4], DeviceType::Cpu);
    let result = tensor_4d.add(&tensor_3d)?;

    println!("4D Tensor (2,1,3,4) + 3D Tensor (5,1,4):");
    println!("  Result shape: {:?}", result.shape().dims());
    println!("  Result size: {} elements", result.numel());

    // Example 4b: Broadcasting with single-element tensors
    let large_tensor = Tensor::from_data(vec![2.0; 24], vec![2, 3, 4], DeviceType::Cpu);
    let single_element = Tensor::from_data(vec![0.5], vec![1], DeviceType::Cpu); // Scalar
    let result = large_tensor.mul(&single_element)?;
    println!("Large tensor (2,3,4) * Scalar:");
    println!(
        "  All elements halved: first few = {:?}",
        &result.data()[0..3]
    );

    // Example 4c: Chain of broadcasting operations
    let a = Tensor::from_data(vec![1.0, 2.0], vec![2, 1], DeviceType::Cpu);
    let b = Tensor::from_data(vec![3.0, 4.0, 5.0], vec![1, 3], DeviceType::Cpu);
    let c = Tensor::from_data(vec![10.0], vec![1], DeviceType::Cpu);

    let intermediate = a.add(&b)?; // (2,1) + (1,3) -> (2,3)
    let final_result = intermediate.mul(&c)?; // (2,3) * (1) -> (2,3)

    println!("Chained broadcasting: ((2,1) + (1,3)) * (1):");
    println!("  Final result: {:?}", final_result.data());

    Ok(())
}

fn broadcasting_utilities_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Example 5a: Shape compatibility checking
    let shapes = vec![
        (vec![3, 4], vec![1, 4]),
        (vec![3, 1], vec![3, 4]),
        (vec![1], vec![3, 4]),
        (vec![3, 4], vec![2, 4]), // Incompatible
    ];

    for (shape1, shape2) in shapes {
        let compatible = BroadcastOps::are_shapes_compatible(&shape1, &shape2)?;
        println!(
            "Shapes {:?} and {:?} compatible: {}",
            shape1, shape2, compatible
        );

        if compatible {
            let broadcast_shape = BroadcastOps::compute_broadcast_shape(&shape1, &shape2)?;
            println!("  Broadcast shape: {:?}", broadcast_shape);
        }
    }

    // Example 5b: Index computation examples
    println!("\nIndex computation examples:");
    let multi_index = vec![1, 2];
    let original_shape = vec![1, 3];
    let broadcast_shape = vec![2, 3];

    let linear_idx =
        BroadcastOps::compute_broadcast_index(&multi_index, &original_shape, &broadcast_shape)?;
    println!(
        "Multi-index {:?} for shape {:?} -> linear index: {}",
        multi_index, original_shape, linear_idx
    );

    // Example 5c: Flat to multi-index conversion
    let shape = vec![2, 3, 4];
    for flat_idx in [0, 5, 11, 23] {
        let multi_idx = BroadcastOps::flat_to_multi_index(flat_idx, &shape);
        println!(
            "Flat index {} for shape {:?} -> multi-index {:?}",
            flat_idx, shape, multi_idx
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcasting_demo() {
        // Test that the demo functions run without errors
        assert!(basic_broadcasting_examples().is_ok());
        assert!(error_handling_examples().is_ok());
        assert!(memory_efficiency_examples().is_ok());
        assert!(complex_broadcasting_examples().is_ok());
        assert!(broadcasting_utilities_examples().is_ok());
    }

    #[test]
    fn test_comprehensive_broadcasting() {
        // Test comprehensive broadcasting scenarios
        let tensor1 = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3, 1], DeviceType::Cpu);
        let tensor2 = Tensor::from_data(vec![4.0, 5.0], vec![1, 2], DeviceType::Cpu);

        let result = tensor1.add(&tensor2).unwrap();
        assert_eq!(result.shape().dims(), &[3, 2]);
        assert_eq!(result.data(), vec![5.0, 6.0, 6.0, 7.0, 7.0, 8.0]);

        // Test memory efficiency check
        assert!(tensor1.shape().is_broadcast_efficient(&tensor2.shape()));
    }
}
