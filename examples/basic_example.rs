//! Basic ToRSh example demonstrating core functionality

use torsh::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ToRSh {} - Basic Example", torsh::VERSION);
    println!("==========================");
    
    // Tensor creation
    println!("\n1. Tensor Creation:");
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu);
    let b = Tensor::from_data(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2], DeviceType::Cpu);
    
    println!("Tensor a: {:?}", a);
    println!("Tensor b: {:?}", b);
    
    // Element-wise operations
    println!("\n2. Element-wise Operations:");
    let add_result = a.add(&b)?;
    let mul_result = a.mul(&b)?;
    
    println!("a + b: {:?}", add_result);
    println!("a * b: {:?}", mul_result);
    
    // Matrix multiplication
    println!("\n3. Matrix Multiplication:");
    let c = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], DeviceType::Cpu);
    let d = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], DeviceType::Cpu);
    
    let matmul_result = c.matmul(&d)?;
    println!("Matrix multiplication result: {:?}", matmul_result);
    
    // Broadcasting
    println!("\n4. Broadcasting:");
    let large = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], DeviceType::Cpu);
    let small = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![1, 3], DeviceType::Cpu);
    
    let broadcast_result = large.add(&small)?;
    println!("Broadcasting add result: {:?}", broadcast_result);
    
    // Reductions
    println!("\n5. Reductions:");
    let sum = a.sum()?;
    let mean = a.mean()?;
    let max = a.max()?;
    
    println!("Sum: {:?}", sum);
    println!("Mean: {:?}", mean);
    println!("Max: {:?}", max);
    
    // Activations
    println!("\n6. Activation Functions:");
    let negative = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2], DeviceType::Cpu);
    let relu_result = negative.relu()?;
    let sigmoid_result = negative.sigmoid()?;
    
    println!("ReLU result: {:?}", relu_result);
    println!("Sigmoid result: {:?}", sigmoid_result);
    
    println!("\n✅ All operations completed successfully!");
    println!("ToRSh is working correctly with:");
    println!("  • Tensor creation and manipulation");
    println!("  • Element-wise operations with broadcasting");
    println!("  • Matrix multiplication");
    println!("  • Reduction operations");
    println!("  • Activation functions");
    
    Ok(())
}