//! Gradient Checkpointing Example
//!
//! This example demonstrates how to use gradient checkpointing to save memory
//! during training of large neural networks.

// TODO: Re-enable when checkpoint functions are properly exported
/*
use torsh_autograd::checkpoint::{
    auto_checkpoint_sequence, checkpoint, checkpoint_sequential,
    configure_checkpointing_with_strategy, get_checkpoint_memory_stats, CheckpointStrategy,
};
*/
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

fn main() -> Result<()> {
    println!("Gradient Checkpointing Example");
    println!("==============================");
    println!("TODO: Re-enable when checkpoint functions are properly exported");

    // TODO: Re-enable when checkpoint functions are properly exported
    /*
    // Example 1: Basic checkpointing
    println!("\n1. Basic Checkpointing");
    basic_checkpointing_example()?;

    // Example 2: Sequential checkpointing
    println!("\n2. Sequential Checkpointing");
    sequential_checkpointing_example()?;

    // Example 3: Transformer layer checkpointing
    println!("\n3. Transformer Layer Checkpointing");
    transformer_checkpointing_example()?;

    // Example 4: Adaptive checkpointing
    println!("\n4. Adaptive Checkpointing");
    adaptive_checkpointing_example()?;
    */

    // Example 5: Memory statistics
    // println!("\n5. Memory Statistics");
    // memory_statistics_example();

    Ok(())
}

/// Demonstrate basic gradient checkpointing
fn basic_checkpointing_example() -> Result<()> {
    // Define a compute-intensive function
    let expensive_operation = |inputs: &[Tensor<f32>]| -> Result<Vec<Tensor<f32>>> {
        let mut result = inputs[0].clone();

        // Simulate expensive computation with multiple operations
        for _ in 0..5 {
            result = result.mul_scalar(1.1)?;
            result = result.add_scalar(0.1)?;
        }

        Ok(vec![result])
    };

    let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu);

    // Without checkpointing (normal execution)
    let normal_result = expensive_operation(&[input.clone()])?;

    // With checkpointing (memory-efficient execution)
    // let checkpointed_result = checkpoint(expensive_operation, &[input.clone()])?;
    let checkpointed_result = normal_result.clone(); // TODO: Re-enable when checkpoint functions are exported

    println!("Normal result:       {:?}", normal_result[0].data());
    println!("Checkpointed result: {:?}", checkpointed_result[0].data());
    println!(
        "Results are identical: {}",
        *normal_result[0].data() == *checkpointed_result[0].data()
    );

    Ok(())
}

/// Demonstrate sequential checkpointing for multiple operations
fn sequential_checkpointing_example() -> Result<()> {
    // Define a sequence of operations
    let relu = |x: &Tensor<f32>| x.relu();
    let layer_norm = |x: &Tensor<f32>| {
        // Simplified layer normalization (just scaling for demo)
        x.mul_scalar(0.5)
    };
    let linear = |x: &Tensor<f32>| x.mul_scalar(2.0)?.add_scalar(1.0);
    let dropout = |x: &Tensor<f32>| x.mul_scalar(0.9); // Simplified dropout

    let operations = [relu, layer_norm, linear, dropout];
    let input = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![4], DeviceType::Cpu);

    // Apply sequential checkpointing with 2 segments
    // let result = checkpoint_sequential(&operations, &input, 2)?;
    let result = input.clone(); // TODO: Re-enable when checkpoint functions are exported

    println!("Input:  {:?}", input.data());
    println!("Output: {:?}", result.data());

    Ok(())
}

/// Demonstrate transformer layer checkpointing pattern
fn transformer_checkpointing_example() -> Result<()> {
    // Simplified transformer layer function
    let transformer_layer = |x: &Tensor<f32>| -> Result<Tensor<f32>> {
        // Attention + FFN in one function for simplicity
        let attended = x.mul_scalar(0.8)?.add_scalar(0.1)?; // Simplified attention
        let expanded = attended.mul_scalar(4.0)?; // FFN expand
        let activated = expanded.relu()?; // FFN activation
        activated.mul_scalar(0.25) // FFN project back
    };

    let input = Tensor::from_data(vec![0.5, 1.0, 1.5, 2.0], vec![4], DeviceType::Cpu);
    let mut current_input = input;

    // Process multiple transformer layers with checkpointing
    for layer_idx in 0..3 {
        // let result = checkpoint(
        //     |inputs| transformer_layer(&inputs[0]).map(|output| vec![output]),
        //     &[current_input.clone()],
        // )?;
        let result = vec![transformer_layer(&current_input)?]; // TODO: Re-enable when checkpoint functions are exported

        current_input = result.into_iter().next().unwrap();
        println!("Layer {} output: {:?}", layer_idx, current_input.data());
    }

    Ok(())
}

/// Demonstrate adaptive checkpointing based on memory budget
fn adaptive_checkpointing_example() -> Result<()> {
    // Configure adaptive checkpointing with 256MB memory budget
    // configure_checkpointing_with_strategy(
    //     10,   // max checkpoints
    //     true, // enabled
    //     CheckpointStrategy::Adaptive {
    //         memory_budget_mb: 256,
    //     },
    // ); // TODO: Re-enable when checkpoint functions are exported

    // Create a sequence of operations that would use significant memory
    let heavy_operation = |x: &Tensor<f32>| -> Result<Tensor<f32>> {
        let expanded = x.mul_scalar(1.5)?;
        let processed = expanded.add_scalar(0.2)?;
        processed.relu()
    };

    let operations = vec![heavy_operation; 8];
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu);

    // let result = auto_checkpoint_sequence(&operations, &input, 256)?;
    let result = input.clone(); // TODO: Re-enable when checkpoint functions are exported

    println!("Input:  {:?}", input.data());
    println!("Final output after 8 operations: {:?}", result.data());

    Ok(())
}

/// Demonstrate memory statistics collection
fn memory_statistics_example() {
    println!("Gradient Checkpointing Memory Statistics:");

    // let stats = get_checkpoint_memory_stats();
    // TODO: Re-enable when checkpoint functions are exported
    println!("  Active checkpoints: {}", 0); // stats.num_active_checkpoints
    println!("  Total memory saved: {} bytes", 0); // stats.total_memory_saved
    println!(
        "  Average memory per checkpoint: {} bytes",
        0 // stats.average_memory_per_checkpoint
    );
    println!("  Total recompute time: {:.3}s", 0.0); // stats.total_recompute_time

    // if stats.total_memory_saved > 0 {
    //     println!(
    //         "  Memory efficiency: {:.1}x",
    //         stats.total_memory_saved as f32 / (stats.num_active_checkpoints as f32 * 1000.0)
    //     );
    // }
}
