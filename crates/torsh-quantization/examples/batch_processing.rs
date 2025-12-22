//! Batch Quantization Processing
//!
//! Demonstrates batch quantization with consistent parameters

use torsh_quantization::{quantize_batch_consistent, QuantConfig};
use torsh_tensor::creation::tensor_1d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Batch Quantization Example ===\n");

    // Create multiple tensors
    let tensor1 = tensor_1d(&vec![1.0, 2.0, 3.0, 4.0])?;
    let tensor2 = tensor_1d(&vec![5.0, 6.0, 7.0, 8.0])?;
    let tensor3 = tensor_1d(&vec![9.0, 10.0, 11.0, 12.0])?;

    let tensors = vec![&tensor1, &tensor2, &tensor3];
    let config = QuantConfig::int8();

    println!(
        "Quantizing {} tensors with consistent parameters...\n",
        tensors.len()
    );

    let results = quantize_batch_consistent(&tensors, &config)?;

    println!("Batch quantization results:");
    println!("  Shared scale: {:.6}", results[0].1);
    println!("  Shared zero_point: {}", results[0].2);

    // Verify consistency
    let consistent = results
        .windows(2)
        .all(|w| (w[0].1 - w[1].1).abs() < 1e-6 && w[0].2 == w[1].2);
    if consistent {
        println!("  ✓ All tensors use identical quantization parameters");
    }

    println!("\n✓ Batch processing complete!");
    Ok(())
}
