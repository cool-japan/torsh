//! Basic Quantization Example
//!
//! Demonstrates fundamental quantization workflow

use torsh_quantization::{
    calculate_quantization_metrics, dequantize, quantize_with_config, QuantConfig,
};
use torsh_tensor::creation::tensor_1d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Quantization Example ===\n");

    // Create a sample tensor
    let data = vec![-10.5, -5.2, -2.1, 0.0, 1.3, 3.7, 5.9, 8.4, 12.6, 15.8];
    let tensor = tensor_1d(&data)?;
    println!("Original data ({} values): {:?}\n", data.len(), data);

    // Quantize using INT8 configuration
    let config = QuantConfig::int8();
    println!("Quantizing with INT8 scheme...");
    let (quantized, scale, zero_point) = quantize_with_config(&tensor, &config)?;
    println!("  Scale: {:.6}, Zero point: {}\n", scale, zero_point);

    // Dequantize back to floating point
    let dequantized = dequantize(&quantized, scale, zero_point)?;

    // Calculate metrics
    let metrics = calculate_quantization_metrics(&tensor, &dequantized, 32, 8)?;
    println!("Quality Metrics:");
    println!("  PSNR: {:.2} dB", metrics.psnr);
    println!("  Compression: {:.2}x", metrics.compression_ratio);
    println!("  MAE: {:.6}\n", metrics.mae);

    println!("✓ Quantization complete!");
    Ok(())
}
