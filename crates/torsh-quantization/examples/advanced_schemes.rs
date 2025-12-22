//! Advanced Quantization Schemes
//!
//! Demonstrates INT8, INT4, Binary, and Ternary quantization

use torsh_quantization::{compare_quantization_configs, QuantConfig};
use torsh_tensor::creation::tensor_1d;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Quantization Schemes ===\n");

    // Create test data
    let data: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
    let tensor = tensor_1d(&data)?;
    println!("Test data: {} values\n", data.len());

    // Compare multiple quantization schemes
    let configs = vec![
        QuantConfig::int8(),
        QuantConfig::int4(),
        QuantConfig::binary(),
        QuantConfig::ternary(),
    ];

    println!("Comparing quantization schemes...\n");
    let comparison = compare_quantization_configs(&tensor, &configs)?;

    println!(
        "{:<15} {:<10} {:<12} {:<10}",
        "Scheme", "PSNR (dB)", "Compression", "Speed (ms)"
    );
    println!("{}", "-".repeat(50));
    for (config, metrics, time_ms) in &comparison {
        let scheme = format!("{:?}", config.scheme);
        println!(
            "{:<15} {:<10.2} {:<12.2}x {:<10.2}",
            scheme, metrics.psnr, metrics.compression_ratio, time_ms
        );
    }

    println!("\n✓ Scheme comparison complete!");
    Ok(())
}
