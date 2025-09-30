//! Example script to run performance comparisons
//!
//! Usage: cargo run --example run_comparisons --release

use torsh_benches::comparisons::benchmark_and_analyze;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting ToRSh performance comparison benchmarks...");

    // Run comprehensive benchmarks and analysis
    benchmark_and_analyze()?;

    println!("\nBenchmarks completed successfully!");
    println!("Check target/comparison_report.md for detailed results");
    println!("Check target/performance_analysis.md for performance analysis");

    Ok(())
}
