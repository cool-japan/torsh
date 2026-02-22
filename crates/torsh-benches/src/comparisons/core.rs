//! Core infrastructure for benchmark comparisons
//!
//! This module provides the fundamental types and utilities for comparing
//! ToRSh performance against other tensor libraries.

use crate::{BenchResult, Benchmarkable};
use std::hint::black_box;
use std::collections::HashMap;

/// Comparison benchmark runner
pub struct ComparisonRunner {
    results: Vec<ComparisonResult>,
}

impl ComparisonRunner {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add comparison results
    pub fn add_result(&mut self, result: ComparisonResult) {
        self.results.push(result);
    }

    /// Get all comparison results
    pub fn results(&self) -> &[ComparisonResult] {
        &self.results
    }

    /// Generate comparison report
    pub fn generate_report(&self, output_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(output_path)?;

        writeln!(file, "# ToRSh Performance Comparison Report\n")?;

        // Group results by operation
        let mut grouped: HashMap<String, Vec<&ComparisonResult>> = HashMap::new();

        for result in &self.results {
            grouped
                .entry(result.operation.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (operation, results) in grouped {
            writeln!(file, "## {}\n", operation)?;
            writeln!(file, "| Library | Size | Time (μs) | Speedup vs ToRSh |")?;
            writeln!(file, "|---------|------|-----------|------------------|")?;

            for result in &results {
                let speedup = if result.library == "torsh" {
                    1.0
                } else {
                    // Find corresponding ToRSh result
                    if let Some(torsh_result) = results
                        .iter()
                        .find(|r| r.library == "torsh" && r.size == result.size)
                    {
                        torsh_result.time_ns / result.time_ns
                    } else {
                        1.0
                    }
                };

                writeln!(
                    file,
                    "| {} | {} | {:.2} | {:.2}x |",
                    result.library,
                    result.size,
                    result.time_ns / 1000.0,
                    speedup
                )?;
            }
            writeln!(file)?;
        }

        Ok(())
    }
}

impl Default for ComparisonRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComparisonResult {
    pub operation: String,
    pub library: String,
    pub size: usize,
    pub time_ns: f64,
    pub throughput: Option<f64>,
    pub memory_usage: Option<usize>,
}

/// Legacy function for backward compatibility
pub fn benchmark_and_compare() -> std::io::Result<()> {
    crate::comparisons::integration::benchmark_and_analyze()
}