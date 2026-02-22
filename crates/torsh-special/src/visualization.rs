//! Visualization tools for function behavior and accuracy analysis
//!
//! This module provides utilities for analyzing and visualizing the behavior
//! of special functions, including accuracy comparisons, convergence analysis,
//! and numerical stability assessment.

use crate::error_functions;
use crate::fast_approximations;
use crate::gamma;
use std::collections::HashMap;
use torsh_core::{device::DeviceType, error::Result as TorshResult};
use torsh_tensor::Tensor;

/// Function behavior analysis results
#[derive(Debug, Clone)]
pub struct FunctionAnalysis {
    /// Function name
    pub name: String,
    /// Input range analyzed
    pub input_range: (f32, f32),
    /// Number of sample points
    pub num_points: usize,
    /// Maximum absolute value in the range
    pub max_value: f32,
    /// Minimum absolute value in the range
    pub min_value: f32,
    /// Points where function has discontinuities or singularities
    pub singularities: Vec<f32>,
    /// Estimated numerical accuracy (relative error)
    pub numerical_accuracy: f32,
    /// Function monotonicity in the range
    pub monotonicity: Monotonicity,
}

/// Monotonicity classification
#[derive(Debug, Clone, PartialEq)]
pub enum Monotonicity {
    Increasing,
    Decreasing,
    NonMonotonic,
    Constant,
}

/// Accuracy comparison between two function implementations
#[derive(Debug, Clone)]
pub struct AccuracyComparison {
    /// Reference function name
    pub reference_name: String,
    /// Test function name  
    pub test_name: String,
    /// Maximum relative error
    pub max_relative_error: f32,
    /// Average relative error
    pub avg_relative_error: f32,
    /// Root mean square error
    pub rms_error: f32,
    /// Points with largest errors
    pub worst_points: Vec<(f32, f32, f32)>, // (input, error, relative_error)
}

/// Generate ASCII plot data for function visualization
#[derive(Debug, Clone)]
pub struct PlotData {
    /// X values
    pub x_values: Vec<f32>,
    /// Y values
    pub y_values: Vec<f32>,
    /// Plot width in characters
    pub width: usize,
    /// Plot height in characters
    pub height: usize,
    /// ASCII representation
    pub ascii_plot: String,
}

/// Analyze the behavior of a special function over a given range
pub fn analyze_function_behavior<F>(
    name: &str,
    func: F,
    range: (f32, f32),
    num_points: usize,
) -> TorshResult<FunctionAnalysis>
where
    F: Fn(&Tensor<f32>) -> TorshResult<Tensor<f32>>,
{
    let device = DeviceType::Cpu;
    let (start, end) = range;

    // Generate input points
    let step = (end - start) / (num_points - 1) as f32;
    let x_values: Vec<f32> = (0..num_points).map(|i| start + i as f32 * step).collect();
    let x_tensor = Tensor::from_data(x_values.clone(), vec![num_points], device)?;

    // Evaluate function
    let result = func(&x_tensor)?;
    let y_values = result.data()?;

    // Analyze properties
    let max_value = y_values
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
    let min_value = y_values.iter().fold(f32::INFINITY, |a, &b| a.min(b.abs()));

    // Detect singularities (large jumps or infinite values)
    let mut singularities = Vec::new();
    for i in 1..y_values.len() {
        let jump = (y_values[i] - y_values[i - 1]).abs();
        if jump > 100.0 || !y_values[i].is_finite() {
            singularities.push(x_values[i]);
        }
    }

    // Assess monotonicity
    let monotonicity = assess_monotonicity(&y_values);

    // Estimate numerical accuracy (using finite differences)
    let numerical_accuracy = estimate_numerical_accuracy(&x_values, &y_values);

    Ok(FunctionAnalysis {
        name: name.to_string(),
        input_range: range,
        num_points,
        max_value,
        min_value,
        singularities,
        numerical_accuracy,
        monotonicity,
    })
}

/// Compare accuracy between two function implementations
pub fn compare_function_accuracy<F1, F2>(
    reference_name: &str,
    reference_func: F1,
    test_name: &str,
    test_func: F2,
    range: (f32, f32),
    num_points: usize,
) -> TorshResult<AccuracyComparison>
where
    F1: Fn(&Tensor<f32>) -> TorshResult<Tensor<f32>>,
    F2: Fn(&Tensor<f32>) -> TorshResult<Tensor<f32>>,
{
    let device = DeviceType::Cpu;
    let (start, end) = range;

    // Generate input points
    let step = (end - start) / (num_points - 1) as f32;
    let x_values: Vec<f32> = (0..num_points).map(|i| start + i as f32 * step).collect();
    let x_tensor = Tensor::from_data(x_values.clone(), vec![num_points], device)?;

    // Evaluate both functions
    let ref_result = reference_func(&x_tensor)?;
    let test_result = test_func(&x_tensor)?;

    let ref_values = ref_result.data()?;
    let test_values = test_result.data()?;

    // Calculate errors
    let mut errors = Vec::new();
    let mut relative_errors = Vec::new();
    let mut worst_points: Vec<(f32, f32, f32)> = Vec::new();

    for i in 0..num_points {
        if ref_values[i].is_finite() && test_values[i].is_finite() && ref_values[i] != 0.0 {
            let error = (test_values[i] - ref_values[i]).abs();
            let rel_error = error / ref_values[i].abs();

            errors.push(error);
            relative_errors.push(rel_error);

            // Track worst points
            if worst_points.len() < 5 || rel_error > worst_points[4].2 {
                worst_points.push((x_values[i], error, rel_error));
                worst_points.sort_by(|a, b| {
                    b.2.partial_cmp(&a.2)
                        .expect("relative error comparison should succeed for finite floats")
                });
                worst_points.truncate(5);
            }
        }
    }

    let max_relative_error = relative_errors.iter().fold(0.0f32, |a, &b| a.max(b));
    let avg_relative_error = relative_errors.iter().sum::<f32>() / relative_errors.len() as f32;
    let rms_error = (errors.iter().map(|&x| x * x).sum::<f32>() / errors.len() as f32).sqrt();

    Ok(AccuracyComparison {
        reference_name: reference_name.to_string(),
        test_name: test_name.to_string(),
        max_relative_error,
        avg_relative_error,
        rms_error,
        worst_points,
    })
}

/// Generate ASCII plot of function behavior
pub fn generate_ascii_plot<F>(
    func: F,
    range: (f32, f32),
    num_points: usize,
    width: usize,
    height: usize,
) -> TorshResult<PlotData>
where
    F: Fn(&Tensor<f32>) -> TorshResult<Tensor<f32>>,
{
    let device = DeviceType::Cpu;
    let (start, end) = range;

    // Generate input points
    let step = (end - start) / (num_points - 1) as f32;
    let x_values: Vec<f32> = (0..num_points).map(|i| start + i as f32 * step).collect();
    let x_tensor = Tensor::from_data(x_values.clone(), vec![num_points], device)?;

    // Evaluate function
    let result = func(&x_tensor)?;
    let y_values = result.data()?;

    // Find y-range
    let y_min = y_values.iter().fold(
        f32::INFINITY,
        |a, &b| {
            if b.is_finite() {
                a.min(b)
            } else {
                a
            }
        },
    );
    let y_max = y_values.iter().fold(
        f32::NEG_INFINITY,
        |a, &b| {
            if b.is_finite() {
                a.max(b)
            } else {
                a
            }
        },
    );

    // Create ASCII plot
    let mut plot = vec![vec![' '; width]; height];

    // Plot axes
    for row in plot.iter_mut().take(height) {
        row[0] = '|'; // Y-axis
    }
    for j in 0..width {
        plot[height - 1][j] = '-'; // X-axis
    }
    plot[height - 1][0] = '+'; // Origin

    // Plot function points
    for i in 0..num_points {
        if y_values[i].is_finite() {
            let x_pos = ((x_values[i] - start) / (end - start) * (width - 1) as f32) as usize;
            let y_pos = ((y_max - y_values[i]) / (y_max - y_min) * (height - 1) as f32) as usize;

            if x_pos < width && y_pos < height {
                plot[y_pos][x_pos] = '*';
            }
        }
    }

    // Convert to string
    let ascii_plot = plot
        .iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(PlotData {
        x_values,
        y_values: y_values.to_vec(),
        width,
        height,
        ascii_plot,
    })
}

/// Benchmark function performance across optimization levels
pub fn benchmark_optimization_levels(
    range: (f32, f32),
    num_points: usize,
    iterations: usize,
) -> TorshResult<HashMap<String, f64>> {
    use std::time::Instant;

    let device = DeviceType::Cpu;
    let (start, end) = range;

    // Generate input data
    let step = (end - start) / (num_points - 1) as f32;
    let x_values: Vec<f32> = (0..num_points).map(|i| start + i as f32 * step).collect();
    let x_tensor = Tensor::from_data(x_values, vec![num_points], device)?;

    let mut results = HashMap::new();

    // Benchmark standard gamma function
    let start_time = Instant::now();
    for _ in 0..iterations {
        let _ = gamma::gamma(&x_tensor)?;
    }
    let gamma_time = start_time.elapsed().as_nanos() as f64 / iterations as f64;
    results.insert("gamma_standard".to_string(), gamma_time);

    // Benchmark fast gamma approximation
    let start_time = Instant::now();
    for _ in 0..iterations {
        let _ = fast_approximations::gamma_fast(&x_tensor)?;
    }
    let gamma_fast_time = start_time.elapsed().as_nanos() as f64 / iterations as f64;
    results.insert("gamma_fast".to_string(), gamma_fast_time);

    // Benchmark standard error function
    let start_time = Instant::now();
    for _ in 0..iterations {
        let _ = error_functions::erf(&x_tensor)?;
    }
    let erf_time = start_time.elapsed().as_nanos() as f64 / iterations as f64;
    results.insert("erf_standard".to_string(), erf_time);

    // Benchmark fast error function approximation
    let start_time = Instant::now();
    for _ in 0..iterations {
        let _ = fast_approximations::erf_fast(&x_tensor)?;
    }
    let erf_fast_time = start_time.elapsed().as_nanos() as f64 / iterations as f64;
    results.insert("erf_fast".to_string(), erf_fast_time);

    Ok(results)
}

/// Assess monotonicity of a function from sample values
fn assess_monotonicity(values: &[f32]) -> Monotonicity {
    if values.len() < 2 {
        return Monotonicity::Constant;
    }

    let mut increasing = 0;
    let mut decreasing = 0;
    let mut constant = 0;

    for i in 1..values.len() {
        if values[i].is_finite() && values[i - 1].is_finite() {
            if values[i] > values[i - 1] {
                increasing += 1;
            } else if values[i] < values[i - 1] {
                decreasing += 1;
            } else {
                constant += 1;
            }
        }
    }

    let total = increasing + decreasing + constant;
    if total == 0 {
        return Monotonicity::Constant;
    }

    let inc_ratio = increasing as f32 / total as f32;
    let dec_ratio = decreasing as f32 / total as f32;

    if inc_ratio > 0.9 {
        Monotonicity::Increasing
    } else if dec_ratio > 0.9 {
        Monotonicity::Decreasing
    } else if inc_ratio < 0.1 && dec_ratio < 0.1 {
        Monotonicity::Constant
    } else {
        Monotonicity::NonMonotonic
    }
}

/// Estimate numerical accuracy using finite differences
fn estimate_numerical_accuracy(x_values: &[f32], y_values: &[f32]) -> f32 {
    if x_values.len() < 3 {
        return 1e-6; // Default assumption
    }

    let mut max_curvature = 0.0f32;

    for i in 1..x_values.len() - 1 {
        if y_values[i - 1].is_finite() && y_values[i].is_finite() && y_values[i + 1].is_finite() {
            let h1 = x_values[i] - x_values[i - 1];
            let h2 = x_values[i + 1] - x_values[i];

            if h1 > 0.0 && h2 > 0.0 {
                // Second derivative approximation
                let d2y =
                    (y_values[i + 1] - y_values[i]) / h2 - (y_values[i] - y_values[i - 1]) / h1;
                let curvature = d2y.abs() / (h1 + h2);
                max_curvature = max_curvature.max(curvature);
            }
        }
    }

    // Estimate accuracy based on curvature and floating-point precision
    let machine_eps = f32::EPSILON;
    let estimated_error = machine_eps * (1.0 + max_curvature);

    estimated_error.min(1e-3).max(machine_eps)
}

/// Print comprehensive function analysis report
pub fn print_analysis_report(analysis: &FunctionAnalysis) {
    println!("═══ Function Analysis Report ═══");
    println!("Function: {}", analysis.name);
    println!(
        "Range: [{:.3}, {:.3}]",
        analysis.input_range.0, analysis.input_range.1
    );
    println!("Sample points: {}", analysis.num_points);
    println!(
        "Value range: [{:.6}, {:.6}]",
        analysis.min_value, analysis.max_value
    );
    println!("Monotonicity: {:?}", analysis.monotonicity);
    println!("Numerical accuracy: {:.2e}", analysis.numerical_accuracy);

    if !analysis.singularities.is_empty() {
        println!("Singularities detected at: {:?}", analysis.singularities);
    } else {
        println!("No singularities detected");
    }

    println!("═══════════════════════════════");
}

/// Print accuracy comparison report
pub fn print_accuracy_report(comparison: &AccuracyComparison) {
    println!("═══ Accuracy Comparison Report ═══");
    println!("Reference: {}", comparison.reference_name);
    println!("Test function: {}", comparison.test_name);
    println!("Max relative error: {:.2e}", comparison.max_relative_error);
    println!(
        "Average relative error: {:.2e}",
        comparison.avg_relative_error
    );
    println!("RMS error: {:.2e}", comparison.rms_error);

    println!("\nWorst accuracy points:");
    for (i, &(x, err, rel_err)) in comparison.worst_points.iter().enumerate() {
        println!(
            "  {}: x={:.4}, error={:.2e}, rel_error={:.2e}",
            i + 1,
            x,
            err,
            rel_err
        );
    }

    println!("═══════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_analysis() -> TorshResult<()> {
        let analysis = analyze_function_behavior("gamma", gamma::gamma, (0.1, 3.0), 50)?;

        assert_eq!(analysis.name, "gamma");
        assert!(analysis.max_value > 0.0);
        assert!(analysis.numerical_accuracy > 0.0);

        Ok(())
    }

    #[test]
    fn test_accuracy_comparison() -> TorshResult<()> {
        let comparison = compare_function_accuracy(
            "gamma_standard",
            gamma::gamma,
            "gamma_fast",
            fast_approximations::gamma_fast,
            (0.5, 2.0),
            20,
        )?;

        assert!(comparison.max_relative_error >= 0.0);
        assert!(comparison.avg_relative_error >= 0.0);
        assert!(comparison.rms_error >= 0.0);

        Ok(())
    }

    #[test]
    fn test_ascii_plot() -> TorshResult<()> {
        let plot = generate_ascii_plot(gamma::gamma, (0.5, 2.0), 20, 40, 20)?;

        assert_eq!(plot.width, 40);
        assert_eq!(plot.height, 20);
        assert!(!plot.ascii_plot.is_empty());
        assert!(plot.ascii_plot.contains('*')); // Should have plot points
        assert!(plot.ascii_plot.contains('|')); // Should have axes

        Ok(())
    }

    #[test]
    fn test_monotonicity_assessment() {
        assert_eq!(
            assess_monotonicity(&[1.0, 2.0, 3.0, 4.0]),
            Monotonicity::Increasing
        );
        assert_eq!(
            assess_monotonicity(&[4.0, 3.0, 2.0, 1.0]),
            Monotonicity::Decreasing
        );
        assert_eq!(
            assess_monotonicity(&[2.0, 2.0, 2.0, 2.0]),
            Monotonicity::Constant
        );
        assert_eq!(
            assess_monotonicity(&[1.0, 3.0, 2.0, 4.0]),
            Monotonicity::NonMonotonic
        );
    }

    #[test]
    fn test_benchmark() -> TorshResult<()> {
        let results = benchmark_optimization_levels((0.5, 2.0), 100, 5)?;

        assert!(results.contains_key("gamma_standard"));
        assert!(results.contains_key("gamma_fast"));
        assert!(results.contains_key("erf_standard"));
        assert!(results.contains_key("erf_fast"));

        // Verify all benchmarks returned positive timing values
        assert!(results["gamma_standard"] > 0.0);
        assert!(results["gamma_fast"] > 0.0);
        assert!(results["erf_standard"] > 0.0);
        assert!(results["erf_fast"] > 0.0);

        // Fast functions should generally be faster, but allow large tolerance for system variability
        // Allow up to 10x slowdown to account for system load, cold cache, etc.
        assert!(results["gamma_fast"] <= results["gamma_standard"] * 10.0);
        assert!(results["erf_fast"] <= results["erf_standard"] * 10.0);

        Ok(())
    }
}
