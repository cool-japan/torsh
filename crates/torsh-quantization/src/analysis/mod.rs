//! Analysis tools for quantization
//!
//! This module provides comprehensive tools for analyzing the impact of quantization on models,
//! including sensitivity analysis, accuracy comparison, performance metrics, and visualization.
//! The modular architecture separates concerns while maintaining a clean, unified interface.

// Module declarations
pub mod benchmarking;
pub mod config;
pub mod sensitivity;
pub mod size;
pub mod speed;
pub mod statistical;
pub mod visualization;

// Re-export main types for convenience
pub use config::{
    AccuracyComparison, AnalysisConfig, EfficiencyWeights, LayerSensitivityResult,
    NormalizationFactors, SensitivityAnalysisResults,
};

pub use sensitivity::SensitivityAnalyzer;

pub use size::{SizeAnalyzer, SizeReport};

pub use speed::SpeedAnalyzer;

pub use statistical::{
    AdvancedStatisticalAnalyzer, ComprehensiveStatisticalReport, RiskLevel, StatisticalSignificance,
};

pub use visualization::VisualizationTool;

pub use benchmarking::{
    BenchmarkConfig, BenchmarkResult, OptimizationCriteria, QuantizationBenchmarker,
};

// Convenience functions for common analysis tasks

/// Perform quick sensitivity analysis for a list of layer names
pub fn quick_sensitivity_analysis(
    layer_names: &[String],
) -> crate::TorshResult<SensitivityAnalysisResults> {
    let analyzer = SensitivityAnalyzer::new(Vec::new()); // Empty test data for heuristic analysis
    analyzer.heuristic_sensitivity_analysis(layer_names)
}

/// Generate size comparison report for multiple schemes
pub fn compare_model_sizes(
    num_parameters: usize,
    schemes: &[crate::QScheme],
) -> std::collections::HashMap<crate::QScheme, SizeReport> {
    SizeAnalyzer::generate_size_report(num_parameters, schemes)
}

/// Create a comprehensive analysis report
pub fn generate_comprehensive_analysis_report(
    num_parameters: usize,
    layer_names: &[String],
    baseline_accuracy: f32,
) -> crate::TorshResult<String> {
    let mut report = String::new();

    // Header
    report.push_str("Comprehensive Quantization Analysis Report\n");
    report.push_str(&"=".repeat(60));
    report.push_str("\n\n");

    // Model statistics
    report.push_str("Model Statistics:\n");
    report.push_str(&format!("- Parameters: {}\n", num_parameters));
    report.push_str(&format!("- Layers: {}\n", layer_names.len()));
    report.push_str(&format!(
        "- Baseline Accuracy: {:.4}\n\n",
        baseline_accuracy
    ));

    // Sensitivity analysis
    report.push_str("Sensitivity Analysis:\n");
    report.push_str(&"-".repeat(30));
    report.push('\n');

    let sensitivity_results = quick_sensitivity_analysis(layer_names)?;
    report.push_str(&sensitivity_results.summary_report());
    report.push_str("\n\n");

    // Size analysis
    report.push_str("Size Analysis:\n");
    report.push_str(&"-".repeat(30));
    report.push('\n');

    let schemes = vec![
        crate::QScheme::PerTensorAffine,
        crate::QScheme::PerChannelAffine,
        crate::QScheme::Int4PerTensor,
        crate::QScheme::Binary,
    ];

    let size_reports = compare_model_sizes(num_parameters, &schemes);
    for (scheme, report_data) in size_reports {
        report.push_str(&format!(
            "{:?}: {:.1}MB -> {:.1}MB ({:.1}x reduction, {:.1}% saved)\n",
            scheme,
            report_data.original_size_mb,
            report_data.quantized_size_mb,
            report_data.reduction_ratio,
            report_data.space_savings_percentage()
        ));
    }
    report.push_str("\n");

    // Performance estimates
    report.push_str("Performance Estimates:\n");
    report.push_str(&"-".repeat(30));
    report.push('\n');

    for scheme in schemes {
        let speed_improvement = SpeedAnalyzer::estimate_speed_improvement(scheme);
        report.push_str(&format!(
            "{:?}: {:.1}x speed improvement\n",
            scheme, speed_improvement
        ));
    }

    Ok(report)
}

/// Create an analysis configuration optimized for production deployment
pub fn production_analysis_config() -> AnalysisConfig {
    AnalysisConfig::conservative()
}

/// Create an analysis configuration optimized for research/experimentation
pub fn research_analysis_config() -> AnalysisConfig {
    AnalysisConfig::aggressive()
}

/// Batch analyze multiple models
pub fn batch_analyze_models(
    model_configs: &[(usize, Vec<String>, f32)], // (num_parameters, layer_names, baseline_accuracy)
) -> crate::TorshResult<Vec<String>> {
    let mut reports = Vec::new();

    for (i, (num_parameters, layer_names, baseline_accuracy)) in model_configs.iter().enumerate() {
        let mut report = format!("Model {} Analysis:\n", i + 1);
        report.push_str(&"=".repeat(40));
        report.push('\n');

        let detailed_report = generate_comprehensive_analysis_report(
            *num_parameters,
            layer_names,
            *baseline_accuracy,
        )?;

        report.push_str(&detailed_report);
        report.push_str("\n\n");

        reports.push(report);
    }

    Ok(reports)
}
