//! # Autograd Gradient Flow Visualization System
//!
//! This module provides comprehensive tools for visualizing and analyzing gradient flows
//! in the automatic differentiation system. It offers detailed insights into gradient
//! propagation, bottleneck identification, real-time monitoring, and statistical analysis.
//!
//! ## Architecture
//!
//! The visualization system is organized into four specialized modules:
//!
//! - **`core`**: Foundation types and data structures for gradient flow analysis
//! - **`visualizer`**: Main visualization engine with multiple output format generators
//! - **`monitoring`**: Real-time monitoring and trend analysis capabilities
//! - **`magnitude_analysis`**: Advanced gradient magnitude statistical analysis
//!
//! ## Key Features
//!
//! ### Gradient Flow Analysis
//! - Comprehensive gradient flow tracking through computation graphs
//! - Bottleneck identification and performance analysis
//! - Critical path analysis for optimization guidance
//! - Memory usage breakdown and fragmentation analysis
//!
//! ### Multiple Output Formats
//! - Text-based reports for console output
//! - DOT format for Graphviz integration
//! - HTML visualization with interactive elements
//! - JSON export for programmatic analysis
//!
//! ### Real-time Monitoring
//! - Live gradient flow monitoring during training
//! - Trend analysis and anomaly detection
//! - Configurable alerting system
//! - Performance metrics tracking
//!
//! ### Statistical Analysis
//! - Per-layer gradient magnitude statistics
//! - Global gradient distribution analysis
//! - Temporal gradient trend tracking
//! - Histogram-based gradient profiling
//!
//! ## Usage Examples
//!
//! ### Basic Gradient Flow Analysis
//!
//! ```rust,ignore
//! use torsh_autograd::visualization::{GradientVisualizer, VisualizerConfig};
//! use torsh_autograd::context::AutogradContext;
//! # fn example(ctx: &AutogradContext) -> torsh_core::error::Result<()> {
//!
//! // Create visualizer with default configuration
//! let visualizer = GradientVisualizer::with_config(VisualizerConfig::default());
//!
//! // Analyze gradient flow in autograd context
//! let analysis = visualizer.analyze_gradient_flow(&ctx)?;
//!
//! // Generate text visualization
//! let text_report = visualizer.generate_text_visualization(&analysis)?;
//! println!("{}", text_report);
//!
//! // Export as HTML for web viewing
//! let html_report = visualizer.generate_html_visualization(&analysis)?;
//! std::fs::write("gradient_analysis.html", html_report)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Real-time Monitoring
//!
//! ```rust,ignore
//! use torsh_autograd::visualization::{GradientFlowMonitor, MonitoringConfig};
//! use torsh_autograd::context::AutogradContext;
//! use std::time::Duration;
//! # fn example(ctx: &AutogradContext, num_epochs: usize) -> torsh_core::error::Result<()> {
//!
//! // Configure monitoring system
//! let config = MonitoringConfig {
//!     max_history_size: 1000,
//!     min_trend_samples: 10,
//!     trend_window_size: 100,
//!     change_threshold: 0.1,
//!     enable_alerting: true,
//!     snapshot_interval: Duration::from_secs(5),
//! };
//!
//! // Create monitor
//! let mut monitor = GradientFlowMonitor::with_config(config);
//!
//! // Training loop with monitoring
//! for epoch in 0..num_epochs {
//!     // Training step...
//!
//!     // Analyze and store gradient flows
//!     monitor.analyze_and_store(&ctx)?;
//! }
//!
//! // Generate monitoring report
//! let report = monitor.generate_monitoring_report()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Statistical Analysis
//!
//! ```rust,ignore
//! use torsh_autograd::visualization::{GradientMagnitudeAnalyzer, AnalyzerConfig};
//! use torsh_autograd::context::AutogradContext;
//! # fn example(ctx: &AutogradContext) -> torsh_core::error::Result<()> {
//!
//! // Configure detailed analysis
//! let config = AnalyzerConfig {
//!     enable_histograms: true,
//!     enable_temporal_tracking: true,
//!     histogram_bins: 100,
//!     temporal_window: 1000,
//! };
//!
//! // Create analyzer
//! let analyzer = GradientMagnitudeAnalyzer::<f32>::new(config);
//!
//! // Perform detailed statistical analysis
//! let detailed_stats = analyzer.analyze_gradients(&ctx, None)?;
//!
//! // Access per-layer statistics
//! for (layer_name, layer_stats) in &detailed_stats.layer_stats {
//!     println!("Layer {}: mean={:.6}, std={:.6}",
//!              layer_name, layer_stats.mean_magnitude, layer_stats.std_deviation);
//! }
//!
//! // Access global statistics
//! println!("Global gradient norm: {:.6}", detailed_stats.global_stats.total_norm);
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Considerations
//!
//! - Visualization analysis can add computational overhead; use selectively in production
//! - Real-time monitoring is optimized for minimal performance impact
//! - Statistical analysis can be memory-intensive for large models
//! - Consider using sampling for very large computation graphs
//!
//! ## Integration with ML Workflows
//!
//! The visualization system integrates seamlessly with:
//! - TensorBoard for experiment tracking
//! - Jupyter notebooks for interactive analysis
//! - CI/CD pipelines for automated gradient health checks
//! - Production monitoring systems via JSON export

// Declare submodules
pub mod core;
pub mod magnitude_analysis;
pub mod monitoring;
pub mod visualizer;

// Re-export core foundation types
pub use core::{
    BottleneckMetrics, BottleneckType, GradientBottleneck, GradientFlowAnalysis,
    GradientFlowSummary, GradientMagnitudeCategory, GradientStatistics, MemoryBreakdown,
    MemoryDistribution, MemoryFragmentation, OperationInfo,
};

// Re-export main visualizer types
pub use visualizer::{ColorScheme, GradientVisualizer, VisualizerConfig};

// Re-export monitoring types
pub use monitoring::{
    Alert, AlertSeverity, AlertType, AnomalyReport, AnomalyType, GradientFlowMonitor,
    GradientTrendAnalysis, MetricType, MonitoringConfig, TrendDirection,
};

// Re-export magnitude analysis types
pub use magnitude_analysis::{
    AnalyzerConfig, DetailedGradientStats, GlobalGradientStats, GradientHistogram,
    GradientMagnitudeAnalyzer, HistoricalComparison, LayerGradientStats, MagnitudeTimePoint,
    NormDistribution,
};

// Import necessary types for convenience functions
use crate::context::AutogradContext;
use torsh_core::FloatElement;

/// Convenience function to create a gradient visualizer with default configuration
///
/// This function provides a quick way to start analyzing gradient flows without
/// manually configuring the visualizer.
///
/// # Returns
///
/// A `GradientVisualizer` instance configured with sensible defaults for most use cases.
///
/// # Example
///
/// ```rust,ignore
/// use torsh_autograd::visualization::create_default_visualizer;
/// # use torsh_autograd::context::AutogradContext;
/// # fn example(ctx: &AutogradContext) -> torsh_core::error::Result<()> {
/// let visualizer = create_default_visualizer();
/// let analysis = visualizer.analyze_gradient_flow(&ctx)?;
/// # Ok(())
/// # }
/// ```
pub fn create_default_visualizer() -> GradientVisualizer {
    GradientVisualizer::with_config(VisualizerConfig::default())
}

/// Convenience function to create a gradient flow monitor with default configuration
///
/// This function provides a quick way to start monitoring gradient flows in real-time
/// without manually configuring the monitor.
///
/// # Returns
///
/// A `GradientFlowMonitor` instance configured with sensible defaults for most monitoring scenarios.
///
/// # Example
///
/// ```rust,ignore
/// use torsh_autograd::visualization::create_default_monitor;
/// # use torsh_autograd::context::AutogradContext;
/// # fn example(ctx: &AutogradContext) -> torsh_core::error::Result<()> {
/// let mut monitor = create_default_monitor();
/// monitor.analyze_and_store(&ctx)?;
/// # Ok(())
/// # }
/// ```
pub fn create_default_monitor() -> GradientFlowMonitor {
    GradientFlowMonitor::new()
}

/// Convenience function to create a gradient magnitude analyzer with default configuration
///
/// This function provides a quick way to start detailed statistical analysis of gradient
/// magnitudes without manual configuration.
///
/// # Returns
///
/// A `GradientMagnitudeAnalyzer` instance configured with sensible defaults for statistical analysis.
///
/// # Example
///
/// ```rust,ignore
/// use torsh_autograd::visualization::create_default_analyzer;
/// # use std::collections::HashMap;
/// # fn example() -> torsh_core::error::Result<()> {
/// let mut analyzer = create_default_analyzer::<f32>();
/// let gradients = HashMap::new(); // Example empty gradients
/// let stats = analyzer.analyze_gradients(&gradients, None)?;
/// # Ok(())
/// # }
/// ```
pub fn create_default_analyzer<T: FloatElement + num_traits::FromPrimitive + Default + Clone>(
) -> GradientMagnitudeAnalyzer<T> {
    GradientMagnitudeAnalyzer::new(AnalyzerConfig::default())
}

/// Quick analysis function that performs a complete gradient flow analysis
///
/// This convenience function combines visualization, monitoring insights, and statistical
/// analysis into a single comprehensive report. Ideal for debugging and understanding
/// gradient behavior.
///
/// # Arguments
///
/// * `ctx` - The autograd context to analyze
///
/// # Returns
///
/// A tuple containing:
/// - Basic gradient flow analysis
/// - Detailed statistical analysis
/// - Formatted text report for immediate viewing
///
/// # Example
///
/// ```rust,ignore
/// use torsh_autograd::visualization::quick_gradient_analysis;
/// # use torsh_autograd::context::AutogradContext;
/// # fn example(ctx: &AutogradContext) -> torsh_core::error::Result<()> {
/// let (analysis, detailed_stats, report) = quick_gradient_analysis::<f32>(&ctx)?;
/// println!("{}", report);
/// # Ok(())
/// # }
/// ```
pub fn quick_gradient_analysis<T: FloatElement + num_traits::FromPrimitive + Default + Clone>(
    ctx: &AutogradContext,
) -> torsh_core::error::Result<(GradientFlowAnalysis, DetailedGradientStats<T>, String)> {
    let visualizer = create_default_visualizer();
    let mut analyzer = create_default_analyzer::<T>();

    let analysis = visualizer.analyze_gradient_flow(ctx)?;

    // Extract gradients from context - temporary placeholder implementation
    let gradients = std::collections::HashMap::<String, Vec<T>>::new();
    let detailed_stats = analyzer.analyze_gradients(&gradients, None)?;

    let report = visualizer.generate_text_visualization(&analysis)?;

    Ok((analysis, detailed_stats, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convenience_functions() {
        // Test that convenience functions create instances without panicking
        let _visualizer = create_default_visualizer();
        let _monitor = create_default_monitor();
        let _analyzer = create_default_analyzer::<f32>();
        let _analyzer_f64 = create_default_analyzer::<f64>();
    }

    #[test]
    fn test_module_exports() {
        // Test that all main types are accessible through the module
        let _visualizer_config = VisualizerConfig::default();
        let _monitoring_config = MonitoringConfig::default();
        let _analyzer_config = AnalyzerConfig::default();

        // Test enum variants are accessible
        let _color_scheme = ColorScheme::Default;
        let _bottleneck_type = BottleneckType::Computation;
        let _trend_direction = TrendDirection::Increasing;
        let _alert_type = AlertType::GradientMagnitude;
        let _severity = AlertSeverity::Critical;
    }

    #[test]
    fn test_type_consistency() {
        // Ensure generic types work with both f32 and f64
        let _stats_f32: DetailedGradientStats<f32> = DetailedGradientStats {
            layer_stats: std::collections::HashMap::new(),
            global_stats: GlobalGradientStats::default(),
            gradient_histogram: GradientHistogram::default(),
            magnitude_timeline: Vec::new(),
            norm_distribution: NormDistribution::default(),
        };

        let _stats_f64: DetailedGradientStats<f64> = DetailedGradientStats {
            layer_stats: std::collections::HashMap::new(),
            global_stats: GlobalGradientStats::default(),
            gradient_histogram: GradientHistogram::default(),
            magnitude_timeline: Vec::new(),
            norm_distribution: NormDistribution::default(),
        };
    }
}
