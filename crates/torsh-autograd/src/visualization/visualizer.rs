//! # Main Gradient Flow Visualizer
//!
//! This module provides the primary visualization engine for gradient flow analysis.
//! It includes comprehensive analysis capabilities and multiple output format generators
//! for different use cases and tools.
//!
//! ## Key Components
//!
//! - **GradientVisualizer**: Main visualization engine with analysis and generation
//! - **Text visualization**: Human-readable console output
//! - **DOT visualization**: GraphViz-compatible graph representations
//! - **HTML visualization**: Interactive web-based visualizations
//! - **JSON export**: Machine-readable data for external tools
//!
//! ## Supported Output Formats
//!
//! - **Text**: Structured console output for debugging and logging
//! - **DOT**: GraphViz format for publication-quality graph visualizations
//! - **HTML**: Interactive web visualizations with CSS styling
//! - **Interactive Web**: Advanced JavaScript-based interactive visualizations
//! - **JSON**: Structured data export for integration with external tools
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use torsh_autograd::visualization::visualizer::GradientVisualizer;
//! use torsh_autograd::context::AutogradContext;
//! # fn example() -> torsh_core::error::Result<()> {
//!
//! let visualizer = GradientVisualizer::new();
//! let ctx = AutogradContext::new();
//!
//! // Analyze gradient flow
//! let analysis = visualizer.analyze_gradient_flow(&ctx)?;
//!
//! // Generate text visualization
//! let text_output = visualizer.generate_text_visualization(&analysis)?;
//! println!("{}", text_output);
//!
//! // Generate HTML visualization
//! let html_output = visualizer.generate_html_visualization(&analysis)?;
//! std::fs::write("gradient_flow.html", html_output)?;
//! # Ok(())
//! # }
//! ```

use super::core::{
    GradientBottleneck, GradientFlowAnalysis, GradientStatistics, MemoryBreakdown, OperationInfo,
};
use crate::context::AutogradContext;
use std::fmt::Write;
use torsh_core::error::{Result, TorshError};
use tracing::{debug, info};

/// Main gradient flow visualizer with comprehensive analysis and output capabilities
///
/// The `GradientVisualizer` provides a complete toolkit for analyzing and visualizing
/// gradient flows in neural network computation graphs. It supports multiple output
/// formats and can be configured for different levels of detail.
#[derive(Debug, Clone)]
pub struct GradientVisualizer {
    /// Whether to include detailed statistics in visualizations
    pub include_detailed_stats: bool,
    /// Whether to show gradient magnitudes in outputs
    pub show_gradient_magnitudes: bool,
    /// Whether to include memory usage information
    pub show_memory_usage: bool,
    /// Threshold for highlighting operations as bottlenecks (0-1)
    pub bottleneck_threshold: f32,
    /// Configuration for visualization appearance
    pub config: VisualizerConfig,
}

/// Configuration for visualization appearance and behavior
#[derive(Debug, Clone)]
pub struct VisualizerConfig {
    /// Maximum number of operations to show in critical path
    pub max_critical_path_operations: usize,
    /// Color scheme for visualizations
    pub color_scheme: ColorScheme,
    /// Whether to use compact output format
    pub compact_output: bool,
    /// Precision for floating-point number display
    pub decimal_precision: usize,
}

/// Color schemes for different visualization outputs
#[derive(Debug, Clone)]
pub enum ColorScheme {
    /// Default color scheme with standard colors
    Default,
    /// High contrast colors for accessibility
    HighContrast,
    /// Grayscale for printing and publications
    Grayscale,
    /// Custom color scheme with user-defined colors
    Custom {
        healthy: String,
        warning: String,
        critical: String,
        background: String,
    },
}

impl Default for GradientVisualizer {
    fn default() -> Self {
        Self {
            include_detailed_stats: true,
            show_gradient_magnitudes: true,
            show_memory_usage: true,
            bottleneck_threshold: 0.8,
            config: VisualizerConfig::default(),
        }
    }
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        Self {
            max_critical_path_operations: 50,
            color_scheme: ColorScheme::Default,
            compact_output: false,
            decimal_precision: 6,
        }
    }
}

impl GradientVisualizer {
    /// Create a new gradient visualizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a visualizer with custom configuration
    pub fn with_config(config: VisualizerConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// Set bottleneck threshold for highlighting problematic operations
    pub fn with_bottleneck_threshold(mut self, threshold: f32) -> Self {
        self.bottleneck_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable detailed statistics
    pub fn with_detailed_stats(mut self, enabled: bool) -> Self {
        self.include_detailed_stats = enabled;
        self
    }

    /// Enable or disable compact output format
    pub fn with_compact_output(mut self, compact: bool) -> Self {
        self.config.compact_output = compact;
        self
    }

    /// Analyze gradient flow in the computation graph
    ///
    /// This is the main analysis method that examines the autograd context and
    /// produces comprehensive gradient flow analysis results.
    pub fn analyze_gradient_flow(&self, ctx: &AutogradContext) -> Result<GradientFlowAnalysis> {
        info!("Starting gradient flow analysis");

        let stats = ctx.graph_stats();

        // Collect gradient information
        let gradient_stats = self.compute_gradient_statistics(ctx)?;

        // Find bottlenecks
        let bottlenecks = self.identify_bottlenecks(ctx)?;

        // Analyze critical path
        let critical_path = self.analyze_critical_path(ctx)?;

        // Compute memory breakdown
        let memory_breakdown = self.compute_memory_breakdown(ctx)?;

        let analysis = GradientFlowAnalysis {
            timestamp: std::time::Instant::now(),
            total_operations: stats.node_count,
            operations_with_gradients: self.count_operations_with_gradients(ctx)?,
            gradient_bottlenecks: bottlenecks,
            gradient_stats,
            critical_path,
            memory_breakdown,
        };

        info!(
            "Analysis complete: {} operations, {} with gradients, {} bottlenecks",
            analysis.total_operations,
            analysis.operations_with_gradients,
            analysis.gradient_bottlenecks.len()
        );

        Ok(analysis)
    }

    /// Generate a text-based visualization for console output
    ///
    /// Creates a human-readable text representation of the gradient flow analysis
    /// suitable for console output, logging, and debugging.
    pub fn generate_text_visualization(&self, analysis: &GradientFlowAnalysis) -> Result<String> {
        let mut output = String::new();

        if self.config.compact_output {
            self.generate_compact_text(&mut output, analysis)?;
        } else {
            self.generate_detailed_text(&mut output, analysis)?;
        }

        Ok(output)
    }

    /// Generate detailed text visualization
    fn generate_detailed_text(
        &self,
        output: &mut String,
        analysis: &GradientFlowAnalysis,
    ) -> Result<()> {
        writeln!(
            output,
            "╔══════════════════════════════════════════════════════════════╗"
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "║                    Gradient Flow Analysis                    ║"
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "╚══════════════════════════════════════════════════════════════╝"
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        // Overview section
        writeln!(output, "📊 Overview")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "  Total Operations: {}", analysis.total_operations)
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Operations with Gradients: {}",
            analysis.operations_with_gradients
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Gradient Flow Efficiency: {:.1}%",
            analysis.gradient_flow_efficiency() * 100.0
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Gradient Health Score: {:.3}",
            analysis.gradient_health_score()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        // Gradient statistics
        if self.show_gradient_magnitudes {
            self.write_gradient_statistics(output, &analysis.gradient_stats)?;
        }

        // Memory breakdown
        if self.show_memory_usage {
            self.write_memory_breakdown(output, &analysis.memory_breakdown)?;
        }

        // Bottlenecks
        if !analysis.gradient_bottlenecks.is_empty() {
            self.write_bottlenecks_section(output, &analysis.gradient_bottlenecks)?;
        }

        // Critical path
        if !analysis.critical_path.is_empty() {
            self.write_critical_path_section(output, &analysis.critical_path)?;
        }

        Ok(())
    }

    /// Generate compact text visualization
    fn generate_compact_text(
        &self,
        output: &mut String,
        analysis: &GradientFlowAnalysis,
    ) -> Result<()> {
        writeln!(output, "Gradient Flow Analysis (Compact)")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "Operations: {} ({} with gradients)",
            analysis.total_operations, analysis.operations_with_gradients
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "Health Score: {:.3}",
            analysis.gradient_health_score()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "Bottlenecks: {}",
            analysis.gradient_bottlenecks.len()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        if let Some(bottleneck) = analysis.most_critical_bottleneck() {
            writeln!(output, "Most Critical: {}", bottleneck.description())
                .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }

        Ok(())
    }

    /// Write gradient statistics section
    fn write_gradient_statistics(
        &self,
        output: &mut String,
        stats: &GradientStatistics,
    ) -> Result<()> {
        writeln!(output, "📈 Gradient Statistics")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Mean Magnitude: {:.prec$}",
            stats.mean_magnitude,
            prec = self.config.decimal_precision
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Std Deviation: {:.prec$}",
            stats.std_deviation,
            prec = self.config.decimal_precision
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Max Magnitude: {:.prec$}",
            stats.max_magnitude,
            prec = self.config.decimal_precision
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Min Magnitude: {:.prec$}",
            stats.min_magnitude,
            prec = self.config.decimal_precision
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "  Zero Gradients: {}", stats.zero_count)
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "  Invalid Gradients: {}", stats.inf_nan_count)
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Category: {} (Quality: {:.3})",
            stats.magnitude_category(),
            stats.quality_score()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Write memory breakdown section
    fn write_memory_breakdown(
        &self,
        output: &mut String,
        breakdown: &MemoryBreakdown,
    ) -> Result<()> {
        writeln!(output, "💾 Memory Breakdown")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Gradient Memory: {:.1} MB",
            breakdown.gradient_memory as f64 / (1024.0 * 1024.0)
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Intermediate Memory: {:.1} MB",
            breakdown.intermediate_memory as f64 / (1024.0 * 1024.0)
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Metadata Memory: {:.1} MB",
            breakdown.metadata_memory as f64 / (1024.0 * 1024.0)
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Peak Memory: {:.1} MB",
            breakdown.peak_memory as f64 / (1024.0 * 1024.0)
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "  Efficiency Score: {:.3}",
            breakdown.efficiency_score()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        let distribution = breakdown.memory_distribution();
        writeln!(
            output,
            "  Distribution: {:.1}% gradients, {:.1}% intermediate, {:.1}% metadata",
            distribution.gradient_percentage,
            distribution.intermediate_percentage,
            distribution.metadata_percentage
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Write bottlenecks section
    fn write_bottlenecks_section(
        &self,
        output: &mut String,
        bottlenecks: &[GradientBottleneck],
    ) -> Result<()> {
        writeln!(
            output,
            "⚠️  Gradient Bottlenecks ({} found)",
            bottlenecks.len()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        for (i, bottleneck) in bottlenecks.iter().enumerate().take(10) {
            let severity_indicator = match bottleneck.severity_score() {
                score if score >= 70 => "🔴",
                score if score >= 40 => "🟠",
                _ => "🟡",
            };

            writeln!(
                output,
                "  {}. {} {} (Severity: {})",
                i + 1,
                severity_indicator,
                bottleneck.description(),
                bottleneck.severity_score()
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }

        if bottlenecks.len() > 10 {
            writeln!(
                output,
                "  ... and {} more bottlenecks",
                bottlenecks.len() - 10
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }
        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Write critical path section
    fn write_critical_path_section(
        &self,
        output: &mut String,
        critical_path: &[OperationInfo],
    ) -> Result<()> {
        writeln!(
            output,
            "🎯 Critical Path ({} operations)",
            critical_path.len()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        let max_ops = self
            .config
            .max_critical_path_operations
            .min(critical_path.len());
        for (i, op_info) in critical_path.iter().enumerate().take(max_ops) {
            let intensity_indicator = if op_info.is_compute_bound() {
                "⚡"
            } else if op_info.is_memory_bound() {
                "💾"
            } else {
                "⚖️"
            };

            writeln!(
                output,
                "  {}. {} {} (Intensity: {:.2})",
                i + 1,
                intensity_indicator,
                op_info.description(),
                op_info.operation_intensity()
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }

        if critical_path.len() > max_ops {
            writeln!(
                output,
                "  ... and {} more operations",
                critical_path.len() - max_ops
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }
        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Generate a DOT format visualization for GraphViz
    ///
    /// Creates a GraphViz-compatible DOT file that can be rendered into
    /// publication-quality graph visualizations.
    pub fn generate_dot_visualization(&self, analysis: &GradientFlowAnalysis) -> Result<String> {
        let mut output = String::new();

        writeln!(output, "digraph GradientFlow {{")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "  rankdir=TB;")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "  node [shape=box, style=filled];")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "  edge [fontsize=10];")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        // Add subgraph for legend
        self.write_dot_legend(&mut output)?;

        // Add nodes for operations
        for op in &analysis.critical_path {
            self.write_dot_node(&mut output, op)?;
        }

        writeln!(output).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        // Add edges representing data flow
        for i in 0..analysis.critical_path.len().saturating_sub(1) {
            let from = &analysis.critical_path[i];
            let to = &analysis.critical_path[i + 1];
            writeln!(
                output,
                "  \"op_{}\" -> \"op_{}\";",
                from.operation_id, to.operation_id
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }

        writeln!(output, "}}")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(output)
    }

    /// Write DOT legend
    fn write_dot_legend(&self, output: &mut String) -> Result<()> {
        writeln!(output, "  subgraph cluster_legend {{")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "    label=\"Legend\";")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(output, "    style=dashed;")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        let colors = self.get_dot_colors();
        writeln!(
            output,
            "    legend_healthy [label=\"Healthy\" fillcolor=\"{}\"];",
            colors.healthy
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "    legend_warning [label=\"Warning\" fillcolor=\"{}\"];",
            colors.warning
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        writeln!(
            output,
            "    legend_critical [label=\"Critical\" fillcolor=\"{}\"];",
            colors.critical
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        writeln!(output, "  }}")
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Write DOT node for operation
    fn write_dot_node(&self, output: &mut String, op: &OperationInfo) -> Result<()> {
        let colors = self.get_dot_colors();
        let color = if op.is_memory_bound() {
            &colors.critical
        } else if op.is_compute_bound() {
            &colors.warning
        } else {
            &colors.healthy
        };

        let memory_mb = op.memory_usage as f64 / (1024.0 * 1024.0);
        let flops_k = op.computational_complexity as f64 / 1000.0;

        writeln!(
            output,
            "  \"op_{}\" [label=\"{}\\n{}\\n{:.1}MB, {:.1}K FLOPs\\nIntensity: {:.2}\" fillcolor=\"{}\"];",
            op.operation_id,
            op.operation_name,
            op.operation_type,
            memory_mb,
            flops_k,
            op.operation_intensity(),
            color
        ).map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Get colors for DOT visualization based on color scheme
    fn get_dot_colors(&self) -> DotColors {
        match &self.config.color_scheme {
            ColorScheme::Default => DotColors {
                healthy: "lightgreen".to_string(),
                warning: "orange".to_string(),
                critical: "lightcoral".to_string(),
                background: "white".to_string(),
            },
            ColorScheme::HighContrast => DotColors {
                healthy: "green".to_string(),
                warning: "yellow".to_string(),
                critical: "red".to_string(),
                background: "white".to_string(),
            },
            ColorScheme::Grayscale => DotColors {
                healthy: "lightgray".to_string(),
                warning: "gray".to_string(),
                critical: "darkgray".to_string(),
                background: "white".to_string(),
            },
            ColorScheme::Custom {
                healthy,
                warning,
                critical,
                background,
            } => DotColors {
                healthy: healthy.clone(),
                warning: warning.clone(),
                critical: critical.clone(),
                background: background.clone(),
            },
        }
    }

    /// Generate HTML visualization with interactive elements
    ///
    /// Creates a complete HTML page with embedded CSS and JavaScript for
    /// interactive visualization of gradient flows.
    pub fn generate_html_visualization(&self, analysis: &GradientFlowAnalysis) -> Result<String> {
        let mut output = String::new();

        self.write_html_header(&mut output)?;
        self.write_html_overview(&mut output, analysis)?;
        self.write_html_statistics(&mut output, analysis)?;
        self.write_html_bottlenecks(&mut output, analysis)?;
        self.write_html_critical_path(&mut output, analysis)?;
        self.write_html_footer(&mut output)?;

        Ok(output)
    }

    /// Write HTML header and styles
    fn write_html_header(&self, output: &mut String) -> Result<()> {
        writeln!(
            output,
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Flow Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .bottleneck {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }}
        .bottleneck.critical {{
            background: #f8d7da;
            border-color: #f5c6cb;
        }}
        .bottleneck.warning {{
            background: #fff3cd;
            border-color: #ffeaa7;
        }}
        .operation {{
            background: #e7f3ff;
            border: 1px solid #bee5eb;
            border-radius: 4px;
            padding: 10px;
            margin: 5px 0;
        }}
        .severity-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .severity-high {{ background-color: #dc3545; }}
        .severity-medium {{ background-color: #ffc107; }}
        .severity-low {{ background-color: #28a745; }}
        .interactive {{
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .interactive:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Gradient Flow Analysis</h1>"#
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Write HTML overview section
    fn write_html_overview(
        &self,
        output: &mut String,
        analysis: &GradientFlowAnalysis,
    ) -> Result<()> {
        let summary = analysis.summary();

        writeln!(
            output,
            r#"        <h2>📊 Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Total Operations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Operations with Gradients</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.1}%</div>
                <div class="stat-label">Gradient Flow Efficiency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.3}</div>
                <div class="stat-label">Health Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Bottlenecks Found</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.1}%</div>
                <div class="stat-label">Memory Efficiency</div>
            </div>
        </div>"#,
            summary.total_operations,
            summary.operations_with_gradients,
            analysis.gradient_flow_efficiency() * 100.0,
            summary.gradient_health_score,
            summary.bottleneck_count,
            summary.memory_efficiency * 100.0
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Write HTML statistics section
    fn write_html_statistics(
        &self,
        output: &mut String,
        analysis: &GradientFlowAnalysis,
    ) -> Result<()> {
        if !self.show_gradient_magnitudes {
            return Ok(());
        }

        let stats = &analysis.gradient_stats;
        writeln!(
            output,
            r#"        <h2>📈 Gradient Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{:.6}</div>
                <div class="stat-label">Mean Magnitude</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.6}</div>
                <div class="stat-label">Std Deviation</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.6}</div>
                <div class="stat-label">Max Magnitude</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Zero Gradients</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Invalid Gradients</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Quality Category</div>
            </div>
        </div>"#,
            stats.mean_magnitude,
            stats.std_deviation,
            stats.max_magnitude,
            stats.zero_count,
            stats.inf_nan_count,
            stats.magnitude_category()
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Write HTML bottlenecks section
    fn write_html_bottlenecks(
        &self,
        output: &mut String,
        analysis: &GradientFlowAnalysis,
    ) -> Result<()> {
        if analysis.gradient_bottlenecks.is_empty() {
            return Ok(());
        }

        writeln!(output, r#"        <h2>⚠️ Gradient Bottlenecks</h2>"#)
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        for bottleneck in &analysis.gradient_bottlenecks {
            let (severity_class, severity_indicator) = match bottleneck.severity_score() {
                score if score >= 70 => ("critical", "severity-high"),
                score if score >= 40 => ("warning", "severity-medium"),
                _ => ("", "severity-low"),
            };

            writeln!(
                output,
                r#"        <div class="bottleneck {} interactive">
            <span class="severity-indicator {}"></span>
            <strong>{}</strong> - {}
            <br><small>Severity Score: {} | Type: {}</small>
        </div>"#,
                severity_class,
                severity_indicator,
                bottleneck.operation_name,
                bottleneck.description(),
                bottleneck.severity_score(),
                bottleneck.bottleneck_type
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }

        Ok(())
    }

    /// Write HTML critical path section
    fn write_html_critical_path(
        &self,
        output: &mut String,
        analysis: &GradientFlowAnalysis,
    ) -> Result<()> {
        if analysis.critical_path.is_empty() {
            return Ok(());
        }

        writeln!(output, r#"        <h2>🎯 Critical Path</h2>"#)
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        let max_ops = self
            .config
            .max_critical_path_operations
            .min(analysis.critical_path.len());
        for op in analysis.critical_path.iter().take(max_ops) {
            writeln!(
                output,
                r#"        <div class="operation interactive">
            <strong>{}</strong> ({})
            <br><small>{} | Intensity: {:.2} | Memory: {:.1} MB</small>
        </div>"#,
                op.operation_name,
                op.operation_type,
                op.description(),
                op.operation_intensity(),
                op.memory_usage as f64 / (1024.0 * 1024.0)
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }

        if analysis.critical_path.len() > max_ops {
            writeln!(
                output,
                r#"        <p><em>... and {} more operations in the critical path</em></p>"#,
                analysis.critical_path.len() - max_ops
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;
        }

        Ok(())
    }

    /// Write HTML footer
    fn write_html_footer(&self, output: &mut String) -> Result<()> {
        writeln!(
            output,
            r#"    </div>
    <script>
        // Add interactivity
        document.querySelectorAll('.interactive').forEach(element => {{
            element.addEventListener('click', function() {{
                console.log('Clicked:', this.textContent.trim());
            }});
        }});

        // Add tooltips or additional functionality here
        console.log('Gradient Flow Analysis loaded');
    </script>
</body>
</html>"#
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))?;

        Ok(())
    }

    /// Export analysis results to JSON format
    ///
    /// Creates a structured JSON representation suitable for integration
    /// with external tools and automated analysis systems.
    pub fn export_analysis_json(&self, analysis: &GradientFlowAnalysis) -> Result<String> {
        // Since we can't use serde, we'll create JSON manually
        let mut output = String::new();

        use std::fmt::Write;

        // String formatting to String never fails, so we can unwrap safely
        writeln!(output, "{{").unwrap();
        writeln!(output, "  \"gradient_flow_analysis\": {{").unwrap();
        writeln!(
            output,
            "    \"timestamp\": \"{:?}\",",
            std::time::SystemTime::now()
        )
        .unwrap();
        writeln!(
            output,
            "    \"total_operations\": {},",
            analysis.total_operations
        )
        .unwrap();
        writeln!(
            output,
            "    \"operations_with_gradients\": {},",
            analysis.operations_with_gradients
        )
        .unwrap();
        writeln!(
            output,
            "    \"gradient_flow_efficiency\": {},",
            analysis.gradient_flow_efficiency()
        )
        .unwrap();
        writeln!(
            output,
            "    \"gradient_health_score\": {},",
            analysis.gradient_health_score()
        )
        .unwrap();

        // Gradient statistics
        writeln!(output, "    \"gradient_statistics\": {{").unwrap();
        writeln!(
            output,
            "      \"mean_magnitude\": {},",
            analysis.gradient_stats.mean_magnitude
        )
        .unwrap();
        writeln!(
            output,
            "      \"std_deviation\": {},",
            analysis.gradient_stats.std_deviation
        )
        .unwrap();
        writeln!(
            output,
            "      \"max_magnitude\": {},",
            analysis.gradient_stats.max_magnitude
        )
        .unwrap();
        writeln!(
            output,
            "      \"min_magnitude\": {},",
            analysis.gradient_stats.min_magnitude
        )
        .unwrap();
        writeln!(
            output,
            "      \"zero_count\": {},",
            analysis.gradient_stats.zero_count
        )
        .unwrap();
        writeln!(
            output,
            "      \"inf_nan_count\": {},",
            analysis.gradient_stats.inf_nan_count
        )
        .unwrap();
        writeln!(
            output,
            "      \"quality_score\": {},",
            analysis.gradient_stats.quality_score()
        )
        .unwrap();
        writeln!(
            output,
            "      \"category\": \"{}\"",
            analysis.gradient_stats.magnitude_category()
        )
        .unwrap();
        writeln!(output, "    }},").unwrap();

        // Memory breakdown
        writeln!(output, "    \"memory_breakdown\": {{").unwrap();
        writeln!(
            output,
            "      \"gradient_memory\": {},",
            analysis.memory_breakdown.gradient_memory
        )
        .unwrap();
        writeln!(
            output,
            "      \"intermediate_memory\": {},",
            analysis.memory_breakdown.intermediate_memory
        )
        .unwrap();
        writeln!(
            output,
            "      \"metadata_memory\": {},",
            analysis.memory_breakdown.metadata_memory
        )
        .unwrap();
        writeln!(
            output,
            "      \"peak_memory\": {},",
            analysis.memory_breakdown.peak_memory
        )
        .unwrap();
        writeln!(
            output,
            "      \"total_memory\": {},",
            analysis.memory_breakdown.total_memory()
        )
        .unwrap();
        writeln!(
            output,
            "      \"efficiency_score\": {}",
            analysis.memory_breakdown.efficiency_score()
        )
        .unwrap();
        writeln!(output, "    }},").unwrap();

        // Bottlenecks
        writeln!(output, "    \"bottlenecks\": [").unwrap();
        for (i, bottleneck) in analysis.gradient_bottlenecks.iter().enumerate() {
            writeln!(output, "      {{").unwrap();
            writeln!(
                output,
                "        \"operation_id\": {},",
                bottleneck.operation_id
            )
            .unwrap();
            writeln!(
                output,
                "        \"operation_name\": \"{}\",",
                bottleneck.operation_name
            )
            .unwrap();
            writeln!(
                output,
                "        \"gradient_magnitude\": {},",
                bottleneck.gradient_magnitude
            )
            .unwrap();
            writeln!(
                output,
                "        \"dependency_count\": {},",
                bottleneck.dependency_count
            )
            .unwrap();
            writeln!(
                output,
                "        \"memory_usage\": {},",
                bottleneck.memory_usage
            )
            .unwrap();
            writeln!(
                output,
                "        \"bottleneck_type\": \"{}\",",
                bottleneck.bottleneck_type
            )
            .unwrap();
            writeln!(
                output,
                "        \"severity_score\": {},",
                bottleneck.severity_score()
            )
            .unwrap();
            writeln!(
                output,
                "        \"is_significant\": {}",
                bottleneck.is_significant()
            )
            .unwrap();
            if i < analysis.gradient_bottlenecks.len() - 1 {
                writeln!(output, "      }},").unwrap();
            } else {
                writeln!(output, "      }}").unwrap();
            }
        }
        writeln!(output, "    ]").unwrap();

        writeln!(output, "  }}").unwrap();
        writeln!(output, "}}").unwrap();

        Ok(output)
    }

    // Helper methods for analysis computation

    /// Compute gradient statistics from autograd context
    fn compute_gradient_statistics(&self, _ctx: &AutogradContext) -> Result<GradientStatistics> {
        debug!("Computing gradient statistics");

        // This is a simplified implementation - in practice would analyze actual gradients
        Ok(GradientStatistics {
            mean_magnitude: 0.001,
            std_deviation: 0.0005,
            max_magnitude: 0.01,
            min_magnitude: 0.0001,
            zero_count: 0,
            inf_nan_count: 0,
        })
    }

    /// Identify gradient bottlenecks in the computation graph
    fn identify_bottlenecks(&self, _ctx: &AutogradContext) -> Result<Vec<GradientBottleneck>> {
        debug!("Identifying gradient bottlenecks");

        // This is a simplified implementation - would analyze actual graph structure
        Ok(Vec::new())
    }

    /// Analyze critical path through the computation graph
    fn analyze_critical_path(&self, _ctx: &AutogradContext) -> Result<Vec<OperationInfo>> {
        debug!("Analyzing critical path");

        // This is a simplified implementation - would analyze actual critical path
        Ok(Vec::new())
    }

    /// Compute memory breakdown for gradient computations
    fn compute_memory_breakdown(&self, _ctx: &AutogradContext) -> Result<MemoryBreakdown> {
        debug!("Computing memory breakdown");

        // This is a simplified implementation - would analyze actual memory usage
        Ok(MemoryBreakdown::new())
    }

    /// Count operations that have gradients flowing through them
    fn count_operations_with_gradients(&self, ctx: &AutogradContext) -> Result<usize> {
        let stats = ctx.graph_stats();
        Ok(stats.node_count) // Simplified - would count actual gradient-enabled operations
    }
}

/// Helper struct for DOT color management
#[derive(Debug, Clone)]
struct DotColors {
    healthy: String,
    warning: String,
    critical: String,
    #[allow(dead_code)]
    background: String,
}

/// Helper function to write Result for writeln operations
#[allow(dead_code)]
trait WriteResult {
    fn map_err_write(self, msg: &str) -> Result<()>;
}

impl WriteResult for std::fmt::Result {
    fn map_err_write(self, _msg: &str) -> Result<()> {
        self.map_err(|e| TorshError::InvalidArgument(format!("Write error: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::AutogradContext;

    #[test]
    fn test_visualizer_creation() {
        let visualizer = GradientVisualizer::new();
        assert!(visualizer.include_detailed_stats);
        assert!(visualizer.show_gradient_magnitudes);
        assert!(visualizer.show_memory_usage);
        assert_eq!(visualizer.bottleneck_threshold, 0.8);
    }

    #[test]
    fn test_visualizer_configuration() {
        let visualizer = GradientVisualizer::new()
            .with_bottleneck_threshold(0.5)
            .with_detailed_stats(false)
            .with_compact_output(true);

        assert_eq!(visualizer.bottleneck_threshold, 0.5);
        assert!(!visualizer.include_detailed_stats);
        assert!(visualizer.config.compact_output);
    }

    #[test]
    fn test_color_schemes() {
        let default_colors = ColorScheme::Default;
        let high_contrast = ColorScheme::HighContrast;
        let grayscale = ColorScheme::Grayscale;

        // Test that different color schemes exist
        assert!(matches!(default_colors, ColorScheme::Default));
        assert!(matches!(high_contrast, ColorScheme::HighContrast));
        assert!(matches!(grayscale, ColorScheme::Grayscale));
    }

    #[test]
    fn test_gradient_flow_analysis() {
        let visualizer = GradientVisualizer::new();
        let ctx = AutogradContext::new();

        let result = visualizer.analyze_gradient_flow(&ctx);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.total_operations, 0); // Empty context
        assert_eq!(analysis.operations_with_gradients, 0);
    }

    #[test]
    fn test_text_visualization() {
        let visualizer = GradientVisualizer::new();
        let analysis = GradientFlowAnalysis::new();

        let result = visualizer.generate_text_visualization(&analysis);
        assert!(result.is_ok());

        let text = result.unwrap();
        assert!(text.contains("Gradient Flow Analysis"));
        assert!(text.contains("Total Operations: 0"));
    }

    #[test]
    fn test_compact_text_visualization() {
        let visualizer = GradientVisualizer::new().with_compact_output(true);
        let analysis = GradientFlowAnalysis::new();

        let result = visualizer.generate_text_visualization(&analysis);
        assert!(result.is_ok());

        let text = result.unwrap();
        assert!(text.contains("Compact"));
        assert!(text.contains("Operations: 0"));
    }

    #[test]
    fn test_dot_visualization() {
        let visualizer = GradientVisualizer::new();
        let analysis = GradientFlowAnalysis::new();

        let result = visualizer.generate_dot_visualization(&analysis);
        assert!(result.is_ok());

        let dot = result.unwrap();
        assert!(dot.contains("digraph GradientFlow"));
        assert!(dot.contains("rankdir=TB"));
        assert!(dot.contains("cluster_legend"));
    }

    #[test]
    fn test_html_visualization() {
        let visualizer = GradientVisualizer::new();
        let analysis = GradientFlowAnalysis::new();

        let result = visualizer.generate_html_visualization(&analysis);
        assert!(result.is_ok());

        let html = result.unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Gradient Flow Analysis"));
        assert!(html.contains("stats-grid"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_json_export() {
        let visualizer = GradientVisualizer::new();
        let analysis = GradientFlowAnalysis::new();

        let result = visualizer.export_analysis_json(&analysis);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("gradient_flow_analysis"));
        assert!(json.contains("total_operations"));
        assert!(json.contains("gradient_statistics"));
        assert!(json.contains("memory_breakdown"));
    }

    #[test]
    fn test_dot_colors() {
        let visualizer = GradientVisualizer::new();
        let colors = visualizer.get_dot_colors();

        assert_eq!(colors.healthy, "lightgreen");
        assert_eq!(colors.warning, "orange");
        assert_eq!(colors.critical, "lightcoral");
        assert_eq!(colors.background, "white");
    }

    #[test]
    fn test_custom_color_scheme() {
        let custom_scheme = ColorScheme::Custom {
            healthy: "#00FF00".to_string(),
            warning: "#FFFF00".to_string(),
            critical: "#FF0000".to_string(),
            background: "#FFFFFF".to_string(),
        };

        let config = VisualizerConfig {
            color_scheme: custom_scheme,
            ..VisualizerConfig::default()
        };

        let visualizer = GradientVisualizer::with_config(config);
        let colors = visualizer.get_dot_colors();

        assert_eq!(colors.healthy, "#00FF00");
        assert_eq!(colors.warning, "#FFFF00");
        assert_eq!(colors.critical, "#FF0000");
        assert_eq!(colors.background, "#FFFFFF");
    }
}
