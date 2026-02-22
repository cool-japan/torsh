//! Advanced HTML reporting for ToRSh benchmarks
//!
//! This module provides comprehensive HTML report generation with interactive
//! visualizations, charts, and detailed analysis of benchmark results.

use crate::{BenchConfig, BenchResult};
use std::fs;
use std::io::{self, Write};

/// HTML report generator with advanced visualization
pub struct HtmlReportGenerator {
    pub results: Vec<BenchResult>,
    pub configs: Vec<BenchConfig>,
    pub metadata: ReportMetadata,
    pub style_config: StyleConfig,
}

/// Report metadata and configuration
#[derive(Debug, Clone)]
pub struct ReportMetadata {
    pub title: String,
    pub description: String,
    pub version: String,
    pub timestamp: String,
    pub environment: EnvironmentInfo,
    pub summary: ReportSummary,
}

/// Environment information
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    pub os: String,
    pub cpu: String,
    pub memory: String,
    pub gpu: Option<String>,
    pub compiler: String,
    pub rust_version: String,
    pub torsh_version: String,
}

/// Report summary statistics
#[derive(Debug, Clone)]
pub struct ReportSummary {
    pub total_benchmarks: usize,
    pub total_execution_time: f64,
    pub fastest_benchmark: Option<String>,
    pub slowest_benchmark: Option<String>,
    pub average_performance: f64,
    pub performance_variance: f64,
}

/// Styling configuration for HTML reports
#[derive(Debug, Clone)]
pub struct StyleConfig {
    pub theme: Theme,
    pub color_scheme: ColorScheme,
    pub chart_style: ChartStyle,
    pub layout: LayoutConfig,
}

#[derive(Debug, Clone)]
pub enum Theme {
    Light,
    Dark,
    Auto,
}

#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub success: String,
    pub warning: String,
    pub danger: String,
    pub background: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ChartStyle {
    pub chart_type: ChartType,
    pub animation: bool,
    pub interactive: bool,
    pub responsive: bool,
}

#[derive(Debug, Clone)]
pub enum ChartType {
    Bar,
    Line,
    Scatter,
    Heatmap,
    Radar,
    Box,
}

#[derive(Debug, Clone)]
pub struct LayoutConfig {
    pub sidebar: bool,
    pub breadcrumbs: bool,
    pub search: bool,
    pub filters: bool,
    pub export_buttons: bool,
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            theme: Theme::Light,
            color_scheme: ColorScheme::default(),
            chart_style: ChartStyle::default(),
            layout: LayoutConfig::default(),
        }
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            primary: "#007bff".to_string(),
            secondary: "#6c757d".to_string(),
            success: "#28a745".to_string(),
            warning: "#ffc107".to_string(),
            danger: "#dc3545".to_string(),
            background: "#ffffff".to_string(),
            text: "#333333".to_string(),
        }
    }
}

impl Default for ChartStyle {
    fn default() -> Self {
        Self {
            chart_type: ChartType::Bar,
            animation: true,
            interactive: true,
            responsive: true,
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            sidebar: true,
            breadcrumbs: true,
            search: true,
            filters: true,
            export_buttons: true,
        }
    }
}

impl HtmlReportGenerator {
    /// Create a new HTML report generator
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            configs: Vec::new(),
            metadata: ReportMetadata::default(),
            style_config: StyleConfig::default(),
        }
    }

    /// Add benchmark results to the report
    pub fn add_results(&mut self, results: Vec<BenchResult>) {
        self.results.extend(results);
    }

    /// Add benchmark configurations to the report
    pub fn add_configs(&mut self, configs: Vec<BenchConfig>) {
        self.configs.extend(configs);
    }

    /// Set report metadata
    pub fn set_metadata(&mut self, metadata: ReportMetadata) {
        self.metadata = metadata;
    }

    /// Set style configuration
    pub fn set_style(&mut self, style_config: StyleConfig) {
        self.style_config = style_config;
    }

    /// Generate comprehensive HTML report
    pub fn generate_report(&self, output_dir: &str) -> io::Result<()> {
        // Create output directory
        fs::create_dir_all(output_dir)?;

        // Generate main report file
        self.generate_main_report(output_dir)?;

        // Generate detailed analysis pages
        self.generate_performance_analysis(output_dir)?;
        self.generate_comparison_charts(output_dir)?;
        self.generate_detailed_results(output_dir)?;
        self.generate_environment_info(output_dir)?;

        // Generate static assets
        self.generate_css(output_dir)?;
        self.generate_javascript(output_dir)?;

        Ok(())
    }

    /// Generate main report HTML file
    fn generate_main_report(&self, output_dir: &str) -> io::Result<()> {
        let file_path = format!("{}/index.html", output_dir);
        let mut file = fs::File::create(&file_path)?;

        writeln!(
            file,
            "{}",
            self.generate_html_header("ToRSh Benchmark Report")
        )?;
        writeln!(file, "<body class=\"{}\">\n", self.get_theme_class())?;

        // Navigation and header
        writeln!(file, "{}", self.generate_navigation())?;
        writeln!(file, "{}", self.generate_header())?;

        // Main content
        writeln!(file, "<main class=\"container-fluid\">")?;
        writeln!(file, "<div class=\"row\">")?;

        // Sidebar (if enabled)
        if self.style_config.layout.sidebar {
            writeln!(file, "{}", self.generate_sidebar())?;
        }

        // Main content area
        writeln!(
            file,
            "<div class=\"col-md-{}\">\n",
            if self.style_config.layout.sidebar {
                9
            } else {
                12
            }
        )?;

        // Summary cards
        writeln!(file, "{}", self.generate_summary_cards())?;

        // Quick performance overview
        writeln!(file, "{}", self.generate_performance_overview())?;

        // Recent benchmarks table
        writeln!(file, "{}", self.generate_recent_benchmarks_table())?;

        // Performance trends chart
        writeln!(file, "{}", self.generate_performance_trends_chart())?;

        writeln!(file, "</div>")?; // End main content area
        writeln!(file, "</div>")?; // End row
        writeln!(file, "</main>")?;

        // Footer
        writeln!(file, "{}", self.generate_footer())?;

        writeln!(file, "{}", self.generate_html_footer())?;

        Ok(())
    }

    /// Generate performance analysis page
    fn generate_performance_analysis(&self, output_dir: &str) -> io::Result<()> {
        let file_path = format!("{}/performance_analysis.html", output_dir);
        let mut file = fs::File::create(&file_path)?;

        writeln!(
            file,
            "{}",
            self.generate_html_header("Performance Analysis")
        )?;
        writeln!(file, "<body class=\"{}\">\n", self.get_theme_class())?;

        writeln!(file, "{}", self.generate_navigation())?;

        writeln!(file, "<main class=\"container-fluid\">")?;
        writeln!(file, "<h1>Performance Analysis</h1>")?;

        // Performance distribution charts
        writeln!(file, "{}", self.generate_performance_distribution())?;

        // Bottleneck analysis
        writeln!(file, "{}", self.generate_bottleneck_analysis())?;

        // Scalability analysis
        writeln!(file, "{}", self.generate_scalability_analysis())?;

        // Memory usage analysis
        writeln!(file, "{}", self.generate_memory_analysis())?;

        writeln!(file, "</main>")?;
        writeln!(file, "{}", self.generate_footer())?;
        writeln!(file, "{}", self.generate_html_footer())?;

        Ok(())
    }

    /// Generate comparison charts page
    fn generate_comparison_charts(&self, output_dir: &str) -> io::Result<()> {
        let file_path = format!("{}/comparison_charts.html", output_dir);
        let mut file = fs::File::create(&file_path)?;

        writeln!(file, "{}", self.generate_html_header("Comparison Charts"))?;
        writeln!(file, "<body class=\"{}\">\n", self.get_theme_class())?;

        writeln!(file, "{}", self.generate_navigation())?;

        writeln!(file, "<main class=\"container-fluid\">")?;
        writeln!(file, "<h1>Comparison Charts</h1>")?;

        // Benchmark comparison charts
        writeln!(file, "{}", self.generate_benchmark_comparison_chart())?;

        // Data type performance comparison
        writeln!(file, "{}", self.generate_dtype_comparison_chart())?;

        // Size scaling comparison
        writeln!(file, "{}", self.generate_size_scaling_chart())?;

        // Cross-platform comparison
        writeln!(file, "{}", self.generate_platform_comparison_chart())?;

        writeln!(file, "</main>")?;
        writeln!(file, "{}", self.generate_footer())?;
        writeln!(file, "{}", self.generate_html_footer())?;

        Ok(())
    }

    /// Generate detailed results page
    fn generate_detailed_results(&self, output_dir: &str) -> io::Result<()> {
        let file_path = format!("{}/detailed_results.html", output_dir);
        let mut file = fs::File::create(&file_path)?;

        writeln!(file, "{}", self.generate_html_header("Detailed Results"))?;
        writeln!(file, "<body class=\"{}\">\n", self.get_theme_class())?;

        writeln!(file, "{}", self.generate_navigation())?;

        writeln!(file, "<main class=\"container-fluid\">")?;
        writeln!(file, "<h1>Detailed Results</h1>")?;

        // Search and filter controls
        if self.style_config.layout.filters {
            writeln!(file, "{}", self.generate_filter_controls())?;
        }

        // Detailed results table
        writeln!(file, "{}", self.generate_detailed_results_table())?;

        writeln!(file, "</main>")?;
        writeln!(file, "{}", self.generate_footer())?;
        writeln!(file, "{}", self.generate_html_footer())?;

        Ok(())
    }

    /// Generate environment information page
    fn generate_environment_info(&self, output_dir: &str) -> io::Result<()> {
        let file_path = format!("{}/environment.html", output_dir);
        let mut file = fs::File::create(&file_path)?;

        writeln!(
            file,
            "{}",
            self.generate_html_header("Environment Information")
        )?;
        writeln!(file, "<body class=\"{}\">\n", self.get_theme_class())?;

        writeln!(file, "{}", self.generate_navigation())?;

        writeln!(file, "<main class=\"container-fluid\">")?;
        writeln!(file, "<h1>Environment Information</h1>")?;

        // System information
        writeln!(file, "{}", self.generate_system_info())?;

        // Benchmark configuration
        writeln!(file, "{}", self.generate_benchmark_config())?;

        writeln!(file, "</main>")?;
        writeln!(file, "{}", self.generate_footer())?;
        writeln!(file, "{}", self.generate_html_footer())?;

        Ok(())
    }

    /// Generate CSS styles
    fn generate_css(&self, output_dir: &str) -> io::Result<()> {
        let file_path = format!("{}/styles.css", output_dir);
        let mut file = fs::File::create(&file_path)?;

        // Base styles
        writeln!(file, "{}", self.generate_base_css())?;

        // Theme-specific styles
        writeln!(file, "{}", self.generate_theme_css())?;

        // Chart styles
        writeln!(file, "{}", self.generate_chart_css())?;

        // Responsive styles
        writeln!(file, "{}", self.generate_responsive_css())?;

        Ok(())
    }

    /// Generate JavaScript functionality
    fn generate_javascript(&self, output_dir: &str) -> io::Result<()> {
        let file_path = format!("{}/scripts.js", output_dir);
        let mut file = fs::File::create(&file_path)?;

        // Chart rendering functions
        writeln!(file, "{}", self.generate_chart_js())?;

        // Interactive features
        writeln!(file, "{}", self.generate_interactive_js())?;

        // Search and filter functions
        writeln!(file, "{}", self.generate_search_filter_js())?;

        // Export functions
        writeln!(file, "{}", self.generate_export_js())?;

        Ok(())
    }

    // HTML component generation methods

    fn generate_html_header(&self, title: &str) -> String {
        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="styles.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
</head>"#,
            title
        )
    }

    fn generate_html_footer(&self) -> String {
        r#"<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<script src="scripts.js"></script>
</body>
</html>"#.to_string()
    }

    fn get_theme_class(&self) -> &str {
        match self.style_config.theme {
            Theme::Light => "theme-light",
            Theme::Dark => "theme-dark",
            Theme::Auto => "theme-auto",
        }
    }

    fn generate_navigation(&self) -> String {
        let export_buttons = if self.style_config.layout.export_buttons {
            "<div class=\"d-flex\">\n                <button class=\"btn btn-outline-light me-2\" onclick=\"exportToPDF()\">\n                    <i class=\"fas fa-file-pdf me-1\"></i>Export PDF\n                </button>\n                <button class=\"btn btn-outline-light\" onclick=\"exportToCSV()\">\n                    <i class=\"fas fa-file-csv me-1\"></i>Export CSV\n                </button>\n            </div>"
        } else {
            ""
        };

        format!("<nav class=\"navbar navbar-expand-lg navbar-dark bg-primary\">\n    <div class=\"container-fluid\">\n        <a class=\"navbar-brand\" href=\"index.html\">\n            <i class=\"fas fa-chart-line me-2\"></i>ToRSh Benchmarks\n        </a>\n        <button class=\"navbar-toggler\" type=\"button\" data-bs-toggle=\"collapse\" data-bs-target=\"#navbarNav\">\n            <span class=\"navbar-toggler-icon\"></span>\n        </button>\n        <div class=\"collapse navbar-collapse\" id=\"navbarNav\">\n            <ul class=\"navbar-nav me-auto\">\n                <li class=\"nav-item\">\n                    <a class=\"nav-link\" href=\"index.html\"><i class=\"fas fa-home me-1\"></i>Overview</a>\n                </li>\n                <li class=\"nav-item\">\n                    <a class=\"nav-link\" href=\"performance_analysis.html\"><i class=\"fas fa-analytics me-1\"></i>Analysis</a>\n                </li>\n                <li class=\"nav-item\">\n                    <a class=\"nav-link\" href=\"comparison_charts.html\"><i class=\"fas fa-chart-bar me-1\"></i>Comparisons</a>\n                </li>\n                <li class=\"nav-item\">\n                    <a class=\"nav-link\" href=\"detailed_results.html\"><i class=\"fas fa-table me-1\"></i>Detailed Results</a>\n                </li>\n                <li class=\"nav-item\">\n                    <a class=\"nav-link\" href=\"environment.html\"><i class=\"fas fa-cog me-1\"></i>Environment</a>\n                </li>\n            </ul>\n            {}\n        </div>\n    </div>\n</nav>", export_buttons)
    }

    fn generate_header(&self) -> String {
        format!(
            r#"<header class="bg-light py-4 mb-4">
    <div class="container-fluid">
        <div class="row align-items-center">
            <div class="col-md-8">
                <h1 class="display-4 mb-2">{}</h1>
                <p class="lead text-muted">{}</p>
                {}
            </div>
            <div class="col-md-4 text-end">
                <div class="card bg-primary text-white">
                    <div class="card-body">
                        <h5 class="card-title"><i class="fas fa-clock me-2"></i>Generated</h5>
                        <p class="card-text">{}</p>
                        <small>Version {}</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</header>"#,
            self.metadata.title,
            self.metadata.description,
            if self.style_config.layout.breadcrumbs {
                r#"<nav aria-label="breadcrumb">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="index.html">Home</a></li>
                        <li class="breadcrumb-item active">Overview</li>
                    </ol>
                </nav>"#
            } else {
                ""
            },
            self.metadata.timestamp,
            self.metadata.version
        )
    }

    fn generate_sidebar(&self) -> String {
        format!(
            r#"<div class="col-md-3">
    <div class="card">
        <div class="card-header">
            <h5><i class="fas fa-filter me-2"></i>Quick Filters</h5>
        </div>
        <div class="card-body">
            <div class="mb-3">
                <label class="form-label">Benchmark Type</label>
                <select class="form-select" id="benchmarkTypeFilter">
                    <option value="">All Types</option>
                    <option value="tensor">Tensor Operations</option>
                    <option value="nn">Neural Networks</option>
                    <option value="autograd">Autograd</option>
                    <option value="custom">Custom Operations</option>
                </select>
            </div>
            <div class="mb-3">
                <label class="form-label">Performance Range</label>
                <input type="range" class="form-range" id="performanceRange" min="0" max="100">
            </div>
            <div class="mb-3">
                <label class="form-label">Data Type</label>
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="f32Check" checked>
                    <label class="form-check-label" for="f32Check">F32</label>
                </div>
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="f64Check">
                    <label class="form-check-label" for="f64Check">F64</label>
                </div>
            </div>
            {}
        </div>
    </div>
</div>"#,
            if self.style_config.layout.search {
                r#"<div class="mb-3">
                <label class="form-label">Search</label>
                <input type="text" class="form-control" id="searchInput" placeholder="Search benchmarks...">
            </div>"#
            } else {
                ""
            }
        )
    }

    fn generate_summary_cards(&self) -> String {
        let summary = &self.metadata.summary;
        format!(
            r#"<div class="row mb-4">
    <div class="col-md-3">
        <div class="card bg-primary text-white h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5>Total Benchmarks</h5>
                        <h2>{}</h2>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-chart-line fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-success text-white h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5>Avg Performance</h5>
                        <h2>{:.2} GFLOPS</h2>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-tachometer-alt fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-warning text-white h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5>Total Time</h5>
                        <h2>{:.1}s</h2>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-clock fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-info text-white h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h5>Variance</h5>
                        <h2>{:.1}%</h2>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-chart-area fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>"#,
            summary.total_benchmarks,
            summary.average_performance,
            summary.total_execution_time,
            summary.performance_variance * 100.0
        )
    }

    fn generate_performance_overview(&self) -> String {
        r#"<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-line me-2"></i>Performance Overview</h5>
            </div>
            <div class="card-body">
                <canvas id="performanceOverviewChart" width="400" height="200"></canvas>
            </div>
        </div>
    </div>
</div>"#
            .to_string()
    }

    fn generate_recent_benchmarks_table(&self) -> String {
        let mut table_rows = String::new();

        // Take the last 10 results for the "recent" table
        for result in self.results.iter().take(10) {
            table_rows.push_str(&format!(
                r#"<tr>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{:?}</td>
                    <td>{:.2} ms</td>
                    <td>{:.2}</td>
                    <td>
                        <span class="badge bg-{}">{}</span>
                    </td>
                </tr>"#,
                result.name,
                result.size,
                result.dtype,
                result.mean_time_ns / 1_000_000.0,
                result.throughput.unwrap_or(0.0),
                if result.mean_time_ns < 10_000_000.0 {
                    "success"
                } else if result.mean_time_ns < 100_000_000.0 {
                    "warning"
                } else {
                    "danger"
                },
                if result.mean_time_ns < 10_000_000.0 {
                    "Fast"
                } else if result.mean_time_ns < 100_000_000.0 {
                    "Medium"
                } else {
                    "Slow"
                }
            ));
        }

        format!(
            r#"<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-table me-2"></i>Recent Benchmarks</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped table-hover">
                        <thead class="table-dark">
                            <tr>
                                <th>Benchmark</th>
                                <th>Size</th>
                                <th>Data Type</th>
                                <th>Time (ms)</th>
                                <th>Throughput</th>
                                <th>Performance</th>
                            </tr>
                        </thead>
                        <tbody>
                            {}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>"#,
            table_rows
        )
    }

    fn generate_performance_trends_chart(&self) -> String {
        r#"<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-area me-2"></i>Performance Trends</h5>
            </div>
            <div class="card-body">
                <canvas id="performanceTrendsChart" width="400" height="200"></canvas>
            </div>
        </div>
    </div>
</div>"#
            .to_string()
    }

    fn generate_footer(&self) -> String {
        format!(
            r#"<footer class="bg-dark text-light py-4 mt-5">
    <div class="container-fluid">
        <div class="row">
            <div class="col-md-6">
                <h5>ToRSh Benchmark Report</h5>
                <p>Generated on {} using ToRSh version {}</p>
                <p>Environment: {} | {}</p>
            </div>
            <div class="col-md-6 text-end">
                <h5>System Information</h5>
                <p>OS: {}</p>
                <p>CPU: {}</p>
                <p>Memory: {}</p>
            </div>
        </div>
        <hr>
        <div class="row">
            <div class="col-12 text-center">
                <p>&copy; 2025 ToRSh Project. All rights reserved.</p>
            </div>
        </div>
    </div>
</footer>"#,
            self.metadata.timestamp,
            self.metadata.version,
            self.metadata.environment.rust_version,
            self.metadata.environment.compiler,
            self.metadata.environment.os,
            self.metadata.environment.cpu,
            self.metadata.environment.memory
        )
    }

    // Additional component generation methods would go here...
    fn generate_performance_distribution(&self) -> String {
        // Implementation for performance distribution analysis
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Performance Distribution</h5>
    </div>
    <div class="card-body">
        <canvas id="performanceDistributionChart"></canvas>
    </div>
</div>"#
            .to_string()
    }

    fn generate_bottleneck_analysis(&self) -> String {
        // Implementation for bottleneck analysis
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Bottleneck Analysis</h5>
    </div>
    <div class="card-body">
        <p>Analysis of performance bottlenecks and optimization opportunities.</p>
    </div>
</div>"#
            .to_string()
    }

    fn generate_scalability_analysis(&self) -> String {
        // Implementation for scalability analysis
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Scalability Analysis</h5>
    </div>
    <div class="card-body">
        <canvas id="scalabilityChart"></canvas>
    </div>
</div>"#
            .to_string()
    }

    fn generate_memory_analysis(&self) -> String {
        // Implementation for memory analysis
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Memory Usage Analysis</h5>
    </div>
    <div class="card-body">
        <canvas id="memoryChart"></canvas>
    </div>
</div>"#
            .to_string()
    }

    fn generate_benchmark_comparison_chart(&self) -> String {
        // Implementation for benchmark comparison
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Benchmark Comparison</h5>
    </div>
    <div class="card-body">
        <canvas id="benchmarkComparisonChart"></canvas>
    </div>
</div>"#
            .to_string()
    }

    fn generate_dtype_comparison_chart(&self) -> String {
        // Implementation for data type comparison
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Data Type Performance Comparison</h5>
    </div>
    <div class="card-body">
        <canvas id="dtypeComparisonChart"></canvas>
    </div>
</div>"#
            .to_string()
    }

    fn generate_size_scaling_chart(&self) -> String {
        // Implementation for size scaling comparison
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Size Scaling Analysis</h5>
    </div>
    <div class="card-body">
        <canvas id="sizeScalingChart"></canvas>
    </div>
</div>"#
            .to_string()
    }

    fn generate_platform_comparison_chart(&self) -> String {
        // Implementation for platform comparison
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Cross-Platform Performance</h5>
    </div>
    <div class="card-body">
        <canvas id="platformComparisonChart"></canvas>
    </div>
</div>"#
            .to_string()
    }

    fn generate_filter_controls(&self) -> String {
        // Implementation for filter controls
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Filters and Search</h5>
    </div>
    <div class="card-body">
        <div class="row">
            <div class="col-md-4">
                <input type="text" class="form-control" placeholder="Search benchmarks..." id="searchInput">
            </div>
            <div class="col-md-3">
                <select class="form-select" id="typeFilter">
                    <option value="">All Types</option>
                    <option value="tensor">Tensor Ops</option>
                    <option value="nn">Neural Networks</option>
                </select>
            </div>
            <div class="col-md-3">
                <select class="form-select" id="sizeFilter">
                    <option value="">All Sizes</option>
                    <option value="small">Small (&lt; 100)</option>
                    <option value="medium">Medium (100-1000)</option>
                    <option value="large">Large (&gt; 1000)</option>
                </select>
            </div>
            <div class="col-md-2">
                <button class="btn btn-primary w-100" onclick="applyFilters()">Apply</button>
            </div>
        </div>
    </div>
</div>"#.to_string()
    }

    fn generate_detailed_results_table(&self) -> String {
        let mut table_rows = String::new();

        for (i, result) in self.results.iter().enumerate() {
            table_rows.push_str(&format!(
                r#"<tr data-index="{}">
                    <td>{}</td>
                    <td>{}</td>
                    <td>{:?}</td>
                    <td>{:.3} ms</td>
                    <td>{:.3} ms</td>
                    <td>{:.2}</td>
                    <td>{:.2} MB</td>
                    <td>{:.2} MB</td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-{}" role="progressbar" style="width: {}%"></div>
                        </div>
                    </td>
                </tr>"#,
                i,
                result.name,
                result.size,
                result.dtype,
                result.mean_time_ns / 1_000_000.0,
                result.std_dev_ns / 1_000_000.0,
                result.throughput.unwrap_or(0.0),
                result.memory_usage.unwrap_or(0) as f64 / (1024.0 * 1024.0),
                result.peak_memory.unwrap_or(0) as f64 / (1024.0 * 1024.0),
                if result.mean_time_ns < 10_000_000.0 { "success" } else if result.mean_time_ns < 100_000_000.0 { "warning" } else { "danger" },
                ((100.0 - (result.mean_time_ns / 1_000_000.0).min(100.0)).max(10.0)) as u32
            ));
        }

        format!(
            r#"<div class="card">
    <div class="card-header">
        <h5>All Benchmark Results</h5>
    </div>
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-striped table-hover" id="detailedResultsTable">
                <thead class="table-dark">
                    <tr>
                        <th>Benchmark</th>
                        <th>Size</th>
                        <th>Data Type</th>
                        <th>Mean Time</th>
                        <th>Std Dev</th>
                        <th>Throughput</th>
                        <th>Memory</th>
                        <th>Peak Memory</th>
                        <th>Performance</th>
                    </tr>
                </thead>
                <tbody>
                    {}
                </tbody>
            </table>
        </div>
    </div>
</div>"#,
            table_rows
        )
    }

    fn generate_system_info(&self) -> String {
        let env = &self.metadata.environment;
        format!(
            r#"<div class="card mb-4">
    <div class="card-header">
        <h5>System Information</h5>
    </div>
    <div class="card-body">
        <div class="row">
            <div class="col-md-6">
                <table class="table">
                    <tr><th>Operating System</th><td>{}</td></tr>
                    <tr><th>CPU</th><td>{}</td></tr>
                    <tr><th>Memory</th><td>{}</td></tr>
                    <tr><th>GPU</th><td>{}</td></tr>
                </table>
            </div>
            <div class="col-md-6">
                <table class="table">
                    <tr><th>Rust Version</th><td>{}</td></tr>
                    <tr><th>Compiler</th><td>{}</td></tr>
                    <tr><th>ToRSh Version</th><td>{}</td></tr>
                </table>
            </div>
        </div>
    </div>
</div>"#,
            env.os,
            env.cpu,
            env.memory,
            env.gpu.as_deref().unwrap_or("None"),
            env.rust_version,
            env.compiler,
            env.torsh_version
        )
    }

    fn generate_benchmark_config(&self) -> String {
        // Implementation for benchmark configuration display
        r#"<div class="card mb-4">
    <div class="card-header">
        <h5>Benchmark Configuration</h5>
    </div>
    <div class="card-body">
        <p>Configuration details for the benchmark suite.</p>
    </div>
</div>"#
            .to_string()
    }

    // CSS and JavaScript generation methods

    fn generate_base_css(&self) -> String {
        r#"/* Base styles for ToRSh benchmark reports */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
}

.theme-light {
    background-color: #f8f9fa;
    color: #333;
}

.theme-dark {
    background-color: #121212;
    color: #e0e0e0;
}

.card {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    border: 1px solid rgba(0, 0, 0, 0.125);
}

.table-hover tbody tr:hover {
    background-color: rgba(0, 0, 0, 0.05);
}

.navbar-brand {
    font-weight: 600;
}

.display-4 {
    font-weight: 300;
}

.progress {
    background-color: #e9ecef;
}

.badge {
    font-size: 0.75em;
}"#
        .to_string()
    }

    fn generate_theme_css(&self) -> String {
        match self.style_config.theme {
            Theme::Dark => r#"
.theme-dark .card {
    background-color: #1e1e1e;
    border-color: #333;
}

.theme-dark .table {
    color: #e0e0e0;
}

.theme-dark .table-dark {
    background-color: #333;
}

.theme-dark .bg-light {
    background-color: #2d2d2d !important;
}
"#
            .to_string(),
            _ => String::new(),
        }
    }

    fn generate_chart_css(&self) -> String {
        r#"/* Chart styling */
canvas {
    max-height: 400px;
}

.chart-container {
    position: relative;
    height: 400px;
    width: 100%;
}

.chart-legend {
    margin-top: 1rem;
}"#
        .to_string()
    }

    fn generate_responsive_css(&self) -> String {
        r#"/* Responsive design */
@media (max-width: 768px) {
    .display-4 {
        font-size: 2rem;
    }
    
    .card-body {
        padding: 1rem;
    }
    
    .table-responsive {
        font-size: 0.875rem;
    }
}

@media (max-width: 576px) {
    .container-fluid {
        padding: 0.5rem;
    }
    
    .card-header h5 {
        font-size: 1rem;
    }
}"#
        .to_string()
    }

    fn generate_chart_js(&self) -> String {
        r#"// Chart.js configuration and rendering functions
const chartColors = {
    primary: '#007bff',
    secondary: '#6c757d',
    success: '#28a745',
    warning: '#ffc107',
    danger: '#dc3545',
    info: '#17a2b8'
};

function createPerformanceOverviewChart() {
    const ctx = document.getElementById('performanceOverviewChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: benchmarkData.labels,
            datasets: [{
                label: 'Execution Time (ms)',
                data: benchmarkData.times,
                backgroundColor: chartColors.primary,
                borderColor: chartColors.primary,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (ms)'
                    }
                }
            }
        }
    });
}

function createPerformanceTrendsChart() {
    const ctx = document.getElementById('performanceTrendsChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: benchmarkData.timestamps,
            datasets: [{
                label: 'Throughput',
                data: benchmarkData.throughput,
                borderColor: chartColors.success,
                backgroundColor: chartColors.success + '20',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Throughput (ops/sec)'
                    }
                }
            }
        }
    });
}

// Initialize charts when page loads
document.addEventListener('DOMContentLoaded', function() {
    createPerformanceOverviewChart();
    createPerformanceTrendsChart();
});"#
            .to_string()
    }

    fn generate_interactive_js(&self) -> String {
        r#"// Interactive features
function toggleTheme() {
    const body = document.body;
    if (body.classList.contains('theme-light')) {
        body.className = body.className.replace('theme-light', 'theme-dark');
        localStorage.setItem('theme', 'dark');
    } else {
        body.className = body.className.replace('theme-dark', 'theme-light');
        localStorage.setItem('theme', 'light');
    }
}

function sortTable(columnIndex) {
    const table = document.getElementById('detailedResultsTable');
    const tbody = table.getElementsByTagName('tbody')[0];
    const rows = Array.from(tbody.getElementsByTagName('tr'));
    
    rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();
        
        if (!isNaN(aValue) && !isNaN(bValue)) {
            return parseFloat(aValue) - parseFloat(bValue);
        }
        return aValue.localeCompare(bValue);
    });
    
    rows.forEach(row => tbody.appendChild(row));
}

function highlightRow(row) {
    row.style.backgroundColor = '#fffbcc';
    setTimeout(() => {
        row.style.backgroundColor = '';
    }, 2000);
}"#
        .to_string()
    }

    fn generate_search_filter_js(&self) -> String {
        r#"// Search and filter functionality
function filterTable() {
    const searchInput = document.getElementById('searchInput');
    const typeFilter = document.getElementById('typeFilter');
    const sizeFilter = document.getElementById('sizeFilter');
    
    if (!searchInput) return;
    
    const searchTerm = searchInput.value.toLowerCase();
    const selectedType = typeFilter ? typeFilter.value : '';
    const selectedSize = sizeFilter ? sizeFilter.value : '';
    
    const table = document.getElementById('detailedResultsTable');
    const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
    
    for (let row of rows) {
        const cells = row.getElementsByTagName('td');
        const benchmarkName = cells[0].textContent.toLowerCase();
        const size = parseInt(cells[1].textContent);
        
        let showRow = true;
        
        // Search filter
        if (searchTerm && !benchmarkName.includes(searchTerm)) {
            showRow = false;
        }
        
        // Type filter
        if (selectedType && !benchmarkName.includes(selectedType)) {
            showRow = false;
        }
        
        // Size filter
        if (selectedSize) {
            if (selectedSize === 'small' && size >= 100) showRow = false;
            if (selectedSize === 'medium' && (size < 100 || size > 1000)) showRow = false;
            if (selectedSize === 'large' && size <= 1000) showRow = false;
        }
        
        row.style.display = showRow ? '' : 'none';
    }
}

function applyFilters() {
    filterTable();
}

// Add event listeners
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', filterTable);
    }
    
    const filters = ['typeFilter', 'sizeFilter'];
    filters.forEach(filterId => {
        const filter = document.getElementById(filterId);
        if (filter) {
            filter.addEventListener('change', filterTable);
        }
    });
});"#
            .to_string()
    }

    fn generate_export_js(&self) -> String {
        r#"// Export functionality
function exportToPDF() {
    window.print();
}

function exportToCSV() {
    const table = document.getElementById('detailedResultsTable');
    if (!table) return;
    
    let csv = [];
    const rows = table.querySelectorAll('tr');
    
    for (let row of rows) {
        const cols = row.querySelectorAll('td, th');
        const csvRow = [];
        for (let col of cols) {
            // Skip the progress bar column
            if (!col.querySelector('.progress')) {
                csvRow.push('"' + col.textContent.trim().replace(/"/g, '""') + '"');
            }
        }
        csv.push(csvRow.join(','));
    }
    
    const csvFile = new Blob([csv.join('\n')], { type: 'text/csv' });
    const downloadLink = document.createElement('a');
    downloadLink.download = 'benchmark_results.csv';
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = 'none';
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}

function exportToJSON() {
    const data = {
        timestamp: new Date().toISOString(),
        results: benchmarkData
    };
    
    const jsonFile = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const downloadLink = document.createElement('a');
    downloadLink.download = 'benchmark_results.json';
    downloadLink.href = window.URL.createObjectURL(jsonFile);
    downloadLink.style.display = 'none';
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}"#
        .to_string()
    }
}

impl Default for ReportMetadata {
    fn default() -> Self {
        Self {
            title: "ToRSh Benchmark Report".to_string(),
            description: "Comprehensive performance analysis of ToRSh tensor operations"
                .to_string(),
            version: "1.0.0".to_string(),
            timestamp: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            environment: EnvironmentInfo::default(),
            summary: ReportSummary::default(),
        }
    }
}

impl Default for EnvironmentInfo {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            cpu: "Unknown CPU".to_string(),
            memory: "Unknown Memory".to_string(),
            gpu: None,
            compiler: "rustc".to_string(),
            rust_version: "1.70.0".to_string(),
            torsh_version: "0.1.0-alpha.2".to_string(),
        }
    }
}

impl Default for ReportSummary {
    fn default() -> Self {
        Self {
            total_benchmarks: 0,
            total_execution_time: 0.0,
            fastest_benchmark: None,
            slowest_benchmark: None,
            average_performance: 0.0,
            performance_variance: 0.0,
        }
    }
}

impl Default for HtmlReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to generate a complete HTML benchmark report
pub fn generate_html_report(
    results: Vec<BenchResult>,
    configs: Vec<BenchConfig>,
    output_dir: &str,
) -> io::Result<()> {
    let mut generator = HtmlReportGenerator::new();
    generator.add_results(results);
    generator.add_configs(configs);
    generator.generate_report(output_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use torsh_core::dtype::DType;

    #[test]
    fn test_html_report_generator_creation() {
        let generator = HtmlReportGenerator::new();
        assert_eq!(generator.results.len(), 0);
        assert_eq!(generator.configs.len(), 0);
    }

    #[test]
    fn test_add_results() {
        let mut generator = HtmlReportGenerator::new();
        let results = vec![BenchResult {
            name: "test_benchmark".to_string(),
            size: 64,
            dtype: DType::F32,
            mean_time_ns: 1_000_000.0,
            std_dev_ns: 100_000.0,
            throughput: Some(1000.0),
            memory_usage: Some(1024),
            peak_memory: Some(2048),
            metrics: HashMap::new(),
        }];

        generator.add_results(results);
        assert_eq!(generator.results.len(), 1);
    }

    #[test]
    fn test_style_config_default() {
        let style_config = StyleConfig::default();
        assert!(matches!(style_config.theme, Theme::Light));
        assert!(style_config.chart_style.animation);
        assert!(style_config.layout.sidebar);
    }

    #[test]
    fn test_color_scheme_default() {
        let color_scheme = ColorScheme::default();
        assert_eq!(color_scheme.primary, "#007bff");
        assert_eq!(color_scheme.background, "#ffffff");
    }

    #[test]
    fn test_report_metadata_default() {
        let metadata = ReportMetadata::default();
        assert_eq!(metadata.title, "ToRSh Benchmark Report");
        assert_eq!(metadata.version, "1.0.0");
    }
}
