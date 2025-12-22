//! # Advanced Gradient Magnitude Analysis
//!
//! This module provides comprehensive statistical analysis of gradient magnitudes
//! with detailed per-layer statistics, global metrics, and advanced correlation
//! analysis. It supports real-time monitoring and historical analysis of gradient
//! patterns for optimization and debugging.
//!
//! ## Key Components
//!
//! - **DetailedGradientStats**: Comprehensive gradient statistics with per-layer breakdown
//! - **GradientMagnitudeAnalyzer**: Main analysis engine with advanced statistical methods
//! - **LayerGradientStats**: Detailed statistics for individual layers or parameters
//! - **GlobalGradientStats**: Cross-layer and global gradient metrics
//! - **Historical analysis**: Time-series tracking and trend analysis
//!
//! ## Statistical Features
//!
//! - **Per-layer analysis**: Individual layer statistics with norms and distributions
//! - **Global metrics**: Cross-layer correlations, effective rank, coherence measures
//! - **Histogram analysis**: Gradient magnitude distributions and outlier detection
//! - **Time-series tracking**: Historical trends and magnitude evolution
//! - **Signal-to-noise analysis**: Quality metrics for gradient information
//! - **Correlation analysis**: Cross-layer gradient relationships
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use torsh_autograd::visualization::magnitude_analysis::{
//!     GradientMagnitudeAnalyzer, AnalyzerConfig
//! };
//! use std::collections::HashMap;
//!
//! // Create analyzer with custom configuration
//! let config = AnalyzerConfig {
//!     histogram_bins: 50,
//!     compute_correlations: true,
//!     track_timeline: true,
//!     ..AnalyzerConfig::default()
//! };
//! let mut analyzer = GradientMagnitudeAnalyzer::<f32>::new(config);
//!
//! // Start analysis session
//! analyzer.start_session("training_run_1".to_string());
//!
//! // Analyze gradients (example with mock data)
//! let mut gradients = HashMap::new();
//! gradients.insert("layer1".to_string(), vec![0.001, 0.002, 0.0015]);
//! gradients.insert("layer2".to_string(), vec![0.003, 0.0025, 0.004]);
//!
//! let stats = analyzer.analyze_gradients(&gradients, Some(100))?;
//! println!("Global L2 norm: {}", stats.global_stats.global_l2_norm);
//!
//! // Generate detailed report
//! let report = analyzer.generate_report(&stats)?;
//! println!("{}", report);
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::HashMap;
use std::time::{Instant, SystemTime};
use torsh_core::dtype::FloatElement;
use torsh_core::error::Result;
use tracing::{debug, info};

/// Comprehensive gradient magnitude statistics with detailed breakdowns
///
/// This struct provides complete statistical analysis of gradient magnitudes
/// across all layers and parameters, including distributions, correlations,
/// and quality metrics.
#[derive(Debug, Clone)]
pub struct DetailedGradientStats<T: FloatElement> {
    /// Per-layer gradient statistics
    pub layer_stats: HashMap<String, LayerGradientStats<T>>,
    /// Global gradient statistics across all layers
    pub global_stats: GlobalGradientStats<T>,
    /// Histogram of gradient magnitudes
    pub gradient_histogram: GradientHistogram<T>,
    /// Time series data for magnitude tracking
    pub magnitude_timeline: Vec<MagnitudeTimePoint<T>>,
    /// Distribution of gradient norms
    pub norm_distribution: NormDistribution<T>,
}

/// Detailed statistics for individual layers or parameter groups
///
/// Provides comprehensive analysis of gradient characteristics for a specific
/// layer, including magnitude statistics, norms, and quality metrics.
#[derive(Debug, Clone)]
pub struct LayerGradientStats<T: FloatElement> {
    /// Layer or parameter group name
    pub layer_name: String,
    /// Total number of parameters in this layer
    pub parameter_count: usize,
    /// Mean gradient magnitude across all parameters
    pub mean_magnitude: T,
    /// Standard deviation of gradient magnitudes
    pub std_deviation: T,
    /// Maximum gradient magnitude in this layer
    pub max_magnitude: T,
    /// Minimum gradient magnitude in this layer
    pub min_magnitude: T,
    /// L1 norm of gradients
    pub l1_norm: T,
    /// L2 norm of gradients
    pub l2_norm: T,
    /// Infinity norm (maximum absolute value)
    pub inf_norm: T,
    /// Percentage of gradients that are exactly zero
    pub zero_percentage: f32,
    /// Percentage of NaN gradients
    pub nan_percentage: f32,
    /// Percentage of infinite gradients
    pub inf_percentage: f32,
    /// Gradient sparsity (percentage of near-zero values)
    pub sparsity: f32,
    /// Signal-to-noise ratio for this layer
    pub signal_to_noise_ratio: T,
}

/// Global gradient statistics across all parameters and layers
///
/// Provides system-wide gradient analysis including cross-layer correlations,
/// effective dimensionality, and overall gradient quality metrics.
#[derive(Debug, Clone)]
pub struct GlobalGradientStats<T: FloatElement> {
    /// Total number of parameters across all layers
    pub total_parameters: usize,
    /// Global mean magnitude across all gradients
    pub global_mean_magnitude: T,
    /// Global L2 norm of all gradients
    pub global_l2_norm: T,
    /// Effective rank of the gradient space
    pub effective_rank: f32,
    /// Gradient coherence measure (0-1, higher is more coherent)
    pub coherence: f32,
    /// Overall signal-to-noise ratio
    pub overall_snr: T,
    /// Cross-layer correlation coefficient
    pub cross_layer_correlation: f32,
    /// Gradient alignment score (measure of optimization direction consistency)
    pub alignment_score: f32,
}

/// Histogram representation of gradient magnitude distributions
///
/// Provides detailed distribution analysis of gradient magnitudes for
/// understanding gradient patterns and detecting outliers.
#[derive(Debug, Clone)]
pub struct GradientHistogram<T: FloatElement> {
    /// Histogram bin boundaries (lower bounds)
    pub bins: Vec<T>,
    /// Count of values in each bin
    pub counts: Vec<usize>,
    /// Total number of values in the histogram
    pub total_count: usize,
    /// Number of histogram bins
    pub num_bins: usize,
}

/// Time-series data point for gradient magnitude tracking
///
/// Represents a single measurement in the historical timeline of
/// gradient magnitude evolution during training.
#[derive(Debug, Clone)]
pub struct MagnitudeTimePoint<T: FloatElement> {
    /// Timestamp when this measurement was taken
    pub timestamp: SystemTime,
    /// Global L2 norm at this time point
    pub global_l2_norm: T,
    /// Mean magnitude across all parameters
    pub mean_magnitude: T,
    /// Maximum magnitude across all parameters
    pub max_magnitude: T,
    /// Training step or iteration number
    pub step: usize,
}

/// Distribution statistics for gradient norms
///
/// Provides statistical analysis of gradient norm distributions including
/// percentiles, moments, and shape characteristics.
#[derive(Debug, Clone)]
pub struct NormDistribution<T: FloatElement> {
    /// 25th percentile of gradient norms
    pub p25: T,
    /// 50th percentile (median) of gradient norms
    pub p50: T,
    /// 75th percentile of gradient norms
    pub p75: T,
    /// 90th percentile of gradient norms
    pub p90: T,
    /// 95th percentile of gradient norms
    pub p95: T,
    /// 99th percentile of gradient norms
    pub p99: T,
    /// Skewness of the distribution
    pub skewness: f32,
    /// Kurtosis of the distribution
    pub kurtosis: f32,
}

/// Advanced gradient magnitude analyzer with statistical capabilities
///
/// The main analysis engine that provides comprehensive gradient magnitude
/// analysis including historical tracking, correlation analysis, and
/// advanced statistical metrics.
#[derive(Debug)]
pub struct GradientMagnitudeAnalyzer<T: FloatElement> {
    /// Configuration for analysis behavior
    config: AnalyzerConfig,
    /// Historical gradient statistics
    historical_stats: Vec<DetailedGradientStats<T>>,
    /// Current analysis session information
    current_session: Option<AnalysisSession>,
    /// Cached computation results for performance
    cache: AnalysisCache<T>,
}

/// Configuration for gradient magnitude analysis
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Number of bins for histogram analysis
    pub histogram_bins: usize,
    /// Whether to compute cross-layer correlations
    pub compute_correlations: bool,
    /// Whether to track magnitude timeline
    pub track_timeline: bool,
    /// Maximum number of historical statistics to keep
    pub max_history_size: usize,
    /// Threshold for considering gradients as zero
    pub zero_threshold: f32,
    /// Threshold for sparsity calculation
    pub sparsity_threshold: f32,
    /// Whether to enable caching for performance
    pub enable_caching: bool,
    /// Number of percentiles to compute for distributions
    pub num_percentiles: usize,
}

/// Information about the current analysis session
#[derive(Debug, Clone)]
struct AnalysisSession {
    /// Session start time
    start_time: Instant,
    /// Unique session identifier
    session_id: String,
    /// Current step counter
    step_counter: usize,
}

/// Cache for expensive computations
#[derive(Debug, Default)]
struct AnalysisCache<T: FloatElement> {
    /// Cached correlation matrix
    correlation_matrix: Option<Vec<Vec<f32>>>,
    /// Cached SVD results for effective rank computation
    svd_cache: HashMap<String, (Vec<T>, Vec<T>, Vec<T>)>,
    /// Last cache update time
    last_update: Option<Instant>,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            histogram_bins: 50,
            compute_correlations: true,
            track_timeline: true,
            max_history_size: 1000,
            zero_threshold: 1e-8,
            sparsity_threshold: 1e-6,
            enable_caching: true,
            num_percentiles: 6, // p25, p50, p75, p90, p95, p99
        }
    }
}

impl<T: FloatElement + num_traits::FromPrimitive + std::default::Default + Clone>
    GradientMagnitudeAnalyzer<T>
{
    /// Create a new gradient magnitude analyzer with the given configuration
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            config,
            historical_stats: Vec::new(),
            current_session: None,
            cache: AnalysisCache::default(),
        }
    }

    /// Start a new analysis session with a unique identifier
    pub fn start_session(&mut self, session_id: String) {
        info!(
            "Starting gradient magnitude analysis session: {}",
            session_id
        );
        self.current_session = Some(AnalysisSession {
            start_time: Instant::now(),
            session_id,
            step_counter: 0,
        });
    }

    /// Analyze gradient magnitudes from a collection of layer gradients
    ///
    /// Performs comprehensive statistical analysis of the provided gradients,
    /// computing per-layer statistics, global metrics, and updating historical data.
    ///
    /// # Arguments
    ///
    /// * `gradients` - HashMap mapping layer names to gradient vectors
    /// * `step` - Optional step number for timeline tracking
    ///
    /// # Returns
    ///
    /// Detailed gradient statistics including all computed metrics
    pub fn analyze_gradients(
        &mut self,
        gradients: &HashMap<String, Vec<T>>,
        step: Option<usize>,
    ) -> Result<DetailedGradientStats<T>> {
        let start_time = Instant::now();

        // Update step counter
        if let Some(session) = &mut self.current_session {
            if let Some(s) = step {
                session.step_counter = s;
            } else {
                session.step_counter += 1;
            }
        }

        // Compute per-layer statistics
        let layer_stats = self.compute_layer_stats(gradients)?;

        // Compute global statistics
        let global_stats = self.compute_global_stats(gradients, &layer_stats)?;

        // Compute gradient histogram
        let gradient_histogram = self.compute_gradient_histogram(gradients)?;

        // Update magnitude timeline
        let magnitude_timeline = if self.config.track_timeline {
            self.update_magnitude_timeline(&global_stats, step.unwrap_or(0))
        } else {
            Vec::new()
        };

        // Compute norm distribution
        let norm_distribution = self.compute_norm_distribution(gradients)?;

        let stats = DetailedGradientStats {
            layer_stats,
            global_stats,
            gradient_histogram,
            magnitude_timeline,
            norm_distribution,
        };

        // Store in history
        self.historical_stats.push(stats.clone());
        while self.historical_stats.len() > self.config.max_history_size {
            self.historical_stats.remove(0);
        }

        let elapsed = start_time.elapsed();
        debug!("Gradient analysis completed in {:?}", elapsed);

        Ok(stats)
    }

    /// Generate a comprehensive report from gradient statistics
    pub fn generate_report(&self, stats: &DetailedGradientStats<T>) -> Result<String> {
        let mut report = String::new();

        report.push_str("╔════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║                Gradient Magnitude Analysis Report              ║\n");
        report.push_str("╚════════════════════════════════════════════════════════════════╝\n\n");

        // Session information
        if let Some(session) = &self.current_session {
            report.push_str(&format!(
                "Session: {} (Step: {})\n",
                session.session_id, session.step_counter
            ));
            report.push_str(&format!("Runtime: {:?}\n\n", session.start_time.elapsed()));
        }

        // Global statistics
        report.push_str("📊 Global Statistics\n");
        report.push_str(&format!(
            "  Total Parameters: {}\n",
            stats.global_stats.total_parameters
        ));
        report.push_str(&format!(
            "  Global Mean Magnitude: {:.6e}\n",
            torsh_core::TensorElement::to_f64(&stats.global_stats.global_mean_magnitude)
                .unwrap_or(0.0)
        ));
        report.push_str(&format!(
            "  Global L2 Norm: {:.6e}\n",
            torsh_core::TensorElement::to_f64(&stats.global_stats.global_l2_norm).unwrap_or(0.0)
        ));
        report.push_str(&format!(
            "  Effective Rank: {:.2}\n",
            stats.global_stats.effective_rank
        ));
        report.push_str(&format!(
            "  Coherence: {:.3}\n",
            stats.global_stats.coherence
        ));
        report.push_str(&format!(
            "  Cross-layer Correlation: {:.3}\n",
            stats.global_stats.cross_layer_correlation
        ));
        report.push_str(&format!(
            "  Alignment Score: {:.3}\n",
            stats.global_stats.alignment_score
        ));
        report.push_str("\n");

        // Per-layer statistics
        report.push_str("🏗️  Layer Statistics\n");
        for (layer_name, layer_stat) in &stats.layer_stats {
            report.push_str(&format!("  {}:\n", layer_name));
            report.push_str(&format!("    Parameters: {}\n", layer_stat.parameter_count));
            report.push_str(&format!(
                "    Mean Magnitude: {:.6e}\n",
                torsh_core::TensorElement::to_f64(&layer_stat.mean_magnitude).unwrap_or(0.0)
            ));
            report.push_str(&format!(
                "    L2 Norm: {:.6e}\n",
                torsh_core::TensorElement::to_f64(&layer_stat.l2_norm).unwrap_or(0.0)
            ));
            report.push_str(&format!("    Sparsity: {:.1}%\n", layer_stat.sparsity));

            // Quality indicators
            if layer_stat.nan_percentage > 0.0 {
                report.push_str(&format!(
                    "    ⚠️  NaN Gradients: {:.1}%\n",
                    layer_stat.nan_percentage
                ));
            }
            if layer_stat.inf_percentage > 0.0 {
                report.push_str(&format!(
                    "    ⚠️  Infinite Gradients: {:.1}%\n",
                    layer_stat.inf_percentage
                ));
            }

            report.push_str("\n");
        }

        // Distribution analysis
        report.push_str("📈 Gradient Distribution\n");
        report.push_str(&format!(
            "  Histogram Bins: {}\n",
            stats.gradient_histogram.num_bins
        ));
        report.push_str(&format!(
            "  Total Values: {}\n",
            stats.gradient_histogram.total_count
        ));
        report.push_str("  Percentiles:\n");
        report.push_str(&format!(
            "    25th: {:.6e}\n",
            torsh_core::TensorElement::to_f64(&stats.norm_distribution.p25).unwrap_or(0.0)
        ));
        report.push_str(&format!(
            "    50th (Median): {:.6e}\n",
            torsh_core::TensorElement::to_f64(&stats.norm_distribution.p50).unwrap_or(0.0)
        ));
        report.push_str(&format!(
            "    75th: {:.6e}\n",
            torsh_core::TensorElement::to_f64(&stats.norm_distribution.p75).unwrap_or(0.0)
        ));
        report.push_str(&format!(
            "    95th: {:.6e}\n",
            torsh_core::TensorElement::to_f64(&stats.norm_distribution.p95).unwrap_or(0.0)
        ));
        report.push_str(&format!(
            "    99th: {:.6e}\n",
            torsh_core::TensorElement::to_f64(&stats.norm_distribution.p99).unwrap_or(0.0)
        ));
        report.push_str(&format!(
            "  Skewness: {:.3}\n",
            stats.norm_distribution.skewness
        ));
        report.push_str(&format!(
            "  Kurtosis: {:.3}\n",
            stats.norm_distribution.kurtosis
        ));
        report.push_str("\n");

        // Historical analysis
        if self.historical_stats.len() > 1 {
            report.push_str("📜 Historical Analysis\n");
            report.push_str(&format!("  Data Points: {}\n", self.historical_stats.len()));

            if let Some(comparison) = self.get_historical_comparison() {
                report.push_str(&format!("  Trend: {}\n", comparison.trend_description));
                report.push_str(&format!(
                    "  Change: {:.1}%\n",
                    comparison.magnitude_change_percent
                ));
            }
        }

        Ok(report)
    }

    /// Get historical comparison with trend analysis
    pub fn get_historical_comparison(&self) -> Option<HistoricalComparison> {
        if self.historical_stats.len() < 2 {
            return None;
        }

        let latest = &self.historical_stats[self.historical_stats.len() - 1];
        let previous = &self.historical_stats[self.historical_stats.len() - 2];

        let latest_magnitude =
            torsh_core::TensorElement::to_f64(&latest.global_stats.global_mean_magnitude)
                .unwrap_or(0.0);
        let previous_magnitude =
            torsh_core::TensorElement::to_f64(&previous.global_stats.global_mean_magnitude)
                .unwrap_or(0.0);

        let change_percent = if previous_magnitude > 0.0 {
            ((latest_magnitude - previous_magnitude) / previous_magnitude) * 100.0
        } else {
            0.0
        };

        let trend_description = match change_percent {
            x if x > 10.0 => "Strongly Increasing",
            x if x > 2.0 => "Increasing",
            x if x > -2.0 => "Stable",
            x if x > -10.0 => "Decreasing",
            _ => "Strongly Decreasing",
        }
        .to_string();

        Some(HistoricalComparison {
            trend_description,
            magnitude_change_percent: change_percent,
            latest_magnitude,
            previous_magnitude,
        })
    }

    /// Clear all historical data and reset the analyzer
    pub fn clear_history(&mut self) {
        self.historical_stats.clear();
        self.cache = AnalysisCache::default();
        info!("Gradient magnitude analyzer history cleared");
    }

    /// Get the number of historical analysis results stored
    pub fn history_size(&self) -> usize {
        self.historical_stats.len()
    }

    // Private helper methods

    /// Compute detailed statistics for each layer
    fn compute_layer_stats(
        &self,
        gradients: &HashMap<String, Vec<T>>,
    ) -> Result<HashMap<String, LayerGradientStats<T>>> {
        let mut layer_stats = HashMap::new();

        for (layer_name, grad_values) in gradients {
            if grad_values.is_empty() {
                continue;
            }

            let parameter_count = grad_values.len();

            // Basic statistics
            let mean_magnitude = self.compute_mean_magnitude(grad_values);
            let std_deviation = self.compute_std_deviation(grad_values, mean_magnitude);
            let max_magnitude = grad_values
                .iter()
                .max_by(|a, b| {
                    torsh_core::TensorElement::to_f64(*a)
                        .unwrap_or(0.0)
                        .partial_cmp(&torsh_core::TensorElement::to_f64(*b).unwrap_or(0.0))
                        .unwrap()
                })
                .copied()
                .unwrap_or_default();
            let min_magnitude = grad_values
                .iter()
                .min_by(|a, b| {
                    torsh_core::TensorElement::to_f64(*a)
                        .unwrap_or(0.0)
                        .partial_cmp(&torsh_core::TensorElement::to_f64(*b).unwrap_or(0.0))
                        .unwrap()
                })
                .copied()
                .unwrap_or_default();

            // Norms
            let l1_norm = self.compute_l1_norm(grad_values);
            let l2_norm = self.compute_l2_norm(grad_values);
            let inf_norm = max_magnitude;

            // Quality metrics
            let zero_count = grad_values
                .iter()
                .filter(|&&x| {
                    torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0).abs()
                        < self.config.zero_threshold as f64
                })
                .count();
            let zero_percentage = (zero_count as f32 / parameter_count as f32) * 100.0;

            let nan_count = grad_values
                .iter()
                .filter(|&&x| {
                    torsh_core::TensorElement::to_f64(&x)
                        .unwrap_or(0.0)
                        .is_nan()
                })
                .count();
            let nan_percentage = (nan_count as f32 / parameter_count as f32) * 100.0;

            let inf_count = grad_values
                .iter()
                .filter(|&&x| {
                    torsh_core::TensorElement::to_f64(&x)
                        .unwrap_or(0.0)
                        .is_infinite()
                })
                .count();
            let inf_percentage = (inf_count as f32 / parameter_count as f32) * 100.0;

            let sparse_count = grad_values
                .iter()
                .filter(|&&x| {
                    torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0).abs()
                        < self.config.sparsity_threshold as f64
                })
                .count();
            let sparsity = (sparse_count as f32 / parameter_count as f32) * 100.0;

            // Signal-to-noise ratio (simplified)
            let signal_to_noise_ratio =
                if torsh_core::TensorElement::to_f64(&std_deviation).unwrap_or(0.0) > 0.0 {
                    <T as torsh_core::TensorElement>::from_f64(
                        torsh_core::TensorElement::to_f64(&mean_magnitude).unwrap_or(0.0)
                            / torsh_core::TensorElement::to_f64(&std_deviation).unwrap_or(1.0),
                    )
                    .unwrap_or_default()
                } else {
                    <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default()
                };

            let stats = LayerGradientStats {
                layer_name: layer_name.clone(),
                parameter_count,
                mean_magnitude,
                std_deviation,
                max_magnitude,
                min_magnitude,
                l1_norm,
                l2_norm,
                inf_norm,
                zero_percentage,
                nan_percentage,
                inf_percentage,
                sparsity,
                signal_to_noise_ratio,
            };

            layer_stats.insert(layer_name.clone(), stats);
        }

        Ok(layer_stats)
    }

    /// Compute global statistics across all layers
    fn compute_global_stats(
        &self,
        gradients: &HashMap<String, Vec<T>>,
        layer_stats: &HashMap<String, LayerGradientStats<T>>,
    ) -> Result<GlobalGradientStats<T>> {
        // Total parameters
        let total_parameters = gradients.values().map(|v| v.len()).sum();

        // Global mean magnitude (weighted by parameter count)
        let total_magnitude_sum: f64 = layer_stats
            .values()
            .map(|stats| {
                torsh_core::TensorElement::to_f64(&stats.mean_magnitude).unwrap_or(0.0)
                    * stats.parameter_count as f64
            })
            .sum();
        let global_mean_magnitude = <T as torsh_core::TensorElement>::from_f64(
            total_magnitude_sum / total_parameters as f64,
        )
        .unwrap_or_default();

        // Global L2 norm
        let global_l2_squared = gradients
            .values()
            .flat_map(|v| v.iter())
            .map(|&x| {
                let val = torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0);
                val * val
            })
            .sum::<f64>();
        let global_l2_norm = <T as torsh_core::TensorElement>::from_f64(global_l2_squared.sqrt())
            .unwrap_or_default();

        // Effective rank (simplified approximation)
        let layer_norms: Vec<f64> = layer_stats
            .values()
            .map(|stats| torsh_core::TensorElement::to_f64(&stats.l2_norm).unwrap_or(0.0))
            .collect();
        let effective_rank = self.compute_effective_rank(&layer_norms);

        // Cross-layer correlation
        let cross_layer_correlation = if self.config.compute_correlations {
            self.compute_cross_layer_correlation(layer_stats)
        } else {
            0.0
        };

        // Overall signal-to-noise ratio
        let overall_snr = {
            let mean_snr = layer_stats
                .values()
                .map(|stats| {
                    torsh_core::TensorElement::to_f64(&stats.signal_to_noise_ratio).unwrap_or(0.0)
                })
                .filter(|&x| x.is_finite())
                .collect::<Vec<_>>();
            if mean_snr.is_empty() {
                <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default()
            } else {
                let avg_snr = mean_snr.iter().sum::<f64>() / mean_snr.len() as f64;
                <T as torsh_core::TensorElement>::from_f64(avg_snr).unwrap_or_default()
            }
        };

        // Gradient alignment score (coherence measure)
        let alignment_score = self.compute_gradient_alignment(gradients);

        Ok(GlobalGradientStats {
            total_parameters,
            global_mean_magnitude,
            global_l2_norm,
            effective_rank,
            coherence: cross_layer_correlation, // Using correlation as coherence measure
            overall_snr,
            cross_layer_correlation,
            alignment_score,
        })
    }

    /// Compute gradient histogram
    fn compute_gradient_histogram(
        &self,
        gradients: &HashMap<String, Vec<T>>,
    ) -> Result<GradientHistogram<T>> {
        let all_values: Vec<T> = gradients.values().flat_map(|v| v.iter()).copied().collect();

        if all_values.is_empty() {
            return Ok(GradientHistogram {
                bins: Vec::new(),
                counts: Vec::new(),
                total_count: 0,
                num_bins: 0,
            });
        }

        let num_bins = self.config.histogram_bins;
        let min_val = all_values
            .iter()
            .min_by(|a, b| {
                torsh_core::TensorElement::to_f64(*a)
                    .unwrap_or(0.0)
                    .partial_cmp(&torsh_core::TensorElement::to_f64(*b).unwrap_or(0.0))
                    .unwrap()
            })
            .copied()
            .unwrap_or_default();
        let max_val = all_values
            .iter()
            .max_by(|a, b| {
                torsh_core::TensorElement::to_f64(*a)
                    .unwrap_or(0.0)
                    .partial_cmp(&torsh_core::TensorElement::to_f64(*b).unwrap_or(0.0))
                    .unwrap()
            })
            .copied()
            .unwrap_or_default();

        let min_f64 = torsh_core::TensorElement::to_f64(&min_val).unwrap_or(0.0);
        let max_f64 = torsh_core::TensorElement::to_f64(&max_val).unwrap_or(0.0);
        let range = max_f64 - min_f64;

        let mut bins = Vec::new();
        let mut counts = vec![0; num_bins];

        // Create bin boundaries
        for i in 0..num_bins {
            let bin_start = min_f64 + (i as f64 * range / num_bins as f64);
            bins.push(<T as torsh_core::TensorElement>::from_f64(bin_start).unwrap_or_default());
        }

        // Count values in each bin
        for &value in &all_values {
            let val_f64 = torsh_core::TensorElement::to_f64(&value).unwrap_or(0.0);
            if range > 0.0 {
                let bin_index = ((val_f64 - min_f64) / range * num_bins as f64) as usize;
                let bin_index = bin_index.min(num_bins - 1);
                counts[bin_index] += 1;
            } else {
                counts[0] += 1;
            }
        }

        Ok(GradientHistogram {
            bins,
            counts,
            total_count: all_values.len(),
            num_bins,
        })
    }

    /// Update magnitude timeline with new data point
    fn update_magnitude_timeline(
        &mut self,
        global_stats: &GlobalGradientStats<T>,
        step: usize,
    ) -> Vec<MagnitudeTimePoint<T>> {
        let mut timeline = self
            .historical_stats
            .iter()
            .map(|stats| &stats.magnitude_timeline)
            .flatten()
            .cloned()
            .collect::<Vec<_>>();

        let new_point = MagnitudeTimePoint {
            timestamp: SystemTime::now(),
            global_l2_norm: global_stats.global_l2_norm,
            mean_magnitude: global_stats.global_mean_magnitude,
            max_magnitude: <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default(), // Would compute actual max
            step,
        };

        timeline.push(new_point);

        // Keep only recent points to manage memory
        if timeline.len() > 1000 {
            timeline.drain(0..timeline.len() - 1000);
        }

        timeline
    }

    /// Compute norm distribution statistics
    fn compute_norm_distribution(
        &self,
        gradients: &HashMap<String, Vec<T>>,
    ) -> Result<NormDistribution<T>> {
        let mut all_values: Vec<f64> = gradients
            .values()
            .flat_map(|v| v.iter())
            .map(|&x| torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0).abs())
            .collect();

        if all_values.is_empty() {
            return Ok(NormDistribution {
                p25: <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default(),
                p50: <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default(),
                p75: <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default(),
                p90: <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default(),
                p95: <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default(),
                p99: <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default(),
                skewness: 0.0,
                kurtosis: 0.0,
            });
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = |p: f64| -> f64 {
            let index = (p * (all_values.len() - 1) as f64) as usize;
            all_values[index.min(all_values.len() - 1)]
        };

        let p25 = <T as torsh_core::TensorElement>::from_f64(percentile(0.25)).unwrap_or_default();
        let p50 = <T as torsh_core::TensorElement>::from_f64(percentile(0.50)).unwrap_or_default();
        let p75 = <T as torsh_core::TensorElement>::from_f64(percentile(0.75)).unwrap_or_default();
        let p90 = <T as torsh_core::TensorElement>::from_f64(percentile(0.90)).unwrap_or_default();
        let p95 = <T as torsh_core::TensorElement>::from_f64(percentile(0.95)).unwrap_or_default();
        let p99 = <T as torsh_core::TensorElement>::from_f64(percentile(0.99)).unwrap_or_default();

        // Compute moments for skewness and kurtosis
        let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
        let variance =
            all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / all_values.len() as f64;
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 0.0 {
            all_values
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / all_values.len() as f64
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            all_values
                .iter()
                .map(|&x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / all_values.len() as f64
                - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        Ok(NormDistribution {
            p25,
            p50,
            p75,
            p90,
            p95,
            p99,
            skewness: skewness as f32,
            kurtosis: kurtosis as f32,
        })
    }

    // Statistical computation helpers

    fn compute_mean_magnitude(&self, values: &[T]) -> T {
        if values.is_empty() {
            return <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default();
        }

        let sum = values
            .iter()
            .map(|&x| torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0).abs())
            .sum::<f64>();
        <T as torsh_core::TensorElement>::from_f64(sum / values.len() as f64).unwrap_or_default()
    }

    fn compute_std_deviation(&self, values: &[T], mean: T) -> T {
        if values.len() <= 1 {
            return <T as torsh_core::TensorElement>::from_f64(0.0).unwrap_or_default();
        }

        let mean_f64 = torsh_core::TensorElement::to_f64(&mean).unwrap_or(0.0);
        let variance = values
            .iter()
            .map(|&x| {
                let val = torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0).abs();
                (val - mean_f64).powi(2)
            })
            .sum::<f64>()
            / (values.len() - 1) as f64;

        <T as torsh_core::TensorElement>::from_f64(variance.sqrt()).unwrap_or_default()
    }

    fn compute_l1_norm(&self, values: &[T]) -> T {
        let sum = values
            .iter()
            .map(|&x| torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0).abs())
            .sum::<f64>();
        <T as torsh_core::TensorElement>::from_f64(sum).unwrap_or_default()
    }

    fn compute_l2_norm(&self, values: &[T]) -> T {
        let sum_squares = values
            .iter()
            .map(|&x| torsh_core::TensorElement::to_f64(&x).unwrap_or(0.0).powi(2))
            .sum::<f64>();
        <T as torsh_core::TensorElement>::from_f64(sum_squares.sqrt()).unwrap_or_default()
    }

    fn compute_effective_rank(&self, norms: &[f64]) -> f32 {
        if norms.is_empty() {
            return 0.0;
        }

        // Simple effective rank approximation
        let total_norm: f64 = norms.iter().sum();
        if total_norm == 0.0 {
            return 0.0;
        }

        let normalized_norms: Vec<f64> = norms.iter().map(|&x| x / total_norm).collect();

        // Shannon entropy as approximation for effective rank
        let entropy: f64 = normalized_norms
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| -x * x.ln())
            .sum();

        entropy.exp() as f32
    }

    fn compute_cross_layer_correlation(
        &self,
        layer_stats: &HashMap<String, LayerGradientStats<T>>,
    ) -> f32 {
        if layer_stats.len() < 2 {
            return 0.0;
        }

        // Simplified correlation using L2 norms
        let norms: Vec<f64> = layer_stats
            .values()
            .map(|stats| torsh_core::TensorElement::to_f64(&stats.l2_norm).unwrap_or(0.0))
            .collect();

        if norms.len() < 2 {
            return 0.0;
        }

        let mean = norms.iter().sum::<f64>() / norms.len() as f64;
        let variance = norms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / norms.len() as f64;

        if variance == 0.0 {
            return 1.0; // Perfect correlation if all values are the same
        }

        // Return coefficient of variation as a proxy for correlation
        (variance.sqrt() / mean.abs()).min(1.0) as f32
    }

    fn compute_gradient_alignment(&self, gradients: &HashMap<String, Vec<T>>) -> f32 {
        // Simplified alignment score based on consistency of gradient directions
        let mut total_magnitude = 0.0;
        let mut aligned_magnitude = 0.0;

        for values in gradients.values() {
            for &value in values {
                let magnitude = torsh_core::TensorElement::to_f64(&value)
                    .unwrap_or(0.0)
                    .abs();
                total_magnitude += magnitude;

                // Simple alignment check - gradients pointing in same direction
                if torsh_core::TensorElement::to_f64(&value).unwrap_or(0.0) > 0.0 {
                    aligned_magnitude += magnitude;
                }
            }
        }

        if total_magnitude > 0.0 {
            (aligned_magnitude / total_magnitude) as f32
        } else {
            0.0
        }
    }
}

/// Historical comparison result
#[derive(Debug, Clone)]
pub struct HistoricalComparison {
    /// Description of the trend
    pub trend_description: String,
    /// Percentage change in magnitude
    pub magnitude_change_percent: f64,
    /// Latest magnitude value
    pub latest_magnitude: f64,
    /// Previous magnitude value
    pub previous_magnitude: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let config = AnalyzerConfig::default();
        let analyzer = GradientMagnitudeAnalyzer::<f32>::new(config);
        assert_eq!(analyzer.history_size(), 0);
    }

    #[test]
    fn test_session_management() {
        let mut analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());
        analyzer.start_session("test_session".to_string());
        assert!(analyzer.current_session.is_some());
    }

    #[test]
    fn test_gradient_analysis() {
        let mut analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());

        let mut gradients = HashMap::new();
        gradients.insert("layer1".to_string(), vec![0.001, 0.002, 0.0015]);
        gradients.insert("layer2".to_string(), vec![0.003, 0.0025, 0.004]);

        let result = analyzer.analyze_gradients(&gradients, Some(1));
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert_eq!(stats.layer_stats.len(), 2);
        assert_eq!(stats.global_stats.total_parameters, 6);
    }

    #[test]
    fn test_histogram_computation() {
        let analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());

        let mut gradients = HashMap::new();
        gradients.insert("test".to_string(), vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let histogram = analyzer.compute_gradient_histogram(&gradients).unwrap();
        assert_eq!(histogram.total_count, 5);
        assert!(histogram.num_bins > 0);
    }

    #[test]
    fn test_layer_statistics() {
        let analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());

        let mut gradients = HashMap::new();
        gradients.insert(
            "test_layer".to_string(),
            vec![0.0, 0.001, 0.002, 0.0, 0.003],
        );

        let layer_stats = analyzer.compute_layer_stats(&gradients).unwrap();
        let stats = &layer_stats["test_layer"];

        assert_eq!(stats.parameter_count, 5);
        assert_eq!(stats.zero_percentage, 40.0); // 2 out of 5 are zero
        assert!(torsh_core::TensorElement::to_f64(&stats.mean_magnitude).unwrap() > 0.0);
    }

    #[test]
    fn test_norm_distribution() {
        let analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());

        let mut gradients = HashMap::new();
        gradients.insert(
            "test".to_string(),
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        );

        let distribution = analyzer.compute_norm_distribution(&gradients).unwrap();

        // Check that percentiles are in ascending order
        assert!(
            torsh_core::TensorElement::to_f64(&distribution.p25).unwrap()
                <= torsh_core::TensorElement::to_f64(&distribution.p50).unwrap()
        );
        assert!(
            torsh_core::TensorElement::to_f64(&distribution.p50).unwrap()
                <= torsh_core::TensorElement::to_f64(&distribution.p75).unwrap()
        );
        assert!(
            torsh_core::TensorElement::to_f64(&distribution.p75).unwrap()
                <= torsh_core::TensorElement::to_f64(&distribution.p95).unwrap()
        );
    }

    #[test]
    fn test_effective_rank_computation() {
        let analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());

        // Test with uniform norms
        let uniform_norms = vec![1.0, 1.0, 1.0, 1.0];
        let rank = analyzer.compute_effective_rank(&uniform_norms);
        assert!(rank > 0.0);

        // Test with single dominant norm
        let dominated_norms = vec![10.0, 0.1, 0.1, 0.1];
        let dominated_rank = analyzer.compute_effective_rank(&dominated_norms);
        assert!(dominated_rank < rank); // Should have lower effective rank
    }

    #[test]
    fn test_statistical_helpers() {
        let analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mean = analyzer.compute_mean_magnitude(&values);
        assert!((torsh_core::TensorElement::to_f64(&mean).unwrap() - 3.0).abs() < 1e-6);

        let l1_norm = analyzer.compute_l1_norm(&values);
        assert!((torsh_core::TensorElement::to_f64(&l1_norm).unwrap() - 15.0).abs() < 1e-6);

        let l2_norm = analyzer.compute_l2_norm(&values);
        assert!(
            (torsh_core::TensorElement::to_f64(&l2_norm).unwrap() - (55.0_f64).sqrt()).abs() < 1e-6
        );
    }

    #[test]
    fn test_clear_history() {
        let mut analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());

        // Add some data
        let mut gradients = HashMap::new();
        gradients.insert("test".to_string(), vec![0.1, 0.2]);
        analyzer.analyze_gradients(&gradients, None).unwrap();

        assert_eq!(analyzer.history_size(), 1);

        analyzer.clear_history();
        assert_eq!(analyzer.history_size(), 0);
    }

    #[test]
    fn test_historical_comparison() {
        let mut analyzer = GradientMagnitudeAnalyzer::<f32>::new(AnalyzerConfig::default());

        let mut gradients1 = HashMap::new();
        gradients1.insert("test".to_string(), vec![0.001, 0.002]);
        analyzer.analyze_gradients(&gradients1, None).unwrap();

        let mut gradients2 = HashMap::new();
        gradients2.insert("test".to_string(), vec![0.002, 0.004]); // Double the magnitude
        analyzer.analyze_gradients(&gradients2, None).unwrap();

        let comparison = analyzer.get_historical_comparison();
        assert!(comparison.is_some());

        let comp = comparison.unwrap();
        assert!(comp.magnitude_change_percent > 0.0); // Should show increase
    }
}

// Default implementations for structs used in tests
impl<T: FloatElement + num_traits::FromPrimitive + Default> Default for GlobalGradientStats<T> {
    fn default() -> Self {
        Self {
            total_parameters: 0,
            global_mean_magnitude: T::default(),
            global_l2_norm: T::default(),
            effective_rank: 0.0,
            coherence: 0.0,
            overall_snr: T::default(),
            cross_layer_correlation: 0.0,
            alignment_score: 0.0,
        }
    }
}

impl<T: FloatElement + num_traits::FromPrimitive + Default> Default for GradientHistogram<T> {
    fn default() -> Self {
        Self {
            bins: Vec::new(),
            counts: Vec::new(),
            total_count: 0,
            num_bins: 0,
        }
    }
}

impl<T: FloatElement + num_traits::FromPrimitive + Default> Default for NormDistribution<T> {
    fn default() -> Self {
        Self {
            p25: T::default(),
            p50: T::default(),
            p75: T::default(),
            p90: T::default(),
            p95: T::default(),
            p99: T::default(),
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}
