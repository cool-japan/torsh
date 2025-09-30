//! Network-Aware Adaptive Gradient Compression
//!
//! This module implements intelligent gradient compression that adapts to real-time
//! network conditions, bandwidth, latency, and congestion patterns. It provides
//! optimal compression ratios based on network performance metrics and training
//! convergence requirements.

use crate::gradient_compression::{CompressionConfig, CompressionMethod, GradientCompressor};
use crate::gradient_compression_enhanced::{CompressionMetrics, EnhancedGradientCompressor};
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use torsh_tensor::Tensor;
use tracing::{debug, info, warn};

/// Network performance metrics for adaptive compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Current bandwidth in MB/s
    pub bandwidth_mbps: f32,
    /// Round-trip latency in milliseconds
    pub latency_ms: f32,
    /// Packet loss percentage (0.0 to 1.0)
    pub packet_loss: f32,
    /// Network congestion factor (0.0 to 1.0)
    pub congestion_factor: f32,
    /// Network stability score (0.0 to 1.0)
    pub stability_score: f32,
    /// Timestamp of measurement (milliseconds since epoch)
    pub timestamp_ms: u64,
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            bandwidth_mbps: 1000.0, // 1 Gbps default
            latency_ms: 1.0,
            packet_loss: 0.0,
            congestion_factor: 0.0,
            stability_score: 1.0,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
}

impl NetworkMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Adaptive compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCompressionConfig {
    /// Target bandwidth utilization (0.0 to 1.0)
    pub target_bandwidth_utilization: f32,
    /// Minimum compression ratio (0.0 to 1.0)
    pub min_compression_ratio: f32,
    /// Maximum compression ratio (0.0 to 1.0)
    pub max_compression_ratio: f32,
    /// Network monitoring interval
    pub monitoring_interval: Duration,
    /// History window size for metrics
    pub history_window_size: usize,
    /// Adaptation sensitivity (0.0 to 1.0)
    pub adaptation_sensitivity: f32,
    /// Convergence quality weight (0.0 to 1.0)
    pub convergence_quality_weight: f32,
    /// Communication efficiency weight (0.0 to 1.0)
    pub communication_efficiency_weight: f32,
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            target_bandwidth_utilization: 0.8,
            min_compression_ratio: 0.01,
            max_compression_ratio: 0.9,
            monitoring_interval: Duration::from_millis(100),
            history_window_size: 50,
            adaptation_sensitivity: 0.3,
            convergence_quality_weight: 0.6,
            communication_efficiency_weight: 0.4,
        }
    }
}

/// Network bandwidth profiler
#[derive(Debug)]
pub struct NetworkProfiler {
    /// Network metrics history
    metrics_history: VecDeque<NetworkMetrics>,
    /// Bandwidth measurement samples
    bandwidth_samples: VecDeque<f32>,
    /// Latency measurement samples
    latency_samples: VecDeque<f32>,
    /// Last measurement time
    last_measurement: Instant,
    /// Configuration
    config: AdaptiveCompressionConfig,
}

impl NetworkProfiler {
    pub fn new(config: AdaptiveCompressionConfig) -> Self {
        Self {
            metrics_history: VecDeque::with_capacity(config.history_window_size),
            bandwidth_samples: VecDeque::with_capacity(config.history_window_size),
            latency_samples: VecDeque::with_capacity(config.history_window_size),
            last_measurement: Instant::now(),
            config,
        }
    }

    /// Measure current network performance
    pub fn measure_network_performance(&mut self) -> TorshResult<NetworkMetrics> {
        let now = Instant::now();

        // Simulate realistic network measurements
        // In production, this would interface with actual network monitoring tools
        let bandwidth = self.estimate_bandwidth()?;
        let latency = self.estimate_latency()?;
        let packet_loss = self.estimate_packet_loss()?;
        let congestion = self.estimate_congestion_factor(bandwidth, latency)?;
        let stability = self.calculate_stability_score()?;

        let metrics = NetworkMetrics {
            bandwidth_mbps: bandwidth,
            latency_ms: latency,
            packet_loss,
            congestion_factor: congestion,
            stability_score: stability,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        // Update history
        self.metrics_history.push_back(metrics.clone());
        if self.metrics_history.len() > self.config.history_window_size {
            self.metrics_history.pop_front();
        }

        self.bandwidth_samples.push_back(bandwidth);
        if self.bandwidth_samples.len() > self.config.history_window_size {
            self.bandwidth_samples.pop_front();
        }

        self.latency_samples.push_back(latency);
        if self.latency_samples.len() > self.config.history_window_size {
            self.latency_samples.pop_front();
        }

        self.last_measurement = now;
        Ok(metrics)
    }

    /// Estimate current bandwidth (MB/s)
    fn estimate_bandwidth(&self) -> TorshResult<f32> {
        // In production, this would measure actual data transfer rates
        // For now, simulate realistic bandwidth with some variation
        let base_bandwidth = 1000.0; // 1 Gbps
        let variation = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            % 100) as f32
            / 100.0;
        let bandwidth = base_bandwidth * (0.8 + 0.4 * variation);
        Ok(bandwidth.max(10.0)) // Minimum 10 MB/s
    }

    /// Estimate current latency (ms)
    fn estimate_latency(&self) -> TorshResult<f32> {
        // In production, this would measure actual round-trip times
        let base_latency = 1.0; // 1ms base latency
        let variation = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            % 50) as f32
            / 50.0;
        let latency = base_latency * (0.5 + variation);
        Ok(latency.max(0.1)) // Minimum 0.1ms
    }

    /// Estimate packet loss percentage
    fn estimate_packet_loss(&self) -> TorshResult<f32> {
        // In production, this would monitor actual packet loss
        let base_loss = 0.001; // 0.1% base loss
        let variation = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            % 20) as f32
            / 20.0;
        Ok(base_loss * variation)
    }

    /// Estimate network congestion factor
    fn estimate_congestion_factor(&self, bandwidth: f32, latency: f32) -> TorshResult<f32> {
        // Congestion increases with latency and decreases with bandwidth
        let normalized_latency = (latency / 10.0).min(1.0); // Normalize to [0,1]
        let normalized_bandwidth = (1000.0 / bandwidth.max(100.0)).min(1.0); // Inverse normalized
        let congestion = (normalized_latency + normalized_bandwidth) / 2.0;
        Ok(congestion.min(1.0))
    }

    /// Calculate network stability score
    fn calculate_stability_score(&self) -> TorshResult<f32> {
        if self.bandwidth_samples.len() < 5 {
            return Ok(1.0); // Assume stable with insufficient data
        }

        // Calculate coefficient of variation for bandwidth
        let mean_bandwidth: f32 =
            self.bandwidth_samples.iter().sum::<f32>() / self.bandwidth_samples.len() as f32;
        let variance: f32 = self
            .bandwidth_samples
            .iter()
            .map(|&x| (x - mean_bandwidth).powi(2))
            .sum::<f32>()
            / self.bandwidth_samples.len() as f32;
        let std_dev = variance.sqrt();
        let cv_bandwidth = if mean_bandwidth > 0.0 {
            std_dev / mean_bandwidth
        } else {
            0.0
        };

        // Calculate coefficient of variation for latency
        let mean_latency: f32 =
            self.latency_samples.iter().sum::<f32>() / self.latency_samples.len() as f32;
        let latency_variance: f32 = self
            .latency_samples
            .iter()
            .map(|&x| (x - mean_latency).powi(2))
            .sum::<f32>()
            / self.latency_samples.len() as f32;
        let latency_std_dev = latency_variance.sqrt();
        let cv_latency = if mean_latency > 0.0 {
            latency_std_dev / mean_latency
        } else {
            0.0
        };

        // Stability score: lower coefficient of variation = higher stability
        let stability = 1.0 - ((cv_bandwidth + cv_latency) / 2.0).min(1.0);
        Ok(stability.max(0.0))
    }

    /// Get average metrics over recent history
    pub fn get_average_metrics(&self) -> Option<NetworkMetrics> {
        if self.metrics_history.is_empty() {
            return None;
        }

        let count = self.metrics_history.len() as f32;
        let avg_bandwidth = self
            .metrics_history
            .iter()
            .map(|m| m.bandwidth_mbps)
            .sum::<f32>()
            / count;
        let avg_latency = self
            .metrics_history
            .iter()
            .map(|m| m.latency_ms)
            .sum::<f32>()
            / count;
        let avg_packet_loss = self
            .metrics_history
            .iter()
            .map(|m| m.packet_loss)
            .sum::<f32>()
            / count;
        let avg_congestion = self
            .metrics_history
            .iter()
            .map(|m| m.congestion_factor)
            .sum::<f32>()
            / count;
        let avg_stability = self
            .metrics_history
            .iter()
            .map(|m| m.stability_score)
            .sum::<f32>()
            / count;

        Some(NetworkMetrics {
            bandwidth_mbps: avg_bandwidth,
            latency_ms: avg_latency,
            packet_loss: avg_packet_loss,
            congestion_factor: avg_congestion,
            stability_score: avg_stability,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }
}

/// Network-aware adaptive gradient compressor
pub struct NetworkAwareCompressor {
    /// Enhanced gradient compressor
    enhanced_compressor: EnhancedGradientCompressor,
    /// Network profiler
    network_profiler: Arc<Mutex<NetworkProfiler>>,
    /// Current compression configuration
    current_config: Arc<Mutex<CompressionConfig>>,
    /// Adaptive configuration
    adaptive_config: AdaptiveCompressionConfig,
    /// Compression performance history
    compression_history: Arc<Mutex<VecDeque<CompressionMetrics>>>,
    /// Training convergence metrics
    convergence_tracker: Arc<Mutex<ConvergenceTracker>>,
}

/// Training convergence tracking
#[derive(Debug)]
struct ConvergenceTracker {
    /// Loss history
    loss_history: VecDeque<f32>,
    /// Gradient norm history
    gradient_norm_history: VecDeque<f32>,
    /// Learning rate history
    learning_rate_history: VecDeque<f32>,
    /// Convergence rate estimate
    convergence_rate: f32,
}

impl ConvergenceTracker {
    fn new(history_size: usize) -> Self {
        Self {
            loss_history: VecDeque::with_capacity(history_size),
            gradient_norm_history: VecDeque::with_capacity(history_size),
            learning_rate_history: VecDeque::with_capacity(history_size),
            convergence_rate: 0.0,
        }
    }

    fn update_convergence_metrics(&mut self, loss: f32, gradient_norm: f32, learning_rate: f32) {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 100 {
            self.loss_history.pop_front();
        }

        self.gradient_norm_history.push_back(gradient_norm);
        if self.gradient_norm_history.len() > 100 {
            self.gradient_norm_history.pop_front();
        }

        self.learning_rate_history.push_back(learning_rate);
        if self.learning_rate_history.len() > 100 {
            self.learning_rate_history.pop_front();
        }

        self.estimate_convergence_rate();
    }

    fn estimate_convergence_rate(&mut self) {
        if self.loss_history.len() < 10 {
            self.convergence_rate = 0.5; // Neutral estimate
            return;
        }

        // Calculate loss improvement rate
        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let old_losses: Vec<f32> = if self.loss_history.len() >= 20 {
            self.loss_history
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .cloned()
                .collect()
        } else {
            recent_losses.clone()
        };

        let recent_avg: f32 = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let old_avg: f32 = old_losses.iter().sum::<f32>() / old_losses.len() as f32;

        // Convergence rate based on loss improvement
        if old_avg > recent_avg && old_avg > 0.0 {
            let improvement_rate = (old_avg - recent_avg) / old_avg;
            self.convergence_rate = improvement_rate.min(1.0).max(0.0);
        } else {
            self.convergence_rate = 0.1; // Slow convergence if loss not improving
        }
    }

    fn get_convergence_quality(&self) -> f32 {
        // Higher convergence rate indicates better training progress
        // Adjust compression more conservatively for better convergence
        self.convergence_rate
    }
}

impl NetworkAwareCompressor {
    /// Create new network-aware compressor
    pub fn new(
        base_config: CompressionConfig,
        adaptive_config: AdaptiveCompressionConfig,
    ) -> TorshResult<Self> {
        let enhanced_compressor = EnhancedGradientCompressor::new(base_config.clone())?;
        let network_profiler = Arc::new(Mutex::new(NetworkProfiler::new(adaptive_config.clone())));
        let current_config = Arc::new(Mutex::new(base_config));
        let compression_history = Arc::new(Mutex::new(VecDeque::with_capacity(
            adaptive_config.history_window_size,
        )));
        let convergence_tracker = Arc::new(Mutex::new(ConvergenceTracker::new(
            adaptive_config.history_window_size,
        )));

        Ok(Self {
            enhanced_compressor,
            network_profiler,
            current_config,
            adaptive_config,
            compression_history,
            convergence_tracker,
        })
    }

    /// Compress gradient with network-aware adaptation
    pub fn compress_gradient_adaptive(
        &mut self,
        gradient: &Tensor,
        training_metrics: Option<TrainingMetrics>,
    ) -> TorshResult<(
        crate::gradient_compression::CompressedGradient,
        CompressionMetrics,
    )> {
        // Update convergence tracking if metrics provided
        if let Some(metrics) = training_metrics {
            let mut tracker = self.convergence_tracker.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "convergence_tracker",
                    format!("Lock error: {}", e),
                )
            })?;
            tracker.update_convergence_metrics(
                metrics.loss,
                metrics.gradient_norm,
                metrics.learning_rate,
            );
        }

        // Measure network performance
        let network_metrics = {
            let mut profiler = self.network_profiler.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "network_profiler",
                    format!("Lock error: {}", e),
                )
            })?;
            profiler.measure_network_performance()?
        };

        // Adapt compression configuration
        let optimal_config = self.calculate_optimal_compression_config(&network_metrics)?;

        // Update current configuration
        {
            let mut config = self.current_config.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "current_config",
                    format!("Lock error: {}", e),
                )
            })?;
            *config = optimal_config.clone();
        }

        // Update enhanced compressor configuration by creating a new one
        // Since config field is private, we need to create a new compressor
        self.enhanced_compressor = EnhancedGradientCompressor::new(optimal_config.clone())?;

        // Perform compression
        let start_time = Instant::now();
        let (compressed_gradient, mut metrics) = self
            .enhanced_compressor
            .compress_gradient_enhanced(gradient, "adaptive_gradient")?;
        let compression_time = start_time.elapsed();

        // Create enhanced metrics with network awareness
        // Since the original metrics struct doesn't have network fields, we keep the existing metrics
        // and add network information to the log messages

        // Store compression metrics history
        {
            let mut history = self.compression_history.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "compression_history",
                    format!("Lock error: {}", e),
                )
            })?;
            history.push_back(metrics.clone());
            if history.len() > self.adaptive_config.history_window_size {
                history.pop_front();
            }
        }

        info!(
            "Network-aware compression: ratio={:.3}, bandwidth={:.1}MB/s, latency={:.2}ms, stability={:.3}, throughput={:.1}MB/s",
            metrics.compression_ratio,
            network_metrics.bandwidth_mbps,
            network_metrics.latency_ms,
            network_metrics.stability_score,
            metrics.throughput_mbps
        );

        Ok((compressed_gradient, metrics))
    }

    /// Calculate optimal compression configuration based on network conditions
    fn calculate_optimal_compression_config(
        &self,
        network_metrics: &NetworkMetrics,
    ) -> TorshResult<CompressionConfig> {
        let convergence_quality = {
            let tracker = self.convergence_tracker.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "convergence_tracker",
                    format!("Lock error: {}", e),
                )
            })?;
            tracker.get_convergence_quality()
        };

        // Calculate optimal compression ratio
        let optimal_ratio =
            self.calculate_optimal_compression_ratio(network_metrics, convergence_quality)?;

        // Select optimal compression method
        let optimal_method =
            self.select_optimal_compression_method(network_metrics, optimal_ratio)?;

        // Create optimized configuration
        let base_config = self.current_config.lock().map_err(|e| {
            TorshDistributedError::communication_error(
                "current_config",
                format!("Lock error: {}", e),
            )
        })?;

        let mut optimal_config = base_config.clone();
        optimal_config.compression_ratio = optimal_ratio;
        optimal_config.method = optimal_method.clone();

        // Adjust error feedback based on network stability
        optimal_config.error_feedback = network_metrics.stability_score > 0.7;
        optimal_config.error_feedback_momentum = if network_metrics.stability_score > 0.8 {
            0.9 // Higher momentum for stable networks
        } else {
            0.7 // Lower momentum for unstable networks
        };

        debug!(
            "Optimal compression config: ratio={:.3}, method={:?}, error_feedback={}",
            optimal_ratio, optimal_method, optimal_config.error_feedback
        );

        Ok(optimal_config)
    }

    /// Calculate optimal compression ratio based on network and convergence conditions
    fn calculate_optimal_compression_ratio(
        &self,
        network_metrics: &NetworkMetrics,
        convergence_quality: f32,
    ) -> TorshResult<f32> {
        // Base compression ratio from bandwidth utilization
        let target_bandwidth =
            self.adaptive_config.target_bandwidth_utilization * network_metrics.bandwidth_mbps;
        let bandwidth_pressure = (1000.0 / target_bandwidth).min(1.0); // Normalized pressure

        // Adjust for latency
        let latency_factor = (network_metrics.latency_ms / 10.0).min(1.0); // Normalize to reasonable range

        // Adjust for packet loss
        let loss_factor = network_metrics.packet_loss * 10.0; // Amplify packet loss impact

        // Adjust for congestion
        let congestion_factor = network_metrics.congestion_factor;

        // Combine network factors
        let network_pressure =
            (bandwidth_pressure + latency_factor + loss_factor + congestion_factor) / 4.0;

        // Balance between network efficiency and convergence quality
        let efficiency_weight = self.adaptive_config.communication_efficiency_weight;
        let convergence_weight = self.adaptive_config.convergence_quality_weight;

        // Higher compression for poor network, lower compression for good convergence
        let network_compression_ratio = network_pressure * 0.8; // Scale to reasonable compression range
        let convergence_compression_ratio = (1.0 - convergence_quality) * 0.5; // Less compression for good convergence

        let optimal_ratio = efficiency_weight * network_compression_ratio
            + convergence_weight * convergence_compression_ratio;

        // Clamp to configured bounds
        let clamped_ratio = optimal_ratio
            .max(self.adaptive_config.min_compression_ratio)
            .min(self.adaptive_config.max_compression_ratio);

        debug!(
            "Compression ratio calculation: network_pressure={:.3}, convergence_quality={:.3}, optimal={:.3}",
            network_pressure, convergence_quality, clamped_ratio
        );

        Ok(clamped_ratio)
    }

    /// Select optimal compression method based on network characteristics
    fn select_optimal_compression_method(
        &self,
        network_metrics: &NetworkMetrics,
        compression_ratio: f32,
    ) -> TorshResult<CompressionMethod> {
        // Choose method based on network characteristics and compression ratio
        if compression_ratio < 0.1 {
            // Very high compression needed
            if network_metrics.stability_score > 0.8 {
                Ok(CompressionMethod::Quantization { bits: 4 }) // Aggressive quantization for stable networks
            } else {
                Ok(CompressionMethod::SignSGD) // Simple sign compression for unstable networks
            }
        } else if compression_ratio < 0.3 {
            // High compression needed
            if network_metrics.latency_ms < 2.0 {
                Ok(CompressionMethod::TopK {
                    k: compression_ratio,
                }) // TopK for low latency
            } else {
                Ok(CompressionMethod::Quantization { bits: 8 }) // Quantization for higher latency
            }
        } else if compression_ratio < 0.7 {
            // Moderate compression
            if network_metrics.bandwidth_mbps > 500.0 {
                Ok(CompressionMethod::TopK {
                    k: compression_ratio,
                }) // TopK for high bandwidth
            } else {
                Ok(CompressionMethod::Threshold { threshold: 0.01 }) // Threshold for limited bandwidth
            }
        } else {
            // Low compression needed
            if network_metrics.stability_score > 0.9 {
                Ok(CompressionMethod::RandomK {
                    k: compression_ratio,
                }) // Random for very stable networks
            } else {
                Ok(CompressionMethod::TopK {
                    k: compression_ratio,
                }) // TopK as safe default
            }
        }
    }

    /// Get current network performance metrics
    pub fn get_network_metrics(&self) -> TorshResult<Option<NetworkMetrics>> {
        let profiler = self.network_profiler.lock().map_err(|e| {
            TorshDistributedError::communication_error(
                "network_profiler",
                format!("Lock error: {}", e),
            )
        })?;
        Ok(profiler.get_average_metrics())
    }

    /// Get compression performance statistics
    pub fn get_compression_statistics(&self) -> TorshResult<CompressionStatistics> {
        let history = self.compression_history.lock().map_err(|e| {
            TorshDistributedError::communication_error(
                "compression_history",
                format!("Lock error: {}", e),
            )
        })?;

        if history.is_empty() {
            return Ok(CompressionStatistics::default());
        }

        let count = history.len() as f32;
        let avg_ratio = history.iter().map(|m| m.compression_ratio).sum::<f32>() / count;
        let avg_time_us =
            history.iter().map(|m| m.compression_time_us).sum::<u64>() / history.len() as u64;
        let avg_throughput = history.iter().map(|m| m.throughput_mbps).sum::<f32>() / count;
        let avg_error = history.iter().map(|m| m.compression_error).sum::<f32>() / count;

        Ok(CompressionStatistics {
            average_compression_ratio: avg_ratio,
            average_compression_time_us: avg_time_us,
            average_throughput_mbps: avg_throughput,
            average_compression_error: avg_error,
            total_compressions: history.len(),
        })
    }

    /// Update adaptive configuration
    pub fn update_adaptive_config(&mut self, config: AdaptiveCompressionConfig) -> TorshResult<()> {
        self.adaptive_config = config.clone();

        // Update network profiler configuration
        let mut profiler = self.network_profiler.lock().map_err(|e| {
            TorshDistributedError::communication_error(
                "network_profiler",
                format!("Lock error: {}", e),
            )
        })?;
        profiler.config = config;

        Ok(())
    }
}

/// Training metrics for convergence tracking
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub gradient_norm: f32,
    pub learning_rate: f32,
}

/// Compression performance statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStatistics {
    pub average_compression_ratio: f32,
    pub average_compression_time_us: u64,
    pub average_throughput_mbps: f32,
    pub average_compression_error: f32,
    pub total_compressions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[tokio::test]
    async fn test_network_profiler() -> TorshResult<()> {
        let config = AdaptiveCompressionConfig::default();
        let mut profiler = NetworkProfiler::new(config);

        let metrics = profiler.measure_network_performance()?;
        assert!(metrics.bandwidth_mbps > 0.0);
        assert!(metrics.latency_ms > 0.0);
        assert!(metrics.packet_loss >= 0.0);
        assert!(metrics.congestion_factor >= 0.0 && metrics.congestion_factor <= 1.0);
        assert!(metrics.stability_score >= 0.0 && metrics.stability_score <= 1.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_network_aware_compression() -> TorshResult<()> {
        let base_config = CompressionConfig::default();
        let adaptive_config = AdaptiveCompressionConfig::default();
        let mut compressor = NetworkAwareCompressor::new(base_config, adaptive_config)?;

        let gradient = ones::<f32>(&[1000, 1000])?;
        let training_metrics = TrainingMetrics {
            loss: 0.5,
            gradient_norm: 1.0,
            learning_rate: 0.001,
        };

        let (compressed, metrics) =
            compressor.compress_gradient_adaptive(&gradient, Some(training_metrics))?;

        assert!(metrics.compression_ratio > 0.0);
        assert!(metrics.compression_ratio <= 1.0);
        assert!(compressed.original_shape == vec![1000, 1000]);

        Ok(())
    }

    #[tokio::test]
    async fn test_adaptive_compression_ratio_calculation() -> TorshResult<()> {
        let base_config = CompressionConfig::default();
        let adaptive_config = AdaptiveCompressionConfig::default();
        let compressor = NetworkAwareCompressor::new(base_config, adaptive_config)?;

        // Test with different network conditions
        let high_bandwidth_metrics = NetworkMetrics {
            bandwidth_mbps: 2000.0,
            latency_ms: 0.5,
            packet_loss: 0.0,
            congestion_factor: 0.1,
            stability_score: 0.95,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        let low_bandwidth_metrics = NetworkMetrics {
            bandwidth_mbps: 100.0,
            latency_ms: 5.0,
            packet_loss: 0.01,
            congestion_factor: 0.8,
            stability_score: 0.6,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        let high_bandwidth_ratio =
            compressor.calculate_optimal_compression_ratio(&high_bandwidth_metrics, 0.8)?;
        let low_bandwidth_ratio =
            compressor.calculate_optimal_compression_ratio(&low_bandwidth_metrics, 0.8)?;

        // Lower bandwidth should result in higher compression
        assert!(low_bandwidth_ratio > high_bandwidth_ratio);

        Ok(())
    }

    #[tokio::test]
    async fn test_compression_method_selection() -> TorshResult<()> {
        let base_config = CompressionConfig::default();
        let adaptive_config = AdaptiveCompressionConfig::default();
        let compressor = NetworkAwareCompressor::new(base_config, adaptive_config)?;

        let stable_network = NetworkMetrics {
            bandwidth_mbps: 1000.0,
            latency_ms: 1.0,
            packet_loss: 0.0,
            congestion_factor: 0.1,
            stability_score: 0.95,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        // Test different compression ratios
        let high_compression_method =
            compressor.select_optimal_compression_method(&stable_network, 0.05)?;
        let moderate_compression_method =
            compressor.select_optimal_compression_method(&stable_network, 0.5)?;

        // Should select appropriate methods for different compression ratios
        match high_compression_method {
            CompressionMethod::Quantization { bits: 4 } | CompressionMethod::SignSGD => {}
            _ => panic!("Unexpected method for high compression"),
        }

        match moderate_compression_method {
            CompressionMethod::TopK { .. } | CompressionMethod::Threshold { .. } => {}
            _ => panic!("Unexpected method for moderate compression"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_convergence_tracking() -> TorshResult<()> {
        let mut tracker = ConvergenceTracker::new(50);

        // Simulate improving training
        for i in 0..20 {
            let loss = 1.0 - (i as f32 * 0.05); // Decreasing loss
            let gradient_norm = 1.0;
            let learning_rate = 0.001;
            tracker.update_convergence_metrics(loss, gradient_norm, learning_rate);
        }

        let quality = tracker.get_convergence_quality();
        assert!(quality > 0.0);
        assert!(quality <= 1.0);

        Ok(())
    }
}
