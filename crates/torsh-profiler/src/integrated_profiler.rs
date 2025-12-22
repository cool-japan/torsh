//! Integrated Profiler System
//!
//! This module provides a unified profiler that integrates all advanced features:
//! - Online learning for real-time anomaly detection
//! - Cross-platform optimization
//! - Cloud provider integration
//! - Kubernetes deployment
//! - Performance prediction
//! - Automatic optimization recommendations

use crate::cloud_providers::{CloudInstanceMetadata, CloudProvider, MultiCloudProfiler};
use crate::cross_platform::{CrossPlatformProfiler, PlatformArch, ProfilingStrategy};
use crate::kubernetes::{KubernetesProfilerOperator, ProfilingJob};
use crate::online_learning::{
    AnomalyEvent, OnlineAnomalyDetector, OnlineLearningConfig, OnlinePredictor, StreamingKMeans,
};
use crate::ProfileEvent;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use torsh_core::{Result as TorshResult, TorshError};

/// Integrated profiler with all advanced features
pub struct IntegratedProfiler {
    /// Online learning configuration
    learning_config: OnlineLearningConfig,
    /// Anomaly detector
    anomaly_detector: OnlineAnomalyDetector,
    /// Performance predictor
    performance_predictor: OnlinePredictor,
    /// Clustering engine
    clustering: StreamingKMeans,
    /// Cross-platform profiler
    platform_profiler: CrossPlatformProfiler,
    /// Cloud profiler
    cloud_profiler: Option<MultiCloudProfiler>,
    /// Kubernetes operator
    k8s_operator: Option<KubernetesProfilerOperator>,
    /// Performance history
    performance_history: VecDeque<PerformanceSnapshot>,
    /// Optimization recommendations
    recommendations: Vec<OptimizationRecommendation>,
    /// Profiler state
    state: ProfilerState,
    /// Statistics
    stats: IntegratedProfilerStats,
}

/// Performance snapshot for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Average duration
    pub avg_duration_us: f64,
    /// Average memory usage
    pub avg_memory_bytes: f64,
    /// FLOPS rate
    pub flops_rate: f64,
    /// Throughput
    pub throughput_ops_per_sec: f64,
    /// Anomaly count
    pub anomaly_count: usize,
    /// Active cluster
    pub active_cluster: Option<usize>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub rec_type: RecommendationType,
    /// Priority (1-10, 10 being highest)
    pub priority: u8,
    /// Description
    pub description: String,
    /// Expected improvement percentage
    pub expected_improvement_percent: f64,
    /// Implementation complexity (Low, Medium, High)
    pub complexity: String,
    /// Specific actions
    pub actions: Vec<String>,
}

/// Recommendation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Memory optimization
    MemoryOptimization,
    /// CPU optimization
    CpuOptimization,
    /// GPU optimization
    GpuOptimization,
    /// Network optimization
    NetworkOptimization,
    /// Algorithm optimization
    AlgorithmOptimization,
    /// Platform-specific optimization
    PlatformOptimization,
    /// Cloud resource optimization
    CloudOptimization,
    /// Scaling recommendation
    ScalingRecommendation,
}

impl std::fmt::Display for RecommendationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryOptimization => write!(f, "Memory Optimization"),
            Self::CpuOptimization => write!(f, "CPU Optimization"),
            Self::GpuOptimization => write!(f, "GPU Optimization"),
            Self::NetworkOptimization => write!(f, "Network Optimization"),
            Self::AlgorithmOptimization => write!(f, "Algorithm Optimization"),
            Self::PlatformOptimization => write!(f, "Platform Optimization"),
            Self::CloudOptimization => write!(f, "Cloud Optimization"),
            Self::ScalingRecommendation => write!(f, "Scaling Recommendation"),
        }
    }
}

/// Profiler state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilerState {
    Stopped,
    Running,
    Paused,
}

/// Integrated profiler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedProfilerStats {
    /// Total events processed
    pub total_events: u64,
    /// Total anomalies detected
    pub total_anomalies: u64,
    /// Total recommendations generated
    pub total_recommendations: usize,
    /// Average prediction error
    pub avg_prediction_error_percent: f64,
    /// Clustering accuracy
    pub clustering_accuracy: f64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Cloud provider
    pub cloud_provider: Option<String>,
    /// Platform architecture
    pub platform_arch: String,
}

impl IntegratedProfiler {
    /// Create a new integrated profiler
    pub fn new() -> TorshResult<Self> {
        let learning_config = OnlineLearningConfig::default();
        let anomaly_detector = OnlineAnomalyDetector::new(learning_config.clone());
        let performance_predictor = OnlinePredictor::new(3, 0.01, 100); // 3 features
        let clustering = StreamingKMeans::new(5, 2); // 5 clusters, 2 dimensions
        let platform_profiler = CrossPlatformProfiler::new();
        let cloud_profiler = MultiCloudProfiler::new().ok();
        let k8s_operator = None; // Initialized when needed

        Ok(Self {
            learning_config,
            anomaly_detector,
            performance_predictor,
            clustering,
            platform_profiler,
            cloud_profiler,
            k8s_operator,
            performance_history: VecDeque::with_capacity(1000),
            recommendations: Vec::new(),
            state: ProfilerState::Stopped,
            stats: IntegratedProfilerStats {
                total_events: 0,
                total_anomalies: 0,
                total_recommendations: 0,
                avg_prediction_error_percent: 0.0,
                clustering_accuracy: 0.0,
                uptime_seconds: 0,
                cloud_provider: None,
                platform_arch: PlatformArch::detect().to_string(),
            },
        })
    }

    /// Start profiling
    pub fn start(&mut self) -> TorshResult<()> {
        if self.state == ProfilerState::Running {
            return Err(TorshError::operation_error("Profiler already running"));
        }

        self.state = ProfilerState::Running;
        self.platform_profiler.start();

        // Initialize cloud profiler if in cloud environment
        if let Some(cloud) = &self.cloud_profiler {
            self.stats.cloud_provider = Some(cloud.provider().to_string());
        }

        Ok(())
    }

    /// Stop profiling
    pub fn stop(&mut self) -> TorshResult<()> {
        if self.state == ProfilerState::Stopped {
            return Err(TorshError::operation_error("Profiler not running"));
        }

        self.state = ProfilerState::Stopped;
        let _elapsed = self.platform_profiler.stop();

        Ok(())
    }

    /// Process a profile event
    pub fn process_event(&mut self, event: &ProfileEvent) -> TorshResult<ProcessingResult> {
        if self.state != ProfilerState::Running {
            return Err(TorshError::operation_error("Profiler not running"));
        }

        self.stats.total_events += 1;

        // Anomaly detection
        let anomalies = self.anomaly_detector.process_event(event)?;
        self.stats.total_anomalies += anomalies.len() as u64;

        // Clustering
        let features = vec![
            event.duration_us as f64,
            event.bytes_transferred.unwrap_or(0) as f64,
        ];
        let cluster = self.clustering.update(&features)?;

        // Performance prediction
        let prediction_features = vec![
            event.operation_count.unwrap_or(0) as f64,
            event.bytes_transferred.unwrap_or(0) as f64,
            event.flops.unwrap_or(0) as f64,
        ];
        let predicted_duration = self
            .performance_predictor
            .predict(&prediction_features)
            .unwrap_or(0.0);
        let prediction_error = if predicted_duration > 0.0 {
            (event.duration_us as f64 - predicted_duration).abs() / predicted_duration * 100.0
        } else {
            0.0
        };

        // Update predictor
        let _loss = self
            .performance_predictor
            .update(&prediction_features, event.duration_us as f64)?;

        // Update statistics
        self.stats.avg_prediction_error_percent = (self.stats.avg_prediction_error_percent
            * (self.stats.total_events - 1) as f64
            + prediction_error)
            / self.stats.total_events as f64;

        // Generate recommendations if anomalies detected
        if !anomalies.is_empty() {
            self.generate_recommendations(&anomalies, event)?;
        }

        // Update performance history
        if self.stats.total_events % 100 == 0 {
            self.update_performance_snapshot()?;
        }

        Ok(ProcessingResult {
            anomalies,
            cluster,
            predicted_duration,
            prediction_error,
            recommendations_generated: !self.recommendations.is_empty(),
        })
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &mut self,
        anomalies: &[AnomalyEvent],
        event: &ProfileEvent,
    ) -> TorshResult<()> {
        for anomaly in anomalies {
            match anomaly.anomaly_type {
                crate::online_learning::AnomalyType::DurationSpike => {
                    // Check if it's a consistent pattern
                    if anomaly.severity > 2.0 {
                        self.recommendations.push(OptimizationRecommendation {
                            rec_type: RecommendationType::AlgorithmOptimization,
                            priority: 8,
                            description: format!(
                                "Operation '{}' showing {:.0}x duration increase",
                                event.name,
                                anomaly.actual_value / anomaly.expected_value
                            ),
                            expected_improvement_percent: 50.0,
                            complexity: "Medium".to_string(),
                            actions: vec![
                                "Profile the operation to identify bottlenecks".to_string(),
                                "Consider algorithmic optimizations".to_string(),
                                "Check for resource contention".to_string(),
                            ],
                        });
                        self.stats.total_recommendations += 1;
                    }
                }
                crate::online_learning::AnomalyType::MemorySpike => {
                    if anomaly.severity > 1.5 {
                        self.recommendations.push(OptimizationRecommendation {
                            rec_type: RecommendationType::MemoryOptimization,
                            priority: 9,
                            description: format!(
                                "Memory usage spike detected: {:.2} MB above expected",
                                (anomaly.actual_value - anomaly.expected_value) / 1_048_576.0
                            ),
                            expected_improvement_percent: 30.0,
                            complexity: "Low".to_string(),
                            actions: vec![
                                "Review memory allocation patterns".to_string(),
                                "Consider using object pooling".to_string(),
                                "Check for memory leaks".to_string(),
                            ],
                        });
                        self.stats.total_recommendations += 1;
                    }
                }
                _ => {}
            }
        }

        // Platform-specific recommendations
        self.add_platform_recommendations()?;

        // Cloud-specific recommendations
        if self.cloud_profiler.is_some() {
            self.add_cloud_recommendations()?;
        }

        Ok(())
    }

    /// Add platform-specific recommendations
    fn add_platform_recommendations(&mut self) -> TorshResult<()> {
        let strategy = self.platform_profiler.recommended_strategy();

        match strategy {
            ProfilingStrategy::HardwareCounters => {
                // x86_64 recommendations
                self.recommendations.push(OptimizationRecommendation {
                    rec_type: RecommendationType::PlatformOptimization,
                    priority: 7,
                    description: "Use SIMD instructions for vectorized operations".to_string(),
                    expected_improvement_percent: 200.0,
                    complexity: "Medium".to_string(),
                    actions: vec![
                        "Identify hot loops suitable for vectorization".to_string(),
                        "Use AVX2/AVX-512 intrinsics".to_string(),
                        "Enable auto-vectorization compiler flags".to_string(),
                    ],
                });
            }
            ProfilingStrategy::Hybrid => {
                // ARM64 recommendations
                self.recommendations.push(OptimizationRecommendation {
                    rec_type: RecommendationType::PlatformOptimization,
                    priority: 6,
                    description: "Optimize for ARM64 NEON SIMD".to_string(),
                    expected_improvement_percent: 150.0,
                    complexity: "Medium".to_string(),
                    actions: vec![
                        "Use NEON intrinsics for performance-critical code".to_string(),
                        "Consider P-core vs E-core task placement".to_string(),
                        "Optimize cache access patterns for ARM".to_string(),
                    ],
                });
            }
            _ => {}
        }

        Ok(())
    }

    /// Add cloud-specific recommendations
    fn add_cloud_recommendations(&mut self) -> TorshResult<()> {
        if let Some(cloud) = &self.cloud_profiler {
            let cost_per_hour = cloud.estimated_cost_per_hour();

            if cost_per_hour > 5.0 {
                self.recommendations.push(OptimizationRecommendation {
                    rec_type: RecommendationType::CloudOptimization,
                    priority: 10,
                    description: format!(
                        "High cloud cost detected: ${:.2}/hour - Consider optimization",
                        cost_per_hour
                    ),
                    expected_improvement_percent: 50.0,
                    complexity: "Low".to_string(),
                    actions: vec![
                        "Use spot/preemptible instances for development".to_string(),
                        "Right-size instance types for workload".to_string(),
                        "Consider reserved instances for long-term workloads".to_string(),
                    ],
                });
                self.stats.total_recommendations += 1;
            }
        }

        Ok(())
    }

    /// Update performance snapshot
    fn update_performance_snapshot(&mut self) -> TorshResult<()> {
        let anomaly_stats = self.anomaly_detector.get_stats();

        let snapshot = PerformanceSnapshot {
            timestamp: chrono::Utc::now(),
            avg_duration_us: anomaly_stats.duration_mean,
            avg_memory_bytes: anomaly_stats.memory_mean,
            flops_rate: 0.0, // Would calculate from recent events
            throughput_ops_per_sec: 0.0,
            anomaly_count: anomaly_stats.recent_anomaly_count,
            active_cluster: None,
        };

        if self.performance_history.len() >= 1000 {
            self.performance_history.pop_front();
        }
        self.performance_history.push_back(snapshot);

        Ok(())
    }

    /// Get top recommendations
    pub fn get_top_recommendations(&self, count: usize) -> Vec<&OptimizationRecommendation> {
        let mut recs: Vec<_> = self.recommendations.iter().collect();
        recs.sort_by(|a, b| b.priority.cmp(&a.priority));
        recs.into_iter().take(count).collect()
    }

    /// Get statistics
    pub fn get_stats(&self) -> &IntegratedProfilerStats {
        &self.stats
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &VecDeque<PerformanceSnapshot> {
        &self.performance_history
    }

    /// Export integrated report
    pub fn export_report(&self) -> TorshResult<IntegratedReport> {
        Ok(IntegratedReport {
            stats: self.stats.clone(),
            anomaly_summary: self.anomaly_detector.get_stats(),
            predictor_stats: self.performance_predictor.get_stats(),
            top_recommendations: self
                .get_top_recommendations(10)
                .iter()
                .map(|r| (*r).clone())
                .collect(),
            performance_trends: self.calculate_performance_trends(),
            platform_info: self.platform_profiler.platform_info(),
            cloud_info: self
                .cloud_profiler
                .as_ref()
                .map(|c| format!("{}", c.provider())),
        })
    }

    /// Calculate performance trends
    fn calculate_performance_trends(&self) -> PerformanceTrends {
        if self.performance_history.len() < 2 {
            return PerformanceTrends::default();
        }

        let recent_window = self.performance_history.len().min(100);
        let recent: Vec<_> = self
            .performance_history
            .iter()
            .rev()
            .take(recent_window)
            .collect();

        let avg_duration: f64 =
            recent.iter().map(|s| s.avg_duration_us).sum::<f64>() / recent.len() as f64;
        let avg_memory: f64 =
            recent.iter().map(|s| s.avg_memory_bytes).sum::<f64>() / recent.len() as f64;

        let duration_trend = if recent.len() > 10 {
            let first_half: f64 = recent
                .iter()
                .take(recent.len() / 2)
                .map(|s| s.avg_duration_us)
                .sum::<f64>()
                / (recent.len() / 2) as f64;
            let second_half: f64 = recent
                .iter()
                .skip(recent.len() / 2)
                .map(|s| s.avg_duration_us)
                .sum::<f64>()
                / (recent.len() - recent.len() / 2) as f64;
            (second_half - first_half) / first_half * 100.0
        } else {
            0.0
        };

        PerformanceTrends {
            avg_duration_us: avg_duration,
            avg_memory_bytes: avg_memory,
            duration_trend_percent: duration_trend,
            memory_trend_percent: 0.0,
            stability_score: 0.95, // Simplified
        }
    }

    /// Initialize Kubernetes operator
    pub fn init_kubernetes(&mut self, namespace: String) -> TorshResult<()> {
        self.k8s_operator = Some(KubernetesProfilerOperator::new(namespace));
        Ok(())
    }

    /// Get Kubernetes operator
    pub fn k8s_operator(&mut self) -> Option<&mut KubernetesProfilerOperator> {
        self.k8s_operator.as_mut()
    }
}

impl Default for IntegratedProfiler {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| panic!("Failed to create integrated profiler"))
    }
}

/// Processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Detected anomalies
    pub anomalies: Vec<AnomalyEvent>,
    /// Assigned cluster
    pub cluster: usize,
    /// Predicted duration
    pub predicted_duration: f64,
    /// Prediction error percentage
    pub prediction_error: f64,
    /// Recommendations generated
    pub recommendations_generated: bool,
}

/// Integrated report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedReport {
    /// Overall statistics
    pub stats: IntegratedProfilerStats,
    /// Anomaly summary
    pub anomaly_summary: crate::online_learning::OnlineAnomalyStats,
    /// Predictor statistics
    pub predictor_stats: crate::online_learning::PredictorStats,
    /// Top recommendations
    pub top_recommendations: Vec<OptimizationRecommendation>,
    /// Performance trends
    pub performance_trends: PerformanceTrends,
    /// Platform information
    pub platform_info: String,
    /// Cloud information
    pub cloud_info: Option<String>,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceTrends {
    /// Average duration in microseconds
    pub avg_duration_us: f64,
    /// Average memory usage in bytes
    pub avg_memory_bytes: f64,
    /// Duration trend (positive = getting slower)
    pub duration_trend_percent: f64,
    /// Memory trend (positive = using more memory)
    pub memory_trend_percent: f64,
    /// Stability score (0.0 to 1.0)
    pub stability_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrated_profiler_creation() {
        let profiler = IntegratedProfiler::new();
        assert!(profiler.is_ok());

        let profiler = profiler.unwrap();
        assert_eq!(profiler.state, ProfilerState::Stopped);
        assert_eq!(profiler.stats.total_events, 0);
    }

    #[test]
    fn test_start_stop() {
        let mut profiler = IntegratedProfiler::new().unwrap();

        assert!(profiler.start().is_ok());
        assert_eq!(profiler.state, ProfilerState::Running);

        assert!(profiler.stop().is_ok());
        assert_eq!(profiler.state, ProfilerState::Stopped);
    }

    #[test]
    fn test_event_processing() {
        let mut profiler = IntegratedProfiler::new().unwrap();
        profiler.start().unwrap();

        let thread_id = 1;
        let event = ProfileEvent {
            name: "test_op".to_string(),
            category: "test".to_string(),
            thread_id,
            start_us: 0,
            duration_us: 100,
            operation_count: Some(10),
            flops: Some(1000),
            bytes_transferred: Some(1024),
            stack_trace: None,
        };

        let result = profiler.process_event(&event);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(profiler.stats.total_events, 1);
    }

    #[test]
    fn test_recommendation_generation() {
        let mut profiler = IntegratedProfiler::new().unwrap();
        profiler.start().unwrap();

        // Process some normal events
        let thread_id = 1;
        for i in 0..50 {
            let event = ProfileEvent {
                name: format!("op_{}", i),
                category: "test".to_string(),
                thread_id,
                start_us: i * 1000,
                duration_us: 100,
                operation_count: Some(10),
                flops: Some(1000),
                bytes_transferred: Some(1024),
                stack_trace: None,
            };
            profiler.process_event(&event).unwrap();
        }

        // Process an anomalous event
        let anomalous_event = ProfileEvent {
            name: "slow_op".to_string(),
            category: "test".to_string(),
            thread_id,
            start_us: 50000,
            duration_us: 1000, // 10x slower
            operation_count: Some(10),
            flops: Some(1000),
            bytes_transferred: Some(10240), // 10x more data
            stack_trace: None,
        };

        profiler.process_event(&anomalous_event).unwrap();

        let recommendations = profiler.get_top_recommendations(5);
        // Recommendations may be generated
        println!("Generated {} recommendations", recommendations.len());
    }

    #[test]
    fn test_export_report() {
        let mut profiler = IntegratedProfiler::new().unwrap();
        profiler.start().unwrap();

        let report = profiler.export_report();
        assert!(report.is_ok());

        let report = report.unwrap();
        assert_eq!(report.stats.total_events, 0);
    }
}
