//! Workload Characterization
//!
//! This module provides comprehensive workload analysis and characterization capabilities
//! to understand application behavior patterns, resource utilization, and performance characteristics.

use crate::{ProfileEvent, TorshError, TorshResult};
use serde::{Deserialize, Serialize};

/// Workload characterization analyzer
pub struct WorkloadCharacterizer {
    /// Collected workload samples
    samples: Vec<WorkloadSample>,
    /// Analysis results
    analysis: Option<WorkloadAnalysis>,
    /// Configuration for characterization
    config: CharacterizationConfig,
}

/// Configuration for workload characterization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterizationConfig {
    /// Minimum sample count for reliable analysis
    pub min_sample_count: usize,
    /// Time window for analysis (seconds)
    pub analysis_window_seconds: u64,
    /// Enable detailed memory analysis
    pub enable_memory_analysis: bool,
    /// Enable compute intensity analysis
    pub enable_compute_analysis: bool,
    /// Enable I/O pattern analysis
    pub enable_io_analysis: bool,
    /// Enable parallelism analysis
    pub enable_parallelism_analysis: bool,
}

/// Individual workload sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSample {
    pub timestamp: u64,
    pub operation_name: String,
    pub category: String,
    pub duration_ms: f64,
    pub cpu_utilization: f64,
    pub memory_mb: f64,
    pub cache_miss_rate: f64,
    pub io_ops_per_sec: f64,
    pub parallel_threads: u32,
    pub flops: u64,
    pub bytes_accessed: u64,
    pub energy_joules: f64,
}

/// Complete workload analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadAnalysis {
    /// Workload type classification
    pub workload_type: WorkloadType,
    /// Resource utilization patterns
    pub resource_patterns: ResourcePatterns,
    /// Compute characteristics
    pub compute_characteristics: ComputeCharacteristics,
    /// Memory access patterns
    pub memory_patterns: MemoryPatterns,
    /// I/O behavior analysis
    pub io_patterns: IOPatterns,
    /// Parallelism analysis
    pub parallelism_analysis: ParallelismAnalysis,
    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Workload stability metrics
    pub stability_metrics: StabilityMetrics,
}

/// Primary workload type classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkloadType {
    /// CPU-intensive computation
    ComputeIntensive,
    /// Memory bandwidth limited
    MemoryBound,
    /// I/O intensive operations
    IOIntensive,
    /// Balanced resource usage
    Balanced,
    /// GPU/accelerator workload
    GPUAccelerated,
    /// Network/communication bound
    NetworkBound,
    /// Cache-sensitive operations
    CacheSensitive,
    /// Irregular/unpredictable access patterns
    Irregular,
}

/// Resource utilization patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePatterns {
    /// Average CPU utilization (0.0 to 1.0)
    pub avg_cpu_utilization: f64,
    /// CPU utilization variance
    pub cpu_utilization_variance: f64,
    /// Average memory usage (MB)
    pub avg_memory_usage_mb: f64,
    /// Memory usage peak factor
    pub memory_peak_factor: f64,
    /// Memory access locality score (0.0 to 1.0)
    pub memory_locality_score: f64,
    /// Cache efficiency score (0.0 to 1.0)
    pub cache_efficiency_score: f64,
    /// I/O throughput (MB/s)
    pub io_throughput_mbps: f64,
    /// Network utilization (0.0 to 1.0)
    pub network_utilization: f64,
}

/// Compute-specific characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCharacteristics {
    /// Arithmetic intensity (FLOPS per byte accessed)
    pub arithmetic_intensity: f64,
    /// Vectorization efficiency (0.0 to 1.0)
    pub vectorization_efficiency: f64,
    /// Instruction level parallelism score
    pub ilp_score: f64,
    /// Branch prediction efficiency (0.0 to 1.0)
    pub branch_prediction_efficiency: f64,
    /// Compute to memory ratio
    pub compute_to_memory_ratio: f64,
    /// Dominant operation types
    pub dominant_operations: Vec<OperationType>,
}

/// Types of operations in the workload
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    /// Integer arithmetic
    IntegerArithmetic,
    /// Floating-point arithmetic
    FloatingPointArithmetic,
    /// Vector/SIMD operations
    VectorOperations,
    /// Matrix operations
    MatrixOperations,
    /// Memory operations
    MemoryOperations,
    /// Branch/control operations
    BranchOperations,
    /// I/O operations
    IOOperations,
    /// Synchronization operations
    SynchronizationOperations,
}

/// Memory access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPatterns {
    /// Sequential access percentage (0.0 to 1.0)
    pub sequential_access_ratio: f64,
    /// Random access percentage (0.0 to 1.0)
    pub random_access_ratio: f64,
    /// Stride access patterns
    pub stride_patterns: Vec<StridePattern>,
    /// Working set size (MB)
    pub working_set_size_mb: f64,
    /// Memory reuse distance distribution
    pub reuse_distance_distribution: Vec<(u64, f64)>,
    /// Cache line utilization efficiency
    pub cache_line_efficiency: f64,
    /// Memory bandwidth utilization (0.0 to 1.0)
    pub memory_bandwidth_utilization: f64,
}

/// Stride access pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StridePattern {
    /// Stride size in bytes
    pub stride_bytes: u64,
    /// Frequency of this stride pattern (0.0 to 1.0)
    pub frequency: f64,
    /// Cache friendliness score (0.0 to 1.0)
    pub cache_friendliness: f64,
}

/// I/O access patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOPatterns {
    /// Read to write ratio
    pub read_write_ratio: f64,
    /// Sequential I/O percentage (0.0 to 1.0)
    pub sequential_io_ratio: f64,
    /// Average I/O request size (bytes)
    pub avg_io_size_bytes: u64,
    /// I/O burst patterns
    pub burst_patterns: Vec<IOBurstPattern>,
    /// Storage device utilization (0.0 to 1.0)
    pub storage_utilization: f64,
}

/// I/O burst pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOBurstPattern {
    /// Duration of burst (ms)
    pub duration_ms: f64,
    /// I/O intensity during burst (ops/sec)
    pub intensity_ops_per_sec: f64,
    /// Idle time between bursts (ms)
    pub idle_time_ms: f64,
}

/// Parallelism characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismAnalysis {
    /// Thread utilization efficiency (0.0 to 1.0)
    pub thread_utilization_efficiency: f64,
    /// Load balancing score (0.0 to 1.0)
    pub load_balancing_score: f64,
    /// Synchronization overhead percentage (0.0 to 1.0)
    pub synchronization_overhead: f64,
    /// Parallel efficiency (speedup / thread_count)
    pub parallel_efficiency: f64,
    /// Critical path analysis
    pub critical_path_analysis: CriticalPathAnalysis,
    /// Communication patterns
    pub communication_patterns: Vec<CommunicationPattern>,
}

/// Critical path analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalPathAnalysis {
    /// Critical path length (ms)
    pub critical_path_length_ms: f64,
    /// Parallelizable portion (0.0 to 1.0)
    pub parallelizable_portion: f64,
    /// Serial bottleneck locations
    pub serial_bottlenecks: Vec<String>,
}

/// Communication pattern between threads/processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPattern {
    /// Pattern type
    pub pattern_type: CommunicationPatternType,
    /// Data volume (bytes)
    pub data_volume_bytes: u64,
    /// Communication frequency (per second)
    pub frequency_per_sec: f64,
    /// Latency impact (ms)
    pub latency_impact_ms: f64,
}

/// Types of communication patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommunicationPatternType {
    /// Point-to-point communication
    PointToPoint,
    /// Broadcast communication
    Broadcast,
    /// Gather/reduce operations
    GatherReduce,
    /// All-to-all communication
    AllToAll,
    /// Producer-consumer
    ProducerConsumer,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity score (0.0 to 1.0)
    pub severity: f64,
    /// Description of the bottleneck
    pub description: String,
    /// Affected operations
    pub affected_operations: Vec<String>,
    /// Performance impact percentage
    pub performance_impact_percent: f64,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU computation bottleneck
    CPUCompute,
    /// Memory bandwidth bottleneck
    MemoryBandwidth,
    /// Cache miss bottleneck
    CacheMiss,
    /// I/O throughput bottleneck
    IOThroughput,
    /// Network latency bottleneck
    NetworkLatency,
    /// Synchronization bottleneck
    Synchronization,
    /// Load balancing bottleneck
    LoadBalancing,
    /// Resource contention bottleneck
    ResourceContention,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: OptimizationType,
    /// Priority level (1-10)
    pub priority: u8,
    /// Description of the recommendation
    pub description: String,
    /// Expected performance improvement (percentage)
    pub expected_improvement_percent: f64,
    /// Implementation complexity (1-10)
    pub implementation_complexity: u8,
    /// Specific actions to take
    pub actions: Vec<String>,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationType {
    /// Algorithmic optimization
    Algorithmic,
    /// Memory access optimization
    MemoryAccess,
    /// Cache optimization
    CacheOptimization,
    /// Vectorization improvement
    Vectorization,
    /// Parallelization enhancement
    Parallelization,
    /// I/O optimization
    IOOptimization,
    /// Data structure optimization
    DataStructure,
    /// Compiler optimization
    Compiler,
    /// Hardware utilization
    HardwareUtilization,
}

/// Workload stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    /// Performance variance coefficient
    pub performance_variance: f64,
    /// Resource usage stability score (0.0 to 1.0)
    pub resource_stability: f64,
    /// Predictability score (0.0 to 1.0)
    pub predictability_score: f64,
    /// Phase change detection
    pub phase_changes: Vec<PhaseChange>,
}

/// Detected phase change in workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseChange {
    /// Timestamp of phase change
    pub timestamp: u64,
    /// Description of the change
    pub description: String,
    /// Magnitude of change (0.0 to 1.0)
    pub magnitude: f64,
    /// Duration of new phase (ms)
    pub duration_ms: f64,
}

impl Default for CharacterizationConfig {
    fn default() -> Self {
        Self {
            min_sample_count: 100,
            analysis_window_seconds: 60,
            enable_memory_analysis: true,
            enable_compute_analysis: true,
            enable_io_analysis: true,
            enable_parallelism_analysis: true,
        }
    }
}

impl Default for WorkloadCharacterizer {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkloadCharacterizer {
    /// Create a new workload characterizer
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            analysis: None,
            config: CharacterizationConfig::default(),
        }
    }

    /// Create a workload characterizer with custom configuration
    pub fn with_config(config: CharacterizationConfig) -> Self {
        Self {
            samples: Vec::new(),
            analysis: None,
            config,
        }
    }

    /// Add a workload sample for analysis
    pub fn add_sample(&mut self, sample: WorkloadSample) -> TorshResult<()> {
        self.samples.push(sample);
        Ok(())
    }

    /// Add samples from profile events
    pub fn add_samples_from_events(&mut self, events: &[ProfileEvent]) -> TorshResult<()> {
        for event in events {
            let sample = WorkloadSample {
                timestamp: event.start_us / 1000, // Convert to ms
                operation_name: event.name.clone(),
                category: event.category.clone(),
                duration_ms: event.duration_us as f64 / 1000.0,
                cpu_utilization: 0.7, // Would be measured from system
                memory_mb: event.bytes_transferred.unwrap_or(0) as f64 / (1024.0 * 1024.0),
                cache_miss_rate: 0.05, // Would be measured from performance counters
                io_ops_per_sec: 0.0,   // Would be measured from I/O monitoring
                parallel_threads: 1,   // Would be detected from threading analysis
                flops: event.flops.unwrap_or(0),
                bytes_accessed: event.bytes_transferred.unwrap_or(0),
                energy_joules: 0.0, // Would be measured from power monitoring
            };
            self.add_sample(sample)?;
        }
        Ok(())
    }

    /// Perform comprehensive workload analysis
    pub fn analyze(&mut self) -> TorshResult<WorkloadAnalysis> {
        if self.samples.len() < self.config.min_sample_count {
            return Err(TorshError::InvalidArgument(format!(
                "Insufficient samples for analysis: {} < {}",
                self.samples.len(),
                self.config.min_sample_count
            )));
        }

        let workload_type = self.classify_workload_type()?;
        let resource_patterns = self.analyze_resource_patterns()?;
        let compute_characteristics = self.analyze_compute_characteristics()?;
        let memory_patterns = self.analyze_memory_patterns()?;
        let io_patterns = self.analyze_io_patterns()?;
        let parallelism_analysis = self.analyze_parallelism()?;
        let bottlenecks = self.identify_bottlenecks()?;
        let recommendations = self.generate_recommendations(&workload_type, &bottlenecks)?;
        let stability_metrics = self.analyze_stability()?;

        let analysis = WorkloadAnalysis {
            workload_type,
            resource_patterns,
            compute_characteristics,
            memory_patterns,
            io_patterns,
            parallelism_analysis,
            bottlenecks,
            recommendations,
            stability_metrics,
        };

        self.analysis = Some(analysis.clone());
        Ok(analysis)
    }

    /// Get the current analysis results
    pub fn get_analysis(&self) -> Option<&WorkloadAnalysis> {
        self.analysis.as_ref()
    }

    /// Export workload characterization results
    pub fn export_analysis(&self, filename: &str) -> TorshResult<()> {
        if let Some(analysis) = &self.analysis {
            let json = serde_json::to_string_pretty(analysis)
                .map_err(|e| TorshError::SerializationError(e.to_string()))?;

            std::fs::write(filename, json).map_err(|e| TorshError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    /// Clear all samples and analysis
    pub fn reset(&mut self) {
        self.samples.clear();
        self.analysis = None;
    }

    // Private analysis methods

    fn classify_workload_type(&self) -> TorshResult<WorkloadType> {
        let avg_cpu =
            self.samples.iter().map(|s| s.cpu_utilization).sum::<f64>() / self.samples.len() as f64;
        let avg_memory =
            self.samples.iter().map(|s| s.memory_mb).sum::<f64>() / self.samples.len() as f64;
        let avg_io =
            self.samples.iter().map(|s| s.io_ops_per_sec).sum::<f64>() / self.samples.len() as f64;
        let avg_flops =
            self.samples.iter().map(|s| s.flops).sum::<u64>() / self.samples.len() as u64;

        // Simple classification heuristics
        if avg_flops > 1000000 && avg_cpu > 0.8 {
            Ok(WorkloadType::ComputeIntensive)
        } else if avg_memory > 1000.0 {
            Ok(WorkloadType::MemoryBound)
        } else if avg_io > 1000.0 {
            Ok(WorkloadType::IOIntensive)
        } else {
            Ok(WorkloadType::Balanced)
        }
    }

    fn analyze_resource_patterns(&self) -> TorshResult<ResourcePatterns> {
        let cpu_utilizations: Vec<f64> = self.samples.iter().map(|s| s.cpu_utilization).collect();
        let avg_cpu = cpu_utilizations.iter().sum::<f64>() / cpu_utilizations.len() as f64;
        let cpu_variance = cpu_utilizations
            .iter()
            .map(|x| (x - avg_cpu).powi(2))
            .sum::<f64>()
            / cpu_utilizations.len() as f64;

        let avg_memory =
            self.samples.iter().map(|s| s.memory_mb).sum::<f64>() / self.samples.len() as f64;
        let max_memory = self.samples.iter().map(|s| s.memory_mb).fold(0.0, f64::max);
        let memory_peak_factor = if avg_memory > 0.0 {
            max_memory / avg_memory
        } else {
            1.0
        };

        Ok(ResourcePatterns {
            avg_cpu_utilization: avg_cpu,
            cpu_utilization_variance: cpu_variance,
            avg_memory_usage_mb: avg_memory,
            memory_peak_factor,
            memory_locality_score: 0.8, // Would be calculated from access patterns
            cache_efficiency_score: 1.0
                - (self.samples.iter().map(|s| s.cache_miss_rate).sum::<f64>()
                    / self.samples.len() as f64),
            io_throughput_mbps: self.samples.iter().map(|s| s.io_ops_per_sec).sum::<f64>()
                / self.samples.len() as f64,
            network_utilization: 0.0, // Would be measured from network monitoring
        })
    }

    fn analyze_compute_characteristics(&self) -> TorshResult<ComputeCharacteristics> {
        let total_flops: u64 = self.samples.iter().map(|s| s.flops).sum();
        let total_bytes: u64 = self.samples.iter().map(|s| s.bytes_accessed).sum();

        let arithmetic_intensity = if total_bytes > 0 {
            total_flops as f64 / total_bytes as f64
        } else {
            0.0
        };

        Ok(ComputeCharacteristics {
            arithmetic_intensity,
            vectorization_efficiency: 0.7, // Would be measured from instruction analysis
            ilp_score: 0.6,                // Would be calculated from instruction dependencies
            branch_prediction_efficiency: 0.9, // Would be measured from hardware counters
            compute_to_memory_ratio: arithmetic_intensity,
            dominant_operations: vec![OperationType::FloatingPointArithmetic], // Would be determined from instruction analysis
        })
    }

    fn analyze_memory_patterns(&self) -> TorshResult<MemoryPatterns> {
        Ok(MemoryPatterns {
            sequential_access_ratio: 0.6, // Would be calculated from access pattern analysis
            random_access_ratio: 0.4,
            stride_patterns: vec![
                StridePattern {
                    stride_bytes: 64,
                    frequency: 0.4,
                    cache_friendliness: 0.9,
                },
                StridePattern {
                    stride_bytes: 1024,
                    frequency: 0.3,
                    cache_friendliness: 0.5,
                },
            ],
            working_set_size_mb: self.samples.iter().map(|s| s.memory_mb).fold(0.0, f64::max),
            reuse_distance_distribution: vec![(64, 0.3), (1024, 0.4), (65536, 0.3)],
            cache_line_efficiency: 0.8,
            memory_bandwidth_utilization: 0.7,
        })
    }

    fn analyze_io_patterns(&self) -> TorshResult<IOPatterns> {
        Ok(IOPatterns {
            read_write_ratio: 2.0, // Would be calculated from I/O trace analysis
            sequential_io_ratio: 0.7,
            avg_io_size_bytes: 4096,
            burst_patterns: vec![IOBurstPattern {
                duration_ms: 100.0,
                intensity_ops_per_sec: 10000.0,
                idle_time_ms: 50.0,
            }],
            storage_utilization: 0.6,
        })
    }

    fn analyze_parallelism(&self) -> TorshResult<ParallelismAnalysis> {
        Ok(ParallelismAnalysis {
            thread_utilization_efficiency: 0.8,
            load_balancing_score: 0.9,
            synchronization_overhead: 0.05,
            parallel_efficiency: 0.85,
            critical_path_analysis: CriticalPathAnalysis {
                critical_path_length_ms: 100.0,
                parallelizable_portion: 0.8,
                serial_bottlenecks: vec!["initialization".to_string(), "finalization".to_string()],
            },
            communication_patterns: vec![CommunicationPattern {
                pattern_type: CommunicationPatternType::PointToPoint,
                data_volume_bytes: 1024,
                frequency_per_sec: 100.0,
                latency_impact_ms: 0.1,
            }],
        })
    }

    fn identify_bottlenecks(&self) -> TorshResult<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();

        // CPU bottleneck analysis
        let avg_cpu =
            self.samples.iter().map(|s| s.cpu_utilization).sum::<f64>() / self.samples.len() as f64;
        if avg_cpu > 0.9 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CPUCompute,
                severity: avg_cpu,
                description: "High CPU utilization indicates compute bottleneck".to_string(),
                affected_operations: self
                    .samples
                    .iter()
                    .filter(|s| s.cpu_utilization > 0.9)
                    .map(|s| s.operation_name.clone())
                    .collect(),
                performance_impact_percent: 25.0,
            });
        }

        // Cache miss bottleneck analysis
        let avg_cache_miss =
            self.samples.iter().map(|s| s.cache_miss_rate).sum::<f64>() / self.samples.len() as f64;
        if avg_cache_miss > 0.1 {
            bottlenecks.push(PerformanceBottleneck {
                bottleneck_type: BottleneckType::CacheMiss,
                severity: avg_cache_miss,
                description: "High cache miss rate indicates memory access inefficiency"
                    .to_string(),
                affected_operations: self
                    .samples
                    .iter()
                    .filter(|s| s.cache_miss_rate > 0.1)
                    .map(|s| s.operation_name.clone())
                    .collect(),
                performance_impact_percent: 15.0,
            });
        }

        Ok(bottlenecks)
    }

    fn generate_recommendations(
        &self,
        workload_type: &WorkloadType,
        bottlenecks: &[PerformanceBottleneck],
    ) -> TorshResult<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();

        match workload_type {
            WorkloadType::ComputeIntensive => {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationType::Vectorization,
                    priority: 8,
                    description:
                        "Consider vectorizing compute-intensive loops for better SIMD utilization"
                            .to_string(),
                    expected_improvement_percent: 20.0,
                    implementation_complexity: 6,
                    actions: vec![
                        "Identify hot loops in compute kernels".to_string(),
                        "Apply SIMD intrinsics or auto-vectorization hints".to_string(),
                        "Ensure data alignment for optimal vector operations".to_string(),
                    ],
                });
            }
            WorkloadType::MemoryBound => {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationType::MemoryAccess,
                    priority: 9,
                    description: "Optimize memory access patterns to reduce bandwidth pressure"
                        .to_string(),
                    expected_improvement_percent: 30.0,
                    implementation_complexity: 7,
                    actions: vec![
                        "Implement data prefetching strategies".to_string(),
                        "Reorganize data structures for better locality".to_string(),
                        "Consider memory pooling to reduce allocation overhead".to_string(),
                    ],
                });
            }
            _ => {}
        }

        // Add recommendations based on identified bottlenecks
        for bottleneck in bottlenecks {
            if bottleneck.bottleneck_type == BottleneckType::CacheMiss {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationType::CacheOptimization,
                    priority: 7,
                    description: "Improve cache efficiency to reduce memory latency".to_string(),
                    expected_improvement_percent: bottleneck.performance_impact_percent * 0.7,
                    implementation_complexity: 5,
                    actions: vec![
                        "Analyze memory access patterns".to_string(),
                        "Implement cache-friendly data layouts".to_string(),
                        "Add prefetch instructions for predictable accesses".to_string(),
                    ],
                });
            }
        }

        Ok(recommendations)
    }

    fn analyze_stability(&self) -> TorshResult<StabilityMetrics> {
        let durations: Vec<f64> = self.samples.iter().map(|s| s.duration_ms).collect();
        let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance = durations
            .iter()
            .map(|x| (x - avg_duration).powi(2))
            .sum::<f64>()
            / durations.len() as f64;
        let performance_variance = variance.sqrt() / avg_duration;

        Ok(StabilityMetrics {
            performance_variance,
            resource_stability: 0.8, // Would be calculated from resource usage variance
            predictability_score: 1.0 - performance_variance.min(1.0),
            phase_changes: Vec::new(), // Would be detected from time series analysis
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_characterizer_creation() {
        let characterizer = WorkloadCharacterizer::new();
        assert_eq!(characterizer.samples.len(), 0);
        assert!(characterizer.analysis.is_none());
    }

    #[test]
    fn test_add_sample() {
        let mut characterizer = WorkloadCharacterizer::new();
        let sample = WorkloadSample {
            timestamp: 0,
            operation_name: "test_op".to_string(),
            category: "test".to_string(),
            duration_ms: 10.0,
            cpu_utilization: 0.8,
            memory_mb: 100.0,
            cache_miss_rate: 0.05,
            io_ops_per_sec: 0.0,
            parallel_threads: 1,
            flops: 1000000,
            bytes_accessed: 1024,
            energy_joules: 0.1,
        };

        characterizer.add_sample(sample).unwrap();
        assert_eq!(characterizer.samples.len(), 1);
    }

    #[test]
    fn test_workload_classification() {
        let mut characterizer = WorkloadCharacterizer::new();

        // Add compute-intensive samples
        for i in 0..100 {
            let sample = WorkloadSample {
                timestamp: i,
                operation_name: format!("compute_op_{i}"),
                category: "compute".to_string(),
                duration_ms: 10.0,
                cpu_utilization: 0.95,
                memory_mb: 50.0,
                cache_miss_rate: 0.02,
                io_ops_per_sec: 0.0,
                parallel_threads: 1,
                flops: 2000000,
                bytes_accessed: 1024,
                energy_joules: 0.2,
            };
            characterizer.add_sample(sample).unwrap();
        }

        let analysis = characterizer.analyze().unwrap();
        assert_eq!(analysis.workload_type, WorkloadType::ComputeIntensive);
    }
}
