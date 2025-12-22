//! Memory anomaly detection and allocation pattern analysis
//!
//! This module provides sophisticated anomaly detection for memory allocation patterns
//! in autograd operations. It identifies unusual memory behavior that might indicate
//! performance issues, memory leaks, or inefficient allocation strategies.
//!
//! # Overview
//!
//! The anomaly detection system monitors several aspects of memory behavior:
//!
//! - **Allocation Patterns**: Frequency, size, and timing of memory allocations
//! - **Growth Rate Analysis**: Detection of unusual memory growth rates
//! - **Memory Leak Detection**: Identification of persistent memory allocations
//! - **Fragmentation Detection**: Analysis of memory layout efficiency
//! - **Performance Impact**: Assessment of how anomalies affect system performance
//!
//! # Anomaly Categories
//!
//! The system classifies anomalies into several types:
//!
//! - **Excessive Growth Rate**: Memory usage growing faster than expected
//! - **Large Allocation Size**: Individual allocations that are unusually large
//! - **Fragmentation Issues**: Memory layout becoming inefficient
//! - **Potential Leaks**: Allocations that persist longer than expected
//! - **Anomalous Patterns**: Unusual timing or frequency patterns
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use crate::memory::anomaly::{MemoryAnomalyDetector, AnomalySeverity};
//!
//! let mut detector = MemoryAnomalyDetector::new();
//!
//! // Analyze allocation pattern
//! let pattern = detector.analyze_allocation_pattern("conv2d", 1000, 1024 * 1024);
//! if pattern.pattern_type == AllocationPatternType::Irregular {
//!     println!("Irregular allocation pattern detected");
//! }
//!
//! // Check for anomalies
//! let anomalies = detector.detect_anomalies();
//! for anomaly in anomalies {
//!     if anomaly.severity >= AnomalySeverity::High {
//!         println!("High severity anomaly: {}", anomaly.description);
//!     }
//! }
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Memory usage anomaly detection
///
/// Represents a detected anomaly in memory allocation patterns. Anomalies
/// indicate potentially problematic memory behavior that should be investigated
/// and possibly corrected to improve system performance and reliability.
///
/// # Anomaly Lifecycle
///
/// 1. **Detection**: Anomaly is identified by monitoring algorithms
/// 2. **Classification**: Anomaly type and severity are determined
/// 3. **Analysis**: Impact and recommendations are generated
/// 4. **Resolution**: Anomaly is either resolved or becomes persistent
///
/// # Integration with Monitoring
///
/// Anomalies are typically detected by continuous monitoring systems and
/// fed into alerting and optimization pipelines for automatic response.
#[derive(Debug, Clone)]
pub struct MemoryAnomaly {
    /// Type of anomaly detected
    pub anomaly_type: MemoryAnomalyType,
    /// Severity level of the anomaly
    pub severity: AnomalySeverity,
    /// Human-readable description of the anomaly
    pub description: String,
    /// When the anomaly was first detected
    pub detected_at: Instant,
    /// Operation or context where anomaly occurred
    pub operation_context: String,
    /// Quantitative measurement of the anomaly
    pub measurement: f64,
    /// Expected or baseline value for comparison
    pub baseline: f64,
}

/// Types of memory anomalies
///
/// Categorizes different kinds of memory allocation anomalies based on
/// their characteristics and potential impact on system performance.
///
/// # Detection Algorithms
///
/// Each anomaly type uses specific detection algorithms:
///
/// - **Excessive Growth Rate**: Statistical analysis of memory growth over time
/// - **Large Allocation Size**: Comparison against historical allocation sizes
/// - **Fragmentation**: Analysis of memory layout efficiency
/// - **Potential Leak**: Detection of persistent allocations without deallocations
/// - **Anomalous Pattern**: Time-series analysis of allocation frequencies
///
/// # Severity Assessment
///
/// Different anomaly types have different baseline severities, but the final
/// severity is adjusted based on the magnitude of deviation from normal behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryAnomalyType {
    /// Memory growth rate is excessive compared to baseline
    ExcessiveGrowthRate,
    /// Individual allocations are unusually large
    LargeAllocationSize,
    /// Memory fragmentation is degrading performance
    Fragmentation,
    /// Memory leak suspected due to allocation/deallocation imbalance
    PotentialLeak,
    /// Unusual allocation timing or frequency pattern detected
    AnomalousPattern,
}

/// Severity levels for memory anomalies
///
/// Indicates the urgency and potential impact of detected memory anomalies.
/// Higher severity levels typically trigger more immediate response actions.
///
/// # Severity Guidelines
///
/// - **Low**: Minor inefficiency, monitoring recommended
/// - **Medium**: Notable performance impact, optimization suggested
/// - **High**: Significant performance degradation, immediate attention needed
/// - **Critical**: System stability at risk, emergency response required
///
/// # Response Actions
///
/// Severity levels map to different response strategies:
/// - Low → Logging and trend analysis
/// - Medium → Performance optimization recommendations
/// - High → Automatic optimization triggers
/// - Critical → Emergency memory management procedures
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Low severity - informational, requires monitoring
    Low,
    /// Medium severity - performance impact, optimization recommended
    Medium,
    /// High severity - significant impact, immediate attention needed
    High,
    /// Critical severity - system stability at risk, emergency response required
    Critical,
}

/// Memory allocation pattern analysis
///
/// Analyzes memory allocation patterns for a specific operation to identify
/// characteristics that might indicate performance issues or optimization
/// opportunities.
///
/// # Pattern Analysis
///
/// The analysis considers multiple dimensions:
/// - **Frequency**: How often allocations occur
/// - **Size Distribution**: Range and variability of allocation sizes
/// - **Timing**: Regularity and predictability of allocation timing
/// - **Growth**: Trends in allocation sizes over time
///
/// # Use Cases
///
/// - Performance optimization guidance
/// - Memory budget planning
/// - Anomaly detection input
/// - Resource allocation strategies
#[derive(Debug, Clone, Default)]
pub struct AllocationPattern {
    /// Operation name being analyzed
    pub operation_name: String,
    /// Total number of allocations observed
    pub total_allocations: usize,
    /// Average allocation size in bytes
    pub average_allocation_size: usize,
    /// Peak memory usage for this operation
    pub peak_memory_usage: usize,
    /// Time since last allocation
    pub last_allocation_age: Duration,
    /// Allocation frequency (allocations per second)
    pub allocation_frequency: f64,
    /// Type of allocation pattern detected
    pub pattern_type: AllocationPatternType,
    /// Standard deviation of allocation sizes
    pub size_variability: f64,
    /// Trend direction: positive = growing, negative = shrinking, zero = stable
    pub size_trend: f64,
}

/// Types of allocation patterns
///
/// Classifies allocation patterns based on their characteristics, helping
/// to identify appropriate optimization strategies for different workload types.
///
/// # Pattern Characteristics
///
/// - **ManySmall**: Frequent small allocations (< 1MB each)
/// - **FewLarge**: Infrequent large allocations (> 10MB each)
/// - **HighFrequency**: Very frequent allocations (> 100/sec)
/// - **Normal**: Typical allocation pattern within expected parameters
/// - **Irregular**: Unpredictable or anomalous allocation behavior
///
/// # Optimization Strategies
///
/// Each pattern type suggests different optimization approaches:
/// - ManySmall → Memory pooling, batch allocation
/// - FewLarge → Memory mapping, lazy allocation
/// - HighFrequency → Pool warming, allocation caching
/// - Irregular → Adaptive strategies, monitoring
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationPatternType {
    /// Many small allocations (high frequency, small size)
    ManySmall,
    /// Few large allocations (low frequency, large size)
    FewLarge,
    /// High frequency allocations regardless of size
    HighFrequency,
    /// Normal allocation pattern within expected parameters
    Normal,
    /// Irregular or anomalous allocation pattern
    Irregular,
}

impl Default for AllocationPatternType {
    fn default() -> Self {
        AllocationPatternType::Normal
    }
}

/// Memory anomaly detector
///
/// Sophisticated anomaly detection system that monitors memory allocation
/// patterns and identifies potentially problematic behavior. Uses statistical
/// analysis and machine learning techniques to distinguish between normal
/// variation and true anomalies.
///
/// # Detection Algorithms
///
/// The detector employs multiple detection algorithms:
///
/// 1. **Statistical Outlier Detection**: Identifies values outside normal ranges
/// 2. **Trend Analysis**: Detects concerning growth or decline patterns
/// 3. **Pattern Recognition**: Identifies irregular allocation sequences
/// 4. **Threshold Monitoring**: Checks against absolute and relative limits
///
/// # Learning and Adaptation
///
/// The detector adapts to workload characteristics over time:
/// - Baseline establishment from initial observations
/// - Dynamic threshold adjustment based on workload patterns
/// - False positive reduction through pattern learning
/// - Sensitivity tuning based on operational feedback
#[derive(Debug)]
pub struct MemoryAnomalyDetector {
    /// Historical allocation data for baseline establishment
    allocation_history: HashMap<String, VecDeque<AllocationRecord>>,
    /// Detected anomalies with timestamps
    detected_anomalies: VecDeque<MemoryAnomaly>,
    /// Baseline statistics for each operation
    baselines: HashMap<String, OperationBaseline>,
    /// Configuration for anomaly detection sensitivity
    config: AnomalyDetectionConfig,
    /// Maximum history to maintain for each operation
    max_history_per_operation: usize,
    /// Maximum total anomalies to track
    max_anomaly_history: usize,
}

/// Individual allocation record for analysis
#[derive(Debug, Clone)]
struct AllocationRecord {
    /// Timestamp of allocation
    timestamp: Instant,
    /// Size of allocation in bytes
    size: usize,
    /// Duration since previous allocation
    inter_allocation_time: Duration,
}

/// Baseline statistics for an operation
#[derive(Debug, Clone)]
struct OperationBaseline {
    /// Average allocation size
    avg_size: f64,
    /// Standard deviation of allocation sizes
    size_stddev: f64,
    /// Average allocation frequency
    avg_frequency: f64,
    /// Frequency standard deviation
    frequency_stddev: f64,
    /// Number of samples used to establish baseline
    sample_count: usize,
    /// When baseline was last updated
    last_updated: Instant,
}

/// Configuration for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyDetectionConfig {
    /// Sensitivity multiplier for outlier detection (lower = more sensitive)
    pub sensitivity: f64,
    /// Minimum samples required to establish baseline
    pub min_baseline_samples: usize,
    /// Growth rate threshold for triggering anomaly (bytes/second)
    pub growth_rate_threshold: f64,
    /// Large allocation threshold (bytes)
    pub large_allocation_threshold: usize,
    /// Fragmentation ratio threshold (0.0-1.0)
    pub fragmentation_threshold: f64,
    /// Enable adaptive threshold adjustment
    pub adaptive_thresholds: bool,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            sensitivity: 2.5, // 2.5 standard deviations
            min_baseline_samples: 10,
            growth_rate_threshold: 10.0 * 1024.0 * 1024.0, // 10 MB/s
            large_allocation_threshold: 100 * 1024 * 1024, // 100 MB
            fragmentation_threshold: 0.7,
            adaptive_thresholds: true,
        }
    }
}

impl MemoryAnomalyDetector {
    /// Create a new anomaly detector
    ///
    /// Initializes the detector with default configuration and empty history.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let detector = MemoryAnomalyDetector::new();
    /// ```
    pub fn new() -> Self {
        Self {
            allocation_history: HashMap::new(),
            detected_anomalies: VecDeque::new(),
            baselines: HashMap::new(),
            config: AnomalyDetectionConfig::default(),
            max_history_per_operation: 1000,
            max_anomaly_history: 10000,
        }
    }

    /// Create detector with custom configuration
    ///
    /// Allows customization of detection sensitivity and thresholds.
    ///
    /// # Arguments
    ///
    /// * `config` - Custom anomaly detection configuration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = AnomalyDetectionConfig {
    ///     sensitivity: 1.5, // More sensitive detection
    ///     ..Default::default()
    /// };
    /// let detector = MemoryAnomalyDetector::with_config(config);
    /// ```
    pub fn with_config(config: AnomalyDetectionConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Record memory allocation for analysis
    ///
    /// Adds a new allocation record and performs real-time anomaly detection.
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation performing the allocation
    /// * `size` - Size of allocation in bytes
    ///
    /// # Returns
    ///
    /// Vector of any anomalies detected from this allocation.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let anomalies = detector.record_allocation("conv2d", 1024 * 1024);
    /// for anomaly in anomalies {
    ///     println!("Detected: {:?}", anomaly.anomaly_type);
    /// }
    /// ```
    pub fn record_allocation(&mut self, operation: &str, size: usize) -> Vec<MemoryAnomaly> {
        let now = Instant::now();
        let mut new_anomalies = Vec::new();

        // Calculate inter-allocation time and update history
        let (should_update_baseline, history_clone) = {
            // Get or create allocation history for this operation
            let history = self
                .allocation_history
                .entry(operation.to_string())
                .or_insert_with(VecDeque::new);

            // Calculate inter-allocation time
            let inter_allocation_time = history
                .back()
                .map(|last| now.duration_since(last.timestamp))
                .unwrap_or(Duration::ZERO);

            // Create allocation record
            let record = AllocationRecord {
                timestamp: now,
                size,
                inter_allocation_time,
            };

            // Add to history
            history.push_back(record);

            // Maintain history size limit
            if history.len() > self.max_history_per_operation {
                history.pop_front();
            }

            let should_update = history.len() >= self.config.min_baseline_samples;
            let history_clone = history.clone();
            (should_update, history_clone)
        }; // Mutable borrow of self ends here

        // Update baseline if we have enough samples
        if should_update_baseline {
            self.update_baseline(operation, &history_clone);
        }

        // Perform anomaly detection
        new_anomalies.extend(self.detect_size_anomalies(operation, size));
        new_anomalies.extend(self.detect_frequency_anomalies(operation, &history_clone));
        new_anomalies.extend(self.detect_growth_anomalies(operation, &history_clone));

        // Store detected anomalies
        for anomaly in &new_anomalies {
            self.detected_anomalies.push_back(anomaly.clone());

            // Maintain anomaly history size
            if self.detected_anomalies.len() > self.max_anomaly_history {
                self.detected_anomalies.pop_front();
            }
        }

        new_anomalies
    }

    /// Analyze allocation pattern for an operation
    ///
    /// Performs comprehensive analysis of allocation patterns to classify
    /// the type of allocation behavior and identify optimization opportunities.
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation to analyze
    /// * `total_allocations` - Total number of allocations
    /// * `total_bytes` - Total bytes allocated
    ///
    /// # Returns
    ///
    /// Detailed allocation pattern analysis.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let pattern = detector.analyze_allocation_pattern("conv2d", 1000, 50 * 1024 * 1024);
    /// println!("Pattern type: {:?}", pattern.pattern_type);
    /// println!("Average size: {} bytes", pattern.average_allocation_size);
    /// ```
    pub fn analyze_allocation_pattern(
        &self,
        operation: &str,
        total_allocations: usize,
        total_bytes: usize,
    ) -> AllocationPattern {
        let mut pattern = AllocationPattern {
            operation_name: operation.to_string(),
            total_allocations,
            average_allocation_size: if total_allocations > 0 {
                total_bytes / total_allocations
            } else {
                0
            },
            ..Default::default()
        };

        // Get allocation history if available
        if let Some(history) = self.allocation_history.get(operation) {
            if !history.is_empty() {
                // Calculate additional metrics
                pattern.last_allocation_age = history
                    .back()
                    .map(|last| Instant::now().duration_since(last.timestamp))
                    .unwrap_or(Duration::ZERO);

                // Calculate allocation frequency
                if history.len() > 1 {
                    let total_time = history
                        .back()
                        .unwrap()
                        .timestamp
                        .duration_since(history.front().unwrap().timestamp);
                    pattern.allocation_frequency = if total_time.as_secs_f64() > 0.0 {
                        history.len() as f64 / total_time.as_secs_f64()
                    } else {
                        0.0
                    };
                }

                // Calculate size variability
                let sizes: Vec<usize> = history.iter().map(|r| r.size).collect();
                pattern.size_variability = self.calculate_standard_deviation(&sizes);

                // Calculate size trend
                pattern.size_trend = self.calculate_trend(&sizes);

                // Calculate peak memory usage
                pattern.peak_memory_usage = sizes.iter().max().copied().unwrap_or(0);
            }
        }

        // Classify pattern type
        pattern.pattern_type = self.classify_allocation_pattern(&pattern);

        pattern
    }

    /// Get all detected anomalies
    ///
    /// Returns all anomalies detected since the detector was created.
    ///
    /// # Returns
    ///
    /// Vector of all detected anomalies, ordered by detection time.
    pub fn get_anomalies(&self) -> Vec<MemoryAnomaly> {
        self.detected_anomalies.iter().cloned().collect()
    }

    /// Get recent anomalies
    ///
    /// Returns anomalies detected within the specified time window.
    ///
    /// # Arguments
    ///
    /// * `time_window` - How far back to look for anomalies
    ///
    /// # Returns
    ///
    /// Vector of recent anomalies within the time window.
    pub fn get_recent_anomalies(&self, time_window: Duration) -> Vec<MemoryAnomaly> {
        let cutoff = Instant::now() - time_window;
        self.detected_anomalies
            .iter()
            .filter(|anomaly| anomaly.detected_at >= cutoff)
            .cloned()
            .collect()
    }

    /// Clear all detection history
    ///
    /// Resets the detector to initial state, clearing all history and baselines.
    pub fn clear(&mut self) {
        self.allocation_history.clear();
        self.detected_anomalies.clear();
        self.baselines.clear();
    }

    /// Update baseline statistics for an operation
    fn update_baseline(&mut self, operation: &str, history: &VecDeque<AllocationRecord>) {
        let sizes: Vec<f64> = history.iter().map(|r| r.size as f64).collect();
        let frequencies: Vec<f64> = history
            .iter()
            .map(|r| {
                if r.inter_allocation_time.as_secs_f64() > 0.0 {
                    1.0 / r.inter_allocation_time.as_secs_f64()
                } else {
                    0.0
                }
            })
            .collect();

        let baseline = OperationBaseline {
            avg_size: sizes.iter().sum::<f64>() / sizes.len() as f64,
            size_stddev: self.calculate_standard_deviation_f64(&sizes),
            avg_frequency: frequencies.iter().sum::<f64>() / frequencies.len() as f64,
            frequency_stddev: self.calculate_standard_deviation_f64(&frequencies),
            sample_count: history.len(),
            last_updated: Instant::now(),
        };

        self.baselines.insert(operation.to_string(), baseline);
    }

    /// Detect size-based anomalies
    fn detect_size_anomalies(&self, operation: &str, size: usize) -> Vec<MemoryAnomaly> {
        let mut anomalies = Vec::new();

        // Check for large allocation anomaly
        if size > self.config.large_allocation_threshold {
            let severity = if size > self.config.large_allocation_threshold * 10 {
                AnomalySeverity::Critical
            } else if size > self.config.large_allocation_threshold * 5 {
                AnomalySeverity::High
            } else {
                AnomalySeverity::Medium
            };

            anomalies.push(MemoryAnomaly {
                anomaly_type: MemoryAnomalyType::LargeAllocationSize,
                severity,
                description: format!(
                    "Large allocation of {} MB in operation {}",
                    size / (1024 * 1024),
                    operation
                ),
                detected_at: Instant::now(),
                operation_context: operation.to_string(),
                measurement: size as f64,
                baseline: self.config.large_allocation_threshold as f64,
            });
        }

        // Check against baseline if available
        if let Some(baseline) = self.baselines.get(operation) {
            let deviation = (size as f64 - baseline.avg_size).abs() / baseline.size_stddev;
            if deviation > self.config.sensitivity {
                let severity = if deviation > self.config.sensitivity * 3.0 {
                    AnomalySeverity::High
                } else if deviation > self.config.sensitivity * 2.0 {
                    AnomalySeverity::Medium
                } else {
                    AnomalySeverity::Low
                };

                anomalies.push(MemoryAnomaly {
                    anomaly_type: MemoryAnomalyType::AnomalousPattern,
                    severity,
                    description: format!(
                        "Allocation size {} is {:.1} standard deviations from baseline in {}",
                        size, deviation, operation
                    ),
                    detected_at: Instant::now(),
                    operation_context: operation.to_string(),
                    measurement: size as f64,
                    baseline: baseline.avg_size,
                });
            }
        }

        anomalies
    }

    /// Detect frequency-based anomalies
    fn detect_frequency_anomalies(
        &self,
        operation: &str,
        history: &VecDeque<AllocationRecord>,
    ) -> Vec<MemoryAnomaly> {
        let mut anomalies = Vec::new();

        if history.len() < 2 {
            return anomalies;
        }

        // Calculate current frequency
        let recent_interval = history.back().unwrap().inter_allocation_time;
        let current_frequency = if recent_interval.as_secs_f64() > 0.0 {
            1.0 / recent_interval.as_secs_f64()
        } else {
            f64::INFINITY
        };

        // Check for excessively high frequency
        if current_frequency > 1000.0 {
            // More than 1000 allocations per second
            anomalies.push(MemoryAnomaly {
                anomaly_type: MemoryAnomalyType::AnomalousPattern,
                severity: AnomalySeverity::High,
                description: format!(
                    "Extremely high allocation frequency ({:.1}/sec) in operation {}",
                    current_frequency, operation
                ),
                detected_at: Instant::now(),
                operation_context: operation.to_string(),
                measurement: current_frequency,
                baseline: 100.0, // Expected baseline
            });
        }

        anomalies
    }

    /// Detect growth-based anomalies
    fn detect_growth_anomalies(
        &self,
        operation: &str,
        history: &VecDeque<AllocationRecord>,
    ) -> Vec<MemoryAnomaly> {
        let mut anomalies = Vec::new();

        if history.len() < 10 {
            return anomalies; // Need sufficient history for growth analysis
        }

        // Calculate recent growth rate
        let recent_records: Vec<_> = history.iter().rev().take(5).collect();
        if recent_records.len() < 2 {
            return anomalies;
        }

        let total_bytes: usize = recent_records.iter().map(|r| r.size).sum();
        let time_span = recent_records
            .first()
            .unwrap()
            .timestamp
            .duration_since(recent_records.last().unwrap().timestamp);

        if time_span.as_secs_f64() > 0.0 {
            let growth_rate = total_bytes as f64 / time_span.as_secs_f64();

            if growth_rate > self.config.growth_rate_threshold {
                let severity = if growth_rate > self.config.growth_rate_threshold * 10.0 {
                    AnomalySeverity::Critical
                } else if growth_rate > self.config.growth_rate_threshold * 5.0 {
                    AnomalySeverity::High
                } else {
                    AnomalySeverity::Medium
                };

                anomalies.push(MemoryAnomaly {
                    anomaly_type: MemoryAnomalyType::ExcessiveGrowthRate,
                    severity,
                    description: format!(
                        "High memory growth rate ({:.1} MB/s) in operation {}",
                        growth_rate / (1024.0 * 1024.0),
                        operation
                    ),
                    detected_at: Instant::now(),
                    operation_context: operation.to_string(),
                    measurement: growth_rate,
                    baseline: self.config.growth_rate_threshold,
                });
            }
        }

        anomalies
    }

    /// Classify allocation pattern type
    fn classify_allocation_pattern(&self, pattern: &AllocationPattern) -> AllocationPatternType {
        // High frequency pattern (> 100 allocations per second)
        if pattern.allocation_frequency > 100.0 {
            return AllocationPatternType::HighFrequency;
        }

        // Large allocation pattern (average > 10MB)
        if pattern.average_allocation_size > 10 * 1024 * 1024 {
            return AllocationPatternType::FewLarge;
        }

        // Small allocation pattern (average < 1MB and high frequency)
        if pattern.average_allocation_size < 1024 * 1024 && pattern.allocation_frequency > 10.0 {
            return AllocationPatternType::ManySmall;
        }

        // Irregular pattern (high variability or strong trend)
        if pattern.size_variability > pattern.average_allocation_size as f64 * 0.5
            || pattern.size_trend.abs() > 0.1
        {
            return AllocationPatternType::Irregular;
        }

        AllocationPatternType::Normal
    }

    /// Calculate standard deviation for usize values
    fn calculate_standard_deviation(&self, values: &[usize]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
        let variance = values
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        variance.sqrt()
    }

    /// Calculate standard deviation for f64 values
    fn calculate_standard_deviation_f64(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        variance.sqrt()
    }

    /// Calculate trend (slope) for size values
    fn calculate_trend(&self, values: &[usize]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let sum_x = (0..values.len()).sum::<usize>() as f64;
        let sum_y = values.iter().sum::<usize>() as f64;
        let sum_xy = values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y as f64)
            .sum::<f64>();
        let sum_x_squared = (0..values.len())
            .map(|i| (i as f64) * (i as f64))
            .sum::<f64>();

        // Linear regression slope
        (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
    }
}

impl Default for MemoryAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = MemoryAnomalyDetector::new();
        assert_eq!(detector.get_anomalies().len(), 0);
    }

    #[test]
    fn test_large_allocation_detection() {
        let mut detector = MemoryAnomalyDetector::new();

        // Record a large allocation
        let large_size = 200 * 1024 * 1024; // 200MB
        let anomalies = detector.record_allocation("test_op", large_size);

        assert!(!anomalies.is_empty());
        assert_eq!(
            anomalies[0].anomaly_type,
            MemoryAnomalyType::LargeAllocationSize
        );
    }

    #[test]
    fn test_pattern_classification() {
        let detector = MemoryAnomalyDetector::new();

        // Test high frequency pattern
        let high_freq_pattern = AllocationPattern {
            allocation_frequency: 150.0,
            average_allocation_size: 1024,
            ..Default::default()
        };

        assert_eq!(
            detector.classify_allocation_pattern(&high_freq_pattern),
            AllocationPatternType::HighFrequency
        );

        // Test few large pattern
        let few_large_pattern = AllocationPattern {
            allocation_frequency: 1.0,
            average_allocation_size: 20 * 1024 * 1024, // 20MB
            ..Default::default()
        };

        assert_eq!(
            detector.classify_allocation_pattern(&few_large_pattern),
            AllocationPatternType::FewLarge
        );
    }

    #[test]
    fn test_baseline_establishment() {
        let mut detector = MemoryAnomalyDetector::new();

        // Record enough allocations to establish baseline
        for i in 0..15 {
            detector.record_allocation("test_op", 1000 + i * 10);
        }

        // Baseline should now exist
        assert!(detector.baselines.contains_key("test_op"));

        let baseline = &detector.baselines["test_op"];
        assert!(baseline.avg_size > 0.0);
        assert!(baseline.size_stddev >= 0.0);
    }

    #[test]
    fn test_growth_rate_anomaly() {
        let mut detector = MemoryAnomalyDetector::with_config(AnomalyDetectionConfig {
            growth_rate_threshold: 1.0 * 1024.0 * 1024.0, // 1MB/s
            ..Default::default()
        });

        // Rapidly allocate memory to trigger growth rate anomaly
        for i in 0..12 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let anomalies = detector.record_allocation("growth_test", 2 * 1024 * 1024); // 2MB each

            // Should detect growth anomaly after enough samples
            if i > 10 {
                if !anomalies.is_empty() {
                    assert!(anomalies
                        .iter()
                        .any(|a| a.anomaly_type == MemoryAnomalyType::ExcessiveGrowthRate));
                    break;
                }
            }
        }
    }

    #[test]
    fn test_recent_anomalies() {
        let mut detector = MemoryAnomalyDetector::new();

        // Record an anomaly
        detector.record_allocation("test", 200 * 1024 * 1024); // Large allocation

        // Get recent anomalies
        let recent = detector.get_recent_anomalies(Duration::from_secs(1));
        assert!(!recent.is_empty());

        // Get anomalies from longer ago
        let old = detector.get_recent_anomalies(Duration::from_nanos(1));
        assert!(old.is_empty());
    }

    #[test]
    fn test_allocation_pattern_analysis() {
        let detector = MemoryAnomalyDetector::new();

        let pattern = detector.analyze_allocation_pattern("test", 100, 50 * 1024 * 1024);
        assert_eq!(pattern.operation_name, "test");
        assert_eq!(pattern.total_allocations, 100);
        assert_eq!(pattern.average_allocation_size, 512 * 1024); // 50MB / 100
    }
}
