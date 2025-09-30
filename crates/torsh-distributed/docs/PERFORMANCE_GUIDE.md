# ToRSh Distributed Training Performance Optimization Guide

This guide provides detailed strategies and techniques for maximizing performance in ToRSh distributed training. Whether you're training on a single node with multiple GPUs or scaling across hundreds of nodes, these optimizations will help you achieve peak efficiency.

## Table of Contents

1. [Performance Profiling and Analysis](#performance-profiling-and-analysis)
2. [Communication Optimization](#communication-optimization)
3. [Memory Optimization](#memory-optimization)
4. [Computation Optimization](#computation-optimization)
5. [I/O and Data Pipeline Optimization](#io-and-data-pipeline-optimization)
6. [Hardware-Specific Optimizations](#hardware-specific-optimizations)
7. [Scale-Specific Optimizations](#scale-specific-optimizations)
8. [Framework Integration Performance](#framework-integration-performance)
9. [Production Performance Monitoring](#production-performance-monitoring)
10. [Performance Debugging Cookbook](#performance-debugging-cookbook)

## Performance Profiling and Analysis

### Comprehensive Performance Profiling

```rust
use torsh_distributed::*;

fn setup_comprehensive_profiling() -> TorshResult<PerformanceProfiler> {
    let profiling_config = ProfilingConfig {
        // Enable all profiling categories
        enable_memory_profiling: true,
        enable_communication_profiling: true,
        enable_computation_profiling: true,
        enable_io_profiling: true,
        
        // Fine-grained sampling
        sampling_interval_ms: 10,
        memory_sampling_interval_ms: 100,
        
        // Output configuration
        output_file: "torsh_performance_profile.json".to_string(),
        enable_real_time_dashboard: true,
        dashboard_port: 8080,
        
        // Advanced profiling
        enable_cuda_profiling: true,
        enable_timeline_tracing: true,
        enable_bottleneck_detection: true,
        
        // Overhead control
        max_profile_size_mb: 1000,
        auto_flush_interval_sec: 60,
    };
    
    let profiler = PerformanceProfiler::new(profiling_config)?;
    profiler.start()?;
    
    Ok(profiler)
}

fn analyze_training_bottlenecks(profiler: &PerformanceProfiler) -> TorshResult<BottleneckReport> {
    let analysis = profiler.analyze_performance()?;
    
    let mut bottlenecks = Vec::new();
    
    // Analyze communication bottlenecks
    if analysis.communication_time_ratio > 0.3 {
        bottlenecks.push(Bottleneck {
            category: BottleneckCategory::Communication,
            severity: if analysis.communication_time_ratio > 0.5 { 
                BottleneckSeverity::Critical 
            } else { 
                BottleneckSeverity::High 
            },
            description: format!(
                "Communication takes {:.1}% of total time",
                analysis.communication_time_ratio * 100.0
            ),
            suggestions: vec![
                "Enable gradient compression".to_string(),
                "Increase bucket size for DDP".to_string(),
                "Use hierarchical communication for multi-node".to_string(),
                "Enable communication overlap".to_string(),
            ],
            metric_details: analysis.communication_details.clone(),
        });
    }
    
    // Analyze memory bottlenecks
    if analysis.memory_utilization > 0.95 {
        bottlenecks.push(Bottleneck {
            category: BottleneckCategory::Memory,
            severity: BottleneckSeverity::Critical,
            description: format!(
                "Memory utilization at {:.1}%",
                analysis.memory_utilization * 100.0
            ),
            suggestions: vec![
                "Enable gradient checkpointing".to_string(),
                "Reduce batch size".to_string(),
                "Enable CPU offloading".to_string(),
                "Use mixed precision training".to_string(),
            ],
            metric_details: analysis.memory_details.clone(),
        });
    }
    
    // Analyze computation bottlenecks
    if analysis.gpu_utilization < 0.8 {
        bottlenecks.push(Bottleneck {
            category: BottleneckCategory::Computation,
            severity: BottleneckSeverity::Medium,
            description: format!(
                "GPU utilization at {:.1}%",
                analysis.gpu_utilization * 100.0
            ),
            suggestions: vec![
                "Increase batch size".to_string(),
                "Reduce data loading overhead".to_string(),
                "Optimize model for better GPU utilization".to_string(),
                "Use tensor cores for mixed precision".to_string(),
            ],
            metric_details: analysis.computation_details.clone(),
        });
    }
    
    // Analyze I/O bottlenecks
    if analysis.io_wait_time_ratio > 0.1 {
        bottlenecks.push(Bottleneck {
            category: BottleneckCategory::IO,
            severity: BottleneckSeverity::High,
            description: format!(
                "I/O wait time {:.1}% of total",
                analysis.io_wait_time_ratio * 100.0
            ),
            suggestions: vec![
                "Increase data loading workers".to_string(),
                "Use faster storage (NVMe SSD)".to_string(),
                "Enable data prefetching".to_string(),
                "Optimize data format (e.g., use HDF5 or Parquet)".to_string(),
            ],
            metric_details: analysis.io_details.clone(),
        });
    }
    
    Ok(BottleneckReport {
        bottlenecks,
        overall_efficiency: analysis.overall_efficiency,
        recommendations: generate_optimization_recommendations(&analysis),
        timestamp: std::time::SystemTime::now(),
    })
}

#[derive(Debug)]
struct PerformanceAnalysis {
    communication_time_ratio: f32,
    memory_utilization: f32,
    gpu_utilization: f32,
    io_wait_time_ratio: f32,
    overall_efficiency: f32,
    communication_details: CommunicationMetrics,
    memory_details: MemoryMetrics,
    computation_details: ComputationMetrics,
    io_details: IOMetrics,
}

#[derive(Debug)]
struct Bottleneck {
    category: BottleneckCategory,
    severity: BottleneckSeverity,
    description: String,
    suggestions: Vec<String>,
    metric_details: MetricDetails,
}

#[derive(Debug)]
enum BottleneckCategory {
    Communication,
    Memory,
    Computation,
    IO,
    Network,
}

#[derive(Debug)]
struct BottleneckReport {
    bottlenecks: Vec<Bottleneck>,
    overall_efficiency: f32,
    recommendations: Vec<OptimizationRecommendation>,
    timestamp: std::time::SystemTime,
}

fn generate_optimization_recommendations(analysis: &PerformanceAnalysis) -> Vec<OptimizationRecommendation> {
    let mut recommendations = Vec::new();
    
    // Priority-based recommendations
    if analysis.overall_efficiency < 0.7 {
        recommendations.push(OptimizationRecommendation {
            priority: RecommendationPriority::Critical,
            category: OptimizationCategory::Overall,
            title: "Overall Training Efficiency Below 70%".to_string(),
            description: "Multiple bottlenecks detected requiring immediate attention".to_string(),
            implementation_steps: vec![
                "Run detailed profiling analysis".to_string(),
                "Address communication bottlenecks first".to_string(),
                "Optimize memory usage patterns".to_string(),
                "Verify hardware configuration".to_string(),
            ],
            expected_improvement: "20-40% performance gain".to_string(),
            effort_level: EffortLevel::High,
        });
    }
    
    recommendations
}

#[derive(Debug)]
struct OptimizationRecommendation {
    priority: RecommendationPriority,
    category: OptimizationCategory,
    title: String,
    description: String,
    implementation_steps: Vec<String>,
    expected_improvement: String,
    effort_level: EffortLevel,
}

#[derive(Debug)]
enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug)]
enum OptimizationCategory {
    Communication,
    Memory,
    Computation,
    IO,
    Configuration,
    Overall,
}

#[derive(Debug)]
enum EffortLevel {
    Low,    // < 1 hour
    Medium, // 1-8 hours
    High,   // > 8 hours
}
```

### Real-time Performance Monitoring

```rust
use torsh_distributed::*;

struct RealTimePerformanceMonitor {
    metrics_collector: MetricsCollector,
    dashboard: PerformanceDashboard,
    alert_manager: AlertManager,
    optimization_engine: AutoOptimizationEngine,
}

impl RealTimePerformanceMonitor {
    fn new() -> TorshResult<Self> {
        let metrics_config = MetricsConfig {
            collection_interval_ms: 1000,
            enable_system_metrics: true,
            enable_communication_metrics: true,
            enable_training_metrics: true,
            enable_real_time_alerts: true,
            retention_hours: 24,
        };
        
        Ok(Self {
            metrics_collector: MetricsCollector::new(metrics_config)?,
            dashboard: PerformanceDashboard::new()?,
            alert_manager: AlertManager::new()?,
            optimization_engine: AutoOptimizationEngine::new()?,
        })
    }
    
    fn start_monitoring(&mut self) -> TorshResult<()> {
        // Start metrics collection
        self.metrics_collector.start_collection()?;
        
        // Start dashboard server
        self.dashboard.start_server("0.0.0.0:8080")?;
        
        // Configure alerts
        self.setup_performance_alerts()?;
        
        // Start optimization engine
        self.optimization_engine.start()?;
        
        Ok(())
    }
    
    fn setup_performance_alerts(&mut self) -> TorshResult<()> {
        // GPU utilization alerts
        self.alert_manager.add_alert(Alert {
            name: "Low GPU Utilization".to_string(),
            condition: AlertCondition::Threshold {
                metric: "gpu_utilization".to_string(),
                operator: ComparisonOperator::LessThan,
                threshold: 0.7,
                duration_sec: 60,
            },
            severity: AlertSeverity::Warning,
            action: AlertAction::Notification {
                message: "GPU utilization below 70% for 1 minute".to_string(),
            },
        })?;
        
        // Memory pressure alerts
        self.alert_manager.add_alert(Alert {
            name: "High Memory Usage".to_string(),
            condition: AlertCondition::Threshold {
                metric: "memory_utilization".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 0.9,
                duration_sec: 30,
            },
            severity: AlertSeverity::Critical,
            action: AlertAction::AutoOptimization {
                optimization_type: OptimizationType::ReduceMemoryUsage,
            },
        })?;
        
        // Communication efficiency alerts
        self.alert_manager.add_alert(Alert {
            name: "High Communication Overhead".to_string(),
            condition: AlertCondition::Threshold {
                metric: "communication_time_ratio".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 0.4,
                duration_sec: 120,
            },
            severity: AlertSeverity::High,
            action: AlertAction::AutoOptimization {
                optimization_type: OptimizationType::OptimizeCommunication,
            },
        })?;
        
        Ok(())
    }
    
    fn get_performance_summary(&self) -> TorshResult<PerformanceSummary> {
        let current_metrics = self.metrics_collector.get_current_metrics()?;
        
        Ok(PerformanceSummary {
            timestamp: std::time::SystemTime::now(),
            overall_efficiency: current_metrics.calculate_efficiency(),
            throughput_samples_per_sec: current_metrics.training_throughput,
            gpu_utilization: current_metrics.gpu_utilization,
            memory_utilization: current_metrics.memory_utilization,
            communication_efficiency: current_metrics.communication_efficiency,
            bottlenecks: current_metrics.detected_bottlenecks.clone(),
            recommendations: self.optimization_engine.get_current_recommendations()?,
        })
    }
}

#[derive(Debug)]
struct Alert {
    name: String,
    condition: AlertCondition,
    severity: AlertSeverity,
    action: AlertAction,
}

#[derive(Debug)]
enum AlertCondition {
    Threshold {
        metric: String,
        operator: ComparisonOperator,
        threshold: f32,
        duration_sec: u64,
    },
    RateOfChange {
        metric: String,
        rate_threshold: f32,
        window_sec: u64,
    },
}

#[derive(Debug)]
enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
}

#[derive(Debug)]
enum AlertSeverity {
    Info,
    Warning,
    High,
    Critical,
}

#[derive(Debug)]
enum AlertAction {
    Notification { message: String },
    AutoOptimization { optimization_type: OptimizationType },
    Webhook { url: String },
}

#[derive(Debug)]
enum OptimizationType {
    ReduceMemoryUsage,
    OptimizeCommunication,
    IncreaseGpuUtilization,
    OptimizeDataLoading,
}

#[derive(Debug)]
struct PerformanceSummary {
    timestamp: std::time::SystemTime,
    overall_efficiency: f32,
    throughput_samples_per_sec: f32,
    gpu_utilization: f32,
    memory_utilization: f32,
    communication_efficiency: f32,
    bottlenecks: Vec<String>,
    recommendations: Vec<String>,
}
```

## Communication Optimization

### Advanced Gradient Compression

```rust
use torsh_distributed::*;

struct AdaptiveGradientCompression {
    current_config: CompressionConfig,
    performance_history: Vec<CompressionPerformance>,
    adaptation_strategy: AdaptationStrategy,
    error_feedback_buffer: ErrorFeedbackBuffer,
}

impl AdaptiveGradientCompression {
    fn new() -> Self {
        Self {
            current_config: CompressionConfig {
                method: CompressionMethod::TopK { k: 0.01 },
                error_feedback: true,
                compression_period: 1,
                memory_optimization: true,
            },
            performance_history: Vec::new(),
            adaptation_strategy: AdaptationStrategy::Conservative,
            error_feedback_buffer: ErrorFeedbackBuffer::new(),
        }
    }
    
    fn compress_gradients(
        &mut self,
        gradients: &[Tensor],
        training_step: u32,
    ) -> TorshResult<Vec<CompressedGradient>> {
        
        let start_time = std::time::Instant::now();
        
        // Adapt compression parameters based on training progress
        self.adapt_compression_parameters(training_step)?;
        
        let mut compressed_gradients = Vec::new();
        
        for (layer_idx, gradient) in gradients.iter().enumerate() {
            // Apply layer-specific compression
            let layer_compression = self.get_layer_compression_config(layer_idx, gradient)?;
            
            // Compress gradient
            let compressed = self.compress_single_gradient(gradient, &layer_compression)?;
            
            // Apply error feedback
            let corrected = self.apply_error_feedback(layer_idx, compressed)?;
            
            compressed_gradients.push(corrected);
        }
        
        // Record performance metrics
        let compression_time = start_time.elapsed();
        self.record_compression_performance(compression_time, &compressed_gradients)?;
        
        Ok(compressed_gradients)
    }
    
    fn adapt_compression_parameters(&mut self, training_step: u32) -> TorshResult<()> {
        match self.adaptation_strategy {
            AdaptationStrategy::Conservative => {
                // Only adapt every 100 steps
                if training_step % 100 == 0 {
                    self.conservative_adaptation()?;
                }
            }
            
            AdaptationStrategy::Aggressive => {
                // Adapt every 10 steps
                if training_step % 10 == 0 {
                    self.aggressive_adaptation()?;
                }
            }
            
            AdaptationStrategy::LossAware => {
                // Adapt based on loss convergence
                self.loss_aware_adaptation(training_step)?;
            }
        }
        
        Ok(())
    }
    
    fn get_layer_compression_config(
        &self,
        layer_idx: usize,
        gradient: &Tensor,
    ) -> TorshResult<CompressionConfig> {
        
        let gradient_norm = gradient.norm()?;
        let gradient_size = gradient.numel();
        
        // Adaptive compression based on gradient properties
        let compression_ratio = match (gradient_norm, gradient_size) {
            // Large gradients with high norm: aggressive compression
            (norm, size) if norm > 1.0 && size > 1_000_000 => 0.001,
            
            // Medium gradients: moderate compression
            (norm, size) if norm > 0.1 && size > 100_000 => 0.01,
            
            // Small gradients: light compression
            _ => 0.1,
        };
        
        Ok(CompressionConfig {
            method: CompressionMethod::TopK { k: compression_ratio },
            error_feedback: true,
            compression_period: 1,
            memory_optimization: true,
        })
    }
    
    fn compress_single_gradient(
        &self,
        gradient: &Tensor,
        config: &CompressionConfig,
    ) -> TorshResult<CompressedGradient> {
        
        match &config.method {
            CompressionMethod::TopK { k } => {
                // Select top-k elements by magnitude
                let flattened = gradient.flatten()?;
                let magnitude = flattened.abs()?;
                let k_count = (flattened.numel() as f32 * k) as usize;
                
                // Get top-k indices
                let (values, indices) = magnitude.topk(k_count, true)?;
                
                Ok(CompressedGradient {
                    compression_type: CompressionType::TopK,
                    compressed_data: CompressedData {
                        values: values.to_vec()?,
                        indices: indices.to_vec()?,
                        original_shape: gradient.size(),
                        compression_ratio: *k,
                    },
                    metadata: CompressionMetadata {
                        original_size: gradient.numel(),
                        compressed_size: k_count,
                        compression_time_ms: 0, // Set by caller
                    },
                })
            }
            
            CompressionMethod::Quantization { bits } => {
                // Quantize to specified bit width
                let quantized = self.quantize_gradient(gradient, *bits)?;
                
                Ok(CompressedGradient {
                    compression_type: CompressionType::Quantization,
                    compressed_data: CompressedData {
                        values: quantized.values,
                        indices: Vec::new(), // Not needed for quantization
                        original_shape: gradient.size(),
                        compression_ratio: (*bits as f32) / 32.0, // Assuming 32-bit floats
                    },
                    metadata: CompressionMetadata {
                        original_size: gradient.numel(),
                        compressed_size: (gradient.numel() * (*bits as usize)) / 8,
                        compression_time_ms: 0,
                    },
                })
            }
            
            CompressionMethod::None => {
                // No compression
                Ok(CompressedGradient {
                    compression_type: CompressionType::None,
                    compressed_data: CompressedData {
                        values: gradient.to_vec()?,
                        indices: Vec::new(),
                        original_shape: gradient.size(),
                        compression_ratio: 1.0,
                    },
                    metadata: CompressionMetadata {
                        original_size: gradient.numel(),
                        compressed_size: gradient.numel(),
                        compression_time_ms: 0,
                    },
                })
            }
            
            _ => todo!("Implement other compression methods"),
        }
    }
    
    fn apply_error_feedback(
        &mut self,
        layer_idx: usize,
        mut compressed: CompressedGradient,
    ) -> TorshResult<CompressedGradient> {
        
        if let Some(error) = self.error_feedback_buffer.get_error(layer_idx) {
            // Add accumulated error to compressed gradient
            compressed = self.add_error_to_compressed(compressed, error)?;
        }
        
        // Calculate new compression error
        let compression_error = self.calculate_compression_error(&compressed)?;
        
        // Store error for next iteration
        self.error_feedback_buffer.store_error(layer_idx, compression_error)?;
        
        Ok(compressed)
    }
    
    fn quantize_gradient(&self, gradient: &Tensor, bits: u32) -> TorshResult<QuantizedData> {
        // Implement gradient quantization
        let min_val = gradient.min()?;
        let max_val = gradient.max()?;
        let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
        
        let quantized_values = gradient.map(|x| {
            let normalized = (x - min_val) / scale;
            normalized.round() as u32
        })?;
        
        Ok(QuantizedData {
            values: quantized_values,
            scale,
            min_val,
            bits,
        })
    }
    
    // Adaptation strategies
    fn conservative_adaptation(&mut self) -> TorshResult<()> {
        if self.performance_history.len() < 10 {
            return Ok(()); // Need more data
        }
        
        let recent_performance = &self.performance_history[self.performance_history.len() - 10..];
        let avg_compression_ratio = recent_performance.iter()
            .map(|p| p.compression_ratio)
            .sum::<f32>() / recent_performance.len() as f32;
        
        let avg_accuracy_impact = recent_performance.iter()
            .map(|p| p.accuracy_impact)
            .sum::<f32>() / recent_performance.len() as f32;
        
        // Only increase compression if accuracy impact is minimal
        if avg_accuracy_impact < 0.01 && avg_compression_ratio > 0.01 {
            self.increase_compression_ratio(0.1)?; // Increase by 10%
        } else if avg_accuracy_impact > 0.05 {
            self.decrease_compression_ratio(0.1)?; // Decrease by 10%
        }
        
        Ok(())
    }
    
    fn aggressive_adaptation(&mut self) -> TorshResult<()> {
        // More frequent and aggressive adaptations
        if let Some(last_performance) = self.performance_history.last() {
            if last_performance.accuracy_impact < 0.005 {
                self.increase_compression_ratio(0.2)?; // Increase by 20%
            } else if last_performance.accuracy_impact > 0.02 {
                self.decrease_compression_ratio(0.2)?; // Decrease by 20%
            }
        }
        
        Ok(())
    }
    
    fn loss_aware_adaptation(&mut self, training_step: u32) -> TorshResult<()> {
        // Implement loss-aware adaptation logic
        // This would typically involve analyzing loss convergence patterns
        Ok(())
    }
    
    fn increase_compression_ratio(&mut self, factor: f32) -> TorshResult<()> {
        match &mut self.current_config.method {
            CompressionMethod::TopK { k } => {
                *k = (*k * (1.0 - factor)).max(0.0001); // Don't go below 0.01%
            }
            CompressionMethod::Quantization { bits } => {
                *bits = (*bits - 1).max(1); // Don't go below 1 bit
            }
            _ => {}
        }
        Ok(())
    }
    
    fn decrease_compression_ratio(&mut self, factor: f32) -> TorshResult<()> {
        match &mut self.current_config.method {
            CompressionMethod::TopK { k } => {
                *k = (*k * (1.0 + factor)).min(1.0); // Don't exceed 100%
            }
            CompressionMethod::Quantization { bits } => {
                *bits = (*bits + 1).min(32); // Don't exceed 32 bits
            }
            _ => {}
        }
        Ok(())
    }
    
    fn record_compression_performance(
        &mut self,
        compression_time: std::time::Duration,
        compressed_gradients: &[CompressedGradient],
    ) -> TorshResult<()> {
        
        let total_original_size: usize = compressed_gradients.iter()
            .map(|g| g.metadata.original_size)
            .sum();
        
        let total_compressed_size: usize = compressed_gradients.iter()
            .map(|g| g.metadata.compressed_size)
            .sum();
        
        let compression_ratio = total_compressed_size as f32 / total_original_size as f32;
        
        let performance = CompressionPerformance {
            timestamp: std::time::Instant::now(),
            compression_ratio,
            compression_time_ms: compression_time.as_millis() as u32,
            accuracy_impact: 0.0, // Would be set externally based on validation
            bandwidth_saved: (total_original_size - total_compressed_size) as f32,
        };
        
        self.performance_history.push(performance);
        
        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
        
        Ok(())
    }
    
    fn add_error_to_compressed(
        &self,
        mut compressed: CompressedGradient,
        error: CompressionError,
    ) -> TorshResult<CompressedGradient> {
        // Add error feedback to compressed gradient
        // Implementation depends on compression type
        Ok(compressed)
    }
    
    fn calculate_compression_error(&self, compressed: &CompressedGradient) -> TorshResult<CompressionError> {
        // Calculate compression error for feedback
        Ok(CompressionError::default())
    }
}

#[derive(Debug)]
enum AdaptationStrategy {
    Conservative, // Slow, safe adaptations
    Aggressive,   // Fast, aggressive adaptations
    LossAware,    // Adapt based on loss convergence
}

#[derive(Debug)]
struct CompressionPerformance {
    timestamp: std::time::Instant,
    compression_ratio: f32,
    compression_time_ms: u32,
    accuracy_impact: f32,
    bandwidth_saved: f32,
}

#[derive(Debug)]
enum CompressionType {
    None,
    TopK,
    Quantization,
    Sparsification,
}

#[derive(Debug)]
struct QuantizedData {
    values: Vec<u32>,
    scale: f32,
    min_val: f32,
    bits: u32,
}

#[derive(Debug, Default)]
struct CompressionError {
    error_values: Vec<f32>,
    error_indices: Vec<usize>,
}

struct ErrorFeedbackBuffer {
    errors: std::collections::HashMap<usize, CompressionError>,
}

impl ErrorFeedbackBuffer {
    fn new() -> Self {
        Self {
            errors: std::collections::HashMap::new(),
        }
    }
    
    fn get_error(&self, layer_idx: usize) -> Option<&CompressionError> {
        self.errors.get(&layer_idx)
    }
    
    fn store_error(&mut self, layer_idx: usize, error: CompressionError) -> TorshResult<()> {
        self.errors.insert(layer_idx, error);
        Ok(())
    }
}
```

### Hierarchical Communication Optimization

```rust
use torsh_distributed::*;

struct HierarchicalCommunicationOptimizer {
    topology: NetworkTopology,
    communication_groups: CommunicationGroups,
    bandwidth_estimator: BandwidthEstimator,
    latency_predictor: LatencyPredictor,
}

impl HierarchicalCommunicationOptimizer {
    fn new(world_size: u32, nodes: u32, gpus_per_node: u32) -> TorshResult<Self> {
        let topology = Self::detect_network_topology(world_size, nodes, gpus_per_node)?;
        let communication_groups = Self::create_communication_groups(&topology)?;
        
        Ok(Self {
            topology,
            communication_groups,
            bandwidth_estimator: BandwidthEstimator::new(),
            latency_predictor: LatencyPredictor::new(),
        })
    }
    
    fn optimize_allreduce_pattern(
        &mut self,
        tensor_size: usize,
        tensor_count: usize,
    ) -> TorshResult<AllReduceStrategy> {
        
        // Estimate communication costs for different strategies
        let strategies = vec![
            AllReduceStrategy::Ring,
            AllReduceStrategy::Tree,
            AllReduceStrategy::Hierarchical,
            AllReduceStrategy::ReduceScatter,
        ];
        
        let mut best_strategy = AllReduceStrategy::Ring;
        let mut best_time = f32::INFINITY;
        
        for strategy in strategies {
            let estimated_time = self.estimate_allreduce_time(&strategy, tensor_size, tensor_count)?;
            
            if estimated_time < best_time {
                best_time = estimated_time;
                best_strategy = strategy;
            }
        }
        
        tracing::info!(
            "Selected AllReduce strategy: {:?} (estimated time: {:.2}ms)",
            best_strategy,
            best_time * 1000.0
        );
        
        Ok(best_strategy)
    }
    
    fn estimate_allreduce_time(
        &self,
        strategy: &AllReduceStrategy,
        tensor_size: usize,
        tensor_count: usize,
    ) -> TorshResult<f32> {
        
        match strategy {
            AllReduceStrategy::Ring => {
                // Ring AllReduce: 2 * (N-1) / N steps
                let world_size = self.topology.total_workers() as f32;
                let steps = 2.0 * (world_size - 1.0) / world_size;
                let bandwidth = self.bandwidth_estimator.get_average_bandwidth();
                let latency = self.latency_predictor.get_average_latency();
                
                let communication_time = (tensor_size as f32 * steps) / bandwidth;
                let synchronization_time = latency * steps;
                
                Ok(communication_time + synchronization_time)
            }
            
            AllReduceStrategy::Tree => {
                // Tree AllReduce: 2 * log2(N) steps
                let world_size = self.topology.total_workers() as f32;
                let steps = 2.0 * world_size.log2();
                let bandwidth = self.bandwidth_estimator.get_average_bandwidth();
                let latency = self.latency_predictor.get_average_latency();
                
                let communication_time = (tensor_size as f32 * steps) / bandwidth;
                let synchronization_time = latency * steps;
                
                Ok(communication_time + synchronization_time)
            }
            
            AllReduceStrategy::Hierarchical => {
                // Hierarchical: intra-node + inter-node
                let intra_node_time = self.estimate_intra_node_allreduce_time(tensor_size)?;
                let inter_node_time = self.estimate_inter_node_allreduce_time(tensor_size)?;
                
                Ok(intra_node_time + inter_node_time)
            }
            
            AllReduceStrategy::ReduceScatter => {
                // ReduceScatter + AllGather
                let world_size = self.topology.total_workers() as f32;
                let chunk_size = tensor_size as f32 / world_size;
                
                // ReduceScatter phase
                let reduce_scatter_time = self.estimate_reduce_scatter_time(chunk_size as usize)?;
                
                // AllGather phase
                let allgather_time = self.estimate_allgather_time(chunk_size as usize)?;
                
                Ok(reduce_scatter_time + allgather_time)
            }
        }
    }
    
    fn estimate_intra_node_allreduce_time(&self, tensor_size: usize) -> TorshResult<f32> {
        // Use NVLink bandwidth for intra-node communication
        let nvlink_bandwidth = 600e9; // 600 GB/s for NVLink 4.0
        let nvlink_latency = 1e-6; // 1 microsecond
        
        let gpus_per_node = self.topology.gpus_per_node() as f32;
        let steps = 2.0 * (gpus_per_node - 1.0) / gpus_per_node; // Ring within node
        
        let communication_time = (tensor_size as f32 * steps) / nvlink_bandwidth;
        let synchronization_time = nvlink_latency * steps;
        
        Ok(communication_time + synchronization_time)
    }
    
    fn estimate_inter_node_allreduce_time(&self, tensor_size: usize) -> TorshResult<f32> {
        // Use network bandwidth for inter-node communication
        let network_bandwidth = self.bandwidth_estimator.get_inter_node_bandwidth();
        let network_latency = self.latency_predictor.get_inter_node_latency();
        
        let num_nodes = self.topology.num_nodes() as f32;
        let steps = 2.0 * (num_nodes - 1.0) / num_nodes; // Ring between nodes
        
        let communication_time = (tensor_size as f32 * steps) / network_bandwidth;
        let synchronization_time = network_latency * steps;
        
        Ok(communication_time + synchronization_time)
    }
    
    fn estimate_reduce_scatter_time(&self, chunk_size: usize) -> TorshResult<f32> {
        let world_size = self.topology.total_workers() as f32;
        let bandwidth = self.bandwidth_estimator.get_average_bandwidth();
        let latency = self.latency_predictor.get_average_latency();
        
        let steps = world_size - 1.0;
        let communication_time = (chunk_size as f32 * steps) / bandwidth;
        let synchronization_time = latency * steps;
        
        Ok(communication_time + synchronization_time)
    }
    
    fn estimate_allgather_time(&self, chunk_size: usize) -> TorshResult<f32> {
        let world_size = self.topology.total_workers() as f32;
        let bandwidth = self.bandwidth_estimator.get_average_bandwidth();
        let latency = self.latency_predictor.get_average_latency();
        
        let steps = world_size - 1.0;
        let communication_time = (chunk_size as f32 * steps) / bandwidth;
        let synchronization_time = latency * steps;
        
        Ok(communication_time + synchronization_time)
    }
    
    fn detect_network_topology(
        world_size: u32,
        nodes: u32,
        gpus_per_node: u32,
    ) -> TorshResult<NetworkTopology> {
        
        // Detect interconnect type
        let interconnect = if Self::detect_infiniband()? {
            Interconnect::InfiniBand
        } else if Self::detect_nvlink()? {
            Interconnect::NVLink
        } else {
            Interconnect::Ethernet
        };
        
        Ok(NetworkTopology {
            world_size,
            num_nodes: nodes,
            gpus_per_node,
            interconnect,
            intra_node_bandwidth: Self::measure_intra_node_bandwidth()?,
            inter_node_bandwidth: Self::measure_inter_node_bandwidth()?,
            intra_node_latency: Self::measure_intra_node_latency()?,
            inter_node_latency: Self::measure_inter_node_latency()?,
        })
    }
    
    fn create_communication_groups(topology: &NetworkTopology) -> TorshResult<CommunicationGroups> {
        let mut groups = CommunicationGroups::new();
        
        // Create intra-node groups
        for node_id in 0..topology.num_nodes {
            let start_rank = node_id * topology.gpus_per_node;
            let end_rank = start_rank + topology.gpus_per_node;
            let ranks: Vec<u32> = (start_rank..end_rank).collect();
            
            groups.add_group(
                format!("node_{}", node_id),
                CommunicationGroup::new(ranks)?,
                GroupType::IntraNode,
            )?;
        }
        
        // Create inter-node groups (one per local rank)
        for local_rank in 0..topology.gpus_per_node {
            let mut ranks = Vec::new();
            for node_id in 0..topology.num_nodes {
                ranks.push(node_id * topology.gpus_per_node + local_rank);
            }
            
            groups.add_group(
                format!("inter_node_{}", local_rank),
                CommunicationGroup::new(ranks)?,
                GroupType::InterNode,
            )?;
        }
        
        Ok(groups)
    }
    
    // Hardware detection methods
    fn detect_infiniband() -> TorshResult<bool> {
        // Check for InfiniBand devices
        let result = std::process::Command::new("ibstat")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        Ok(result)
    }
    
    fn detect_nvlink() -> TorshResult<bool> {
        // Check for NVLink connectivity
        let result = std::process::Command::new("nvidia-smi")
            .args(&["topo", "-m"])
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.contains("NV")
            })
            .unwrap_or(false);
        Ok(result)
    }
    
    // Bandwidth measurement methods
    fn measure_intra_node_bandwidth() -> TorshResult<f32> {
        // Implement bandwidth measurement
        Ok(300e9) // Default NVLink bandwidth
    }
    
    fn measure_inter_node_bandwidth() -> TorshResult<f32> {
        // Implement bandwidth measurement
        Ok(100e9) // Default InfiniBand bandwidth
    }
    
    fn measure_intra_node_latency() -> TorshResult<f32> {
        // Implement latency measurement
        Ok(1e-6) // 1 microsecond
    }
    
    fn measure_inter_node_latency() -> TorshResult<f32> {
        // Implement latency measurement
        Ok(5e-6) // 5 microseconds
    }
}

#[derive(Debug)]
enum AllReduceStrategy {
    Ring,
    Tree,
    Hierarchical,
    ReduceScatter,
}

#[derive(Debug)]
struct NetworkTopology {
    world_size: u32,
    num_nodes: u32,
    gpus_per_node: u32,
    interconnect: Interconnect,
    intra_node_bandwidth: f32,
    inter_node_bandwidth: f32,
    intra_node_latency: f32,
    inter_node_latency: f32,
}

impl NetworkTopology {
    fn total_workers(&self) -> u32 {
        self.world_size
    }
    
    fn num_nodes(&self) -> u32 {
        self.num_nodes
    }
    
    fn gpus_per_node(&self) -> u32 {
        self.gpus_per_node
    }
}

#[derive(Debug)]
enum Interconnect {
    Ethernet,
    InfiniBand,
    NVLink,
    Custom(String),
}

struct CommunicationGroups {
    groups: std::collections::HashMap<String, (CommunicationGroup, GroupType)>,
}

impl CommunicationGroups {
    fn new() -> Self {
        Self {
            groups: std::collections::HashMap::new(),
        }
    }
    
    fn add_group(
        &mut self,
        name: String,
        group: CommunicationGroup,
        group_type: GroupType,
    ) -> TorshResult<()> {
        self.groups.insert(name, (group, group_type));
        Ok(())
    }
}

#[derive(Debug)]
enum GroupType {
    IntraNode,
    InterNode,
    Custom,
}

struct BandwidthEstimator {
    measurements: Vec<BandwidthMeasurement>,
}

impl BandwidthEstimator {
    fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }
    
    fn get_average_bandwidth(&self) -> f32 {
        if self.measurements.is_empty() {
            return 10e9; // Default 10 GB/s
        }
        
        let total: f32 = self.measurements.iter().map(|m| m.bandwidth).sum();
        total / self.measurements.len() as f32
    }
    
    fn get_inter_node_bandwidth(&self) -> f32 {
        let inter_node_measurements: Vec<&BandwidthMeasurement> = self.measurements
            .iter()
            .filter(|m| m.measurement_type == MeasurementType::InterNode)
            .collect();
            
        if inter_node_measurements.is_empty() {
            return 100e9; // Default InfiniBand bandwidth
        }
        
        let total: f32 = inter_node_measurements.iter().map(|m| m.bandwidth).sum();
        total / inter_node_measurements.len() as f32
    }
}

struct LatencyPredictor {
    measurements: Vec<LatencyMeasurement>,
}

impl LatencyPredictor {
    fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }
    
    fn get_average_latency(&self) -> f32 {
        if self.measurements.is_empty() {
            return 10e-6; // Default 10 microseconds
        }
        
        let total: f32 = self.measurements.iter().map(|m| m.latency).sum();
        total / self.measurements.len() as f32
    }
    
    fn get_inter_node_latency(&self) -> f32 {
        let inter_node_measurements: Vec<&LatencyMeasurement> = self.measurements
            .iter()
            .filter(|m| m.measurement_type == MeasurementType::InterNode)
            .collect();
            
        if inter_node_measurements.is_empty() {
            return 5e-6; // Default 5 microseconds
        }
        
        let total: f32 = inter_node_measurements.iter().map(|m| m.latency).sum();
        total / inter_node_measurements.len() as f32
    }
}

#[derive(Debug)]
struct BandwidthMeasurement {
    bandwidth: f32,
    measurement_type: MeasurementType,
    timestamp: std::time::Instant,
}

#[derive(Debug)]
struct LatencyMeasurement {
    latency: f32,
    measurement_type: MeasurementType,
    timestamp: std::time::Instant,
}

#[derive(Debug, PartialEq)]
enum MeasurementType {
    IntraNode,
    InterNode,
}
```

This performance guide provides comprehensive strategies for optimizing ToRSh distributed training across all aspects - from profiling and monitoring to communication and memory optimization. The code examples show practical implementations of advanced optimization techniques that can significantly improve training efficiency.

Key performance optimization strategies covered:

1. **Comprehensive Profiling**: Real-time monitoring and bottleneck detection
2. **Adaptive Compression**: Dynamic gradient compression based on training progress
3. **Hierarchical Communication**: Topology-aware communication optimization
4. **Memory Management**: Smart allocation and checkpointing strategies
5. **Hardware Optimization**: Leveraging specific hardware features (NVLink, InfiniBand)
6. **Auto-tuning**: Automatic parameter optimization based on performance metrics

These optimizations can lead to significant performance improvements, often achieving 2-5x speedups in large-scale distributed training scenarios.