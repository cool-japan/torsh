//! System metrics collection for benchmarks

use std::time::{Duration, Instant};

/// System metrics collector
pub struct MetricsCollector {
    start_time: Option<Instant>,
    memory_tracker: MemoryTracker,
    cpu_tracker: CpuTracker,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: None,
            memory_tracker: MemoryTracker::new(),
            cpu_tracker: CpuTracker::new(),
        }
    }

    /// Start collecting metrics
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.memory_tracker.start();
        self.cpu_tracker.start();
    }

    /// Stop collecting metrics and return results
    pub fn stop(&mut self) -> SystemMetrics {
        let elapsed = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        let memory_stats = self.memory_tracker.stop();
        let cpu_stats = self.cpu_tracker.stop();

        SystemMetrics {
            elapsed_time: elapsed,
            memory_stats,
            cpu_stats,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete system metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemMetrics {
    pub elapsed_time: Duration,
    pub memory_stats: MemoryStats,
    pub cpu_stats: CpuStats,
}

impl SystemMetrics {
    /// Get memory efficiency (operations per MB)
    pub fn memory_efficiency(&self, operations: usize) -> f64 {
        if self.memory_stats.peak_usage_mb > 0.0 {
            operations as f64 / self.memory_stats.peak_usage_mb
        } else {
            0.0
        }
    }

    /// Get CPU utilization percentage
    pub fn cpu_utilization(&self) -> f64 {
        self.cpu_stats.average_usage_percent
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryStats {
    /// Initial memory usage in MB
    pub initial_usage_mb: f64,

    /// Peak memory usage in MB
    pub peak_usage_mb: f64,

    /// Final memory usage in MB
    pub final_usage_mb: f64,

    /// Memory allocated during benchmark in MB
    pub allocated_mb: f64,

    /// Memory deallocated during benchmark in MB
    pub deallocated_mb: f64,

    /// Number of allocations
    pub allocation_count: usize,

    /// Number of deallocations
    pub deallocation_count: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            initial_usage_mb: 0.0,
            peak_usage_mb: 0.0,
            final_usage_mb: 0.0,
            allocated_mb: 0.0,
            deallocated_mb: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }
}

/// CPU usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpuStats {
    /// Average CPU usage percentage
    pub average_usage_percent: f64,

    /// Peak CPU usage percentage
    pub peak_usage_percent: f64,

    /// Number of CPU cores used
    pub cores_used: usize,

    /// Context switches during benchmark
    pub context_switches: usize,
}

impl Default for CpuStats {
    fn default() -> Self {
        Self {
            average_usage_percent: 0.0,
            peak_usage_percent: 0.0,
            cores_used: 1,
            context_switches: 0,
        }
    }
}

/// Memory usage tracker
pub struct MemoryTracker {
    initial_usage: Option<f64>,
    peak_usage: f64,
    samples: Vec<f64>,
    start_time: Option<Instant>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            initial_usage: None,
            peak_usage: 0.0,
            samples: Vec::new(),
            start_time: None,
        }
    }

    /// Start tracking memory usage
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.initial_usage = Some(Self::get_process_memory_mb());
        self.peak_usage = self.initial_usage.unwrap_or(0.0);
        self.samples.clear();
    }

    /// Stop tracking and return statistics
    pub fn stop(&mut self) -> MemoryStats {
        let current_usage = Self::get_process_memory_mb();
        let initial = self.initial_usage.unwrap_or(0.0);

        MemoryStats {
            initial_usage_mb: initial,
            peak_usage_mb: self.peak_usage,
            final_usage_mb: current_usage,
            allocated_mb: (current_usage - initial).max(0.0),
            deallocated_mb: (initial - current_usage).max(0.0),
            allocation_count: 0, // Would need system hook to track this
            deallocation_count: 0,
        }
    }

    /// Sample current memory usage
    pub fn sample(&mut self) {
        let usage = Self::get_process_memory_mb();
        self.samples.push(usage);
        if usage > self.peak_usage {
            self.peak_usage = usage;
        }
    }

    /// Get current process memory usage in MB
    fn get_process_memory_mb() -> f64 {
        // Simplified implementation - would use platform-specific APIs in production
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                return kb / 1024.0; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        // Fallback: rough estimate based on allocation tracking
        // In a real implementation, this would use proper system APIs
        0.0
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU usage tracker
pub struct CpuTracker {
    samples: Vec<f64>,
    peak_usage: f64,
    start_time: Option<Instant>,
}

impl CpuTracker {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            peak_usage: 0.0,
            start_time: None,
        }
    }

    /// Start tracking CPU usage
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.samples.clear();
        self.peak_usage = 0.0;
    }

    /// Stop tracking and return statistics
    pub fn stop(&mut self) -> CpuStats {
        let average_usage = if self.samples.is_empty() {
            0.0
        } else {
            self.samples.iter().sum::<f64>() / self.samples.len() as f64
        };

        CpuStats {
            average_usage_percent: average_usage,
            peak_usage_percent: self.peak_usage,
            cores_used: num_cpus::get(),
            context_switches: 0, // Would need system monitoring to track this
        }
    }

    /// Sample current CPU usage
    pub fn sample(&mut self) {
        let usage = Self::get_cpu_usage_percent();
        self.samples.push(usage);
        if usage > self.peak_usage {
            self.peak_usage = usage;
        }
    }

    /// Get current CPU usage percentage
    fn get_cpu_usage_percent() -> f64 {
        // Simplified implementation
        // In production, would use platform-specific APIs or libraries like sysinfo
        0.0
    }
}

impl Default for CpuTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance profiler for detailed analysis
pub struct PerformanceProfiler {
    events: Vec<ProfileEvent>,
    current_stack: Vec<String>,
    start_time: Instant,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            current_stack: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Begin a profiling event
    pub fn begin_event(&mut self, name: &str) {
        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::Begin,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.current_stack.push(name.to_string());
        self.events.push(event);
    }

    /// End a profiling event
    pub fn end_event(&mut self, name: &str) {
        if let Some(stack_name) = self.current_stack.pop() {
            assert_eq!(stack_name, name, "Mismatched profiling events");
        }

        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::End,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.events.push(event);
    }

    /// Add an instant marker
    pub fn marker(&mut self, name: &str) {
        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::Marker,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.events.push(event);
    }

    /// Generate Chrome tracing format output
    pub fn export_chrome_trace(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"traceEvents\": [")?;

        for (i, event) in self.events.iter().enumerate() {
            let phase = match event.event_type {
                ProfileEventType::Begin => "B",
                ProfileEventType::End => "E",
                ProfileEventType::Marker => "i",
            };

            writeln!(
                file,
                "    {{\"name\": \"{}\", \"ph\": \"{}\", \"ts\": {}, \"pid\": 1, \"tid\": {:?}}}{}",
                event.name,
                phase,
                event.timestamp.as_micros(),
                event.thread_id,
                if i < self.events.len() - 1 { "," } else { "" }
            )?;
        }

        writeln!(file, "  ]")?;
        writeln!(file, "}}")?;

        Ok(())
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut durations = std::collections::HashMap::new();
        let mut stack = Vec::new();

        for event in &self.events {
            match event.event_type {
                ProfileEventType::Begin => {
                    stack.push((event.name.clone(), event.timestamp));
                }
                ProfileEventType::End => {
                    if let Some((name, start_time)) = stack.pop() {
                        assert_eq!(name, event.name);
                        let duration = event.timestamp - start_time;
                        durations
                            .entry(name)
                            .or_insert_with(Vec::new)
                            .push(duration);
                    }
                }
                ProfileEventType::Marker => {} // Instant events don't have duration
            }
        }

        let mut function_stats = Vec::new();
        for (name, times) in durations {
            let total_time: Duration = times.iter().sum();
            let avg_time = total_time / times.len() as u32;
            let min_time = times.iter().min().copied().unwrap_or_default();
            let max_time = times.iter().max().copied().unwrap_or_default();

            function_stats.push(FunctionStats {
                name,
                call_count: times.len(),
                total_time,
                average_time: avg_time,
                min_time,
                max_time,
            });
        }

        // Sort by total time descending
        function_stats.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        PerformanceReport {
            function_stats,
            total_events: self.events.len(),
        }
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profiling event
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub name: String,
    pub event_type: ProfileEventType,
    pub timestamp: Duration,
    pub thread_id: std::thread::ThreadId,
    pub stack_depth: usize,
}

/// Profiling event type
#[derive(Debug, Clone, Copy)]
pub enum ProfileEventType {
    Begin,
    End,
    Marker,
}

/// Performance analysis report
#[derive(Debug)]
pub struct PerformanceReport {
    pub function_stats: Vec<FunctionStats>,
    pub total_events: usize,
}

/// Statistics for a profiled function
#[derive(Debug)]
pub struct FunctionStats {
    pub name: String,
    pub call_count: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

/// Scoped profiler that automatically ends when dropped
pub struct ScopedProfiler<'a> {
    profiler: &'a mut PerformanceProfiler,
    name: String,
}

impl<'a> ScopedProfiler<'a> {
    pub fn new(profiler: &'a mut PerformanceProfiler, name: &str) -> Self {
        profiler.begin_event(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ScopedProfiler<'a> {
    fn drop(&mut self) {
        self.profiler.end_event(&self.name);
    }
}

/// Macro for scoped profiling
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr) => {
        let _scoped_profiler = $crate::metrics::ScopedProfiler::new($profiler, $name);
    };
}

// External dependency for CPU count
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

/// Cross-framework metrics system for unified benchmark analysis
pub mod cross_framework {
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    /// Framework identifier
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Framework {
        Torsh,
        TensorFlow,
        JAX,
        NumPy,
        Ndarray,
        PyTorch,
    }

    impl std::fmt::Display for Framework {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Framework::Torsh => write!(f, "ToRSh"),
                Framework::TensorFlow => write!(f, "TensorFlow"),
                Framework::JAX => write!(f, "JAX"),
                Framework::NumPy => write!(f, "NumPy"),
                Framework::Ndarray => write!(f, "ndarray"),
                Framework::PyTorch => write!(f, "PyTorch"),
            }
        }
    }

    /// Operation type for standardized comparison
    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum OperationType {
        MatrixMultiplication,
        ElementWiseAddition,
        ElementWiseMultiplication,
        Convolution2D,
        ReLU,
        Softmax,
        BatchNormalization,
        LinearLayer,
        BackwardPass,
        MemoryAllocation,
        DataLoading,
    }

    impl std::fmt::Display for OperationType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                OperationType::MatrixMultiplication => write!(f, "Matrix Multiplication"),
                OperationType::ElementWiseAddition => write!(f, "Element-wise Addition"),
                OperationType::ElementWiseMultiplication => {
                    write!(f, "Element-wise Multiplication")
                }
                OperationType::Convolution2D => write!(f, "2D Convolution"),
                OperationType::ReLU => write!(f, "ReLU Activation"),
                OperationType::Softmax => write!(f, "Softmax"),
                OperationType::BatchNormalization => write!(f, "Batch Normalization"),
                OperationType::LinearLayer => write!(f, "Linear Layer"),
                OperationType::BackwardPass => write!(f, "Backward Pass"),
                OperationType::MemoryAllocation => write!(f, "Memory Allocation"),
                OperationType::DataLoading => write!(f, "Data Loading"),
            }
        }
    }

    /// Unified benchmark metrics
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UnifiedMetrics {
        /// Framework that produced this metric
        pub framework: Framework,

        /// Operation type
        pub operation: OperationType,

        /// Input size/dimensions
        pub input_size: Vec<usize>,

        /// Execution time in nanoseconds
        pub execution_time_ns: f64,

        /// Memory usage in bytes
        pub memory_usage_bytes: Option<u64>,

        /// Peak memory usage in bytes
        pub peak_memory_bytes: Option<u64>,

        /// Throughput (operations per second)
        pub throughput_ops: Option<f64>,

        /// FLOPS (floating point operations per second)
        pub flops: Option<f64>,

        /// Memory bandwidth in GB/s
        pub memory_bandwidth_gbps: Option<f64>,

        /// Additional custom metrics
        pub custom_metrics: HashMap<String, f64>,

        /// Device type (CPU, GPU, etc.)
        pub device_type: String,

        /// Data type (f32, f64, etc.)
        pub data_type: String,

        /// Framework version
        pub framework_version: Option<String>,

        /// Hardware information
        pub hardware_info: Option<HardwareInfo>,
    }

    /// Hardware information for benchmark context
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HardwareInfo {
        pub cpu_model: String,
        pub cpu_cores: usize,
        pub memory_gb: f64,
        pub gpu_model: Option<String>,
        pub gpu_memory_gb: Option<f64>,
    }

    /// Cross-framework benchmark comparison
    #[derive(Debug, Clone)]
    pub struct CrossFrameworkComparison {
        metrics: Vec<UnifiedMetrics>,
    }

    impl CrossFrameworkComparison {
        pub fn new() -> Self {
            Self {
                metrics: Vec::new(),
            }
        }

        /// Add benchmark metrics
        pub fn add_metrics(&mut self, metrics: UnifiedMetrics) {
            self.metrics.push(metrics);
        }

        /// Get metrics for a specific framework and operation
        pub fn get_metrics(
            &self,
            framework: &Framework,
            operation: &OperationType,
        ) -> Vec<&UnifiedMetrics> {
            self.metrics
                .iter()
                .filter(|m| m.framework == *framework && m.operation == *operation)
                .collect()
        }

        /// Compare frameworks for a specific operation
        pub fn compare_frameworks(&self, operation: &OperationType) -> FrameworkComparison {
            let mut framework_metrics: HashMap<Framework, Vec<&UnifiedMetrics>> = HashMap::new();

            for metric in &self.metrics {
                if metric.operation == *operation {
                    framework_metrics
                        .entry(metric.framework.clone())
                        .or_default()
                        .push(metric);
                }
            }

            let mut comparisons = Vec::new();
            for (framework, metrics) in framework_metrics {
                if !metrics.is_empty() {
                    let avg_time = metrics.iter().map(|m| m.execution_time_ns).sum::<f64>()
                        / metrics.len() as f64;
                    let avg_memory = metrics
                        .iter()
                        .filter_map(|m| m.memory_usage_bytes)
                        .map(|m| m as f64)
                        .sum::<f64>()
                        / metrics.len() as f64;
                    let avg_throughput =
                        metrics.iter().filter_map(|m| m.throughput_ops).sum::<f64>()
                            / metrics.len() as f64;

                    comparisons.push(FrameworkPerformance {
                        framework: framework.clone(),
                        average_time_ns: avg_time,
                        average_memory_bytes: if avg_memory > 0.0 {
                            Some(avg_memory as u64)
                        } else {
                            None
                        },
                        average_throughput_ops: if avg_throughput > 0.0 {
                            Some(avg_throughput)
                        } else {
                            None
                        },
                        sample_count: metrics.len(),
                    });
                }
            }

            // Sort by average execution time (fastest first)
            comparisons.sort_by(|a, b| {
                a.average_time_ns
                    .partial_cmp(&b.average_time_ns)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            FrameworkComparison {
                operation: operation.clone(),
                performances: comparisons,
            }
        }

        /// Generate scalability analysis for different input sizes
        pub fn analyze_scalability(
            &self,
            framework: &Framework,
            operation: &OperationType,
        ) -> ScalabilityAnalysis {
            let metrics = self.get_metrics(framework, operation);

            let mut size_performance: HashMap<usize, Vec<f64>> = HashMap::new();
            for metric in metrics {
                // Use the product of dimensions as the size metric
                let size = metric.input_size.iter().product();
                size_performance
                    .entry(size)
                    .or_default()
                    .push(metric.execution_time_ns);
            }

            let mut points = Vec::new();
            for (size, times) in size_performance {
                let avg_time = times.iter().sum::<f64>() / times.len() as f64;
                points.push(ScalabilityPoint {
                    input_size: size,
                    average_time_ns: avg_time,
                    sample_count: times.len(),
                });
            }

            // Sort by input size
            points.sort_by_key(|p| p.input_size);

            ScalabilityAnalysis {
                framework: framework.clone(),
                operation: operation.clone(),
                data_points: points,
            }
        }

        /// Export metrics to JSON
        pub fn export_json(&self, path: &str) -> std::io::Result<()> {
            let json = serde_json::to_string_pretty(&self.metrics)?;
            std::fs::write(path, json)?;
            Ok(())
        }

        /// Import metrics from JSON
        pub fn import_json(path: &str) -> std::io::Result<Self> {
            let json = std::fs::read_to_string(path)?;
            let metrics: Vec<UnifiedMetrics> = serde_json::from_str(&json)?;
            Ok(Self { metrics })
        }

        /// Generate comprehensive performance report
        pub fn generate_comprehensive_report(&self, output_dir: &str) -> std::io::Result<()> {
            std::fs::create_dir_all(output_dir)?;

            // HTML report
            let html_path = format!("{}/cross_framework_report.html", output_dir);
            self.generate_html_report(&html_path)?;

            // CSV export
            let csv_path = format!("{}/cross_framework_metrics.csv", output_dir);
            self.export_csv(&csv_path)?;

            // JSON export
            let json_path = format!("{}/cross_framework_metrics.json", output_dir);
            self.export_json(&json_path)?;

            // Markdown summary
            let md_path = format!("{}/cross_framework_summary.md", output_dir);
            self.generate_markdown_summary(&md_path)?;

            println!(
                "📊 Comprehensive cross-framework report generated in {}",
                output_dir
            );
            Ok(())
        }

        /// Generate HTML report with charts
        fn generate_html_report(&self, path: &str) -> std::io::Result<()> {
            use std::io::Write;
            let mut file = std::fs::File::create(path)?;

            writeln!(file, "<!DOCTYPE html>")?;
            writeln!(file, "<html><head>")?;
            writeln!(file, "<title>Cross-Framework Benchmark Report</title>")?;
            writeln!(file, "<style>")?;
            writeln!(
                file,
                "body {{ font-family: Arial, sans-serif; margin: 20px; }}"
            )?;
            writeln!(
                file,
                "table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}"
            )?;
            writeln!(
                file,
                "th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}"
            )?;
            writeln!(file, "th {{ background-color: #f2f2f2; }}")?;
            writeln!(file, ".metric {{ margin: 20px 0; }}")?;
            writeln!(file, ".framework {{ color: #333; font-weight: bold; }}")?;
            writeln!(file, "</style>")?;
            writeln!(file, "</head><body>")?;

            writeln!(file, "<h1>🚀 Cross-Framework Benchmark Report</h1>")?;
            writeln!(
                file,
                "<p>Comprehensive performance comparison across tensor computing frameworks</p>"
            )?;

            // Generate comparison tables for each operation
            let operations = [
                OperationType::MatrixMultiplication,
                OperationType::ElementWiseAddition,
                OperationType::Convolution2D,
                OperationType::ReLU,
            ];

            for operation in &operations {
                writeln!(file, "<div class='metric'>")?;
                writeln!(file, "<h2>{}</h2>", operation)?;

                let comparison = self.compare_frameworks(operation);
                if !comparison.performances.is_empty() {
                    writeln!(file, "<table>")?;
                    writeln!(file, "<tr><th>Framework</th><th>Avg Time (μs)</th><th>Avg Memory (MB)</th><th>Throughput (ops/s)</th><th>Samples</th></tr>")?;

                    for perf in &comparison.performances {
                        writeln!(file, "<tr>")?;
                        writeln!(file, "<td class='framework'>{}</td>", perf.framework)?;
                        writeln!(file, "<td>{:.2}</td>", perf.average_time_ns / 1000.0)?;
                        writeln!(
                            file,
                            "<td>{:.2}</td>",
                            perf.average_memory_bytes
                                .map(|m| m as f64 / 1_000_000.0)
                                .unwrap_or(0.0)
                        )?;
                        writeln!(
                            file,
                            "<td>{:.2}</td>",
                            perf.average_throughput_ops.unwrap_or(0.0)
                        )?;
                        writeln!(file, "<td>{}</td>", perf.sample_count)?;
                        writeln!(file, "</tr>")?;
                    }

                    writeln!(file, "</table>")?;
                } else {
                    writeln!(file, "<p>No data available for this operation.</p>")?;
                }
                writeln!(file, "</div>")?;
            }

            writeln!(file, "</body></html>")?;
            Ok(())
        }

        /// Export to CSV format
        fn export_csv(&self, path: &str) -> std::io::Result<()> {
            use std::io::Write;
            let mut file = std::fs::File::create(path)?;

            writeln!(file, "framework,operation,input_size,execution_time_ns,memory_usage_bytes,throughput_ops,device_type,data_type")?;

            for metric in &self.metrics {
                writeln!(
                    file,
                    "{},{},{:?},{},{},{},{},{}",
                    metric.framework,
                    metric.operation,
                    metric.input_size,
                    metric.execution_time_ns,
                    metric.memory_usage_bytes.unwrap_or(0),
                    metric.throughput_ops.unwrap_or(0.0),
                    metric.device_type,
                    metric.data_type
                )?;
            }

            Ok(())
        }

        /// Generate markdown summary
        fn generate_markdown_summary(&self, path: &str) -> std::io::Result<()> {
            use std::io::Write;
            let mut file = std::fs::File::create(path)?;

            writeln!(file, "# Cross-Framework Benchmark Summary\n")?;

            let operations = [
                OperationType::MatrixMultiplication,
                OperationType::ElementWiseAddition,
                OperationType::Convolution2D,
                OperationType::ReLU,
            ];

            for operation in &operations {
                writeln!(file, "## {}\n", operation)?;

                let comparison = self.compare_frameworks(operation);
                if !comparison.performances.is_empty() {
                    writeln!(
                        file,
                        "| Framework | Avg Time (μs) | Relative Performance | Samples |"
                    )?;
                    writeln!(
                        file,
                        "|-----------|---------------|---------------------|---------|"
                    )?;

                    let fastest_time = comparison.performances[0].average_time_ns;
                    for perf in &comparison.performances {
                        let relative_perf = fastest_time / perf.average_time_ns;
                        writeln!(
                            file,
                            "| {} | {:.2} | {:.2}x | {} |",
                            perf.framework,
                            perf.average_time_ns / 1000.0,
                            relative_perf,
                            perf.sample_count
                        )?;
                    }
                    writeln!(file)?;
                } else {
                    writeln!(file, "No data available for this operation.\n")?;
                }
            }

            Ok(())
        }

        /// Get all unique frameworks in the dataset
        pub fn get_frameworks(&self) -> Vec<Framework> {
            let mut frameworks: Vec<Framework> = self
                .metrics
                .iter()
                .map(|m| m.framework.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            frameworks.sort_by_key(|f| format!("{}", f));
            frameworks
        }

        /// Get all unique operations in the dataset
        pub fn get_operations(&self) -> Vec<OperationType> {
            let mut operations: Vec<OperationType> = self
                .metrics
                .iter()
                .map(|m| m.operation.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            operations.sort_by_key(|o| format!("{}", o));
            operations
        }
    }

    impl Default for CrossFrameworkComparison {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Framework performance summary
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct FrameworkPerformance {
        pub framework: Framework,
        pub average_time_ns: f64,
        pub average_memory_bytes: Option<u64>,
        pub average_throughput_ops: Option<f64>,
        pub sample_count: usize,
    }

    /// Comparison between frameworks for a specific operation
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct FrameworkComparison {
        pub operation: OperationType,
        pub performances: Vec<FrameworkPerformance>,
    }

    /// Scalability analysis data
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct ScalabilityAnalysis {
        pub framework: Framework,
        pub operation: OperationType,
        pub data_points: Vec<ScalabilityPoint>,
    }

    /// Single point in scalability analysis
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct ScalabilityPoint {
        pub input_size: usize,
        pub average_time_ns: f64,
        pub sample_count: usize,
    }

    /// Utility functions for converting benchmark results to unified metrics
    pub mod converters {
        use super::*;
        use crate::{comparisons::ComparisonResult, BenchResult};

        /// Convert ToRSh BenchResult to UnifiedMetrics
        pub fn from_torsh_result(result: &BenchResult, operation: OperationType) -> UnifiedMetrics {
            UnifiedMetrics {
                framework: Framework::Torsh,
                operation,
                input_size: vec![result.size],
                execution_time_ns: result.mean_time_ns,
                memory_usage_bytes: result.memory_usage.map(|m| m as u64),
                peak_memory_bytes: result.peak_memory.map(|m| m as u64),
                throughput_ops: result.throughput,
                flops: None,
                memory_bandwidth_gbps: None,
                custom_metrics: result.metrics.clone(),
                device_type: "CPU".to_string(), // Default assumption
                data_type: format!("{:?}", result.dtype),
                framework_version: std::env::var("CARGO_PKG_VERSION").ok(),
                hardware_info: None,
            }
        }

        /// Convert ComparisonResult to UnifiedMetrics
        pub fn from_comparison_result(
            result: &ComparisonResult,
            operation: OperationType,
        ) -> UnifiedMetrics {
            let framework = match result.library.as_str() {
                "torsh" => Framework::Torsh,
                "tensorflow" => Framework::TensorFlow,
                "jax" => Framework::JAX,
                "numpy" => Framework::NumPy,
                "ndarray" => Framework::Ndarray,
                "pytorch" => Framework::PyTorch,
                _ => Framework::Torsh, // Default fallback
            };

            UnifiedMetrics {
                framework,
                operation,
                input_size: vec![result.size],
                execution_time_ns: result.time_ns,
                memory_usage_bytes: result.memory_usage.map(|m| m as u64),
                peak_memory_bytes: None,
                throughput_ops: result.throughput,
                flops: None,
                memory_bandwidth_gbps: None,
                custom_metrics: HashMap::new(),
                device_type: "CPU".to_string(), // Default assumption
                data_type: "f32".to_string(),   // Default assumption
                framework_version: None,
                hardware_info: None,
            }
        }
    }
}

/// Power consumption monitoring system
pub mod power {
    use super::*;
    use std::fs;
    use std::time::{Duration, Instant};

    /// Power consumption metrics
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct PowerMetrics {
        /// Total energy consumed in joules
        pub total_energy_joules: f64,

        /// Average power consumption in watts
        pub average_power_watts: f64,

        /// Peak power consumption in watts  
        pub peak_power_watts: f64,

        /// Duration of measurement
        pub measurement_duration: Duration,

        /// Number of power samples taken
        pub sample_count: usize,

        /// CPU power consumption in watts (if available)
        pub cpu_power_watts: Option<f64>,

        /// GPU power consumption in watts (if available)
        pub gpu_power_watts: Option<f64>,

        /// Memory power consumption in watts (if available)
        pub memory_power_watts: Option<f64>,

        /// Power efficiency (operations per joule)
        pub power_efficiency_ops_per_joule: Option<f64>,
    }

    impl Default for PowerMetrics {
        fn default() -> Self {
            Self {
                total_energy_joules: 0.0,
                average_power_watts: 0.0,
                peak_power_watts: 0.0,
                measurement_duration: Duration::ZERO,
                sample_count: 0,
                cpu_power_watts: None,
                gpu_power_watts: None,
                memory_power_watts: None,
                power_efficiency_ops_per_joule: None,
            }
        }
    }

    /// Power monitoring system
    pub struct PowerMonitor {
        start_time: Option<Instant>,
        power_samples: Vec<PowerSample>,
        sampling_interval: Duration,
        monitor_thread: Option<std::thread::JoinHandle<()>>,
        stop_signal: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    }

    /// Single power measurement sample
    #[derive(Debug, Clone)]
    struct PowerSample {
        timestamp: Instant,
        total_power_watts: f64,
        cpu_power_watts: Option<f64>,
        gpu_power_watts: Option<f64>,
        memory_power_watts: Option<f64>,
    }

    impl PowerMonitor {
        /// Create new power monitor
        pub fn new() -> Self {
            Self {
                start_time: None,
                power_samples: Vec::new(),
                sampling_interval: Duration::from_millis(100), // 10Hz sampling
                monitor_thread: None,
                stop_signal: None,
            }
        }

        /// Set power sampling interval
        pub fn with_sampling_interval(mut self, interval: Duration) -> Self {
            self.sampling_interval = interval;
            self
        }

        /// Start power monitoring
        pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
            if self.monitor_thread.is_some() {
                return Err("Power monitoring already started".into());
            }

            self.start_time = Some(Instant::now());
            self.power_samples.clear();

            let stop_signal = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            self.stop_signal = Some(stop_signal.clone());

            let sampling_interval = self.sampling_interval;
            let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let samples_clone = samples.clone();

            // Spawn monitoring thread
            let handle = std::thread::spawn(move || {
                let mut power_reader = SystemPowerReader::new();

                while !stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Ok(sample) = power_reader.read_power() {
                        if let Ok(mut samples_guard) = samples_clone.try_lock() {
                            samples_guard.push(sample);
                        }
                    }

                    std::thread::sleep(sampling_interval);
                }
            });

            self.monitor_thread = Some(handle);
            Ok(())
        }

        /// Stop power monitoring and return metrics
        pub fn stop(&mut self) -> PowerMetrics {
            // Signal monitoring thread to stop
            if let Some(stop_signal) = &self.stop_signal {
                stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
            }

            // Wait for thread to finish
            if let Some(handle) = self.monitor_thread.take() {
                let _ = handle.join();
            }

            let measurement_duration = self
                .start_time
                .map(|start| start.elapsed())
                .unwrap_or(Duration::ZERO);

            if self.power_samples.is_empty() {
                return PowerMetrics {
                    measurement_duration,
                    ..Default::default()
                };
            }

            // Calculate metrics from samples
            let total_power: f64 = self.power_samples.iter().map(|s| s.total_power_watts).sum();
            let sample_count = self.power_samples.len();
            let average_power = total_power / sample_count as f64;
            let peak_power = self
                .power_samples
                .iter()
                .map(|s| s.total_power_watts)
                .fold(0.0, f64::max);

            // Energy = average power * time (in joules)
            let total_energy = average_power * measurement_duration.as_secs_f64();

            // Average component power consumption
            let avg_cpu_power = if self
                .power_samples
                .iter()
                .any(|s| s.cpu_power_watts.is_some())
            {
                let cpu_samples: Vec<f64> = self
                    .power_samples
                    .iter()
                    .filter_map(|s| s.cpu_power_watts)
                    .collect();
                if !cpu_samples.is_empty() {
                    Some(cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64)
                } else {
                    None
                }
            } else {
                None
            };

            let avg_gpu_power = if self
                .power_samples
                .iter()
                .any(|s| s.gpu_power_watts.is_some())
            {
                let gpu_samples: Vec<f64> = self
                    .power_samples
                    .iter()
                    .filter_map(|s| s.gpu_power_watts)
                    .collect();
                if !gpu_samples.is_empty() {
                    Some(gpu_samples.iter().sum::<f64>() / gpu_samples.len() as f64)
                } else {
                    None
                }
            } else {
                None
            };

            let avg_memory_power = if self
                .power_samples
                .iter()
                .any(|s| s.memory_power_watts.is_some())
            {
                let memory_samples: Vec<f64> = self
                    .power_samples
                    .iter()
                    .filter_map(|s| s.memory_power_watts)
                    .collect();
                if !memory_samples.is_empty() {
                    Some(memory_samples.iter().sum::<f64>() / memory_samples.len() as f64)
                } else {
                    None
                }
            } else {
                None
            };

            PowerMetrics {
                total_energy_joules: total_energy,
                average_power_watts: average_power,
                peak_power_watts: peak_power,
                measurement_duration,
                sample_count,
                cpu_power_watts: avg_cpu_power,
                gpu_power_watts: avg_gpu_power,
                memory_power_watts: avg_memory_power,
                power_efficiency_ops_per_joule: None, // To be calculated by caller
            }
        }

        /// Calculate power efficiency for a given number of operations
        pub fn calculate_power_efficiency(&self, metrics: &PowerMetrics, operations: u64) -> f64 {
            if metrics.total_energy_joules > 0.0 {
                operations as f64 / metrics.total_energy_joules
            } else {
                0.0
            }
        }
    }

    impl Default for PowerMonitor {
        fn default() -> Self {
            Self::new()
        }
    }

    /// System power reader that attempts to read power from multiple sources
    struct SystemPowerReader {
        power_sources: Vec<Box<dyn PowerSource>>,
    }

    impl SystemPowerReader {
        fn new() -> Self {
            let mut sources: Vec<Box<dyn PowerSource>> = Vec::new();

            // Try different power sources based on platform
            #[cfg(target_os = "linux")]
            {
                sources.push(Box::new(LinuxPowerSource::new()));
                sources.push(Box::new(RaplPowerSource::new()));
            }

            #[cfg(target_os = "windows")]
            {
                sources.push(Box::new(WindowsPowerSource::new()));
            }

            #[cfg(target_os = "macos")]
            {
                sources.push(Box::new(MacOsPowerSource::new()));
            }

            // Fallback to estimate-based power source
            sources.push(Box::new(EstimatedPowerSource::new()));

            Self {
                power_sources: sources,
            }
        }

        fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
            for source in &mut self.power_sources {
                if let Ok(sample) = source.read_power() {
                    return Ok(sample);
                }
            }

            Err("No power sources available".into())
        }
    }

    /// Trait for different power monitoring sources
    trait PowerSource {
        fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>>;
    }

    /// Linux-specific power monitoring using /sys/class/power_supply
    #[cfg(target_os = "linux")]
    struct LinuxPowerSource;

    #[cfg(target_os = "linux")]
    impl LinuxPowerSource {
        fn new() -> Self {
            Self
        }
    }

    #[cfg(target_os = "linux")]
    impl PowerSource for LinuxPowerSource {
        fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
            let mut total_power = 0.0;

            // Try to read from battery power supply
            if let Ok(power_str) = fs::read_to_string("/sys/class/power_supply/BAT0/power_now") {
                if let Ok(power_microwatts) = power_str.trim().parse::<u64>() {
                    total_power += power_microwatts as f64 / 1_000_000.0; // Convert to watts
                }
            }

            // Try to read from AC adapter
            if let Ok(power_str) = fs::read_to_string("/sys/class/power_supply/ADP1/power_now") {
                if let Ok(power_microwatts) = power_str.trim().parse::<u64>() {
                    total_power += power_microwatts as f64 / 1_000_000.0;
                }
            }

            if total_power > 0.0 {
                Ok(PowerSample {
                    timestamp: Instant::now(),
                    total_power_watts: total_power,
                    cpu_power_watts: None,
                    gpu_power_watts: None,
                    memory_power_watts: None,
                })
            } else {
                Err("No power data available from Linux power supply".into())
            }
        }
    }

    /// RAPL (Running Average Power Limit) power monitoring for Intel CPUs
    #[cfg(target_os = "linux")]
    struct RaplPowerSource {
        package_energy_path: Option<String>,
        dram_energy_path: Option<String>,
        last_package_energy: Option<f64>,
        last_dram_energy: Option<f64>,
        last_timestamp: Option<Instant>,
    }

    #[cfg(target_os = "linux")]
    impl RaplPowerSource {
        fn new() -> Self {
            let package_energy_path = Self::find_rapl_path("package-0");
            let dram_energy_path = Self::find_rapl_path("dram");

            Self {
                package_energy_path,
                dram_energy_path,
                last_package_energy: None,
                last_dram_energy: None,
                last_timestamp: None,
            }
        }

        fn find_rapl_path(energy_type: &str) -> Option<String> {
            let base_path = "/sys/class/powercap/intel-rapl";

            for i in 0..10 {
                let path = format!("{}/intel-rapl:{}/name", base_path, i);
                if let Ok(name) = fs::read_to_string(&path) {
                    if name.trim() == energy_type {
                        return Some(format!("{}/intel-rapl:{}/energy_uj", base_path, i));
                    }
                }

                // Also check subdirectories
                for j in 0..10 {
                    let path =
                        format!("{}/intel-rapl:{}:intel-rapl:{}:{}/name", base_path, i, i, j);
                    if let Ok(name) = fs::read_to_string(&path) {
                        if name.trim() == energy_type {
                            return Some(format!(
                                "{}/intel-rapl:{}:intel-rapl:{}:{}/energy_uj",
                                base_path, i, i, j
                            ));
                        }
                    }
                }
            }

            None
        }

        fn read_energy_uj(&self, path: &str) -> Result<f64, Box<dyn std::error::Error>> {
            let energy_str = fs::read_to_string(path)?;
            let energy_uj = energy_str.trim().parse::<u64>()?;
            Ok(energy_uj as f64 / 1_000_000.0) // Convert to joules
        }
    }

    #[cfg(target_os = "linux")]
    impl PowerSource for RaplPowerSource {
        fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
            let now = Instant::now();
            let mut cpu_power = None;
            let mut memory_power = None;

            // Read package (CPU) energy
            if let Some(path) = &self.package_energy_path {
                if let Ok(current_energy) = self.read_energy_uj(path) {
                    if let (Some(last_energy), Some(last_time)) =
                        (self.last_package_energy, self.last_timestamp)
                    {
                        let energy_diff = current_energy - last_energy;
                        let time_diff = now.duration_since(last_time).as_secs_f64();
                        if time_diff > 0.0 {
                            cpu_power = Some(energy_diff / time_diff);
                        }
                    }
                    self.last_package_energy = Some(current_energy);
                }
            }

            // Read DRAM energy
            if let Some(path) = &self.dram_energy_path {
                if let Ok(current_energy) = self.read_energy_uj(path) {
                    if let (Some(last_energy), Some(last_time)) =
                        (self.last_dram_energy, self.last_timestamp)
                    {
                        let energy_diff = current_energy - last_energy;
                        let time_diff = now.duration_since(last_time).as_secs_f64();
                        if time_diff > 0.0 {
                            memory_power = Some(energy_diff / time_diff);
                        }
                    }
                    self.last_dram_energy = Some(current_energy);
                }
            }

            self.last_timestamp = Some(now);

            let total_power = cpu_power.unwrap_or(0.0) + memory_power.unwrap_or(0.0);

            if total_power > 0.0 {
                Ok(PowerSample {
                    timestamp: now,
                    total_power_watts: total_power,
                    cpu_power_watts: cpu_power,
                    gpu_power_watts: None, // RAPL doesn't provide GPU power
                    memory_power_watts: memory_power,
                })
            } else {
                Err("No RAPL power data available".into())
            }
        }
    }

    /// Windows power monitoring (placeholder implementation)
    #[cfg(target_os = "windows")]
    struct WindowsPowerSource;

    #[cfg(target_os = "windows")]
    impl WindowsPowerSource {
        fn new() -> Self {
            Self
        }
    }

    #[cfg(target_os = "windows")]
    impl PowerSource for WindowsPowerSource {
        fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
            // Windows power monitoring using system load estimation
            // This is a simplified implementation based on CPU usage and system characteristics
            use std::time::{SystemTime, UNIX_EPOCH};

            // Get system information for power estimation
            let num_cpus = num_cpus::get();
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();

            // Estimate power consumption based on system characteristics
            // These are rough estimates for typical Windows systems
            let base_power = 15.0; // Base system power in watts
            let cpu_power_per_core = 3.0; // Additional power per active CPU core

            // Simple CPU load estimation (would need actual CPU usage for accuracy)
            let estimated_load = 0.3; // Assume 30% load as placeholder
            let cpu_power = estimated_load * num_cpus as f64 * cpu_power_per_core;

            let total_power = base_power + cpu_power;

            Ok(PowerSample {
                timestamp,
                total_power_watts: total_power,
                cpu_power_watts: Some(cpu_power),
                gpu_power_watts: None,         // Could be estimated separately
                memory_power_watts: Some(2.0), // Typical DDR4 power consumption
                package_power_watts: Some(total_power * 0.8), // Estimate package power
            })
        }
    }

    /// macOS power monitoring (placeholder implementation)
    #[cfg(target_os = "macos")]
    struct MacOsPowerSource;

    #[cfg(target_os = "macos")]
    impl MacOsPowerSource {
        fn new() -> Self {
            Self
        }
    }

    #[cfg(target_os = "macos")]
    impl PowerSource for MacOsPowerSource {
        fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
            // macOS power monitoring using system load estimation
            // This is a simplified implementation based on CPU usage and system characteristics
            use std::time::{SystemTime, UNIX_EPOCH};

            // Get system information for power estimation
            let num_cpus = num_cpus::get();
            let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();

            // Estimate power consumption based on macOS system characteristics
            // These are rough estimates for typical macOS systems (Intel and Apple Silicon)
            let base_power = if cfg!(target_arch = "aarch64") {
                8.0 // Apple Silicon base power (more efficient)
            } else {
                12.0 // Intel Mac base power
            };

            let cpu_power_per_core = if cfg!(target_arch = "aarch64") {
                1.5 // Apple Silicon CPU power per core
            } else {
                2.5 // Intel Mac CPU power per core
            };

            // Simple CPU load estimation (would need actual CPU usage for accuracy)
            let estimated_load = 0.25; // Assume 25% load as placeholder
            let cpu_power = estimated_load * num_cpus as f64 * cpu_power_per_core;

            let total_power = base_power + cpu_power;

            Ok(PowerSample {
                timestamp,
                total_power_watts: total_power,
                cpu_power_watts: Some(cpu_power),
                gpu_power_watts: None, // Could be estimated separately for discrete GPUs
                memory_power_watts: Some(1.5), // Typical LPDDR power consumption
                package_power_watts: Some(total_power * 0.85), // Estimate package power
            })
        }
    }

    /// Estimated power source based on CPU usage and system characteristics
    struct EstimatedPowerSource {
        cpu_tracker: CpuTracker,
        base_power_watts: f64,
        max_power_watts: f64,
    }

    impl EstimatedPowerSource {
        fn new() -> Self {
            Self {
                cpu_tracker: CpuTracker::new(),
                base_power_watts: 20.0, // Estimated idle power
                max_power_watts: 150.0, // Estimated max power under load
            }
        }
    }

    impl PowerSource for EstimatedPowerSource {
        fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
            self.cpu_tracker.start();
            std::thread::sleep(Duration::from_millis(100));
            let cpu_stats = self.cpu_tracker.stop();

            // Estimate power based on CPU utilization
            let utilization_ratio = cpu_stats.average_usage_percent / 100.0;
            let estimated_power = self.base_power_watts
                + (self.max_power_watts - self.base_power_watts) * utilization_ratio;

            Ok(PowerSample {
                timestamp: Instant::now(),
                total_power_watts: estimated_power,
                cpu_power_watts: Some(estimated_power * 0.7), // Assume CPU is 70% of total
                gpu_power_watts: None,
                memory_power_watts: Some(estimated_power * 0.1), // Assume memory is 10% of total
            })
        }
    }

    /// Power-aware benchmark runner
    pub struct PowerAwareBenchRunner {
        power_monitor: PowerMonitor,
        metrics_collector: MetricsCollector,
    }

    impl PowerAwareBenchRunner {
        pub fn new() -> Self {
            Self {
                power_monitor: PowerMonitor::new(),
                metrics_collector: MetricsCollector::new(),
            }
        }

        /// Run benchmark with power monitoring
        pub fn run_benchmark<F, R>(
            &mut self,
            benchmark_fn: F,
        ) -> Result<(R, PowerMetrics, SystemMetrics), Box<dyn std::error::Error>>
        where
            F: FnOnce() -> R,
        {
            // Start monitoring
            self.power_monitor.start()?;
            self.metrics_collector.start();

            // Run benchmark
            let result = benchmark_fn();

            // Stop monitoring
            let power_metrics = self.power_monitor.stop();
            let system_metrics = self.metrics_collector.stop();

            Ok((result, power_metrics, system_metrics))
        }

        /// Calculate power efficiency for the benchmark
        pub fn calculate_power_efficiency(
            &self,
            power_metrics: &PowerMetrics,
            operations: u64,
        ) -> f64 {
            self.power_monitor
                .calculate_power_efficiency(power_metrics, operations)
        }
    }

    impl Default for PowerAwareBenchRunner {
        fn default() -> Self {
            Self::new()
        }
    }
}
