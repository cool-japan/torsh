//! Trace visualization for JIT compilation
//!
//! This module provides comprehensive trace visualization capabilities for JIT-compiled
//! code, including execution flow diagrams, performance heatmaps, and interactive visualization.

use crate::{JitError, JitResult};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Trace visualization manager
#[derive(Debug)]
pub struct TraceVisualizationManager {
    /// Visualization sessions
    sessions: IndexMap<String, VisualizationSession>,

    /// Visualization configuration
    config: VisualizationConfig,

    /// Trace data collectors
    collectors: Vec<TraceCollector>,

    /// Visualization renderers
    renderers: HashMap<OutputFormat, Box<dyn VisualizationRenderer>>,

    /// Statistics about visualization
    stats: VisualizationStats,
}

/// Visualization session
#[derive(Debug, Clone)]
pub struct VisualizationSession {
    /// Session ID
    pub id: String,

    /// Session name
    pub name: String,

    /// Start time
    pub start_time: Instant,

    /// Collected traces
    pub traces: Vec<ExecutionTrace>,

    /// Call graphs
    pub call_graphs: Vec<CallGraph>,

    /// Performance data
    pub performance_data: PerformanceData,

    /// Session metadata
    pub metadata: HashMap<String, String>,

    /// Session status
    pub status: SessionStatus,
}

/// Session status
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Collecting,
    Processing,
    Ready,
    Error(String),
}

/// Execution trace
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Trace ID
    pub id: String,

    /// Function name
    pub function_name: String,

    /// Execution events
    pub events: Vec<TraceEvent>,

    /// Total execution time
    pub total_time: Duration,

    /// Thread ID
    pub thread_id: u64,

    /// CPU utilization
    pub cpu_utilization: f32,

    /// Memory usage
    pub memory_usage: MemoryUsage,
}

/// Trace event
#[derive(Debug, Clone)]
pub enum TraceEvent {
    /// Function entry
    FunctionEntry {
        timestamp: Instant,
        function_name: String,
        address: u64,
        parameters: Vec<TraceValue>,
    },

    /// Function exit
    FunctionExit {
        timestamp: Instant,
        function_name: String,
        return_value: Option<TraceValue>,
        duration: Duration,
    },

    /// Kernel launch
    KernelLaunch {
        timestamp: Instant,
        kernel_name: String,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
    },

    /// Kernel completion
    KernelComplete {
        timestamp: Instant,
        kernel_name: String,
        duration: Duration,
        occupancy: f32,
    },

    /// Memory operation
    MemoryOp {
        timestamp: Instant,
        operation: MemoryOperation,
        address: u64,
        size: usize,
        duration: Duration,
    },

    /// Synchronization event
    Synchronization {
        timestamp: Instant,
        sync_type: SynchronizationType,
        duration: Duration,
    },

    /// Custom event
    Custom {
        timestamp: Instant,
        name: String,
        data: HashMap<String, TraceValue>,
    },
}

/// Trace value types
#[derive(Debug, Clone)]
pub enum TraceValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    String(String),
    Pointer(u64),
    Array(Vec<TraceValue>),
    Struct(HashMap<String, TraceValue>),
}

/// Memory operations for tracing
#[derive(Debug, Clone)]
pub enum MemoryOperation {
    Alloc,
    Free,
    Read,
    Write,
    Copy,
}

/// Synchronization types
#[derive(Debug, Clone)]
pub enum SynchronizationType {
    Barrier,
    Mutex,
    Semaphore,
    CondVar,
    Atomic,
}

/// Memory usage information
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Peak memory usage
    pub peak: usize,

    /// Current memory usage
    pub current: usize,

    /// Total allocations
    pub total_allocations: u64,

    /// Total deallocations
    pub total_deallocations: u64,

    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
}

/// Call graph representation
#[derive(Debug, Clone)]
pub struct CallGraph {
    /// Graph nodes (functions)
    pub nodes: IndexMap<String, CallGraphNode>,

    /// Graph edges (function calls)
    pub edges: Vec<CallGraphEdge>,

    /// Root functions
    pub roots: Vec<String>,

    /// Graph statistics
    pub stats: CallGraphStats,
}

/// Call graph node
#[derive(Debug, Clone)]
pub struct CallGraphNode {
    /// Function name
    pub name: String,

    /// Function address
    pub address: u64,

    /// Total execution time
    pub total_time: Duration,

    /// Number of calls
    pub call_count: u64,

    /// Average execution time per call
    pub avg_time: Duration,

    /// CPU utilization
    pub cpu_utilization: f32,

    /// Memory usage
    pub memory_usage: MemoryUsage,

    /// Node metadata
    pub metadata: HashMap<String, String>,
}

/// Call graph edge
#[derive(Debug, Clone)]
pub struct CallGraphEdge {
    /// Caller function
    pub from: String,

    /// Callee function
    pub to: String,

    /// Number of calls
    pub call_count: u64,

    /// Total time spent in callee from this caller
    pub total_time: Duration,

    /// Edge weight (for visualization)
    pub weight: f64,
}

/// Call graph statistics
#[derive(Debug, Clone, Default)]
pub struct CallGraphStats {
    /// Total number of nodes
    pub node_count: usize,

    /// Total number of edges
    pub edge_count: usize,

    /// Maximum depth
    pub max_depth: usize,

    /// Average fan-out
    pub avg_fanout: f64,

    /// Critical path length
    pub critical_path_length: Duration,
}

/// Performance data for visualization
#[derive(Debug, Clone, Default)]
pub struct PerformanceData {
    /// Timeline data
    pub timeline: Vec<TimelineEvent>,

    /// Heatmap data
    pub heatmaps: HashMap<String, Heatmap>,

    /// Performance counters
    pub counters: HashMap<String, Counter>,

    /// Histogram data
    pub histograms: HashMap<String, Histogram>,

    /// Flamegraph data
    pub flamegraph: Option<Flamegraph>,
}

/// Timeline event
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Timestamp
    pub timestamp: Instant,

    /// Duration
    pub duration: Duration,

    /// Event name
    pub name: String,

    /// Event category
    pub category: String,

    /// Thread ID
    pub thread_id: u64,

    /// Process ID
    pub process_id: u64,

    /// Event arguments
    pub args: HashMap<String, TraceValue>,
}

/// Heatmap data
#[derive(Debug, Clone)]
pub struct Heatmap {
    /// Heatmap name
    pub name: String,

    /// Data points (x, y, intensity)
    pub data: Vec<(f64, f64, f64)>,

    /// X-axis label
    pub x_label: String,

    /// Y-axis label
    pub y_label: String,

    /// Color scale
    pub color_scale: ColorScale,
}

/// Color scale for heatmaps
#[derive(Debug, Clone)]
pub enum ColorScale {
    /// Heat scale (blue to red)
    Heat,

    /// Viridis scale
    Viridis,

    /// Plasma scale
    Plasma,

    /// Custom scale
    Custom(Vec<(f64, String)>),
}

/// Performance counter
#[derive(Debug, Clone)]
pub struct Counter {
    /// Counter name
    pub name: String,

    /// Current value
    pub value: f64,

    /// Unit
    pub unit: String,

    /// History
    pub history: Vec<(Instant, f64)>,
}

/// Histogram data
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Histogram name
    pub name: String,

    /// Bins
    pub bins: Vec<HistogramBin>,

    /// Total count
    pub total_count: u64,

    /// Statistics
    pub stats: HistogramStats,
}

/// Histogram bin
#[derive(Debug, Clone)]
pub struct HistogramBin {
    /// Bin start value
    pub start: f64,

    /// Bin end value
    pub end: f64,

    /// Count in this bin
    pub count: u64,
}

/// Histogram statistics
#[derive(Debug, Clone)]
pub struct HistogramStats {
    /// Mean
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Minimum value
    pub min: f64,

    /// Maximum value
    pub max: f64,

    /// Percentiles
    pub percentiles: HashMap<u8, f64>,
}

/// Flamegraph data
#[derive(Debug, Clone)]
pub struct Flamegraph {
    /// Root node
    pub root: FlamegraphNode,

    /// Total duration
    pub total_duration: Duration,

    /// Color scheme
    pub color_scheme: ColorScheme,
}

/// Flamegraph node
#[derive(Debug, Clone)]
pub struct FlamegraphNode {
    /// Function name
    pub name: String,

    /// Self time
    pub self_time: Duration,

    /// Total time (including children)
    pub total_time: Duration,

    /// Children nodes
    pub children: Vec<FlamegraphNode>,

    /// Sample count
    pub sample_count: u64,
}

/// Color schemes for visualization
#[derive(Debug, Clone)]
pub enum ColorScheme {
    /// Default scheme
    Default,

    /// High contrast
    HighContrast,

    /// Colorblind friendly
    ColorblindFriendly,

    /// Custom scheme
    Custom(Vec<String>),
}

/// Trace data collector
#[derive(Debug, Clone)]
pub struct TraceCollector {
    /// Collector name
    pub name: String,

    /// Collection interval
    pub interval: Duration,

    /// Enabled flag
    pub enabled: bool,

    /// Filter criteria
    pub filters: Vec<TraceFilter>,
}

/// Trace filter
#[derive(Debug)]
pub enum TraceFilter {
    /// Function name filter
    FunctionName(String),

    /// Thread ID filter
    ThreadId(u64),

    /// Duration filter
    MinDuration(Duration),

    /// Custom filter (function pointer for Debug compatibility)
    Custom(fn(&TraceEvent) -> bool),
}

/// Visualization renderer trait
pub trait VisualizationRenderer: Send + Sync + std::fmt::Debug {
    /// Render visualization
    fn render(&self, session: &VisualizationSession, output_path: &str) -> JitResult<()>;

    /// Get supported output format
    fn output_format(&self) -> OutputFormat;

    /// Get renderer name
    fn name(&self) -> &str;
}

/// Output formats for visualization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// HTML with interactive JavaScript
    Html,

    /// SVG vector graphics
    Svg,

    /// PNG raster image
    Png,

    /// JSON data format
    Json,

    /// Chrome tracing format
    ChromeTracing,

    /// Flamegraph format
    Flamegraph,
}

/// Visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Enable trace visualization
    pub enabled: bool,

    /// Default output format
    pub default_format: OutputFormat,

    /// Output directory
    pub output_directory: String,

    /// Maximum trace events per session
    pub max_events: usize,

    /// Enable real-time visualization
    pub real_time: bool,

    /// Sampling rate for real-time visualization
    pub real_time_sampling_rate: f64,

    /// Color scheme
    pub color_scheme: ColorScheme,

    /// Enable interactive features
    pub interactive: bool,
}

/// Visualization statistics
#[derive(Debug, Clone, Default)]
pub struct VisualizationStats {
    /// Total sessions created
    pub total_sessions: u64,

    /// Total traces collected
    pub total_traces: u64,

    /// Total visualizations generated
    pub total_visualizations: u64,

    /// Average processing time
    pub avg_processing_time: Duration,

    /// Total file size generated
    pub total_file_size: u64,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_format: OutputFormat::Html,
            output_directory: "/tmp/torsh_visualizations".to_string(),
            max_events: 1_000_000,
            real_time: false,
            real_time_sampling_rate: 0.1, // 10% sampling
            color_scheme: ColorScheme::Default,
            interactive: true,
        }
    }
}

impl TraceVisualizationManager {
    /// Create a new trace visualization manager
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            sessions: IndexMap::new(),
            config,
            collectors: Vec::new(),
            renderers: HashMap::new(),
            stats: VisualizationStats::default(),
        }
    }

    /// Create a new manager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(VisualizationConfig::default())
    }

    /// Start a new visualization session
    pub fn start_session(&mut self, name: &str) -> JitResult<String> {
        if !self.config.enabled {
            return Err(JitError::RuntimeError(
                "Trace visualization disabled".to_string(),
            ));
        }

        let session_id = format!("viz_session_{}", self.sessions.len() + 1);
        let session = VisualizationSession {
            id: session_id.clone(),
            name: name.to_string(),
            start_time: Instant::now(),
            traces: Vec::new(),
            call_graphs: Vec::new(),
            performance_data: PerformanceData::default(),
            metadata: HashMap::new(),
            status: SessionStatus::Collecting,
        };

        self.sessions.insert(session_id.clone(), session);
        self.stats.total_sessions += 1;

        Ok(session_id)
    }

    /// Stop a visualization session
    pub fn stop_session(&mut self, session_id: &str) -> JitResult<()> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.status = SessionStatus::Processing;

            // Process collected data
            self.process_session_data(session_id)?;

            if let Some(session) = self.sessions.get_mut(session_id) {
                session.status = SessionStatus::Ready;
            }
        } else {
            return Err(JitError::RuntimeError(format!(
                "Session {} not found",
                session_id
            )));
        }

        Ok(())
    }

    /// Add a trace event to a session
    pub fn add_trace_event(&mut self, session_id: &str, event: TraceEvent) -> JitResult<()> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.traces.is_empty() {
                session.traces.push(ExecutionTrace {
                    id: "default_trace".to_string(),
                    function_name: "main".to_string(),
                    events: Vec::new(),
                    total_time: Duration::default(),
                    thread_id: 0,
                    cpu_utilization: 0.0,
                    memory_usage: MemoryUsage::default(),
                });
            }

            if let Some(trace) = session.traces.first_mut() {
                trace.events.push(event);
                self.stats.total_traces += 1;
            }
        }

        Ok(())
    }

    /// Process session data to generate visualizations
    fn process_session_data(&mut self, session_id: &str) -> JitResult<()> {
        // Extract traces to avoid borrowing issues
        let traces = if let Some(session) = self.sessions.get(session_id) {
            session.traces.clone()
        } else {
            return Ok(());
        };

        // Generate data outside of mutable borrow
        let call_graph = self.generate_call_graph(&traces)?;
        let performance_data = self.generate_performance_data(&traces)?;
        let timeline = self.generate_timeline(&traces)?;
        let flamegraph = Some(self.generate_flamegraph(&traces)?);

        // Update session
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.call_graphs.push(call_graph);
            session.performance_data = performance_data;
            session.performance_data.timeline = timeline;
            session.performance_data.flamegraph = flamegraph;
        }

        Ok(())
    }

    /// Generate call graph from traces
    fn generate_call_graph(&self, traces: &[ExecutionTrace]) -> JitResult<CallGraph> {
        let mut nodes = IndexMap::new();
        let mut edges = Vec::new();
        let mut call_stack: Vec<String> = Vec::new();

        for trace in traces {
            for event in &trace.events {
                match event {
                    TraceEvent::FunctionEntry { function_name, .. } => {
                        // Add node if not exists
                        if !nodes.contains_key(function_name) {
                            nodes.insert(
                                function_name.clone(),
                                CallGraphNode {
                                    name: function_name.clone(),
                                    address: 0,
                                    total_time: Duration::default(),
                                    call_count: 0,
                                    avg_time: Duration::default(),
                                    cpu_utilization: 0.0,
                                    memory_usage: MemoryUsage::default(),
                                    metadata: HashMap::new(),
                                },
                            );
                        }

                        // Add edge from parent if exists
                        if let Some(parent) = call_stack.last() {
                            edges.push(CallGraphEdge {
                                from: parent.clone(),
                                to: function_name.clone(),
                                call_count: 1,
                                total_time: Duration::default(),
                                weight: 1.0,
                            });
                        }

                        call_stack.push(function_name.clone());
                    }
                    TraceEvent::FunctionExit {
                        function_name,
                        duration,
                        ..
                    } => {
                        if let Some(node) = nodes.get_mut(function_name) {
                            node.total_time += *duration;
                            node.call_count += 1;
                            node.avg_time = node.total_time / node.call_count as u32;
                        }

                        call_stack.pop();
                    }
                    _ => {}
                }
            }
        }

        let stats = CallGraphStats {
            node_count: nodes.len(),
            edge_count: edges.len(),
            max_depth: 0,
            avg_fanout: if !nodes.is_empty() {
                edges.len() as f64 / nodes.len() as f64
            } else {
                0.0
            },
            critical_path_length: Duration::default(),
        };

        Ok(CallGraph {
            nodes,
            edges,
            roots: vec!["main".to_string()],
            stats,
        })
    }

    /// Generate performance data from traces
    fn generate_performance_data(&self, traces: &[ExecutionTrace]) -> JitResult<PerformanceData> {
        let mut performance_data = PerformanceData::default();

        // Generate histograms
        let mut execution_times = Vec::new();
        for trace in traces {
            for event in &trace.events {
                if let TraceEvent::FunctionExit { duration, .. } = event {
                    execution_times.push(duration.as_nanos() as f64);
                }
            }
        }

        if !execution_times.is_empty() {
            let histogram = self.create_histogram("execution_times", &execution_times)?;
            performance_data
                .histograms
                .insert("execution_times".to_string(), histogram);
        }

        Ok(performance_data)
    }

    /// Generate timeline from traces
    fn generate_timeline(&self, traces: &[ExecutionTrace]) -> JitResult<Vec<TimelineEvent>> {
        let mut timeline = Vec::new();

        for trace in traces {
            for event in &trace.events {
                match event {
                    TraceEvent::FunctionEntry {
                        timestamp,
                        function_name,
                        ..
                    } => {
                        timeline.push(TimelineEvent {
                            timestamp: *timestamp,
                            duration: Duration::default(),
                            name: function_name.clone(),
                            category: "function".to_string(),
                            thread_id: trace.thread_id,
                            process_id: 0,
                            args: HashMap::new(),
                        });
                    }
                    _ => {}
                }
            }
        }

        // Sort by timestamp
        timeline.sort_by_key(|event| event.timestamp);

        Ok(timeline)
    }

    /// Generate flamegraph from traces
    fn generate_flamegraph(&self, traces: &[ExecutionTrace]) -> JitResult<Flamegraph> {
        let root = FlamegraphNode {
            name: "root".to_string(),
            self_time: Duration::default(),
            total_time: traces.iter().map(|t| t.total_time).sum(),
            children: Vec::new(),
            sample_count: traces.len() as u64,
        };

        Ok(Flamegraph {
            root,
            total_duration: traces.iter().map(|t| t.total_time).sum(),
            color_scheme: ColorScheme::Default,
        })
    }

    /// Create histogram from data
    fn create_histogram(&self, name: &str, data: &[f64]) -> JitResult<Histogram> {
        if data.is_empty() {
            return Ok(Histogram {
                name: name.to_string(),
                bins: Vec::new(),
                total_count: 0,
                stats: HistogramStats {
                    mean: 0.0,
                    std_dev: 0.0,
                    min: 0.0,
                    max: 0.0,
                    percentiles: HashMap::new(),
                },
            });
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        let mean = sorted_data.iter().sum::<f64>() / sorted_data.len() as f64;

        let variance =
            sorted_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / sorted_data.len() as f64;
        let std_dev = variance.sqrt();

        // Create bins
        let bin_count = 20;
        let bin_width = (max - min) / bin_count as f64;
        let mut bins = Vec::new();

        for i in 0..bin_count {
            let start = min + i as f64 * bin_width;
            let end = min + (i + 1) as f64 * bin_width;
            let count = sorted_data
                .iter()
                .filter(|&&x| x >= start && x < end)
                .count() as u64;

            bins.push(HistogramBin { start, end, count });
        }

        Ok(Histogram {
            name: name.to_string(),
            bins,
            total_count: data.len() as u64,
            stats: HistogramStats {
                mean,
                std_dev,
                min,
                max,
                percentiles: HashMap::new(),
            },
        })
    }

    /// Render visualization for a session
    pub fn render_visualization(
        &self,
        session_id: &str,
        format: OutputFormat,
        output_path: &str,
    ) -> JitResult<()> {
        if let Some(session) = self.sessions.get(session_id) {
            if let Some(renderer) = self.renderers.get(&format) {
                renderer.render(session, output_path)?;
            } else {
                // Use default renderer
                self.render_default(session, format, output_path)?;
            }
        } else {
            return Err(JitError::RuntimeError(format!(
                "Session {} not found",
                session_id
            )));
        }

        Ok(())
    }

    /// Default renderer implementation
    fn render_default(
        &self,
        session: &VisualizationSession,
        format: OutputFormat,
        output_path: &str,
    ) -> JitResult<()> {
        match format {
            OutputFormat::Json => {
                let json_data = format!(
                    r#"{{"session": "{}", "status": "{:?}", "traces": {}}}"#,
                    session.name,
                    session.status,
                    session.traces.len()
                );
                std::fs::write(output_path, json_data)
                    .map_err(|e| JitError::RuntimeError(format!("Failed to write JSON: {}", e)))?;
            }
            OutputFormat::Html => {
                let html_content = self.generate_html_visualization(session)?;
                std::fs::write(output_path, html_content)
                    .map_err(|e| JitError::RuntimeError(format!("Failed to write HTML: {}", e)))?;
            }
            _ => {
                return Err(JitError::RuntimeError(format!(
                    "Unsupported format: {:?}",
                    format
                )));
            }
        }

        Ok(())
    }

    /// Generate HTML visualization
    fn generate_html_visualization(&self, session: &VisualizationSession) -> JitResult<String> {
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>ToRSh JIT Trace Visualization - {}</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background-color: #f0f0f0; padding: 10px; }}
        .content {{ padding: 20px; }}
        .metric {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ToRSh JIT Trace Visualization</h1>
        <h2>Session: {}</h2>
    </div>
    <div class="content">
        <div class="metric">Status: {:?}</div>
        <div class="metric">Traces: {}</div>
        <div class="metric">Call Graphs: {}</div>
        <div class="metric">Timeline Events: {}</div>
    </div>
</body>
</html>"#,
            session.name,
            session.name,
            session.status,
            session.traces.len(),
            session.call_graphs.len(),
            session.performance_data.timeline.len()
        );

        Ok(html)
    }

    /// Get session
    pub fn get_session(&self, session_id: &str) -> Option<&VisualizationSession> {
        self.sessions.get(session_id)
    }

    /// Get statistics
    pub fn get_stats(&self) -> &VisualizationStats {
        &self.stats
    }

    /// Add renderer for a format
    pub fn add_renderer(&mut self, format: OutputFormat, renderer: Box<dyn VisualizationRenderer>) {
        self.renderers.insert(format, renderer);
    }
}

// Implement Clone for TraceFilter (needed for TraceCollector Clone)
impl Clone for TraceFilter {
    fn clone(&self) -> Self {
        match self {
            TraceFilter::FunctionName(name) => TraceFilter::FunctionName(name.clone()),
            TraceFilter::ThreadId(id) => TraceFilter::ThreadId(*id),
            TraceFilter::MinDuration(duration) => TraceFilter::MinDuration(*duration),
            TraceFilter::Custom(func) => TraceFilter::Custom(*func),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_manager_creation() {
        let manager = TraceVisualizationManager::with_defaults();
        assert!(manager.config.enabled);
        assert_eq!(manager.config.default_format, OutputFormat::Html);
    }

    #[test]
    fn test_session_lifecycle() {
        let mut manager = TraceVisualizationManager::with_defaults();

        let session_id = manager.start_session("test_session").unwrap();
        assert!(!session_id.is_empty());

        let session = manager.get_session(&session_id).unwrap();
        assert_eq!(session.name, "test_session");
        assert_eq!(session.status, SessionStatus::Collecting);

        manager.stop_session(&session_id).unwrap();

        let session = manager.get_session(&session_id).unwrap();
        assert_eq!(session.status, SessionStatus::Ready);
    }

    #[test]
    fn test_trace_event_addition() {
        let mut manager = TraceVisualizationManager::with_defaults();
        let session_id = manager.start_session("test_session").unwrap();

        let event = TraceEvent::FunctionEntry {
            timestamp: Instant::now(),
            function_name: "test_function".to_string(),
            address: 0x1000,
            parameters: Vec::new(),
        };

        manager.add_trace_event(&session_id, event).unwrap();

        let session = manager.get_session(&session_id).unwrap();
        assert_eq!(session.traces.len(), 1);
        assert_eq!(session.traces[0].events.len(), 1);
    }

    #[test]
    fn test_html_generation() {
        let mut manager = TraceVisualizationManager::with_defaults();
        let session_id = manager.start_session("test_session").unwrap();

        manager.stop_session(&session_id).unwrap();

        let session = manager.get_session(&session_id).unwrap();
        let html = manager.generate_html_visualization(session).unwrap();

        assert!(html.contains("ToRSh JIT Trace Visualization"));
        assert!(html.contains("test_session"));
    }
}
