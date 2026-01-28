//! Debugging utilities for distributed training systems
//!
//! This module provides comprehensive debugging tools including operation tracing,
//! state inspection, diagnostic tools, and automated troubleshooting capabilities.

use crate::metrics::get_global_metrics_collector;
use crate::profiling::get_global_profiler;
use crate::{TorshDistributedError, TorshResult};
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Logging levels for debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Debug event for tracking system operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugEvent {
    /// Unique event identifier
    pub event_id: u64,
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Log level
    pub level: LogLevel,
    /// Source module/component
    pub source: String,
    /// Rank that generated the event
    pub rank: u32,
    /// Event message
    pub message: String,
    /// Additional context data
    pub context: HashMap<String, String>,
    /// Call stack trace (if available)
    pub call_stack: Vec<String>,
    /// Duration (for operation events)
    pub duration: Option<Duration>,
}

impl DebugEvent {
    /// Create a new debug event
    pub fn new(level: LogLevel, source: String, rank: u32, message: String) -> Self {
        Self {
            event_id: 0, // Will be set by the debugger
            timestamp: SystemTime::now(),
            level,
            source,
            rank,
            message,
            context: HashMap::new(),
            call_stack: Vec::new(),
            duration: None,
        }
    }

    /// Add context information
    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }

    /// Add call stack
    pub fn with_call_stack(mut self, stack: Vec<String>) -> Self {
        self.call_stack = stack;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Format as a human-readable string
    pub fn format(&self) -> String {
        let timestamp_str = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .map(|d| format!("{:.3}", d.as_secs_f64()))
            .unwrap_or_else(|_| "unknown".to_string());

        let duration_str = self
            .duration
            .map(|d| format!(" [{}ms]", d.as_millis()))
            .unwrap_or_default();

        format!(
            "[{}] {} [{}:{}] {}{}\n",
            timestamp_str, self.level, self.source, self.rank, self.message, duration_str
        )
    }
}

/// System state snapshot for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStateSnapshot {
    /// Timestamp when snapshot was taken
    pub timestamp: SystemTime,
    /// Process group information
    pub process_group: ProcessGroupState,
    /// Communication state
    pub communication: CommunicationState,
    /// Resource utilization
    pub resources: ResourceState,
    /// Active operations
    pub active_operations: Vec<ActiveOperation>,
    /// Recent errors
    pub recent_errors: Vec<DebugEvent>,
}

/// Process group state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessGroupState {
    /// Current rank
    pub rank: u32,
    /// World size
    pub world_size: u32,
    /// Backend type
    pub backend: String,
    /// Process group health status
    pub health_status: String,
    /// Active process count
    pub active_processes: u32,
    /// Failed processes
    pub failed_processes: Vec<u32>,
}

/// Communication state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationState {
    /// Pending operations count
    pub pending_operations: u32,
    /// Failed operations count
    pub failed_operations: u32,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// Current bandwidth utilization (MB/s)
    pub bandwidth_mbps: f64,
    /// Communication queue length
    pub queue_length: u32,
    /// Last successful communication timestamp
    pub last_success: Option<SystemTime>,
}

/// Resource state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceState {
    /// CPU usage percentage
    pub cpu_usage_pct: f64,
    /// Memory usage percentage
    pub memory_usage_pct: f64,
    /// GPU usage percentage (if available)
    pub gpu_usage_pct: Option<f64>,
    /// Network I/O (bytes/sec)
    pub network_io_bps: u64,
    /// Disk I/O (bytes/sec)
    pub disk_io_bps: u64,
    /// Memory pressure indicator
    pub memory_pressure: String,
}

/// Active operation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveOperation {
    /// Operation type
    pub operation_type: String,
    /// Start time
    pub start_time: SystemTime,
    /// Expected duration
    pub expected_duration: Option<Duration>,
    /// Progress percentage (0-100)
    pub progress_pct: f64,
    /// Rank(s) involved
    pub ranks: Vec<u32>,
    /// Operation status
    pub status: String,
}

/// Diagnostic check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// Check name
    pub check_name: String,
    /// Whether the check passed
    pub passed: bool,
    /// Severity if check failed
    pub severity: LogLevel,
    /// Description of the issue (if any)
    pub description: String,
    /// Suggested remediation
    pub remediation: Vec<String>,
    /// Supporting data
    pub data: HashMap<String, String>,
}

/// Configuration for debugging utilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Whether debugging is enabled
    pub enabled: bool,
    /// Minimum log level to capture
    pub min_log_level: LogLevel,
    /// Maximum number of events to keep in memory
    pub max_events: usize,
    /// Whether to capture call stacks
    pub capture_call_stacks: bool,
    /// Whether to enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Snapshot interval (seconds)
    pub snapshot_interval_secs: u64,
    /// Auto-diagnosis interval (seconds)
    pub auto_diagnosis_interval_secs: u64,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_log_level: LogLevel::Info,
            max_events: 1000,
            capture_call_stacks: false, // Expensive operation
            real_time_monitoring: true,
            snapshot_interval_secs: 30,
            auto_diagnosis_interval_secs: 60,
        }
    }
}

/// Comprehensive debugging system for distributed training
pub struct DistributedDebugger {
    /// Configuration
    config: RwLock<DebugConfig>,
    /// Event counter for unique IDs
    event_counter: Mutex<u64>,
    /// Circular buffer of debug events
    events: Mutex<VecDeque<DebugEvent>>,
    /// System state snapshots
    snapshots: Mutex<VecDeque<SystemStateSnapshot>>,
    /// Active operation tracking
    active_operations: Mutex<HashMap<String, ActiveOperation>>,
    /// Diagnostic results history
    diagnostic_history: Mutex<Vec<DiagnosticResult>>,
    /// Performance statistics
    stats: Mutex<DebuggerStats>,
}

/// Statistics for the debugger itself
#[derive(Debug, Default, Serialize, Deserialize)]
struct DebuggerStats {
    events_captured: u64,
    snapshots_taken: u64,
    diagnostics_run: u64,
    errors_detected: u64,
}

impl DistributedDebugger {
    /// Create a new distributed debugger
    pub fn new() -> Self {
        Self::with_config(DebugConfig::default())
    }

    /// Create a new distributed debugger with custom configuration
    pub fn with_config(config: DebugConfig) -> Self {
        Self {
            config: RwLock::new(config),
            event_counter: Mutex::new(0),
            events: Mutex::new(VecDeque::new()),
            snapshots: Mutex::new(VecDeque::new()),
            active_operations: Mutex::new(HashMap::new()),
            diagnostic_history: Mutex::new(Vec::new()),
            stats: Mutex::new(DebuggerStats::default()),
        }
    }

    /// Log a debug event
    pub fn log_event(&self, mut event: DebugEvent) -> TorshResult<()> {
        let config = self
            .config
            .read()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;

        if !config.enabled || event.level < config.min_log_level {
            return Ok(());
        }

        // Assign unique event ID
        {
            let mut counter = self
                .event_counter
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            *counter += 1;
            event.event_id = *counter;
        }

        // Capture call stack if enabled
        if config.capture_call_stacks {
            // In a real implementation, you would capture the actual call stack
            event.call_stack = vec!["main".to_string(), "debug_function".to_string()];
        }

        // Store event
        {
            let mut events = self
                .events
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            events.push_back(event.clone());

            // Maintain circular buffer
            if events.len() > config.max_events {
                events.pop_front();
            }
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            stats.events_captured += 1;
            if event.level >= LogLevel::Error {
                stats.errors_detected += 1;
            }
        }

        // Print to console if critical
        if event.level >= LogLevel::Critical {
            info!("CRITICAL: {}", event.format());
        }

        Ok(())
    }

    /// Take a system state snapshot
    pub fn take_snapshot(&self) -> TorshResult<SystemStateSnapshot> {
        let snapshot = SystemStateSnapshot {
            timestamp: SystemTime::now(),
            process_group: self.capture_process_group_state()?,
            communication: self.capture_communication_state()?,
            resources: self.capture_resource_state()?,
            active_operations: self.get_active_operations(),
            recent_errors: self.get_recent_errors(10)?,
        };

        // Store snapshot
        {
            let mut snapshots = self
                .snapshots
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            snapshots.push_back(snapshot.clone());

            // Keep only last 20 snapshots
            if snapshots.len() > 20 {
                snapshots.pop_front();
            }
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            stats.snapshots_taken += 1;
        }

        Ok(snapshot)
    }

    /// Capture process group state
    fn capture_process_group_state(&self) -> TorshResult<ProcessGroupState> {
        // In a real implementation, this would query the actual process group
        Ok(ProcessGroupState {
            rank: 0,                     // Would get from actual process group
            world_size: 1,               // Would get from actual process group
            backend: "Mock".to_string(), // Would get from actual process group
            health_status: "Healthy".to_string(),
            active_processes: 1,
            failed_processes: Vec::new(),
        })
    }

    /// Capture communication state
    fn capture_communication_state(&self) -> TorshResult<CommunicationState> {
        let metrics_collector = get_global_metrics_collector();

        if let Ok(comm_history) = metrics_collector.get_communication_history() {
            if let Some(latest) = comm_history.last() {
                return Ok(CommunicationState {
                    pending_operations: 0, // Would track from actual communication system
                    failed_operations: latest.value.failed_operations as u32,
                    avg_latency_ms: latest.value.avg_latency_ms,
                    bandwidth_mbps: latest.value.avg_bandwidth_mbps,
                    queue_length: 0, // Would get from actual queue
                    last_success: Some(latest.timestamp),
                });
            }
        }

        Ok(CommunicationState {
            pending_operations: 0,
            failed_operations: 0,
            avg_latency_ms: 0.0,
            bandwidth_mbps: 0.0,
            queue_length: 0,
            last_success: None,
        })
    }

    /// Capture resource state
    fn capture_resource_state(&self) -> TorshResult<ResourceState> {
        let metrics_collector = get_global_metrics_collector();

        if let Ok(system_history) = metrics_collector.get_system_history() {
            if let Some(latest) = system_history.last() {
                return Ok(ResourceState {
                    cpu_usage_pct: latest.value.cpu_usage_pct,
                    memory_usage_pct: latest.value.memory_usage_pct,
                    gpu_usage_pct: latest.value.gpu_usage_pct,
                    network_io_bps: latest.value.network_bytes_rx + latest.value.network_bytes_tx,
                    disk_io_bps: latest.value.disk_bytes_read + latest.value.disk_bytes_write,
                    memory_pressure: if latest.value.memory_usage_pct > 90.0 {
                        "High"
                    } else {
                        "Normal"
                    }
                    .to_string(),
                });
            }
        }

        Ok(ResourceState {
            cpu_usage_pct: 0.0,
            memory_usage_pct: 0.0,
            gpu_usage_pct: None,
            network_io_bps: 0,
            disk_io_bps: 0,
            memory_pressure: "Unknown".to_string(),
        })
    }

    /// Get active operations
    fn get_active_operations(&self) -> Vec<ActiveOperation> {
        self.active_operations
            .lock()
            .map(|ops| ops.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Track an active operation
    pub fn start_operation(&self, operation_type: String, ranks: Vec<u32>) -> TorshResult<String> {
        let operation_id = format!(
            "{}_{}",
            operation_type,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        let operation = ActiveOperation {
            operation_type: operation_type.clone(),
            start_time: SystemTime::now(),
            expected_duration: None,
            progress_pct: 0.0,
            ranks,
            status: "Running".to_string(),
        };

        {
            let mut active_ops = self
                .active_operations
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            active_ops.insert(operation_id.clone(), operation);
        }

        self.log_event(
            DebugEvent::new(
                LogLevel::Debug,
                "DistributedDebugger".to_string(),
                0, // Would get actual rank
                format!("Started operation: {}", operation_type),
            )
            .with_context("operation_id".to_string(), operation_id.clone()),
        )?;

        Ok(operation_id)
    }

    /// Update operation progress
    pub fn update_operation_progress(
        &self,
        operation_id: &str,
        progress_pct: f64,
    ) -> TorshResult<()> {
        let mut active_ops = self
            .active_operations
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;

        if let Some(operation) = active_ops.get_mut(operation_id) {
            operation.progress_pct = progress_pct;
        }

        Ok(())
    }

    /// Complete an operation
    pub fn complete_operation(&self, operation_id: &str, success: bool) -> TorshResult<()> {
        let mut active_ops = self
            .active_operations
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;

        if let Some(operation) = active_ops.remove(operation_id) {
            let duration = SystemTime::now()
                .duration_since(operation.start_time)
                .unwrap_or_default();

            self.log_event(
                DebugEvent::new(
                    if success {
                        LogLevel::Debug
                    } else {
                        LogLevel::Error
                    },
                    "DistributedDebugger".to_string(),
                    0, // Would get actual rank
                    format!(
                        "Completed operation: {} ({})",
                        operation.operation_type,
                        if success { "SUCCESS" } else { "FAILED" }
                    ),
                )
                .with_context("operation_id".to_string(), operation_id.to_string())
                .with_duration(duration),
            )?;
        }

        Ok(())
    }

    /// Get recent error events
    fn get_recent_errors(&self, count: usize) -> TorshResult<Vec<DebugEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;

        Ok(events
            .iter()
            .filter(|e| e.level >= LogLevel::Error)
            .rev()
            .take(count)
            .cloned()
            .collect())
    }

    /// Run comprehensive system diagnostics
    pub fn run_diagnostics(&self) -> TorshResult<Vec<DiagnosticResult>> {
        let mut results = vec![
            // Communication health check
            self.check_communication_health()?,
        ];

        // Resource utilization check
        results.push(self.check_resource_utilization()?);

        // Bottleneck detection check
        results.push(self.check_bottlenecks()?);

        // Error rate check
        results.push(self.check_error_rate()?);

        // Process group health check
        results.push(self.check_process_group_health()?);

        // Store results
        {
            let mut diagnostic_history = self
                .diagnostic_history
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            diagnostic_history.extend(results.clone());

            // Keep only last 100 diagnostic results
            let current_len = diagnostic_history.len();
            if current_len > 100 {
                diagnostic_history.drain(0..current_len - 100);
            }
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            stats.diagnostics_run += 1;
        }

        Ok(results)
    }

    /// Check communication health
    fn check_communication_health(&self) -> TorshResult<DiagnosticResult> {
        let profiler = get_global_profiler();
        let all_stats = profiler.get_all_operation_stats()?;

        let total_failed = 0u64;
        let mut total_operations = 0u64;
        let mut max_latency: f64 = 0.0;

        for stats in all_stats.values() {
            total_operations += stats.count;
            max_latency = max_latency.max(stats.max_latency.as_secs_f64() * 1000.0);
            // Note: We don't have a direct failed count in the current profiler,
            // so this is a placeholder
        }

        let failure_rate = if total_operations > 0 {
            total_failed as f64 / total_operations as f64
        } else {
            0.0
        };
        let passed = failure_rate < 0.01 && max_latency < 1000.0; // Less than 1% failures and < 1s max latency

        Ok(DiagnosticResult {
            check_name: "Communication Health".to_string(),
            passed,
            severity: if !passed {
                LogLevel::Error
            } else {
                LogLevel::Info
            },
            description: if passed {
                "Communication system is healthy".to_string()
            } else {
                format!(
                    "Communication issues detected: {:.2}% failure rate, {:.1}ms max latency",
                    failure_rate * 100.0,
                    max_latency
                )
            },
            remediation: if !passed {
                vec![
                    "Check network connectivity".to_string(),
                    "Verify NCCL/MPI configuration".to_string(),
                    "Monitor bandwidth utilization".to_string(),
                ]
            } else {
                vec![]
            },
            data: {
                let mut data = HashMap::new();
                data.insert("failure_rate".to_string(), failure_rate.to_string());
                data.insert("max_latency_ms".to_string(), max_latency.to_string());
                data.insert("total_operations".to_string(), total_operations.to_string());
                data
            },
        })
    }

    /// Check resource utilization
    fn check_resource_utilization(&self) -> TorshResult<DiagnosticResult> {
        let state = self.capture_resource_state()?;

        let high_cpu = state.cpu_usage_pct > 95.0;
        let high_memory = state.memory_usage_pct > 90.0;
        let high_gpu = state.gpu_usage_pct.is_some_and(|gpu| gpu > 98.0);

        let passed = !high_cpu && !high_memory && !high_gpu;

        let mut issues = Vec::new();
        if high_cpu {
            issues.push(format!("High CPU usage: {:.1}%", state.cpu_usage_pct));
        }
        if high_memory {
            issues.push(format!("High memory usage: {:.1}%", state.memory_usage_pct));
        }
        if high_gpu {
            issues.push(format!(
                "High GPU usage: {:.1}%",
                state.gpu_usage_pct.unwrap_or(0.0)
            ));
        }

        Ok(DiagnosticResult {
            check_name: "Resource Utilization".to_string(),
            passed,
            severity: if !passed {
                LogLevel::Warn
            } else {
                LogLevel::Info
            },
            description: if passed {
                "Resource utilization is normal".to_string()
            } else {
                format!("Resource pressure detected: {}", issues.join(", "))
            },
            remediation: if !passed {
                vec![
                    "Scale to more resources if available".to_string(),
                    "Optimize memory usage with gradient checkpointing".to_string(),
                    "Consider model sharding or parallelism".to_string(),
                ]
            } else {
                vec![]
            },
            data: {
                let mut data = HashMap::new();
                data.insert("cpu_usage_pct".to_string(), state.cpu_usage_pct.to_string());
                data.insert(
                    "memory_usage_pct".to_string(),
                    state.memory_usage_pct.to_string(),
                );
                if let Some(gpu_usage) = state.gpu_usage_pct {
                    data.insert("gpu_usage_pct".to_string(), gpu_usage.to_string());
                }
                data
            },
        })
    }

    /// Check for bottlenecks
    fn check_bottlenecks(&self) -> TorshResult<DiagnosticResult> {
        crate::bottleneck_detection::with_global_bottleneck_detector(|detector| {
            let recent_bottlenecks = detector
                .get_bottleneck_history()
                .iter()
                .filter(|b| b.detected_at > SystemTime::now() - Duration::from_secs(300)) // Last 5 minutes
                .collect::<Vec<_>>();

            let critical_bottlenecks = recent_bottlenecks
                .iter()
                .filter(|b| {
                    matches!(
                        b.severity,
                        crate::bottleneck_detection::BottleneckSeverity::Critical
                            | crate::bottleneck_detection::BottleneckSeverity::High
                    )
                })
                .count();

            let passed = critical_bottlenecks == 0;

            Ok(DiagnosticResult {
                check_name: "Bottleneck Detection".to_string(),
                passed,
                severity: if critical_bottlenecks > 0 {
                    LogLevel::Error
                } else {
                    LogLevel::Info
                },
                description: if passed {
                    "No critical bottlenecks detected".to_string()
                } else {
                    format!(
                        "{} critical bottlenecks detected in the last 5 minutes",
                        critical_bottlenecks
                    )
                },
                remediation: if !passed {
                    vec![
                        "Review bottleneck analysis for specific recommendations".to_string(),
                        "Consider load balancing adjustments".to_string(),
                        "Optimize communication patterns".to_string(),
                    ]
                } else {
                    vec![]
                },
                data: {
                    let mut data = HashMap::new();
                    data.insert(
                        "recent_bottlenecks".to_string(),
                        recent_bottlenecks.len().to_string(),
                    );
                    data.insert(
                        "critical_bottlenecks".to_string(),
                        critical_bottlenecks.to_string(),
                    );
                    data
                },
            })
        })
    }

    /// Check error rate
    fn check_error_rate(&self) -> TorshResult<DiagnosticResult> {
        let events = self
            .events
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;

        let recent_events = events
            .iter()
            .filter(|e| e.timestamp > SystemTime::now() - Duration::from_secs(300)) // Last 5 minutes
            .collect::<Vec<_>>();

        let error_events = recent_events
            .iter()
            .filter(|e| e.level >= LogLevel::Error)
            .count();

        let error_rate = if !recent_events.is_empty() {
            error_events as f64 / recent_events.len() as f64
        } else {
            0.0
        };

        let passed = error_rate < 0.05; // Less than 5% error rate

        Ok(DiagnosticResult {
            check_name: "Error Rate".to_string(),
            passed,
            severity: if !passed {
                LogLevel::Error
            } else {
                LogLevel::Info
            },
            description: if passed {
                "Error rate is within normal limits".to_string()
            } else {
                format!(
                    "High error rate detected: {:.1}% ({} errors in {} events)",
                    error_rate * 100.0,
                    error_events,
                    recent_events.len()
                )
            },
            remediation: if !passed {
                vec![
                    "Review recent error messages for patterns".to_string(),
                    "Check system logs for underlying issues".to_string(),
                    "Verify configuration and environment setup".to_string(),
                ]
            } else {
                vec![]
            },
            data: {
                let mut data = HashMap::new();
                data.insert("error_rate".to_string(), error_rate.to_string());
                data.insert("error_count".to_string(), error_events.to_string());
                data.insert("total_events".to_string(), recent_events.len().to_string());
                data
            },
        })
    }

    /// Check process group health
    fn check_process_group_health(&self) -> TorshResult<DiagnosticResult> {
        let state = self.capture_process_group_state()?;

        let passed = state.failed_processes.is_empty() && state.health_status == "Healthy";

        Ok(DiagnosticResult {
            check_name: "Process Group Health".to_string(),
            passed,
            severity: if !passed {
                LogLevel::Critical
            } else {
                LogLevel::Info
            },
            description: if passed {
                format!(
                    "Process group is healthy ({}/{} processes active)",
                    state.active_processes, state.world_size
                )
            } else {
                format!(
                    "Process group issues: {} failed processes, status: {}",
                    state.failed_processes.len(),
                    state.health_status
                )
            },
            remediation: if !passed {
                vec![
                    "Restart failed processes if possible".to_string(),
                    "Check network connectivity between nodes".to_string(),
                    "Verify resource availability on all nodes".to_string(),
                ]
            } else {
                vec![]
            },
            data: {
                let mut data = HashMap::new();
                data.insert("world_size".to_string(), state.world_size.to_string());
                data.insert(
                    "active_processes".to_string(),
                    state.active_processes.to_string(),
                );
                data.insert(
                    "failed_processes".to_string(),
                    state.failed_processes.len().to_string(),
                );
                data.insert("health_status".to_string(), state.health_status);
                data
            },
        })
    }

    /// Generate comprehensive debug report
    pub fn generate_debug_report(&self) -> TorshResult<String> {
        let mut report = String::new();
        report.push_str("=== Distributed Training Debug Report ===\n\n");

        // System state
        if let Ok(snapshot) = self.take_snapshot() {
            report.push_str("=== Current System State ===\n");
            report.push_str(&format!("Timestamp: {:?}\n", snapshot.timestamp));
            report.push_str(&format!(
                "Process Group: Rank {}/{}, Backend: {}, Status: {}\n",
                snapshot.process_group.rank,
                snapshot.process_group.world_size,
                snapshot.process_group.backend,
                snapshot.process_group.health_status
            ));
            report.push_str(&format!(
                "Resources: CPU {:.1}%, Memory {:.1}%",
                snapshot.resources.cpu_usage_pct, snapshot.resources.memory_usage_pct
            ));
            if let Some(gpu) = snapshot.resources.gpu_usage_pct {
                report.push_str(&format!(", GPU {:.1}%", gpu));
            }
            report.push('\n');
            report.push_str(&format!(
                "Communication: {:.1}ms avg latency, {:.1} MB/s bandwidth\n",
                snapshot.communication.avg_latency_ms, snapshot.communication.bandwidth_mbps
            ));
            report.push_str(&format!(
                "Active Operations: {}\n\n",
                snapshot.active_operations.len()
            ));
        }

        // Diagnostic results
        if let Ok(diagnostics) = self.run_diagnostics() {
            report.push_str("=== Diagnostic Results ===\n");
            for diagnostic in &diagnostics {
                let status = if diagnostic.passed { "PASS" } else { "FAIL" };
                report.push_str(&format!(
                    "[{}] {}: {}\n",
                    status, diagnostic.check_name, diagnostic.description
                ));

                if !diagnostic.remediation.is_empty() {
                    report.push_str("  Recommended Actions:\n");
                    for action in &diagnostic.remediation {
                        report.push_str(&format!("  - {}\n", action));
                    }
                }
            }
            report.push('\n');
        }

        // Recent errors
        if let Ok(errors) = self.get_recent_errors(5) {
            if !errors.is_empty() {
                report.push_str("=== Recent Errors ===\n");
                for error in &errors {
                    report.push_str(&error.format());
                }
                report.push('\n');
            }
        }

        // Statistics
        if let Ok(stats) = self.stats.lock() {
            report.push_str("=== Debugger Statistics ===\n");
            report.push_str(&format!("Events Captured: {}\n", stats.events_captured));
            report.push_str(&format!("Snapshots Taken: {}\n", stats.snapshots_taken));
            report.push_str(&format!("Diagnostics Run: {}\n", stats.diagnostics_run));
            report.push_str(&format!("Errors Detected: {}\n", stats.errors_detected));
        }

        Ok(report)
    }

    /// Export debug data to JSON
    pub fn export_debug_data(&self) -> TorshResult<String> {
        #[derive(Serialize)]
        struct DebugExport {
            config: DebugConfig,
            events: Vec<DebugEvent>,
            snapshots: Vec<SystemStateSnapshot>,
            diagnostic_history: Vec<DiagnosticResult>,
            statistics: Option<DebuggerStats>,
        }

        let config = self
            .config
            .read()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?
            .clone();
        let events = self
            .events
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?
            .iter()
            .cloned()
            .collect();
        let snapshots = self
            .snapshots
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?
            .iter()
            .cloned()
            .collect();
        let diagnostic_history = self
            .diagnostic_history
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?
            .clone();
        let statistics = self.stats.lock().ok().map(|s| DebuggerStats {
            events_captured: s.events_captured,
            snapshots_taken: s.snapshots_taken,
            diagnostics_run: s.diagnostics_run,
            errors_detected: s.errors_detected,
        });

        let export = DebugExport {
            config,
            events,
            snapshots,
            diagnostic_history,
            statistics,
        };

        serde_json::to_string_pretty(&export).map_err(|e| {
            TorshDistributedError::backend_error(
                "debugging",
                format!("JSON serialization failed: {}", e),
            )
        })
    }

    /// Clear all debug data
    pub fn clear(&self) -> TorshResult<()> {
        {
            let mut events = self
                .events
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            events.clear();
        }

        {
            let mut snapshots = self
                .snapshots
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            snapshots.clear();
        }

        {
            let mut active_ops = self
                .active_operations
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            active_ops.clear();
        }

        {
            let mut diagnostic_history = self
                .diagnostic_history
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            diagnostic_history.clear();
        }

        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("debugging", "Lock poisoned"))?;
            *stats = DebuggerStats::default();
        }

        Ok(())
    }
}

impl Default for DistributedDebugger {
    fn default() -> Self {
        Self::new()
    }
}

/// Global debugger instance
static GLOBAL_DEBUGGER: std::sync::OnceLock<Arc<DistributedDebugger>> = std::sync::OnceLock::new();

/// Get the global debugger instance
pub fn get_global_debugger() -> &'static Arc<DistributedDebugger> {
    GLOBAL_DEBUGGER.get_or_init(|| Arc::new(DistributedDebugger::new()))
}

/// Initialize the global debugger with custom configuration
pub fn init_global_debugger(config: DebugConfig) -> TorshResult<()> {
    let debugger = Arc::new(DistributedDebugger::with_config(config));
    GLOBAL_DEBUGGER.set(debugger).map_err(|_| {
        TorshDistributedError::backend_error("debugging", "Global debugger already initialized")
    })?;
    Ok(())
}

/// Convenience macros for debugging
#[macro_export]
macro_rules! debug_log {
    ($level:expr, $source:expr, $rank:expr, $msg:expr) => {
        let debugger = $crate::debugging::get_global_debugger();
        let event = $crate::debugging::DebugEvent::new($level, $source.to_string(), $rank, $msg.to_string());
        let _ = debugger.log_event(event);
    };
    ($level:expr, $source:expr, $rank:expr, $msg:expr, $($key:expr => $value:expr),+) => {
        let debugger = $crate::debugging::get_global_debugger();
        let mut event = $crate::debugging::DebugEvent::new($level, $source.to_string(), $rank, $msg.to_string());
        $(
            event = event.with_context($key.to_string(), $value.to_string());
        )+
        let _ = debugger.log_event(event);
    };
}

#[macro_export]
macro_rules! debug_trace_operation {
    ($op_type:expr, $ranks:expr, $code:block) => {{
        let debugger = $crate::debugging::get_global_debugger();
        let op_id = debugger.start_operation($op_type.to_string(), $ranks).unwrap_or_default();
        let result = $code;
        let _ = debugger.complete_operation(&op_id, true); // Assume success, real impl would check result
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_event_creation() {
        let event = DebugEvent::new(
            LogLevel::Info,
            "test_module".to_string(),
            0,
            "Test message".to_string(),
        )
        .with_context("key".to_string(), "value".to_string())
        .with_duration(Duration::from_millis(100));

        assert_eq!(event.level, LogLevel::Info);
        assert_eq!(event.source, "test_module");
        assert_eq!(event.message, "Test message");
        assert_eq!(event.context.get("key"), Some(&"value".to_string()));
        assert_eq!(event.duration, Some(Duration::from_millis(100)));
    }

    #[test]
    fn test_debugger_creation() {
        let debugger = DistributedDebugger::new();
        assert!(debugger.get_active_operations().is_empty());
    }

    #[test]
    fn test_event_logging() {
        let debugger = DistributedDebugger::new();
        let event = DebugEvent::new(
            LogLevel::Info,
            "test".to_string(),
            0,
            "Test event".to_string(),
        );

        debugger.log_event(event).unwrap();

        let events = debugger.events.lock().expect("lock should not be poisoned");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].message, "Test event");
    }

    #[test]
    fn test_operation_tracking() {
        let debugger = DistributedDebugger::new();

        let op_id = debugger
            .start_operation("test_op".to_string(), vec![0, 1])
            .unwrap();
        assert!(debugger.get_active_operations().len() == 1);

        debugger.update_operation_progress(&op_id, 50.0).unwrap();
        debugger.complete_operation(&op_id, true).unwrap();

        assert!(debugger.get_active_operations().is_empty());
    }

    #[test]
    fn test_snapshot_taking() {
        let debugger = DistributedDebugger::new();
        let snapshot = debugger.take_snapshot().unwrap();

        assert_eq!(snapshot.process_group.backend, "Mock");
        assert!(snapshot.recent_errors.is_empty());
    }

    #[test]
    fn test_diagnostics() {
        let debugger = DistributedDebugger::new();
        let results = debugger.run_diagnostics().unwrap();

        assert!(!results.is_empty());
        assert!(results
            .iter()
            .any(|r| r.check_name == "Communication Health"));
        assert!(results
            .iter()
            .any(|r| r.check_name == "Resource Utilization"));
    }

    #[test]
    fn test_debug_report_generation() {
        let debugger = DistributedDebugger::new();
        let report = debugger.generate_debug_report().unwrap();

        assert!(report.contains("Distributed Training Debug Report"));
        assert!(report.contains("Current System State"));
        assert!(report.contains("Diagnostic Results"));
    }

    #[test]
    fn test_json_export() {
        let debugger = DistributedDebugger::new();
        let event = DebugEvent::new(
            LogLevel::Info,
            "test".to_string(),
            0,
            "Export test".to_string(),
        );
        debugger.log_event(event).unwrap();

        let json = debugger.export_debug_data().unwrap();
        assert!(json.contains("Export test"));
        assert!(json.contains("events"));
        assert!(json.contains("config"));
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Critical > LogLevel::Error);
        assert!(LogLevel::Error > LogLevel::Warn);
        assert!(LogLevel::Warn > LogLevel::Info);
        assert!(LogLevel::Info > LogLevel::Debug);
        assert!(LogLevel::Debug > LogLevel::Trace);
    }
}
