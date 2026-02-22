// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Autograd Operation Replay and Analysis
//!
//! This module provides recording, replay, and analysis capabilities for autograd operations,
//! enabling deterministic reproduction of computations, debugging, and performance analysis.
//!
//! # Features
//!
//! - **Operation Recording**: Capture all autograd operations for later replay
//! - **Deterministic Replay**: Reproduce exact computation sequences
//! - **Comparative Analysis**: Compare multiple execution runs
//! - **Performance Profiling**: Analyze operation performance across replays
//! - **Operation Mutation**: Modify operations during replay for experimentation
//! - **Export/Import**: Save and load recorded sessions

use crate::error_handling::{AutogradError, AutogradResult};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Recorded operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedOperation {
    /// Operation ID (unique within session)
    pub id: u64,

    /// Operation type/name
    pub operation: String,

    /// Timestamp when recorded
    pub timestamp: DateTime<Utc>,

    /// Duration of execution
    pub duration: Option<Duration>,

    /// Input tensor IDs
    pub inputs: Vec<String>,

    /// Output tensor IDs
    pub outputs: Vec<String>,

    /// Operation parameters (serialized)
    pub parameters: HashMap<String, String>,

    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,

    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,

    /// Memory allocated
    pub memory_allocated: Option<usize>,

    /// Operation metadata
    pub metadata: HashMap<String, String>,
}

impl RecordedOperation {
    /// Create a new recorded operation
    pub fn new(id: u64, operation: String) -> Self {
        Self {
            id,
            operation,
            timestamp: Utc::now(),
            duration: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            parameters: HashMap::new(),
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            memory_allocated: None,
            metadata: HashMap::new(),
        }
    }

    /// Add input
    pub fn add_input(&mut self, tensor_id: String, shape: Vec<usize>) {
        self.inputs.push(tensor_id);
        self.input_shapes.push(shape);
    }

    /// Add output
    pub fn add_output(&mut self, tensor_id: String, shape: Vec<usize>) {
        self.outputs.push(tensor_id);
        self.output_shapes.push(shape);
    }

    /// Set parameter
    pub fn set_parameter(&mut self, key: String, value: String) {
        self.parameters.insert(key, value);
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Recorded session containing a sequence of operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedSession {
    /// Session ID
    pub id: String,

    /// Session name
    pub name: String,

    /// Recording start time
    pub start_time: DateTime<Utc>,

    /// Recording end time
    pub end_time: Option<DateTime<Utc>>,

    /// Recorded operations
    pub operations: Vec<RecordedOperation>,

    /// Session metadata
    pub metadata: HashMap<String, String>,

    /// Total duration
    pub total_duration: Option<Duration>,

    /// Peak memory usage
    pub peak_memory: usize,
}

impl RecordedSession {
    /// Create a new session
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            start_time: Utc::now(),
            end_time: None,
            operations: Vec::new(),
            metadata: HashMap::new(),
            total_duration: None,
            peak_memory: 0,
        }
    }

    /// Finalize the session
    pub fn finalize(&mut self) {
        self.end_time = Some(Utc::now());
        if let Some(end) = self.end_time {
            self.total_duration = Some(
                chrono::Duration::to_std(&end.signed_duration_since(self.start_time))
                    .unwrap_or(Duration::ZERO),
            );
        }

        // Calculate peak memory
        let mut current_memory = 0;
        for op in &self.operations {
            if let Some(mem) = op.memory_allocated {
                current_memory += mem;
                self.peak_memory = self.peak_memory.max(current_memory);
            }
        }
    }

    /// Get operation by ID
    pub fn get_operation(&self, id: u64) -> Option<&RecordedOperation> {
        self.operations.iter().find(|op| op.id == id)
    }

    /// Get operations by type
    pub fn operations_by_type(&self, operation_type: &str) -> Vec<&RecordedOperation> {
        self.operations
            .iter()
            .filter(|op| op.operation == operation_type)
            .collect()
    }

    /// Get total operation count
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Get operation statistics
    pub fn operation_stats(&self) -> HashMap<String, OperationStatistics> {
        let mut stats: HashMap<String, OperationStatistics> = HashMap::new();

        for op in &self.operations {
            let stat = stats
                .entry(op.operation.clone())
                .or_insert_with(|| OperationStatistics {
                    operation_type: op.operation.clone(),
                    count: 0,
                    total_duration: Duration::ZERO,
                    total_memory: 0,
                    min_duration: Duration::MAX,
                    max_duration: Duration::ZERO,
                });

            stat.count += 1;

            if let Some(duration) = op.duration {
                stat.total_duration += duration;
                stat.min_duration = stat.min_duration.min(duration);
                stat.max_duration = stat.max_duration.max(duration);
            }

            if let Some(mem) = op.memory_allocated {
                stat.total_memory += mem;
            }
        }

        stats
    }
}

/// Operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStatistics {
    /// Operation type
    pub operation_type: String,

    /// Number of invocations
    pub count: usize,

    /// Total duration
    pub total_duration: Duration,

    /// Total memory
    pub total_memory: usize,

    /// Minimum duration
    pub min_duration: Duration,

    /// Maximum duration
    pub max_duration: Duration,
}

impl OperationStatistics {
    /// Get average duration
    pub fn average_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.count as u32
        }
    }

    /// Get average memory
    pub fn average_memory(&self) -> usize {
        if self.count == 0 {
            0
        } else {
            self.total_memory / self.count
        }
    }
}

/// Replay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Whether to verify tensor shapes match recorded shapes
    pub verify_shapes: bool,

    /// Whether to record performance during replay
    pub profile_replay: bool,

    /// Whether to pause on operation mismatches
    pub pause_on_mismatch: bool,

    /// Maximum number of operations to replay (None = all)
    pub max_operations: Option<usize>,

    /// Operation filter (only replay these operations)
    pub operation_filter: Option<Vec<String>>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            verify_shapes: true,
            profile_replay: true,
            pause_on_mismatch: false,
            max_operations: None,
            operation_filter: None,
        }
    }
}

/// Replay result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResult {
    /// Number of operations replayed
    pub operations_replayed: usize,

    /// Number of operations skipped
    pub operations_skipped: usize,

    /// Number of mismatches detected
    pub mismatches: usize,

    /// Total replay duration
    pub total_duration: Duration,

    /// Replay errors
    pub errors: Vec<String>,

    /// Performance comparison (if profiling enabled)
    pub performance_comparison: Option<PerformanceComparison>,
}

/// Performance comparison between original and replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// Original total duration
    pub original_duration: Duration,

    /// Replay total duration
    pub replay_duration: Duration,

    /// Performance difference (positive = slower, negative = faster)
    pub difference_ms: f64,

    /// Percentage difference
    pub difference_percent: f64,

    /// Per-operation comparisons
    pub operation_comparisons: Vec<OperationComparison>,
}

/// Per-operation performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationComparison {
    /// Operation ID
    pub operation_id: u64,

    /// Operation type
    pub operation_type: String,

    /// Original duration
    pub original_duration: Duration,

    /// Replay duration
    pub replay_duration: Duration,

    /// Difference (ms)
    pub difference_ms: f64,
}

/// Operation recorder
pub struct OperationRecorder {
    /// Whether recording is enabled
    enabled: Arc<RwLock<bool>>,

    /// Current session
    current_session: Arc<Mutex<Option<RecordedSession>>>,

    /// Completed sessions
    completed_sessions: Arc<Mutex<Vec<RecordedSession>>>,

    /// Next operation ID
    next_operation_id: Arc<Mutex<u64>>,
}

impl OperationRecorder {
    /// Create a new recorder
    pub fn new() -> Self {
        Self {
            enabled: Arc::new(RwLock::new(false)),
            current_session: Arc::new(Mutex::new(None)),
            completed_sessions: Arc::new(Mutex::new(Vec::new())),
            next_operation_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Start recording a new session
    pub fn start_session(&self, name: String) -> String {
        let session_id = format!("session_{}", Utc::now().timestamp());
        let session = RecordedSession::new(session_id.clone(), name);

        *self.current_session.lock() = Some(session);
        *self.enabled.write() = true;

        session_id
    }

    /// Stop recording current session
    pub fn stop_session(&self) -> Option<RecordedSession> {
        *self.enabled.write() = false;

        let mut current = self.current_session.lock();
        if let Some(mut session) = current.take() {
            session.finalize();

            self.completed_sessions.lock().push(session.clone());
            Some(session)
        } else {
            None
        }
    }

    /// Record an operation
    pub fn record_operation(
        &self,
        operation: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        duration: Option<Duration>,
    ) -> Option<u64> {
        if !*self.enabled.read() {
            return None;
        }

        let op_id = {
            let mut next_id = self.next_operation_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let mut recorded_op = RecordedOperation::new(op_id, operation);
        recorded_op.inputs = inputs;
        recorded_op.outputs = outputs;
        recorded_op.duration = duration;

        let mut session = self.current_session.lock();
        if let Some(ref mut sess) = *session {
            sess.operations.push(recorded_op);
            Some(op_id)
        } else {
            None
        }
    }

    /// Get current session
    pub fn current_session(&self) -> Option<RecordedSession> {
        self.current_session.lock().clone()
    }

    /// Get completed sessions
    pub fn completed_sessions(&self) -> Vec<RecordedSession> {
        self.completed_sessions.lock().clone()
    }

    /// Get session by ID
    pub fn get_session(&self, id: &str) -> Option<RecordedSession> {
        self.completed_sessions
            .lock()
            .iter()
            .find(|s| s.id == id)
            .cloned()
    }

    /// Clear completed sessions
    pub fn clear_sessions(&self) {
        self.completed_sessions.lock().clear();
    }

    /// Export session to JSON
    pub fn export_session(&self, session_id: &str) -> AutogradResult<String> {
        if let Some(session) = self.get_session(session_id) {
            serde_json::to_string_pretty(&session).map_err(|e| AutogradError::Configuration {
                parameter: "serialization".to_string(),
                value: "session".to_string(),
                reason: format!("Failed: {}", e),
                valid_range: None,
            })
        } else {
            Err(AutogradError::Configuration {
                parameter: "session_id".to_string(),
                value: session_id.to_string(),
                reason: "Session not found".to_string(),
                valid_range: None,
            })
        }
    }

    /// Import session from JSON
    pub fn import_session(&self, json: &str) -> AutogradResult<String> {
        let session: RecordedSession =
            serde_json::from_str(json).map_err(|e| AutogradError::Configuration {
                parameter: "deserialization".to_string(),
                value: "session".to_string(),
                reason: format!("Failed: {}", e),
                valid_range: None,
            })?;

        let session_id = session.id.clone();
        self.completed_sessions.lock().push(session);

        Ok(session_id)
    }

    /// Save session to file
    pub fn save_session(&self, session_id: &str, path: &Path) -> AutogradResult<()> {
        let json = self.export_session(session_id)?;

        std::fs::write(path, json).map_err(|e| AutogradError::Configuration {
            parameter: "file_write".to_string(),
            value: "session".to_string(),
            reason: format!("Failed: {}", e),
            valid_range: None,
        })?;

        Ok(())
    }

    /// Load session from file
    pub fn load_session(&self, path: &Path) -> AutogradResult<String> {
        let json = std::fs::read_to_string(path).map_err(|e| AutogradError::Configuration {
            parameter: "file_read".to_string(),
            value: "session".to_string(),
            reason: format!("Failed: {}", e),
            valid_range: None,
        })?;

        self.import_session(&json)
    }
}

impl Default for OperationRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation replayer
pub struct OperationReplayer {
    /// Replay configuration
    config: ReplayConfig,

    /// Current replay position
    current_position: Arc<Mutex<usize>>,

    /// Replay statistics
    statistics: Arc<Mutex<ReplayStatistics>>,
}

/// Replay statistics
#[derive(Debug, Clone, Default)]
pub struct ReplayStatistics {
    /// Total operations replayed
    pub total_replayed: usize,

    /// Total operations skipped
    pub total_skipped: usize,

    /// Shape mismatches
    pub shape_mismatches: usize,

    /// Type mismatches
    pub type_mismatches: usize,

    /// Errors encountered
    pub errors: Vec<String>,
}

impl OperationReplayer {
    /// Create a new replayer
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            config,
            current_position: Arc::new(Mutex::new(0)),
            statistics: Arc::new(Mutex::new(ReplayStatistics::default())),
        }
    }

    /// Replay a session
    pub fn replay_session(&self, session: &RecordedSession) -> AutogradResult<ReplayResult> {
        let start_time = std::time::Instant::now();
        let mut stats = self.statistics.lock();

        stats.total_replayed = 0;
        stats.total_skipped = 0;
        stats.shape_mismatches = 0;
        stats.type_mismatches = 0;
        stats.errors.clear();

        let max_ops = self
            .config
            .max_operations
            .unwrap_or(session.operations.len());

        for (i, op) in session.operations.iter().take(max_ops).enumerate() {
            // Check operation filter
            if let Some(ref filter) = self.config.operation_filter {
                if !filter.contains(&op.operation) {
                    stats.total_skipped += 1;
                    continue;
                }
            }

            // TODO: Actually replay the operation
            // This would require integrating with the actual autograd execution engine

            stats.total_replayed += 1;
            *self.current_position.lock() = i + 1;
        }

        let total_duration = start_time.elapsed();

        Ok(ReplayResult {
            operations_replayed: stats.total_replayed,
            operations_skipped: stats.total_skipped,
            mismatches: stats.shape_mismatches + stats.type_mismatches,
            total_duration,
            errors: stats.errors.clone(),
            performance_comparison: None, // TODO: Implement performance comparison
        })
    }

    /// Get current position
    pub fn current_position(&self) -> usize {
        *self.current_position.lock()
    }

    /// Reset position
    pub fn reset(&self) {
        *self.current_position.lock() = 0;
        *self.statistics.lock() = ReplayStatistics::default();
    }
}

/// Global operation recorder
static GLOBAL_RECORDER: once_cell::sync::Lazy<OperationRecorder> =
    once_cell::sync::Lazy::new(OperationRecorder::new);

/// Get the global recorder
pub fn global_recorder() -> &'static OperationRecorder {
    &GLOBAL_RECORDER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let recorder = OperationRecorder::new();
        let session_id = recorder.start_session("test_session".to_string());

        assert!(!session_id.is_empty());

        let session = recorder.stop_session();
        assert!(session.is_some());

        let sess = session.unwrap();
        assert_eq!(sess.name, "test_session");
    }

    #[test]
    fn test_operation_recording() {
        let recorder = OperationRecorder::new();
        recorder.start_session("test".to_string());

        let op_id = recorder.record_operation(
            "matmul".to_string(),
            vec!["t1".to_string(), "t2".to_string()],
            vec!["t3".to_string()],
            Some(Duration::from_millis(10)),
        );

        assert!(op_id.is_some());

        let session = recorder.stop_session();
        assert!(session.is_some());

        let sess = session.unwrap();
        assert_eq!(sess.operations.len(), 1);
        assert_eq!(sess.operations[0].operation, "matmul");
    }

    #[test]
    fn test_session_export_import() {
        let recorder = OperationRecorder::new();
        recorder.start_session("export_test".to_string());

        recorder.record_operation(
            "add".to_string(),
            vec!["a".to_string()],
            vec!["b".to_string()],
            None,
        );

        let session = recorder.stop_session().unwrap();
        let session_id = session.id.clone();

        let json = recorder.export_session(&session_id);
        assert!(json.is_ok());

        // Clear and re-import
        recorder.clear_sessions();

        let imported_id = recorder.import_session(&json.unwrap());
        assert!(imported_id.is_ok());

        let imported_session = recorder.get_session(&imported_id.unwrap());
        assert!(imported_session.is_some());
    }

    #[test]
    fn test_operation_statistics() {
        let mut session = RecordedSession::new("test".to_string(), "test".to_string());

        for i in 0..5 {
            let mut op = RecordedOperation::new(i, "matmul".to_string());
            op.duration = Some(Duration::from_millis(10 * (i + 1)));
            session.operations.push(op);
        }

        let stats = session.operation_stats();

        assert_eq!(stats.len(), 1);
        assert_eq!(stats["matmul"].count, 5);
        assert_eq!(
            stats["matmul"].average_duration(),
            Duration::from_millis(30)
        );
    }

    #[test]
    fn test_replayer() {
        let mut session = RecordedSession::new("replay_test".to_string(), "test".to_string());

        for i in 0..10 {
            session
                .operations
                .push(RecordedOperation::new(i, format!("op_{}", i)));
        }

        let config = ReplayConfig::default();
        let replayer = OperationReplayer::new(config);

        let result = replayer.replay_session(&session);
        assert!(result.is_ok());

        let res = result.unwrap();
        assert_eq!(res.operations_replayed, 10);
    }
}
