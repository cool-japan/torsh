// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient Computation Tracing
//!
//! This module provides comprehensive tracing support for gradient computation paths,
//! enabling detailed analysis of gradient flow, performance profiling, and debugging.
//!
//! # Features
//!
//! - **Distributed Tracing**: OpenTelemetry-compatible tracing spans
//! - **Gradient Path Tracking**: Track gradients through computation graph
//! - **Performance Attribution**: Identify performance bottlenecks per operation
//! - **Causality Tracking**: Understand dependencies and data flow
//! - **Sampling**: Configurable trace sampling for production use
//! - **Export**: Export traces in multiple formats (JSON, OpenTelemetry, Jaeger)

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable tracing
    pub enabled: bool,

    /// Trace sampling rate (0.0-1.0)
    pub sampling_rate: f64,

    /// Maximum trace depth
    pub max_depth: usize,

    /// Maximum spans per trace
    pub max_spans_per_trace: usize,

    /// Enable gradient value tracking
    pub track_gradient_values: bool,

    /// Enable tensor shape tracking
    pub track_tensor_shapes: bool,

    /// Enable memory tracking
    pub track_memory_usage: bool,

    /// Export format
    pub export_format: TraceExportFormat,

    /// Trace retention duration (seconds)
    pub retention_duration_secs: u64,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_rate: 0.1, // 10% sampling in production
            max_depth: 1000,
            max_spans_per_trace: 10000,
            track_gradient_values: false, // Expensive
            track_tensor_shapes: true,
            track_memory_usage: true,
            export_format: TraceExportFormat::Json,
            retention_duration_secs: 3600, // 1 hour
        }
    }
}

/// Trace export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceExportFormat {
    /// JSON format
    Json,

    /// OpenTelemetry format
    OpenTelemetry,

    /// Jaeger format
    Jaeger,

    /// Zipkin format
    Zipkin,
}

/// Gradient computation tracer
pub struct GradientTracer {
    config: TracingConfig,
    active_traces: Arc<RwLock<HashMap<String, Trace>>>,
    completed_traces: Arc<RwLock<VecDeque<Trace>>>,
    span_stack: Arc<RwLock<Vec<String>>>,
}

/// Trace represents a complete gradient computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// Trace ID
    pub trace_id: String,

    /// Trace name
    pub name: String,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// End time
    pub end_time: Option<DateTime<Utc>>,

    /// Duration (milliseconds)
    pub duration_ms: Option<f64>,

    /// Root span
    pub root_span: Option<Span>,

    /// All spans in trace
    pub spans: HashMap<String, Span>,

    /// Trace metadata
    pub metadata: HashMap<String, String>,

    /// Total memory allocated
    pub total_memory_bytes: u64,

    /// Status
    pub status: TraceStatus,

    /// Error information
    pub error: Option<String>,
}

/// Span represents a single operation in the trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Span ID
    pub span_id: String,

    /// Parent span ID
    pub parent_span_id: Option<String>,

    /// Trace ID
    pub trace_id: String,

    /// Operation name
    pub operation_name: String,

    /// Operation type
    pub operation_type: String,

    /// Start time
    pub start_time: DateTime<Utc>,

    /// End time
    pub end_time: Option<DateTime<Utc>>,

    /// Duration (microseconds for precision)
    pub duration_us: Option<u64>,

    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,

    /// Events within this span
    pub events: Vec<SpanEvent>,

    /// Tags
    pub tags: HashMap<String, String>,

    /// Status
    pub status: SpanStatus,

    /// Child span IDs
    pub child_spans: Vec<String>,
}

/// Attribute value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<AttributeValue>),
}

/// Span event (point-in-time occurrence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name
    pub name: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,
}

/// Trace status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceStatus {
    /// In progress
    InProgress,

    /// Completed successfully
    Completed,

    /// Failed
    Failed,

    /// Cancelled
    Cancelled,
}

/// Span status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// OK
    Ok,

    /// Error
    Error,

    /// Unset
    Unset,
}

impl GradientTracer {
    /// Create a new gradient tracer
    pub fn new(config: TracingConfig) -> Self {
        Self {
            config,
            active_traces: Arc::new(RwLock::new(HashMap::new())),
            completed_traces: Arc::new(RwLock::new(VecDeque::new())),
            span_stack: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start a new trace
    pub fn start_trace(&self, name: String) -> Option<String> {
        if !self.config.enabled {
            return None;
        }

        // Apply sampling
        if !self.should_sample() {
            return None;
        }

        let trace_id = uuid::Uuid::new_v4().to_string();

        let trace = Trace {
            trace_id: trace_id.clone(),
            name,
            start_time: Utc::now(),
            end_time: None,
            duration_ms: None,
            root_span: None,
            spans: HashMap::new(),
            metadata: HashMap::new(),
            total_memory_bytes: 0,
            status: TraceStatus::InProgress,
            error: None,
        };

        self.active_traces.write().insert(trace_id.clone(), trace);

        Some(trace_id)
    }

    /// End a trace
    pub fn end_trace(&self, trace_id: &str) {
        let mut active = self.active_traces.write();

        if let Some(mut trace) = active.remove(trace_id) {
            trace.end_time = Some(Utc::now());
            trace.duration_ms = Some(
                (trace.end_time.expect("end_time was just set") - trace.start_time)
                    .num_milliseconds() as f64,
            );
            trace.status = TraceStatus::Completed;

            // Add to completed traces
            let mut completed = self.completed_traces.write();
            completed.push_back(trace);

            // Enforce retention limit
            let retention_cutoff =
                Utc::now() - chrono::Duration::seconds(self.config.retention_duration_secs as i64);

            while let Some(oldest) = completed.front() {
                if oldest.end_time.unwrap_or(Utc::now()) < retention_cutoff {
                    completed.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Start a new span
    pub fn start_span(
        &self,
        trace_id: &str,
        operation_name: String,
        operation_type: String,
    ) -> Option<String> {
        if !self.config.enabled {
            return None;
        }

        let mut active = self.active_traces.write();

        if let Some(trace) = active.get_mut(trace_id) {
            // Check span limit
            if trace.spans.len() >= self.config.max_spans_per_trace {
                return None;
            }

            let span_id = uuid::Uuid::new_v4().to_string();

            // Get parent span from stack
            let parent_span_id = self.span_stack.read().last().cloned();

            let span = Span {
                span_id: span_id.clone(),
                parent_span_id: parent_span_id.clone(),
                trace_id: trace_id.to_string(),
                operation_name,
                operation_type,
                start_time: Utc::now(),
                end_time: None,
                duration_us: None,
                attributes: HashMap::new(),
                events: Vec::new(),
                tags: HashMap::new(),
                status: SpanStatus::Unset,
                child_spans: Vec::new(),
            };

            // Update parent's children
            if let Some(parent_id) = parent_span_id {
                if let Some(parent) = trace.spans.get_mut(&parent_id) {
                    parent.child_spans.push(span_id.clone());
                }
            } else {
                // This is the root span
                trace.root_span = Some(span.clone());
            }

            trace.spans.insert(span_id.clone(), span);

            // Push to span stack
            self.span_stack.write().push(span_id.clone());

            Some(span_id)
        } else {
            None
        }
    }

    /// End a span
    pub fn end_span(&self, trace_id: &str, span_id: &str) {
        let mut active = self.active_traces.write();

        if let Some(trace) = active.get_mut(trace_id) {
            if let Some(span) = trace.spans.get_mut(span_id) {
                span.end_time = Some(Utc::now());
                span.duration_us = Some(
                    (span.end_time.expect("end_time was just set") - span.start_time)
                        .num_microseconds()
                        .unwrap_or(0) as u64,
                );
                span.status = SpanStatus::Ok;

                // Pop from span stack
                let mut stack = self.span_stack.write();
                if stack.last() == Some(&span_id.to_string()) {
                    stack.pop();
                }
            }
        }
    }

    /// Add attribute to span
    pub fn add_span_attribute(
        &self,
        trace_id: &str,
        span_id: &str,
        key: String,
        value: AttributeValue,
    ) {
        let mut active = self.active_traces.write();

        if let Some(trace) = active.get_mut(trace_id) {
            if let Some(span) = trace.spans.get_mut(span_id) {
                span.attributes.insert(key, value);
            }
        }
    }

    /// Add event to span
    pub fn add_span_event(
        &self,
        trace_id: &str,
        span_id: &str,
        event_name: String,
        attributes: HashMap<String, AttributeValue>,
    ) {
        let mut active = self.active_traces.write();

        if let Some(trace) = active.get_mut(trace_id) {
            if let Some(span) = trace.spans.get_mut(span_id) {
                span.events.push(SpanEvent {
                    name: event_name,
                    timestamp: Utc::now(),
                    attributes,
                });
            }
        }
    }

    /// Record gradient computation
    pub fn record_gradient_computation(
        &self,
        trace_id: &str,
        span_id: &str,
        tensor_id: &str,
        gradient_norm: Option<f64>,
        tensor_shape: Option<Vec<usize>>,
    ) {
        if let Some(gradient_norm) = gradient_norm {
            if self.config.track_gradient_values {
                self.add_span_attribute(
                    trace_id,
                    span_id,
                    "gradient_norm".to_string(),
                    AttributeValue::Float(gradient_norm),
                );
            }
        }

        if let Some(shape) = tensor_shape {
            if self.config.track_tensor_shapes {
                let shape_str = format!("{:?}", shape);
                self.add_span_attribute(
                    trace_id,
                    span_id,
                    "tensor_shape".to_string(),
                    AttributeValue::String(shape_str),
                );
            }
        }

        self.add_span_attribute(
            trace_id,
            span_id,
            "tensor_id".to_string(),
            AttributeValue::String(tensor_id.to_string()),
        );
    }

    /// Get completed traces
    pub fn get_completed_traces(&self, limit: Option<usize>) -> Vec<Trace> {
        let completed = self.completed_traces.read();
        let limit = limit.unwrap_or(100);

        completed.iter().rev().take(limit).cloned().collect()
    }

    /// Get trace by ID
    pub fn get_trace(&self, trace_id: &str) -> Option<Trace> {
        // Check active traces first
        if let Some(trace) = self.active_traces.read().get(trace_id) {
            return Some(trace.clone());
        }

        // Check completed traces
        let completed = self.completed_traces.read();
        completed.iter().find(|t| t.trace_id == trace_id).cloned()
    }

    /// Export trace to JSON
    pub fn export_trace_json(&self, trace_id: &str) -> Option<String> {
        self.get_trace(trace_id)
            .and_then(|trace| serde_json::to_string_pretty(&trace).ok())
    }

    /// Get trace statistics
    pub fn get_trace_statistics(&self, trace_id: &str) -> Option<TraceStatistics> {
        self.get_trace(trace_id).map(|trace| {
            let total_spans = trace.spans.len();
            let completed_spans = trace
                .spans
                .values()
                .filter(|s| s.end_time.is_some())
                .count();

            let total_duration_us: u64 = trace.spans.values().filter_map(|s| s.duration_us).sum();

            let avg_span_duration_us = if completed_spans > 0 {
                total_duration_us / completed_spans as u64
            } else {
                0
            };

            let max_depth = self.calculate_trace_depth(&trace);

            TraceStatistics {
                trace_id: trace_id.to_string(),
                total_spans,
                completed_spans,
                max_depth,
                total_duration_us,
                avg_span_duration_us,
                total_memory_bytes: trace.total_memory_bytes,
            }
        })
    }

    // Private helper methods

    fn should_sample(&self) -> bool {
        use scirs2_core::random::thread_rng;
        let mut rng = thread_rng();
        rng.random::<f64>() < self.config.sampling_rate
    }

    fn calculate_trace_depth(&self, trace: &Trace) -> usize {
        fn span_depth(span: &Span, spans: &HashMap<String, Span>) -> usize {
            let child_depths: Vec<usize> = span
                .child_spans
                .iter()
                .filter_map(|child_id| spans.get(child_id))
                .map(|child| span_depth(child, spans))
                .collect();

            1 + child_depths.into_iter().max().unwrap_or(0)
        }

        trace
            .root_span
            .as_ref()
            .map(|root| span_depth(root, &trace.spans))
            .unwrap_or(0)
    }
}

/// Trace statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStatistics {
    /// Trace ID
    pub trace_id: String,

    /// Total spans
    pub total_spans: usize,

    /// Completed spans
    pub completed_spans: usize,

    /// Maximum depth
    pub max_depth: usize,

    /// Total duration (microseconds)
    pub total_duration_us: u64,

    /// Average span duration (microseconds)
    pub avg_span_duration_us: u64,

    /// Total memory used (bytes)
    pub total_memory_bytes: u64,
}

/// RAII span guard for automatic span management
pub struct SpanGuard {
    tracer: Arc<GradientTracer>,
    trace_id: String,
    span_id: String,
    #[allow(dead_code)]
    start_instant: Instant,
}

impl SpanGuard {
    /// Create a new span guard
    pub fn new(tracer: Arc<GradientTracer>, trace_id: String, span_id: String) -> Self {
        Self {
            tracer,
            trace_id,
            span_id,
            start_instant: Instant::now(),
        }
    }

    /// Add attribute to this span
    pub fn add_attribute(&self, key: String, value: AttributeValue) {
        self.tracer
            .add_span_attribute(&self.trace_id, &self.span_id, key, value);
    }

    /// Add event to this span
    pub fn add_event(&self, name: String, attributes: HashMap<String, AttributeValue>) {
        self.tracer
            .add_span_event(&self.trace_id, &self.span_id, name, attributes);
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        self.tracer.end_span(&self.trace_id, &self.span_id);
    }
}

/// Global gradient tracer instance
static GLOBAL_TRACER: OnceLock<Arc<GradientTracer>> = OnceLock::new();

/// Get global gradient tracer
pub fn get_global_tracer() -> Arc<GradientTracer> {
    GLOBAL_TRACER
        .get_or_init(|| Arc::new(GradientTracer::new(TracingConfig::default())))
        .clone()
}

/// Initialize global gradient tracer
pub fn init_global_tracer(config: TracingConfig) {
    let _ = GLOBAL_TRACER.set(Arc::new(GradientTracer::new(config)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_creation() {
        let tracer = GradientTracer::new(TracingConfig {
            sampling_rate: 1.0,
            ..Default::default()
        });

        let trace_id = tracer.start_trace("test_trace".to_string());
        assert!(trace_id.is_some());

        let trace_id = trace_id.unwrap();
        let trace = tracer.get_trace(&trace_id);
        assert!(trace.is_some());

        let trace = trace.unwrap();
        assert_eq!(trace.name, "test_trace");
        assert_eq!(trace.status, TraceStatus::InProgress);
    }

    #[test]
    fn test_span_creation() {
        let tracer = GradientTracer::new(TracingConfig {
            sampling_rate: 1.0,
            ..Default::default()
        });

        let trace_id = tracer.start_trace("test_trace".to_string()).unwrap();

        let span_id = tracer
            .start_span(&trace_id, "test_op".to_string(), "forward".to_string())
            .unwrap();

        let trace = tracer.get_trace(&trace_id).unwrap();
        assert_eq!(trace.spans.len(), 1);
        assert!(trace.spans.contains_key(&span_id));
    }

    #[test]
    fn test_nested_spans() {
        let tracer = GradientTracer::new(TracingConfig {
            sampling_rate: 1.0,
            ..Default::default()
        });

        let trace_id = tracer.start_trace("test_trace".to_string()).unwrap();

        let parent_span = tracer
            .start_span(&trace_id, "parent".to_string(), "op".to_string())
            .unwrap();

        let child_span = tracer
            .start_span(&trace_id, "child".to_string(), "op".to_string())
            .unwrap();

        tracer.end_span(&trace_id, &child_span);
        tracer.end_span(&trace_id, &parent_span);

        let trace = tracer.get_trace(&trace_id).unwrap();
        let parent = trace.spans.get(&parent_span).unwrap();
        assert_eq!(parent.child_spans.len(), 1);
        assert_eq!(parent.child_spans[0], child_span);
    }

    #[test]
    fn test_span_attributes() {
        let tracer = GradientTracer::new(TracingConfig {
            sampling_rate: 1.0,
            ..Default::default()
        });

        let trace_id = tracer.start_trace("test_trace".to_string()).unwrap();
        let span_id = tracer
            .start_span(&trace_id, "test_op".to_string(), "forward".to_string())
            .unwrap();

        tracer.add_span_attribute(
            &trace_id,
            &span_id,
            "test_attr".to_string(),
            AttributeValue::String("test_value".to_string()),
        );

        let trace = tracer.get_trace(&trace_id).unwrap();
        let span = trace.spans.get(&span_id).unwrap();
        assert!(span.attributes.contains_key("test_attr"));
    }

    #[test]
    fn test_trace_statistics() {
        let tracer = GradientTracer::new(TracingConfig {
            sampling_rate: 1.0,
            ..Default::default()
        });

        let trace_id = tracer.start_trace("test_trace".to_string()).unwrap();

        let span1 = tracer
            .start_span(&trace_id, "op1".to_string(), "forward".to_string())
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        tracer.end_span(&trace_id, &span1);

        let span2 = tracer
            .start_span(&trace_id, "op2".to_string(), "backward".to_string())
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        tracer.end_span(&trace_id, &span2);

        let stats = tracer.get_trace_statistics(&trace_id).unwrap();
        assert_eq!(stats.total_spans, 2);
        assert_eq!(stats.completed_spans, 2);
        assert!(stats.total_duration_us > 0);
    }

    #[test]
    fn test_span_guard() {
        let tracer = Arc::new(GradientTracer::new(TracingConfig {
            sampling_rate: 1.0,
            ..Default::default()
        }));

        let trace_id = tracer.start_trace("test_trace".to_string()).unwrap();

        {
            let span_id = tracer
                .start_span(&trace_id, "guarded_op".to_string(), "op".to_string())
                .unwrap();

            let _guard = SpanGuard::new(tracer.clone(), trace_id.clone(), span_id.clone());

            // Span should be active
            let trace = tracer.get_trace(&trace_id).unwrap();
            let span = trace.spans.get(&span_id).unwrap();
            assert!(span.end_time.is_none());
        }

        // After guard drops, span should be ended
        std::thread::sleep(std::time::Duration::from_millis(10));
        let trace = tracer.get_trace(&trace_id).unwrap();
        let span = trace.spans.values().next().unwrap();
        assert!(span.end_time.is_some());
    }
}
