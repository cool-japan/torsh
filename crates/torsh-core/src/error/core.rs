//! Core error infrastructure for ToRSh
//!
//! This module provides fundamental error handling infrastructure including
//! location tracking, debug context, formatting utilities, and base traits.

use std::backtrace::{Backtrace, BacktraceStatus};
use std::fmt;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Helper struct for formatting shapes in error messages
pub struct ShapeDisplay<'a>(&'a [usize]);

impl fmt::Display for ShapeDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

/// Helper function to format shapes nicely
pub fn format_shape(shape: &[usize]) -> ShapeDisplay<'_> {
    ShapeDisplay(shape)
}

/// Source location information for error tracking
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    /// File where the error occurred
    pub file: String,
    /// Line number where the error occurred
    pub line: u32,
    /// Column number where the error occurred
    pub column: u32,
    /// Function or method name (if available)
    pub function: Option<String>,
}

impl fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(function) = &self.function {
            write!(f, "{}:{} in {}", self.file, self.line, function)
        } else {
            write!(f, "{}:{}", self.file, self.line)
        }
    }
}

impl From<&'static Location<'static>> for ErrorLocation {
    fn from(location: &'static Location<'static>) -> Self {
        Self {
            file: location.file().to_string(),
            line: location.line(),
            column: location.column(),
            function: None,
        }
    }
}

impl ErrorLocation {
    /// Add function information to the location
    pub fn with_function(mut self, function: &str) -> Self {
        self.function = Some(function.to_string());
        self
    }
}

/// Enhanced debug context for error reporting
#[derive(Debug, Clone)]
pub struct ErrorDebugContext {
    pub thread_info: ThreadInfo,
    pub backtrace: Option<String>,
    pub metadata: std::collections::HashMap<String, String>,
    pub timestamp: SystemTime,
    pub error_id: u64,
}

/// Thread information for error context
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    pub thread_id: String,
    pub thread_name: Option<String>,
}

impl Default for ErrorDebugContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorDebugContext {
    /// Create a new error debug context with full information
    pub fn new() -> Self {
        static ERROR_COUNTER: AtomicU64 = AtomicU64::new(0);

        Self {
            thread_info: ThreadInfo::current(),
            backtrace: capture_stack_trace(),
            metadata: std::collections::HashMap::new(),
            timestamp: SystemTime::now(),
            error_id: ERROR_COUNTER.fetch_add(1, Ordering::SeqCst),
        }
    }

    /// Create a minimal error debug context (less overhead)
    pub fn minimal() -> Self {
        static ERROR_COUNTER: AtomicU64 = AtomicU64::new(0);

        Self {
            thread_info: ThreadInfo::current(),
            backtrace: None,
            metadata: std::collections::HashMap::new(),
            timestamp: SystemTime::now(),
            error_id: ERROR_COUNTER.fetch_add(1, Ordering::SeqCst),
        }
    }

    /// Add metadata to the error context
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Format debug information as a human-readable string
    pub fn format_debug_info(&self) -> String {
        let mut info = String::new();

        info.push_str(&format!("Error ID: {}\n", self.error_id));

        if let Ok(elapsed) = self.timestamp.duration_since(UNIX_EPOCH) {
            info.push_str(&format!(
                "Timestamp: {:.3}s since epoch\n",
                elapsed.as_secs_f64()
            ));
        }

        info.push_str(&format!(
            "Thread: {} ({})\n",
            self.thread_info.thread_id,
            self.thread_info.thread_name.as_deref().unwrap_or("unnamed")
        ));

        if !self.metadata.is_empty() {
            info.push_str("Metadata:\n");
            for (key, value) in &self.metadata {
                info.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        if let Some(ref backtrace) = self.backtrace {
            info.push_str("Backtrace:\n");
            info.push_str(backtrace);
        }

        info
    }
}

impl ThreadInfo {
    /// Get current thread information
    pub fn current() -> Self {
        Self {
            thread_id: format!("{:?}", std::thread::current().id()),
            thread_name: std::thread::current().name().map(|s| s.to_string()),
        }
    }
}

/// Capture stack trace if available
pub fn capture_stack_trace() -> Option<String> {
    let backtrace = Backtrace::capture();
    match backtrace.status() {
        BacktraceStatus::Captured => Some(format!("{backtrace}")),
        BacktraceStatus::Disabled => {
            // Provide helpful message when backtraces are disabled
            Some(
                "Backtrace disabled. Set RUST_BACKTRACE=1 environment variable to enable."
                    .to_string(),
            )
        }
        BacktraceStatus::Unsupported => {
            Some("Backtrace not supported on this platform".to_string())
        }
        _ => None,
    }
}

/// Capture minimal stack trace with limited frames
pub fn capture_minimal_stack_trace(max_frames: usize) -> Option<String> {
    let backtrace = Backtrace::capture();
    match backtrace.status() {
        BacktraceStatus::Captured => {
            let full_trace = format!("{backtrace}");
            let lines: Vec<&str> = full_trace.lines().take(max_frames).collect();
            Some(lines.join("\n"))
        }
        _ => capture_stack_trace(),
    }
}

/// Error category enumeration for classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Shape,
    Memory,
    Computation,
    Device,
    DataType,
    Io,
    Configuration,
    Threading,
    Network,
    UserInput,
    Internal,
}

/// Error severity levels for prioritization
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Shape => write!(f, "Shape"),
            ErrorCategory::Memory => write!(f, "Memory"),
            ErrorCategory::Computation => write!(f, "Computation"),
            ErrorCategory::Device => write!(f, "Device"),
            ErrorCategory::DataType => write!(f, "DataType"),
            ErrorCategory::Io => write!(f, "I/O"),
            ErrorCategory::Configuration => write!(f, "Configuration"),
            ErrorCategory::Threading => write!(f, "Threading"),
            ErrorCategory::Network => write!(f, "Network"),
            ErrorCategory::UserInput => write!(f, "UserInput"),
            ErrorCategory::Internal => write!(f, "Internal"),
        }
    }
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "Low"),
            ErrorSeverity::Medium => write!(f, "Medium"),
            ErrorSeverity::High => write!(f, "High"),
            ErrorSeverity::Critical => write!(f, "Critical"),
        }
    }
}

// Macros are defined in error.rs to avoid duplication

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_display() {
        let shape = [2, 3, 4];
        let display = format_shape(&shape);
        assert_eq!(format!("{}", display), "[2, 3, 4]");

        let empty_shape = [];
        let empty_display = format_shape(&empty_shape);
        assert_eq!(format!("{}", empty_display), "[]");
    }

    #[test]
    fn test_error_location() {
        let location = ErrorLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: 10,
            function: Some("test_function".to_string()),
        };

        assert_eq!(format!("{}", location), "test.rs:42 in test_function");

        let location_no_fn = ErrorLocation {
            file: "test.rs".to_string(),
            line: 42,
            column: 10,
            function: None,
        };

        assert_eq!(format!("{}", location_no_fn), "test.rs:42");
    }

    #[test]
    fn test_error_debug_context() {
        let mut context = ErrorDebugContext::minimal();
        context = context.with_metadata("operation", "tensor_add");

        assert!(context.metadata.contains_key("operation"));
        assert_eq!(context.metadata["operation"], "tensor_add");

        let debug_info = context.format_debug_info();
        assert!(debug_info.contains("Error ID:"));
        assert!(debug_info.contains("Thread:"));
        assert!(debug_info.contains("operation: tensor_add"));
    }

    #[test]
    fn test_thread_info() {
        let thread_info = ThreadInfo::current();
        assert!(!thread_info.thread_id.is_empty());
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(format!("{}", ErrorCategory::Shape), "Shape");
        assert_eq!(format!("{}", ErrorCategory::Memory), "Memory");
        assert_eq!(format!("{}", ErrorSeverity::Critical), "Critical");
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Low < ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium < ErrorSeverity::High);
        assert!(ErrorSeverity::High < ErrorSeverity::Critical);
    }
}
