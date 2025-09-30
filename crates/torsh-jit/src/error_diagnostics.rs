//! Error diagnostics for JIT compilation
//!
//! This module provides comprehensive error diagnostics capabilities for JIT compilation,
//! including detailed error messages, source location tracking, and recovery suggestions.

use crate::{JitError, JitResult};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Error diagnostics manager
#[derive(Debug)]
pub struct ErrorDiagnosticsManager {
    /// Diagnostic configuration
    config: DiagnosticsConfig,

    /// Error history
    error_history: Vec<DiagnosticError>,

    /// Error patterns for analysis
    error_patterns: HashMap<String, ErrorPattern>,

    /// Recovery suggestions database
    recovery_suggestions: HashMap<ErrorCategory, Vec<RecoverySuggestion>>,

    /// Context information
    context_stack: Vec<DiagnosticContext>,

    /// Statistics
    stats: DiagnosticsStats,
}

/// Diagnostic error with enhanced information
#[derive(Debug)]
pub struct DiagnosticError {
    /// Unique error ID
    pub id: String,

    /// Error timestamp
    pub timestamp: Instant,

    /// Error category
    pub category: ErrorCategory,

    /// Error severity
    pub severity: ErrorSeverity,

    /// Error message
    pub message: String,

    /// Source location
    pub source_location: Option<SourceLocation>,

    /// Stack trace
    pub stack_trace: Vec<StackFrame>,

    /// Error context
    pub context: DiagnosticContext,

    /// Related errors
    pub related_errors: Vec<String>,

    /// Recovery suggestions
    pub suggestions: Vec<RecoverySuggestion>,

    /// Error metadata
    pub metadata: HashMap<String, String>,

    /// Underlying error
    pub underlying_error: Option<Box<JitError>>,
}

/// Error categories for classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Graph construction errors
    GraphConstruction,

    /// Type inference errors
    TypeInference,

    /// Shape inference errors
    ShapeInference,

    /// Optimization errors
    Optimization,

    /// Code generation errors
    CodeGeneration,

    /// Runtime errors
    Runtime,

    /// Memory errors
    Memory,

    /// Resource errors
    Resource,

    /// User input errors
    UserInput,

    /// Internal compiler errors
    Internal,

    /// External dependency errors
    External,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    /// Information messages
    Info,

    /// Warning messages
    Warning,

    /// Error messages
    Error,

    /// Fatal error messages
    Fatal,

    /// Internal compiler error
    Ice, // Internal Compiler Error
}

/// Source location for diagnostics
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File path
    pub file: String,

    /// Line number (1-based)
    pub line: u32,

    /// Column number (1-based)
    pub column: u32,

    /// Length of the error span
    pub length: Option<u32>,

    /// Source code snippet
    pub snippet: Option<String>,
}

/// Stack frame for error traces
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Function name
    pub function: String,

    /// File path
    pub file: Option<String>,

    /// Line number
    pub line: Option<u32>,

    /// Address
    pub address: Option<u64>,

    /// Module name
    pub module: Option<String>,
}

/// Diagnostic context
#[derive(Debug, Clone)]
pub struct DiagnosticContext {
    /// Operation being performed
    pub operation: String,

    /// Input description
    pub input: String,

    /// Expected result
    pub expected: Option<String>,

    /// Actual result
    pub actual: Option<String>,

    /// Environment information
    pub environment: EnvironmentInfo,

    /// Additional context data
    pub data: HashMap<String, String>,
}

/// Environment information
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    /// Rust version
    pub rust_version: String,

    /// ToRSh version
    pub torsh_version: String,

    /// Target architecture
    pub target_arch: String,

    /// Operating system
    pub target_os: String,

    /// Available memory
    pub available_memory: Option<u64>,

    /// CPU information
    pub cpu_info: Option<String>,

    /// GPU information
    pub gpu_info: Option<String>,
}

/// Error pattern for recognition
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Pattern name
    pub name: String,

    /// Pattern description
    pub description: String,

    /// Matching criteria
    pub criteria: Vec<MatchCriterion>,

    /// Common causes
    pub common_causes: Vec<String>,

    /// Suggested solutions
    pub solutions: Vec<RecoverySuggestion>,

    /// Frequency of occurrence
    pub frequency: u64,
}

/// Criteria for matching error patterns
#[derive(Debug, Clone)]
pub enum MatchCriterion {
    /// Message contains text
    MessageContains(String),

    /// Error category matches
    CategoryEquals(ErrorCategory),

    /// Source location matches pattern
    LocationMatches(String),

    /// Stack trace contains function
    StackContains(String),

    /// Custom matcher
    Custom(fn(&DiagnosticError) -> bool),
}

/// Recovery suggestion
#[derive(Debug, Clone)]
pub struct RecoverySuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,

    /// Suggestion message
    pub message: String,

    /// Detailed explanation
    pub explanation: Option<String>,

    /// Code example (if applicable)
    pub code_example: Option<String>,

    /// Link to documentation
    pub doc_link: Option<String>,

    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,

    /// Automatic fix available
    pub auto_fix: Option<AutoFix>,
}

/// Types of suggestions
#[derive(Debug, Clone)]
pub enum SuggestionType {
    /// Quick fix
    QuickFix,

    /// Code change
    CodeChange,

    /// Configuration change
    ConfigChange,

    /// Environment setup
    EnvironmentSetup,

    /// Documentation reference
    Documentation,

    /// Workaround
    Workaround,

    /// Investigation required
    Investigation,
}

/// Automatic fix information
#[derive(Debug, Clone)]
pub struct AutoFix {
    /// Fix description
    pub description: String,

    /// Fix function
    pub fix_fn: fn(&DiagnosticError) -> JitResult<()>,

    /// Side effects
    pub side_effects: Vec<String>,

    /// Requires user confirmation
    pub requires_confirmation: bool,
}

/// Diagnostics configuration
#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    /// Enable error diagnostics
    pub enabled: bool,

    /// Maximum error history size
    pub max_history_size: usize,

    /// Enable stack trace collection
    pub collect_stack_traces: bool,

    /// Enable source snippet extraction
    pub extract_source_snippets: bool,

    /// Error reporting level
    pub reporting_level: ErrorSeverity,

    /// Enable pattern matching
    pub enable_pattern_matching: bool,

    /// Enable recovery suggestions
    pub enable_suggestions: bool,

    /// Maximum suggestions per error
    pub max_suggestions: usize,

    /// Color output for terminal
    pub color_output: bool,

    /// Verbose output
    pub verbose: bool,
}

/// Diagnostics statistics
#[derive(Debug, Clone, Default)]
pub struct DiagnosticsStats {
    /// Total errors recorded
    pub total_errors: u64,

    /// Errors by category
    pub errors_by_category: HashMap<ErrorCategory, u64>,

    /// Errors by severity
    pub errors_by_severity: HashMap<ErrorSeverity, u64>,

    /// Pattern matches
    pub pattern_matches: u64,

    /// Suggestions provided
    pub suggestions_provided: u64,

    /// Auto-fixes applied
    pub auto_fixes_applied: u64,
}

/// Error formatter for different output formats
pub struct ErrorFormatter {
    /// Formatting configuration
    config: FormatterConfig,
}

/// Formatter configuration
#[derive(Debug, Clone)]
pub struct FormatterConfig {
    /// Include source snippets
    pub include_source: bool,

    /// Include stack traces
    pub include_stack_trace: bool,

    /// Include suggestions
    pub include_suggestions: bool,

    /// Use colors
    pub use_colors: bool,

    /// Maximum line length
    pub max_line_length: usize,

    /// Indentation size
    pub indent_size: usize,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_history_size: 1000,
            collect_stack_traces: true,
            extract_source_snippets: true,
            reporting_level: ErrorSeverity::Warning,
            enable_pattern_matching: true,
            enable_suggestions: true,
            max_suggestions: 5,
            color_output: true,
            verbose: false,
        }
    }
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            include_source: true,
            include_stack_trace: true,
            include_suggestions: true,
            use_colors: true,
            max_line_length: 120,
            indent_size: 2,
        }
    }
}

impl ErrorDiagnosticsManager {
    /// Create a new error diagnostics manager
    pub fn new(config: DiagnosticsConfig) -> Self {
        let mut manager = Self {
            config,
            error_history: Vec::new(),
            error_patterns: HashMap::new(),
            recovery_suggestions: HashMap::new(),
            context_stack: Vec::new(),
            stats: DiagnosticsStats::default(),
        };

        manager.initialize_default_patterns();
        manager.initialize_default_suggestions();
        manager
    }

    /// Create a new manager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(DiagnosticsConfig::default())
    }

    /// Record an error with diagnostics
    pub fn record_error(&mut self, error: JitError) -> DiagnosticError {
        let mut diagnostic_error = self.create_diagnostic_error(error);

        // Update statistics
        self.stats.total_errors += 1;
        *self
            .stats
            .errors_by_category
            .entry(diagnostic_error.category.clone())
            .or_insert(0) += 1;
        *self
            .stats
            .errors_by_severity
            .entry(diagnostic_error.severity.clone())
            .or_insert(0) += 1;

        // Try to match error patterns
        if self.config.enable_pattern_matching {
            self.match_error_patterns(&diagnostic_error);
        }

        // Add recovery suggestions
        if self.config.enable_suggestions {
            self.add_recovery_suggestions(&mut diagnostic_error);
        }

        // Store in history (clone for history, but we need to return the original)
        let diagnostic_error_copy = DiagnosticError {
            id: diagnostic_error.id.clone(),
            timestamp: diagnostic_error.timestamp,
            category: diagnostic_error.category.clone(),
            severity: diagnostic_error.severity.clone(),
            message: diagnostic_error.message.clone(),
            source_location: diagnostic_error.source_location.clone(),
            stack_trace: diagnostic_error.stack_trace.clone(),
            context: diagnostic_error.context.clone(),
            related_errors: diagnostic_error.related_errors.clone(),
            suggestions: diagnostic_error.suggestions.clone(),
            metadata: diagnostic_error.metadata.clone(),
            underlying_error: None, // Don't store the underlying error in history to avoid clone issues
        };

        if self.error_history.len() >= self.config.max_history_size {
            self.error_history.remove(0);
        }
        self.error_history.push(diagnostic_error_copy);

        diagnostic_error
    }

    /// Create a diagnostic error from a JIT error
    fn create_diagnostic_error(&self, error: JitError) -> DiagnosticError {
        let error_id = format!("err_{}", self.stats.total_errors);
        let category = self.categorize_error(&error);
        let severity = self.determine_severity(&error);
        let message = error.to_string();

        let context = self
            .context_stack
            .last()
            .cloned()
            .unwrap_or_else(|| DiagnosticContext {
                operation: "unknown".to_string(),
                input: "unknown".to_string(),
                expected: None,
                actual: None,
                environment: self.get_environment_info(),
                data: HashMap::new(),
            });

        DiagnosticError {
            id: error_id,
            timestamp: Instant::now(),
            category,
            severity,
            message,
            source_location: None,
            stack_trace: self.collect_stack_trace(),
            context,
            related_errors: Vec::new(),
            suggestions: Vec::new(),
            metadata: HashMap::new(),
            underlying_error: Some(Box::new(error)),
        }
    }

    /// Categorize an error
    fn categorize_error(&self, error: &JitError) -> ErrorCategory {
        match error {
            JitError::GraphError(_) => ErrorCategory::GraphConstruction,
            JitError::OptimizationError(_) => ErrorCategory::Optimization,
            JitError::CodeGenError(_) => ErrorCategory::CodeGeneration,
            JitError::RuntimeError(_) => ErrorCategory::Runtime,
            JitError::UnsupportedOp(_) => ErrorCategory::UserInput,
            JitError::CompilationError(_) => ErrorCategory::CodeGeneration,
            JitError::AnalysisError(_) => ErrorCategory::TypeInference,
            JitError::BackendError(_) => ErrorCategory::External,
            JitError::FusionError(_) => ErrorCategory::Optimization,
            JitError::AbstractInterpretationError(_) => ErrorCategory::TypeInference,
        }
    }

    /// Determine error severity
    fn determine_severity(&self, error: &JitError) -> ErrorSeverity {
        match error {
            JitError::GraphError(_) => ErrorSeverity::Error,
            JitError::OptimizationError(_) => ErrorSeverity::Warning,
            JitError::CodeGenError(_) => ErrorSeverity::Error,
            JitError::RuntimeError(_) => ErrorSeverity::Error,
            JitError::UnsupportedOp(_) => ErrorSeverity::Error,
            JitError::CompilationError(_) => ErrorSeverity::Error,
            JitError::AnalysisError(_) => ErrorSeverity::Warning,
            JitError::BackendError(_) => ErrorSeverity::Fatal,
            JitError::FusionError(_) => ErrorSeverity::Warning,
            JitError::AbstractInterpretationError(_) => ErrorSeverity::Warning,
        }
    }

    /// Collect stack trace
    fn collect_stack_trace(&self) -> Vec<StackFrame> {
        if !self.config.collect_stack_traces {
            return Vec::new();
        }

        // TODO: Implement actual stack trace collection
        vec![StackFrame {
            function: "jit_compile".to_string(),
            file: Some("lib.rs".to_string()),
            line: Some(100),
            address: None,
            module: Some("torsh_jit".to_string()),
        }]
    }

    /// Get environment information
    fn get_environment_info(&self) -> EnvironmentInfo {
        EnvironmentInfo {
            rust_version: "1.75.0".to_string(), // TODO: Get actual version
            torsh_version: "0.1.0-alpha.1".to_string(),
            target_arch: std::env::consts::ARCH.to_string(),
            target_os: std::env::consts::OS.to_string(),
            available_memory: None, // TODO: Get actual memory info
            cpu_info: None,         // TODO: Get CPU info
            gpu_info: None,         // TODO: Get GPU info
        }
    }

    /// Match error patterns
    fn match_error_patterns(&mut self, error: &DiagnosticError) {
        for (_pattern_name, pattern) in &self.error_patterns {
            if self.matches_pattern(error, pattern) {
                self.stats.pattern_matches += 1;
                // TODO: Apply pattern-specific handling
            }
        }
    }

    /// Check if error matches a pattern
    fn matches_pattern(&self, error: &DiagnosticError, pattern: &ErrorPattern) -> bool {
        for criterion in &pattern.criteria {
            match criterion {
                MatchCriterion::MessageContains(text) => {
                    if !error.message.contains(text) {
                        return false;
                    }
                }
                MatchCriterion::CategoryEquals(category) => {
                    if error.category != *category {
                        return false;
                    }
                }
                MatchCriterion::LocationMatches(_pattern) => {
                    // TODO: Implement location pattern matching
                }
                MatchCriterion::StackContains(function) => {
                    if !error
                        .stack_trace
                        .iter()
                        .any(|frame| frame.function.contains(function))
                    {
                        return false;
                    }
                }
                MatchCriterion::Custom(matcher) => {
                    if !matcher(error) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Add recovery suggestions to an error
    fn add_recovery_suggestions(&mut self, error: &mut DiagnosticError) {
        if let Some(suggestions) = self.recovery_suggestions.get(&error.category) {
            for suggestion in suggestions.iter().take(self.config.max_suggestions) {
                error.suggestions.push(suggestion.clone());
                self.stats.suggestions_provided += 1;
            }
        }
    }

    /// Initialize default error patterns
    fn initialize_default_patterns(&mut self) {
        // Type mismatch pattern
        let type_mismatch = ErrorPattern {
            name: "type_mismatch".to_string(),
            description: "Type mismatch in operation".to_string(),
            criteria: vec![
                MatchCriterion::MessageContains("type".to_string()),
                MatchCriterion::CategoryEquals(ErrorCategory::TypeInference),
            ],
            common_causes: vec![
                "Incorrect input types".to_string(),
                "Missing type annotations".to_string(),
            ],
            solutions: vec![],
            frequency: 0,
        };

        self.error_patterns
            .insert("type_mismatch".to_string(), type_mismatch);

        // Shape mismatch pattern
        let shape_mismatch = ErrorPattern {
            name: "shape_mismatch".to_string(),
            description: "Shape mismatch in tensor operation".to_string(),
            criteria: vec![
                MatchCriterion::MessageContains("shape".to_string()),
                MatchCriterion::CategoryEquals(ErrorCategory::ShapeInference),
            ],
            common_causes: vec![
                "Incompatible tensor shapes".to_string(),
                "Missing shape information".to_string(),
            ],
            solutions: vec![],
            frequency: 0,
        };

        self.error_patterns
            .insert("shape_mismatch".to_string(), shape_mismatch);
    }

    /// Initialize default recovery suggestions
    fn initialize_default_suggestions(&mut self) {
        // Type inference suggestions
        let type_suggestions = vec![RecoverySuggestion {
            suggestion_type: SuggestionType::CodeChange,
            message: "Check input types and add explicit type annotations".to_string(),
            explanation: Some(
                "Type inference failed. Consider adding explicit type information.".to_string(),
            ),
            code_example: Some("tensor.cast(DType::F32)".to_string()),
            doc_link: Some("https://docs.rs/torsh/latest/torsh/".to_string()),
            confidence: 0.8,
            auto_fix: None,
        }];

        self.recovery_suggestions
            .insert(ErrorCategory::TypeInference, type_suggestions);

        // Shape inference suggestions
        let shape_suggestions = vec![
            RecoverySuggestion {
                suggestion_type: SuggestionType::CodeChange,
                message: "Verify tensor shapes are compatible for the operation".to_string(),
                explanation: Some("Shape inference failed. Check that tensor dimensions match operation requirements.".to_string()),
                code_example: Some("tensor.reshape(&[batch_size, channels, height, width])".to_string()),
                doc_link: Some("https://docs.rs/torsh/latest/torsh/".to_string()),
                confidence: 0.9,
                auto_fix: None,
            },
        ];

        self.recovery_suggestions
            .insert(ErrorCategory::ShapeInference, shape_suggestions);
    }

    /// Push diagnostic context
    pub fn push_context(&mut self, context: DiagnosticContext) {
        self.context_stack.push(context);
    }

    /// Pop diagnostic context
    pub fn pop_context(&mut self) -> Option<DiagnosticContext> {
        self.context_stack.pop()
    }

    /// Get error history
    pub fn get_error_history(&self) -> &[DiagnosticError] {
        &self.error_history
    }

    /// Get statistics
    pub fn get_stats(&self) -> &DiagnosticsStats {
        &self.stats
    }

    /// Format error for display
    pub fn format_error(&self, error: &DiagnosticError, format_config: &FormatterConfig) -> String {
        let formatter = ErrorFormatter::new(format_config.clone());
        formatter.format(error)
    }

    /// Get similar errors from history
    pub fn get_similar_errors(&self, error: &DiagnosticError) -> Vec<&DiagnosticError> {
        self.error_history
            .iter()
            .filter(|e| e.category == error.category && e.severity == error.severity)
            .collect()
    }

    /// Export diagnostics data
    pub fn export_diagnostics(&self, output_path: &str) -> JitResult<()> {
        let diagnostics_data = format!(
            r#"{{"total_errors": {}, "errors_by_category": {:?}, "patterns": {}}}"#,
            self.stats.total_errors,
            self.stats.errors_by_category,
            self.error_patterns.len()
        );

        std::fs::write(output_path, diagnostics_data)
            .map_err(|e| JitError::RuntimeError(format!("Failed to export diagnostics: {}", e)))?;

        Ok(())
    }
}

impl ErrorFormatter {
    /// Create a new error formatter
    pub fn new(config: FormatterConfig) -> Self {
        Self { config }
    }

    /// Format a diagnostic error
    pub fn format(&self, error: &DiagnosticError) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!(
            "{}[{}] {}: {}\n",
            self.color_for_severity(&error.severity),
            error.severity.as_str(),
            error.category.as_str(),
            error.message
        ));

        // Source location
        if let Some(location) = &error.source_location {
            output.push_str(&format!(
                "  --> {}:{}:{}\n",
                location.file, location.line, location.column
            ));

            if self.config.include_source {
                if let Some(snippet) = &location.snippet {
                    output.push_str(&format!("   |\n   | {}\n   |\n", snippet));
                }
            }
        }

        // Context
        output.push_str(&format!(
            "  Context: {} ({})\n",
            error.context.operation, error.context.input
        ));

        // Stack trace
        if self.config.include_stack_trace && !error.stack_trace.is_empty() {
            output.push_str("  Stack trace:\n");
            for frame in &error.stack_trace {
                output.push_str(&format!(
                    "    at {} ({}:{})\n",
                    frame.function,
                    frame.file.as_ref().unwrap_or(&"unknown".to_string()),
                    frame.line.unwrap_or(0)
                ));
            }
        }

        // Suggestions
        if self.config.include_suggestions && !error.suggestions.is_empty() {
            output.push_str("  Suggestions:\n");
            for suggestion in &error.suggestions {
                output.push_str(&format!("    - {}\n", suggestion.message));
                if let Some(explanation) = &suggestion.explanation {
                    output.push_str(&format!("      {}\n", explanation));
                }
            }
        }

        output.push_str(&self.reset_color());
        output
    }

    /// Get color for severity level
    fn color_for_severity(&self, severity: &ErrorSeverity) -> &str {
        if !self.config.use_colors {
            return "";
        }

        match severity {
            ErrorSeverity::Info => "\x1b[36m",    // Cyan
            ErrorSeverity::Warning => "\x1b[33m", // Yellow
            ErrorSeverity::Error => "\x1b[31m",   // Red
            ErrorSeverity::Fatal => "\x1b[35m",   // Magenta
            ErrorSeverity::Ice => "\x1b[41m",     // Red background
        }
    }

    /// Reset color
    fn reset_color(&self) -> &str {
        if self.config.use_colors {
            "\x1b[0m"
        } else {
            ""
        }
    }
}

impl ErrorSeverity {
    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            ErrorSeverity::Info => "INFO",
            ErrorSeverity::Warning => "WARNING",
            ErrorSeverity::Error => "ERROR",
            ErrorSeverity::Fatal => "FATAL",
            ErrorSeverity::Ice => "ICE",
        }
    }
}

impl ErrorCategory {
    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            ErrorCategory::GraphConstruction => "GRAPH",
            ErrorCategory::TypeInference => "TYPE",
            ErrorCategory::ShapeInference => "SHAPE",
            ErrorCategory::Optimization => "OPT",
            ErrorCategory::CodeGeneration => "CODEGEN",
            ErrorCategory::Runtime => "RUNTIME",
            ErrorCategory::Memory => "MEMORY",
            ErrorCategory::Resource => "RESOURCE",
            ErrorCategory::UserInput => "INPUT",
            ErrorCategory::Internal => "INTERNAL",
            ErrorCategory::External => "EXTERNAL",
        }
    }
}

impl fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {}",
            self.severity.as_str(),
            self.category.as_str(),
            self.message
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_manager_creation() {
        let manager = ErrorDiagnosticsManager::with_defaults();
        assert!(manager.config.enabled);
        assert_eq!(manager.config.max_history_size, 1000);
    }

    #[test]
    fn test_error_recording() {
        let mut manager = ErrorDiagnosticsManager::with_defaults();
        let error = JitError::RuntimeError("Test error".to_string());

        let diagnostic_error = manager.record_error(error);
        assert_eq!(diagnostic_error.category, ErrorCategory::Runtime);
        assert_eq!(diagnostic_error.severity, ErrorSeverity::Error);
        assert!(diagnostic_error.message.contains("Test error"));
    }

    #[test]
    fn test_error_categorization() {
        let manager = ErrorDiagnosticsManager::with_defaults();

        let graph_error = JitError::GraphError("Graph error".to_string());
        assert_eq!(
            manager.categorize_error(&graph_error),
            ErrorCategory::GraphConstruction
        );

        let runtime_error = JitError::RuntimeError("Runtime error".to_string());
        assert_eq!(
            manager.categorize_error(&runtime_error),
            ErrorCategory::Runtime
        );
    }

    #[test]
    fn test_error_formatting() {
        let mut manager = ErrorDiagnosticsManager::with_defaults();
        let error = JitError::RuntimeError("Test error".to_string());
        let diagnostic_error = manager.record_error(error);

        let formatter_config = FormatterConfig::default();
        let formatted = manager.format_error(&diagnostic_error, &formatter_config);

        assert!(formatted.contains("ERROR"));
        assert!(formatted.contains("RUNTIME"));
        assert!(formatted.contains("Test error"));
    }

    #[test]
    fn test_context_stack() {
        let mut manager = ErrorDiagnosticsManager::with_defaults();

        let context = DiagnosticContext {
            operation: "test_operation".to_string(),
            input: "test_input".to_string(),
            expected: None,
            actual: None,
            environment: manager.get_environment_info(),
            data: HashMap::new(),
        };

        manager.push_context(context.clone());
        assert_eq!(manager.context_stack.len(), 1);

        let popped = manager.pop_context().unwrap();
        assert_eq!(popped.operation, "test_operation");
        assert_eq!(manager.context_stack.len(), 0);
    }
}
