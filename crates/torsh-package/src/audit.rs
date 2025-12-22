//! Audit logging for security compliance and tracking
//!
//! This module provides comprehensive audit logging for all package operations
//! including downloads, uploads, access control changes, and security events.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

/// Audit event type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Package downloaded
    PackageDownload,
    /// Package uploaded/published
    PackageUpload,
    /// Package deleted
    PackageDelete,
    /// Package version yanked
    PackageYank,
    /// Package version unyanked
    PackageUnyank,
    /// User authentication
    UserAuthentication,
    /// User authorization check
    UserAuthorization,
    /// Access granted
    AccessGranted,
    /// Access denied
    AccessDenied,
    /// Role assigned
    RoleAssigned,
    /// Role revoked
    RoleRevoked,
    /// Permission changed
    PermissionChanged,
    /// Security violation detected
    SecurityViolation,
    /// Package integrity check
    IntegrityCheck,
    /// Package signature verification
    SignatureVerification,
    /// Configuration change
    ConfigurationChange,
    /// System event
    SystemEvent,
}

/// Audit event severity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuditSeverity {
    /// Informational event
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical security event
    Critical,
}

/// Audit event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event ID (unique)
    pub id: String,
    /// Event type
    pub event_type: AuditEventType,
    /// Severity level
    pub severity: AuditSeverity,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// User ID who performed the action
    pub user_id: Option<String>,
    /// IP address of the client
    pub ip_address: Option<String>,
    /// User agent string
    pub user_agent: Option<String>,
    /// Action performed
    pub action: String,
    /// Resource affected (e.g., package name)
    pub resource: Option<String>,
    /// Result of the action (success/failure)
    pub result: ActionResult,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Error message if action failed
    pub error: Option<String>,
}

/// Action result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionResult {
    /// Action succeeded
    Success,
    /// Action failed
    Failure,
    /// Action was denied
    Denied,
}

/// Audit log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log file path
    pub log_path: PathBuf,
    /// Maximum log file size in bytes before rotation
    pub max_file_size: u64,
    /// Number of rotated files to keep
    pub max_files: usize,
    /// Log format
    pub format: AuditLogFormat,
    /// Minimum severity to log
    pub min_severity: AuditSeverity,
    /// Enable real-time log streaming
    pub stream_enabled: bool,
    /// Buffer size for log entries
    pub buffer_size: usize,
}

/// Audit log format
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditLogFormat {
    /// JSON format (one event per line)
    Json,
    /// CSV format
    Csv,
    /// Plain text format
    Text,
    /// Syslog format
    Syslog,
}

/// Audit logger
pub struct AuditLogger {
    /// Configuration
    config: AuditLogConfig,
    /// Event buffer
    buffer: Vec<AuditEvent>,
    /// Event listeners for real-time streaming
    listeners: Vec<Box<dyn AuditListener>>,
    /// Statistics
    statistics: AuditStatistics,
}

/// Audit listener for real-time event streaming
pub trait AuditListener: Send + Sync {
    /// Called when an event is logged
    fn on_event(&mut self, event: &AuditEvent);

    /// Called on flush
    fn on_flush(&mut self);
}

/// Audit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStatistics {
    /// Total events logged
    pub total_events: u64,
    /// Events by type
    pub events_by_type: HashMap<String, u64>,
    /// Events by severity
    pub events_by_severity: HashMap<String, u64>,
    /// Failed actions
    pub failed_actions: u64,
    /// Security violations
    pub security_violations: u64,
    /// Unique users
    pub unique_users: u64,
}

/// Audit query filter
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Filter by event type
    pub event_types: Vec<AuditEventType>,
    /// Filter by severity
    pub min_severity: Option<AuditSeverity>,
    /// Filter by user ID
    pub user_id: Option<String>,
    /// Filter by resource
    pub resource: Option<String>,
    /// Filter by result
    pub result: Option<ActionResult>,
    /// Start time for query range
    pub start_time: Option<DateTime<Utc>>,
    /// End time for query range
    pub end_time: Option<DateTime<Utc>>,
    /// Maximum number of results
    pub limit: Option<usize>,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(event_type: AuditEventType, action: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            event_type,
            severity: AuditSeverity::Info,
            timestamp: Utc::now(),
            user_id: None,
            ip_address: None,
            user_agent: None,
            action,
            resource: None,
            result: ActionResult::Success,
            metadata: HashMap::new(),
            error: None,
        }
    }

    /// Set severity
    pub fn with_severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set user ID
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Set IP address
    pub fn with_ip(mut self, ip: String) -> Self {
        self.ip_address = Some(ip);
        self
    }

    /// Set resource
    pub fn with_resource(mut self, resource: String) -> Self {
        self.resource = Some(resource);
        self
    }

    /// Set result
    pub fn with_result(mut self, result: ActionResult) -> Self {
        self.result = result;
        self
    }

    /// Set error message
    pub fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Format as JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to serialize event: {}", e)))
    }

    /// Format as text
    pub fn to_text(&self) -> String {
        format!(
            "[{}] {} - {} - {} - {} - User: {:?} - Resource: {:?} - Result: {:?}{}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S"),
            self.severity_str(),
            self.event_type_str(),
            self.action,
            self.id,
            self.user_id,
            self.resource,
            self.result,
            self.error
                .as_ref()
                .map_or(String::new(), |e| format!(" - Error: {}", e))
        )
    }

    /// Get severity as string
    fn severity_str(&self) -> &str {
        match self.severity {
            AuditSeverity::Info => "INFO",
            AuditSeverity::Warning => "WARN",
            AuditSeverity::Error => "ERROR",
            AuditSeverity::Critical => "CRIT",
        }
    }

    /// Get event type as string
    fn event_type_str(&self) -> &str {
        match self.event_type {
            AuditEventType::PackageDownload => "DOWNLOAD",
            AuditEventType::PackageUpload => "UPLOAD",
            AuditEventType::PackageDelete => "DELETE",
            AuditEventType::PackageYank => "YANK",
            AuditEventType::PackageUnyank => "UNYANK",
            AuditEventType::UserAuthentication => "AUTH",
            AuditEventType::UserAuthorization => "AUTHZ",
            AuditEventType::AccessGranted => "ACCESS_GRANTED",
            AuditEventType::AccessDenied => "ACCESS_DENIED",
            AuditEventType::RoleAssigned => "ROLE_ASSIGN",
            AuditEventType::RoleRevoked => "ROLE_REVOKE",
            AuditEventType::PermissionChanged => "PERM_CHANGE",
            AuditEventType::SecurityViolation => "SECURITY_VIOLATION",
            AuditEventType::IntegrityCheck => "INTEGRITY_CHECK",
            AuditEventType::SignatureVerification => "SIGNATURE_VERIFY",
            AuditEventType::ConfigurationChange => "CONFIG_CHANGE",
            AuditEventType::SystemEvent => "SYSTEM",
        }
    }
}

impl Default for AuditLogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_path: PathBuf::from("/var/log/torsh/audit.log"),
            max_file_size: 100 * 1024 * 1024, // 100 MB
            max_files: 10,
            format: AuditLogFormat::Json,
            min_severity: AuditSeverity::Info,
            stream_enabled: false,
            buffer_size: 1000,
        }
    }
}

impl AuditLogConfig {
    /// Create new configuration
    pub fn new<P: AsRef<Path>>(log_path: P) -> Self {
        Self {
            log_path: log_path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_file_size == 0 {
            return Err(TorshError::InvalidArgument(
                "Max file size must be greater than zero".to_string(),
            ));
        }

        if self.max_files == 0 {
            return Err(TorshError::InvalidArgument(
                "Max files must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for AuditStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditStatistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            total_events: 0,
            events_by_type: HashMap::new(),
            events_by_severity: HashMap::new(),
            failed_actions: 0,
            security_violations: 0,
            unique_users: 0,
        }
    }

    /// Update statistics with an event
    pub fn update(&mut self, event: &AuditEvent) {
        self.total_events += 1;

        // Count by type
        let type_key = format!("{:?}", event.event_type);
        *self.events_by_type.entry(type_key).or_insert(0) += 1;

        // Count by severity
        let severity_key = format!("{:?}", event.severity);
        *self.events_by_severity.entry(severity_key).or_insert(0) += 1;

        // Count failures
        if event.result == ActionResult::Failure {
            self.failed_actions += 1;
        }

        // Count security violations
        if event.event_type == AuditEventType::SecurityViolation {
            self.security_violations += 1;
        }
    }
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditLogConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            buffer: Vec::new(),
            listeners: Vec::new(),
            statistics: AuditStatistics::new(),
        })
    }

    /// Log an event
    pub fn log(&mut self, event: AuditEvent) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check severity filter
        if event.severity < self.config.min_severity {
            return Ok(());
        }

        // Update statistics
        self.statistics.update(&event);

        // Notify listeners
        for listener in &mut self.listeners {
            listener.on_event(&event);
        }

        // Add to buffer
        self.buffer.push(event);

        // Flush if buffer is full
        if self.buffer.len() >= self.config.buffer_size {
            self.flush()?;
        }

        Ok(())
    }

    /// Log package download
    pub fn log_download(&mut self, user_id: &str, package: &str, version: &str) -> Result<()> {
        let event = AuditEvent::new(
            AuditEventType::PackageDownload,
            format!("Download package {}", package),
        )
        .with_user(user_id.to_string())
        .with_resource(format!("{}:{}", package, version))
        .with_severity(AuditSeverity::Info);

        self.log(event)
    }

    /// Log package upload
    pub fn log_upload(&mut self, user_id: &str, package: &str, version: &str) -> Result<()> {
        let event = AuditEvent::new(
            AuditEventType::PackageUpload,
            format!("Upload package {}", package),
        )
        .with_user(user_id.to_string())
        .with_resource(format!("{}:{}", package, version))
        .with_severity(AuditSeverity::Info);

        self.log(event)
    }

    /// Log access denial
    pub fn log_access_denied(&mut self, user_id: &str, resource: &str, reason: &str) -> Result<()> {
        let event = AuditEvent::new(
            AuditEventType::AccessDenied,
            format!("Access denied to {}", resource),
        )
        .with_user(user_id.to_string())
        .with_resource(resource.to_string())
        .with_result(ActionResult::Denied)
        .with_severity(AuditSeverity::Warning)
        .add_metadata("reason".to_string(), reason.to_string());

        self.log(event)
    }

    /// Log security violation
    pub fn log_security_violation(
        &mut self,
        user_id: Option<&str>,
        violation: &str,
        details: &str,
    ) -> Result<()> {
        let mut event = AuditEvent::new(
            AuditEventType::SecurityViolation,
            format!("Security violation: {}", violation),
        )
        .with_severity(AuditSeverity::Critical)
        .with_result(ActionResult::Failure)
        .add_metadata("details".to_string(), details.to_string());

        if let Some(uid) = user_id {
            event = event.with_user(uid.to_string());
        }

        self.log(event)
    }

    /// Flush buffered events to disk
    pub fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // In production, would write to file
        // For now, just clear the buffer
        self.buffer.clear();

        // Notify listeners
        for listener in &mut self.listeners {
            listener.on_flush();
        }

        Ok(())
    }

    /// Add an event listener
    pub fn add_listener(&mut self, listener: Box<dyn AuditListener>) {
        self.listeners.push(listener);
    }

    /// Query events (simplified - in production would query from persistent storage)
    pub fn query(&self, _query: &AuditQuery) -> Vec<AuditEvent> {
        // In production, would query from log files or database
        // For now, return buffered events
        self.buffer.clone()
    }

    /// Get statistics
    pub fn get_statistics(&self) -> &AuditStatistics {
        &self.statistics
    }

    /// Get event count by type
    pub fn get_event_count(&self, event_type: &AuditEventType) -> u64 {
        let key = format!("{:?}", event_type);
        self.statistics
            .events_by_type
            .get(&key)
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent::new(
            AuditEventType::PackageDownload,
            "Download test-package".to_string(),
        )
        .with_user("user1".to_string())
        .with_resource("test-package:1.0.0".to_string())
        .with_severity(AuditSeverity::Info);

        assert_eq!(event.event_type, AuditEventType::PackageDownload);
        assert_eq!(event.user_id, Some("user1".to_string()));
        assert_eq!(event.result, ActionResult::Success);
    }

    #[test]
    fn test_audit_event_formatting() {
        let event = AuditEvent::new(AuditEventType::PackageDownload, "Download test".to_string());

        let json = event.to_json().unwrap();
        assert!(json.contains("PackageDownload"));

        let text = event.to_text();
        assert!(text.contains("DOWNLOAD"));
        assert!(text.contains("INFO"));
    }

    #[test]
    fn test_audit_logger() {
        let config = AuditLogConfig::new(PathBuf::from("/tmp/test-audit.log"));
        let mut logger = AuditLogger::new(config).unwrap();

        let event = AuditEvent::new(AuditEventType::PackageDownload, "Test download".to_string());

        logger.log(event).unwrap();
        assert_eq!(logger.statistics.total_events, 1);
        assert_eq!(logger.buffer.len(), 1);
    }

    #[test]
    fn test_log_download() {
        let config = AuditLogConfig::new(PathBuf::from("/tmp/test-audit.log"));
        let mut logger = AuditLogger::new(config).unwrap();

        logger
            .log_download("user1", "test-package", "1.0.0")
            .unwrap();

        assert_eq!(logger.get_event_count(&AuditEventType::PackageDownload), 1);
    }

    #[test]
    fn test_log_access_denied() {
        let config = AuditLogConfig::new(PathBuf::from("/tmp/test-audit.log"));
        let mut logger = AuditLogger::new(config).unwrap();

        logger
            .log_access_denied("user1", "test-package", "Insufficient permissions")
            .unwrap();

        assert_eq!(logger.get_event_count(&AuditEventType::AccessDenied), 1);
    }

    #[test]
    fn test_security_violation_logging() {
        let config = AuditLogConfig::new(PathBuf::from("/tmp/test-audit.log"));
        let mut logger = AuditLogger::new(config).unwrap();

        logger
            .log_security_violation(Some("user1"), "Suspicious activity", "Details here")
            .unwrap();

        assert_eq!(logger.statistics.security_violations, 1);
    }

    #[test]
    fn test_statistics_update() {
        let mut stats = AuditStatistics::new();

        let event1 = AuditEvent::new(AuditEventType::PackageDownload, "Download".to_string());
        let event2 = AuditEvent::new(AuditEventType::PackageUpload, "Upload".to_string())
            .with_result(ActionResult::Failure);

        stats.update(&event1);
        stats.update(&event2);

        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.failed_actions, 1);
    }

    #[test]
    fn test_buffer_flush() {
        let mut config = AuditLogConfig::new(PathBuf::from("/tmp/test-audit.log"));
        config.buffer_size = 2;

        let mut logger = AuditLogger::new(config).unwrap();

        logger
            .log(AuditEvent::new(
                AuditEventType::PackageDownload,
                "Test1".to_string(),
            ))
            .unwrap();
        assert_eq!(logger.buffer.len(), 1);

        logger
            .log(AuditEvent::new(
                AuditEventType::PackageDownload,
                "Test2".to_string(),
            ))
            .unwrap();

        // Buffer should be flushed after reaching buffer_size
        assert_eq!(logger.buffer.len(), 0);
    }
}
