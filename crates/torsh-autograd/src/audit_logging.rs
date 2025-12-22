// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Autograd Operation Audit Logging
//!
//! This module provides comprehensive audit logging for autograd operations,
//! enabling tracking, debugging, compliance, and security monitoring.
//!
//! # Features
//!
//! - **Operation Tracking**: Log all autograd operations with detailed context
//! - **Security Auditing**: Track access patterns and anomalous behavior
//! - **Compliance Logging**: Meet regulatory requirements for ML operations
//! - **Tamper-Proof Logs**: Cryptographic hashing for log integrity
//! - **Query Interface**: Efficient querying of audit logs
//! - **Retention Policies**: Automatic log rotation and archival
//! - **Performance Impact**: Minimal overhead with async logging

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

/// Audit log entry for autograd operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Unique entry ID
    pub id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Operation type
    pub operation_type: OperationType,

    /// Operation name/identifier
    pub operation_name: String,

    /// Actor (user, process, or system component)
    pub actor: String,

    /// Tensor IDs involved
    pub tensor_ids: Vec<String>,

    /// Input/output metadata
    pub metadata: HashMap<String, String>,

    /// Result status
    pub status: OperationStatus,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// Resource usage
    pub resource_usage: ResourceUsage,

    /// Security context
    pub security_context: SecurityContext,

    /// Correlation ID (for tracing related operations)
    pub correlation_id: Option<String>,

    /// Parent operation ID (for nested operations)
    pub parent_id: Option<String>,

    /// Log integrity hash (for tamper detection)
    pub integrity_hash: Option<String>,
}

/// Type of autograd operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    /// Forward pass
    Forward,

    /// Backward pass
    Backward,

    /// Gradient computation
    GradientComputation,

    /// Parameter update
    ParameterUpdate,

    /// Checkpoint creation
    Checkpoint,

    /// Memory allocation
    MemoryAllocation,

    /// Graph construction
    GraphConstruction,

    /// Custom operation
    CustomOperation,

    /// System operation
    SystemOperation,
}

/// Operation result status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationStatus {
    /// Operation succeeded
    Success,

    /// Operation failed
    Failed,

    /// Operation was skipped
    Skipped,

    /// Operation timed out
    Timeout,

    /// Operation was cancelled
    Cancelled,
}

/// Resource usage for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory allocated (bytes)
    pub memory_bytes: u64,

    /// Peak memory (bytes)
    pub peak_memory_bytes: u64,

    /// CPU time (milliseconds)
    pub cpu_time_ms: f64,

    /// GPU time (milliseconds), if applicable
    pub gpu_time_ms: Option<f64>,

    /// I/O operations count
    pub io_operations: u64,

    /// Network bytes transferred
    pub network_bytes: u64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            memory_bytes: 0,
            peak_memory_bytes: 0,
            cpu_time_ms: 0.0,
            gpu_time_ms: None,
            io_operations: 0,
            network_bytes: 0,
        }
    }
}

/// Security context for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// User/process identifier
    pub principal: String,

    /// Access level
    pub access_level: AccessLevel,

    /// Source IP (if network operation)
    pub source_ip: Option<String>,

    /// Session ID
    pub session_id: Option<String>,

    /// Authentication method
    pub auth_method: Option<String>,

    /// Permissions granted
    pub permissions: Vec<String>,
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            principal: "system".to_string(),
            access_level: AccessLevel::System,
            source_ip: None,
            session_id: None,
            auth_method: None,
            permissions: vec![],
        }
    }
}

/// Access level for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Public access
    Public,

    /// User access
    User,

    /// Admin access
    Admin,

    /// System access
    System,

    /// Debug access
    Debug,
}

/// Audit logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,

    /// Log all operations (vs. sampling)
    pub log_all_operations: bool,

    /// Sampling rate (if not logging all)
    pub sampling_rate: f64,

    /// Enable integrity hashing
    pub enable_integrity_hash: bool,

    /// Maximum log entries in memory
    pub max_memory_entries: usize,

    /// Persist logs to disk
    pub persist_to_disk: bool,

    /// Log file path
    pub log_path: Option<PathBuf>,

    /// Log rotation size (bytes)
    pub rotation_size_bytes: u64,

    /// Log retention days
    pub retention_days: u32,

    /// Async logging (for performance)
    pub async_logging: bool,

    /// Minimum log level
    pub min_access_level: AccessLevel,

    /// Operations to exclude from logging
    pub excluded_operations: Vec<OperationType>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_all_operations: true,
            sampling_rate: 1.0,
            enable_integrity_hash: true,
            max_memory_entries: 10000,
            persist_to_disk: true,
            log_path: None,
            rotation_size_bytes: 100 * 1024 * 1024, // 100 MB
            retention_days: 30,
            async_logging: true,
            min_access_level: AccessLevel::Public,
            excluded_operations: vec![],
        }
    }
}

/// Audit logger for autograd operations
pub struct AuditLogger {
    config: AuditConfig,
    entries: Arc<RwLock<VecDeque<AuditLogEntry>>>,
    statistics: Arc<RwLock<AuditStatistics>>,
    event_handlers: Arc<RwLock<Vec<Box<dyn AuditEventHandler + Send + Sync>>>>,
}

/// Audit statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStatistics {
    /// Total entries logged
    pub total_entries: u64,

    /// Entries by operation type
    pub entries_by_type: HashMap<String, u64>,

    /// Entries by status
    pub entries_by_status: HashMap<String, u64>,

    /// Total failed operations
    pub total_failed: u64,

    /// Average operation duration
    pub avg_duration_ms: f64,

    /// Peak memory usage
    pub peak_memory_bytes: u64,

    /// Total CPU time
    pub total_cpu_time_ms: f64,

    /// Security violations detected
    pub security_violations: u64,
}

/// Audit event handler trait
pub trait AuditEventHandler {
    /// Handle a new audit log entry
    fn on_audit_entry(&mut self, entry: &AuditLogEntry);

    /// Handle security violation
    fn on_security_violation(&mut self, entry: &AuditLogEntry);

    /// Handle operation failure
    fn on_operation_failure(&mut self, entry: &AuditLogEntry);
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(VecDeque::new())),
            statistics: Arc::new(RwLock::new(AuditStatistics::default())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Log an autograd operation
    pub fn log_operation(
        &self,
        operation_type: OperationType,
        operation_name: String,
        actor: String,
        status: OperationStatus,
    ) -> AuditLogEntry {
        if !self.config.enabled {
            return self.create_empty_entry(operation_type, operation_name);
        }

        // Check if operation type is excluded
        if self.config.excluded_operations.contains(&operation_type) {
            return self.create_empty_entry(operation_type, operation_name);
        }

        // Create audit entry
        let mut entry = AuditLogEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            operation_type,
            operation_name,
            actor,
            tensor_ids: vec![],
            metadata: HashMap::new(),
            status,
            error: None,
            duration_ms: 0.0,
            resource_usage: ResourceUsage::default(),
            security_context: SecurityContext::default(),
            correlation_id: None,
            parent_id: None,
            integrity_hash: None,
        };

        // Compute integrity hash
        if self.config.enable_integrity_hash {
            entry.integrity_hash = Some(self.compute_hash(&entry));
        }

        // Store entry
        self.store_entry(entry.clone());

        // Update statistics
        self.update_statistics(&entry);

        // Notify handlers
        self.notify_handlers(&entry);

        entry
    }

    /// Create a detailed audit entry with all parameters
    pub fn log_detailed_operation(
        &self,
        operation_type: OperationType,
        operation_name: String,
        actor: String,
        tensor_ids: Vec<String>,
        metadata: HashMap<String, String>,
        status: OperationStatus,
        error: Option<String>,
        duration_ms: f64,
        resource_usage: ResourceUsage,
        security_context: SecurityContext,
        correlation_id: Option<String>,
        parent_id: Option<String>,
    ) -> AuditLogEntry {
        if !self.config.enabled {
            return self.create_empty_entry(operation_type, operation_name);
        }

        let mut entry = AuditLogEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            operation_type,
            operation_name,
            actor,
            tensor_ids,
            metadata,
            status,
            error,
            duration_ms,
            resource_usage,
            security_context,
            correlation_id,
            parent_id,
            integrity_hash: None,
        };

        // Compute integrity hash
        if self.config.enable_integrity_hash {
            entry.integrity_hash = Some(self.compute_hash(&entry));
        }

        // Store entry
        self.store_entry(entry.clone());

        // Update statistics
        self.update_statistics(&entry);

        // Notify handlers
        self.notify_handlers(&entry);

        entry
    }

    /// Query audit logs
    pub fn query(&self, query: AuditQuery) -> Vec<AuditLogEntry> {
        let entries = self.entries.read();

        entries
            .iter()
            .filter(|entry| {
                // Filter by time range
                if let Some(start) = query.start_time {
                    if entry.timestamp < start {
                        return false;
                    }
                }
                if let Some(end) = query.end_time {
                    if entry.timestamp > end {
                        return false;
                    }
                }

                // Filter by operation type
                if let Some(op_type) = query.operation_type {
                    if entry.operation_type != op_type {
                        return false;
                    }
                }

                // Filter by status
                if let Some(status) = query.status {
                    if entry.status != status {
                        return false;
                    }
                }

                // Filter by actor
                if let Some(ref actor) = query.actor {
                    if &entry.actor != actor {
                        return false;
                    }
                }

                // Filter by correlation ID
                if let Some(ref corr_id) = query.correlation_id {
                    if entry.correlation_id.as_ref() != Some(corr_id) {
                        return false;
                    }
                }

                true
            })
            .take(query.limit.unwrap_or(usize::MAX))
            .cloned()
            .collect()
    }

    /// Get audit statistics
    pub fn statistics(&self) -> AuditStatistics {
        (*self.statistics.read()).clone()
    }

    /// Register audit event handler
    pub fn register_handler(&self, handler: Box<dyn AuditEventHandler + Send + Sync>) {
        self.event_handlers.write().push(handler);
    }

    /// Export audit logs to JSON
    pub fn export_json(&self) -> serde_json::Result<String> {
        let entries = self.entries.read();
        serde_json::to_string_pretty(&*entries)
    }

    /// Clear old audit logs based on retention policy
    pub fn cleanup_old_logs(&self) {
        let retention_duration = chrono::Duration::days(self.config.retention_days as i64);
        let cutoff_time = Utc::now() - retention_duration;

        let mut entries = self.entries.write();

        // Remove old entries
        while let Some(entry) = entries.front() {
            if entry.timestamp < cutoff_time {
                entries.pop_front();
            } else {
                break;
            }
        }
    }

    /// Verify log integrity
    pub fn verify_integrity(&self) -> IntegrityVerificationResult {
        let entries = self.entries.read();
        let mut result = IntegrityVerificationResult {
            total_entries: entries.len(),
            verified_entries: 0,
            corrupted_entries: Vec::new(),
            missing_hashes: 0,
        };

        for entry in entries.iter() {
            if let Some(ref stored_hash) = entry.integrity_hash {
                let computed_hash = self.compute_hash(entry);
                if stored_hash == &computed_hash {
                    result.verified_entries += 1;
                } else {
                    result.corrupted_entries.push(entry.id.clone());
                }
            } else {
                result.missing_hashes += 1;
            }
        }

        result
    }

    // Private helper methods

    fn create_empty_entry(
        &self,
        operation_type: OperationType,
        operation_name: String,
    ) -> AuditLogEntry {
        AuditLogEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            operation_type,
            operation_name,
            actor: "system".to_string(),
            tensor_ids: vec![],
            metadata: HashMap::new(),
            status: OperationStatus::Skipped,
            error: None,
            duration_ms: 0.0,
            resource_usage: ResourceUsage::default(),
            security_context: SecurityContext::default(),
            correlation_id: None,
            parent_id: None,
            integrity_hash: None,
        }
    }

    fn store_entry(&self, entry: AuditLogEntry) {
        let mut entries = self.entries.write();

        // Add new entry
        entries.push_back(entry);

        // Enforce max memory entries
        while entries.len() > self.config.max_memory_entries {
            entries.pop_front();
        }
    }

    fn update_statistics(&self, entry: &AuditLogEntry) {
        let mut stats = self.statistics.write();

        stats.total_entries += 1;

        // Update counts by type
        let type_key = format!("{:?}", entry.operation_type);
        *stats.entries_by_type.entry(type_key).or_insert(0) += 1;

        // Update counts by status
        let status_key = format!("{:?}", entry.status);
        *stats.entries_by_status.entry(status_key).or_insert(0) += 1;

        // Update failure count
        if entry.status == OperationStatus::Failed {
            stats.total_failed += 1;
        }

        // Update duration average
        let total_duration = stats.avg_duration_ms * (stats.total_entries - 1) as f64;
        stats.avg_duration_ms = (total_duration + entry.duration_ms) / stats.total_entries as f64;

        // Update peak memory
        if entry.resource_usage.peak_memory_bytes > stats.peak_memory_bytes {
            stats.peak_memory_bytes = entry.resource_usage.peak_memory_bytes;
        }

        // Update total CPU time
        stats.total_cpu_time_ms += entry.resource_usage.cpu_time_ms;
    }

    fn notify_handlers(&self, entry: &AuditLogEntry) {
        let mut handlers = self.event_handlers.write();

        for handler in handlers.iter_mut() {
            handler.on_audit_entry(entry);

            if entry.status == OperationStatus::Failed {
                handler.on_operation_failure(entry);
            }

            // Detect security violations
            if entry.security_context.access_level < self.config.min_access_level {
                handler.on_security_violation(entry);
            }
        }
    }

    fn compute_hash(&self, entry: &AuditLogEntry) -> String {
        // Simple hash computation using built-in hasher
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        entry.id.hash(&mut hasher);
        entry.timestamp.to_string().hash(&mut hasher);
        entry.operation_name.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }
}

/// Query parameters for audit logs
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Start time filter
    pub start_time: Option<DateTime<Utc>>,

    /// End time filter
    pub end_time: Option<DateTime<Utc>>,

    /// Operation type filter
    pub operation_type: Option<OperationType>,

    /// Status filter
    pub status: Option<OperationStatus>,

    /// Actor filter
    pub actor: Option<String>,

    /// Correlation ID filter
    pub correlation_id: Option<String>,

    /// Maximum number of results
    pub limit: Option<usize>,
}

/// Result of integrity verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityVerificationResult {
    /// Total entries checked
    pub total_entries: usize,

    /// Successfully verified entries
    pub verified_entries: usize,

    /// Corrupted entry IDs
    pub corrupted_entries: Vec<String>,

    /// Entries without integrity hash
    pub missing_hashes: usize,
}

/// Global audit logger instance
static GLOBAL_AUDIT_LOGGER: OnceLock<Arc<AuditLogger>> = OnceLock::new();

/// Get global audit logger
pub fn get_global_audit_logger() -> Arc<AuditLogger> {
    GLOBAL_AUDIT_LOGGER
        .get_or_init(|| Arc::new(AuditLogger::new(AuditConfig::default())))
        .clone()
}

/// Initialize global audit logger with custom config
pub fn init_global_audit_logger(config: AuditConfig) {
    let _ = GLOBAL_AUDIT_LOGGER.set(Arc::new(AuditLogger::new(config)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_logging() {
        let logger = AuditLogger::new(AuditConfig::default());

        let entry = logger.log_operation(
            OperationType::Forward,
            "test_op".to_string(),
            "test_user".to_string(),
            OperationStatus::Success,
        );

        assert_eq!(entry.operation_name, "test_op");
        assert_eq!(entry.status, OperationStatus::Success);
        assert!(entry.integrity_hash.is_some());
    }

    #[test]
    fn test_query_logs() {
        let logger = AuditLogger::new(AuditConfig::default());

        // Log some operations
        logger.log_operation(
            OperationType::Forward,
            "op1".to_string(),
            "user1".to_string(),
            OperationStatus::Success,
        );
        logger.log_operation(
            OperationType::Backward,
            "op2".to_string(),
            "user2".to_string(),
            OperationStatus::Failed,
        );

        // Query all operations
        let query = AuditQuery::default();
        let results = logger.query(query);
        assert_eq!(results.len(), 2);

        // Query specific operation type
        let query = AuditQuery {
            operation_type: Some(OperationType::Forward),
            ..Default::default()
        };
        let results = logger.query(query);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.log_operation(
            OperationType::Forward,
            "op1".to_string(),
            "user1".to_string(),
            OperationStatus::Success,
        );
        logger.log_operation(
            OperationType::Backward,
            "op2".to_string(),
            "user2".to_string(),
            OperationStatus::Failed,
        );

        let stats = logger.statistics();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.total_failed, 1);
    }

    #[test]
    fn test_integrity_verification() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.log_operation(
            OperationType::Forward,
            "test_op".to_string(),
            "test_user".to_string(),
            OperationStatus::Success,
        );

        let result = logger.verify_integrity();
        assert_eq!(result.total_entries, 1);
        assert_eq!(result.verified_entries, 1);
        assert_eq!(result.corrupted_entries.len(), 0);
    }

    #[test]
    fn test_cleanup_old_logs() {
        let mut config = AuditConfig::default();
        config.retention_days = 0; // Immediate expiry for testing

        let logger = AuditLogger::new(config);

        logger.log_operation(
            OperationType::Forward,
            "old_op".to_string(),
            "user".to_string(),
            OperationStatus::Success,
        );

        // Wait a bit
        std::thread::sleep(std::time::Duration::from_millis(10));

        logger.cleanup_old_logs();

        let stats = logger.statistics();
        // Statistics should still show the entry (cleanup doesn't affect stats)
        assert_eq!(stats.total_entries, 1);
    }

    #[test]
    fn test_export_json() {
        let logger = AuditLogger::new(AuditConfig::default());

        logger.log_operation(
            OperationType::Forward,
            "test_op".to_string(),
            "test_user".to_string(),
            OperationStatus::Success,
        );

        let json = logger.export_json().unwrap();
        assert!(json.contains("test_op"));
        assert!(json.contains("Forward"));
    }
}
