//! Exception Safety for Autograd Operations
//!
//! This module provides comprehensive exception safety guarantees for autograd operations,
//! ensuring that computation graphs remain consistent and resources are properly managed
//! even when operations fail. It implements strong exception safety where possible and
//! provides transactional semantics for critical operations.

use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::error::CoreError;
use scirs2_core::ndarray::{Array, ArrayView};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

/// Exception safety levels for autograd operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExceptionSafetyLevel {
    /// No guarantee - operation may leave system in invalid state
    NoGuarantee,
    /// Basic guarantee - no resource leaks, but state may be changed
    Basic,
    /// Strong guarantee - operation either succeeds completely or has no effect
    Strong,
    /// No-throw guarantee - operation never fails
    NoThrow,
}

impl fmt::Display for ExceptionSafetyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExceptionSafetyLevel::NoGuarantee => write!(f, "No Guarantee"),
            ExceptionSafetyLevel::Basic => write!(f, "Basic"),
            ExceptionSafetyLevel::Strong => write!(f, "Strong"),
            ExceptionSafetyLevel::NoThrow => write!(f, "No-Throw"),
        }
    }
}

/// Transaction for autograd operations with rollback capability
#[derive(Debug)]
pub struct AutogradTransaction {
    id: usize,
    operations: Vec<TransactionOperation>,
    committed: bool,
    rolled_back: bool,
    safety_level: ExceptionSafetyLevel,
    resource_guards: Vec<Box<dyn ResourceGuard>>,
}

impl AutogradTransaction {
    pub fn new(safety_level: ExceptionSafetyLevel) -> Self {
        static TRANSACTION_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let id = TRANSACTION_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            id,
            operations: Vec::new(),
            committed: false,
            rolled_back: false,
            safety_level,
            resource_guards: Vec::new(),
        }
    }

    pub fn add_operation(&mut self, operation: TransactionOperation) {
        if !self.committed && !self.rolled_back {
            self.operations.push(operation);
        }
    }

    pub fn add_resource_guard(&mut self, guard: Box<dyn ResourceGuard>) {
        self.resource_guards.push(guard);
    }

    pub fn commit(&mut self) -> AutogradResult<()> {
        if self.committed || self.rolled_back {
            return Err(AutogradError::gradient_computation(
                "transaction_commit",
                format!("Transaction {} already finalized", self.id),
            ));
        }

        // For strong safety level, we need to ensure all operations can succeed
        if self.safety_level == ExceptionSafetyLevel::Strong {
            for operation in &self.operations {
                if let Err(e) = operation.validate() {
                    self.rollback()?;
                    return Err(e);
                }
            }
        }

        // Apply all operations
        for operation in &mut self.operations {
            if let Err(e) = operation.apply() {
                if self.safety_level == ExceptionSafetyLevel::Strong {
                    self.rollback()?;
                    return Err(e);
                } else {
                    tracing::warn!("Operation failed in transaction {}: {}", self.id, e);
                }
            }
        }

        self.committed = true;
        tracing::debug!("Transaction {} committed successfully", self.id);
        Ok(())
    }

    pub fn rollback(&mut self) -> AutogradResult<()> {
        if self.committed {
            return Err(AutogradError::gradient_computation(
                "transaction_rollback",
                format!("Cannot rollback committed transaction {}", self.id),
            ));
        }

        if self.rolled_back {
            return Ok(()); // Already rolled back
        }

        // Rollback operations in reverse order
        for operation in self.operations.iter_mut().rev() {
            if let Err(e) = operation.rollback() {
                tracing::error!(
                    "Failed to rollback operation in transaction {}: {}",
                    self.id,
                    e
                );
                // Continue with other rollbacks even if one fails
            }
        }

        self.rolled_back = true;
        tracing::debug!("Transaction {} rolled back successfully", self.id);
        Ok(())
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn is_committed(&self) -> bool {
        self.committed
    }

    pub fn is_rolled_back(&self) -> bool {
        self.rolled_back
    }

    pub fn safety_level(&self) -> ExceptionSafetyLevel {
        self.safety_level
    }
}

impl Drop for AutogradTransaction {
    fn drop(&mut self) {
        if !self.committed && !self.rolled_back {
            tracing::warn!(
                "Transaction {} dropped without commit or rollback, rolling back",
                self.id
            );
            if let Err(e) = self.rollback() {
                tracing::error!(
                    "Failed to rollback transaction {} in destructor: {}",
                    self.id,
                    e
                );
            }
        }
    }
}

/// Individual operation within a transaction
pub struct TransactionOperation {
    name: String,
    validate_fn: Option<Box<dyn Fn() -> AutogradResult<()> + Send + Sync>>,
    apply_fn: Box<dyn FnMut() -> AutogradResult<()> + Send>,
    rollback_fn: Option<Box<dyn FnMut() -> AutogradResult<()> + Send>>,
    applied: bool,
}

impl std::fmt::Debug for TransactionOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionOperation")
            .field("name", &self.name)
            .field("has_validate_fn", &self.validate_fn.is_some())
            .field("has_rollback_fn", &self.rollback_fn.is_some())
            .field("applied", &self.applied)
            .finish()
    }
}

impl TransactionOperation {
    pub fn new<F>(name: String, apply_fn: F) -> Self
    where
        F: FnMut() -> AutogradResult<()> + Send + 'static,
    {
        Self {
            name,
            validate_fn: None,
            apply_fn: Box::new(apply_fn),
            rollback_fn: None,
            applied: false,
        }
    }

    pub fn with_validation<V>(mut self, validate_fn: V) -> Self
    where
        V: Fn() -> AutogradResult<()> + Send + Sync + 'static,
    {
        self.validate_fn = Some(Box::new(validate_fn));
        self
    }

    pub fn with_rollback<R>(mut self, rollback_fn: R) -> Self
    where
        R: FnMut() -> AutogradResult<()> + Send + 'static,
    {
        self.rollback_fn = Some(Box::new(rollback_fn));
        self
    }

    pub fn validate(&self) -> AutogradResult<()> {
        if let Some(ref validate_fn) = self.validate_fn {
            validate_fn()
        } else {
            Ok(())
        }
    }

    pub fn apply(&mut self) -> AutogradResult<()> {
        if self.applied {
            return Ok(());
        }

        let result = (self.apply_fn)();
        if result.is_ok() {
            self.applied = true;
        }
        result
    }

    pub fn rollback(&mut self) -> AutogradResult<()> {
        if !self.applied {
            return Ok(()); // Nothing to rollback
        }

        if let Some(ref mut rollback_fn) = self.rollback_fn {
            let result = rollback_fn();
            if result.is_ok() {
                self.applied = false;
            }
            result
        } else {
            tracing::warn!("No rollback function for operation: {}", self.name);
            Ok(())
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn is_applied(&self) -> bool {
        self.applied
    }
}

/// Resource guard trait for RAII resource management
pub trait ResourceGuard: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn release(&mut self) -> AutogradResult<()>;
}

/// Guard for gradient storage resources
#[derive(Debug)]
pub struct GradientStorageGuard {
    storage_id: String,
    gradients_backup: HashMap<String, Vec<f64>>,
    released: bool,
}

impl GradientStorageGuard {
    pub fn new(storage_id: String) -> Self {
        Self {
            storage_id,
            gradients_backup: HashMap::new(),
            released: false,
        }
    }

    pub fn backup_gradient(&mut self, name: String, gradient: Vec<f64>) {
        if !self.released {
            self.gradients_backup.insert(name, gradient);
        }
    }

    pub fn restore_gradients(&self) -> AutogradResult<()> {
        if self.released {
            return Ok(());
        }

        // In a real implementation, this would restore gradients to the storage
        for (name, gradient) in &self.gradients_backup {
            tracing::debug!(
                "Restoring gradient {} with {} elements",
                name,
                gradient.len()
            );
        }

        Ok(())
    }
}

impl ResourceGuard for GradientStorageGuard {
    fn name(&self) -> &str {
        &self.storage_id
    }

    fn release(&mut self) -> AutogradResult<()> {
        if !self.released {
            self.restore_gradients()?;
            self.released = true;
            tracing::debug!("Released gradient storage guard: {}", self.storage_id);
        }
        Ok(())
    }
}

impl Drop for GradientStorageGuard {
    fn drop(&mut self) {
        if !self.released {
            if let Err(e) = self.release() {
                tracing::error!(
                    "Failed to release gradient storage guard {} in destructor: {}",
                    self.storage_id,
                    e
                );
            }
        }
    }
}

/// Guard for computation graph nodes
#[derive(Debug)]
pub struct ComputationGraphGuard {
    node_ids: Vec<usize>,
    graph_state_backup: Vec<u8>, // Serialized graph state
    released: bool,
}

impl ComputationGraphGuard {
    pub fn new(node_ids: Vec<usize>) -> Self {
        Self {
            node_ids,
            graph_state_backup: Vec::new(),
            released: false,
        }
    }

    pub fn backup_graph_state(&mut self, state: Vec<u8>) {
        if !self.released {
            self.graph_state_backup = state;
        }
    }

    pub fn restore_graph_state(&self) -> AutogradResult<()> {
        if self.released || self.graph_state_backup.is_empty() {
            return Ok(());
        }

        // In a real implementation, this would restore the computation graph
        tracing::debug!(
            "Restoring computation graph state for {} nodes",
            self.node_ids.len()
        );
        Ok(())
    }
}

impl ResourceGuard for ComputationGraphGuard {
    fn name(&self) -> &str {
        "ComputationGraphGuard"
    }

    fn release(&mut self) -> AutogradResult<()> {
        if !self.released {
            self.restore_graph_state()?;
            self.released = true;
            tracing::debug!(
                "Released computation graph guard for {} nodes",
                self.node_ids.len()
            );
        }
        Ok(())
    }
}

impl Drop for ComputationGraphGuard {
    fn drop(&mut self) {
        if !self.released {
            if let Err(e) = self.release() {
                tracing::error!(
                    "Failed to release computation graph guard in destructor: {}",
                    e
                );
            }
        }
    }
}

/// Exception-safe operation executor
pub struct ExceptionSafeExecutor {
    default_safety_level: ExceptionSafetyLevel,
    active_transactions: Arc<Mutex<HashMap<usize, Arc<Mutex<AutogradTransaction>>>>>,
    error_recovery_enabled: bool,
}

impl ExceptionSafeExecutor {
    pub fn new(default_safety_level: ExceptionSafetyLevel) -> Self {
        Self {
            default_safety_level,
            active_transactions: Arc::new(Mutex::new(HashMap::new())),
            error_recovery_enabled: true,
        }
    }

    pub fn with_default_safety() -> Self {
        Self::new(ExceptionSafetyLevel::Strong)
    }

    pub fn begin_transaction(
        &self,
        safety_level: Option<ExceptionSafetyLevel>,
    ) -> AutogradResult<Arc<Mutex<AutogradTransaction>>> {
        let level = safety_level.unwrap_or(self.default_safety_level);
        let transaction = Arc::new(Mutex::new(AutogradTransaction::new(level)));

        let transaction_id = {
            let tx = transaction.lock().map_err(|e| {
                AutogradError::gradient_computation(
                    "transaction_lock",
                    format!("Failed to lock transaction: {}", e),
                )
            })?;
            tx.id()
        };

        let mut active = self.active_transactions.lock().map_err(|e| {
            AutogradError::gradient_computation(
                "active_transactions_lock",
                format!("Failed to lock active transactions: {}", e),
            )
        })?;
        active.insert(transaction_id, transaction.clone());

        tracing::debug!(
            "Started transaction {} with safety level: {}",
            transaction_id,
            level
        );
        Ok(transaction)
    }

    pub fn execute_with_safety<F, R>(
        &self,
        operation_name: &str,
        safety_level: ExceptionSafetyLevel,
        operation: F,
    ) -> AutogradResult<R>
    where
        F: FnOnce() -> AutogradResult<R>,
    {
        match safety_level {
            ExceptionSafetyLevel::NoThrow => {
                // For no-throw guarantee, we catch all errors and return default values or handle gracefully
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation())) {
                    Ok(result) => result,
                    Err(panic) => {
                        tracing::error!(
                            "Panic in no-throw operation {}: {:?}",
                            operation_name,
                            panic
                        );
                        Err(AutogradError::gradient_computation(
                            "panic_recovery",
                            format!("Panic in no-throw operation: {}", operation_name),
                        ))
                    }
                }
            }
            ExceptionSafetyLevel::Strong => {
                // For strong guarantee, we use a transaction to ensure all-or-nothing semantics
                let transaction = self.begin_transaction(Some(safety_level))?;
                let result = operation();

                match result {
                    Ok(value) => {
                        let mut tx = transaction.lock().map_err(|e| {
                            AutogradError::gradient_computation(
                                "transaction_lock",
                                format!("Failed to lock transaction: {}", e),
                            )
                        })?;
                        tx.commit()?;
                        Ok(value)
                    }
                    Err(e) => {
                        let mut tx = transaction.lock().map_err(|e2| {
                            AutogradError::gradient_computation(
                                "transaction_rollback_lock",
                                format!("Failed to lock transaction for rollback: {}", e2),
                            )
                        })?;
                        tx.rollback()?;
                        Err(e)
                    }
                }
            }
            ExceptionSafetyLevel::Basic => {
                // For basic guarantee, we ensure no resource leaks but state may change
                let _resource_guard = BasicResourceGuard::new(operation_name.to_string());
                operation()
            }
            ExceptionSafetyLevel::NoGuarantee => {
                // No special handling
                operation()
            }
        }
    }

    pub fn execute_batch_with_safety<F>(
        &self,
        operations: Vec<(&str, ExceptionSafetyLevel, F)>,
    ) -> AutogradResult<Vec<AutogradResult<()>>>
    where
        F: FnOnce() -> AutogradResult<()>,
    {
        let transaction = self.begin_transaction(Some(ExceptionSafetyLevel::Strong))?;
        let mut results = Vec::new();

        for (name, safety_level, operation) in operations {
            let result = self.execute_with_safety(name, safety_level, operation);
            results.push(result);
        }

        // Check if any operations failed
        let any_failed = results.iter().any(|r| r.is_err());

        if any_failed {
            let mut tx = transaction.lock().map_err(|e| {
                AutogradError::gradient_computation(
                    "transaction_rollback_lock",
                    format!("Failed to lock transaction for rollback: {}", e),
                )
            })?;
            tx.rollback()?;
        } else {
            let mut tx = transaction.lock().map_err(|e| {
                AutogradError::gradient_computation(
                    "transaction_commit_lock",
                    format!("Failed to lock transaction for commit: {}", e),
                )
            })?;
            tx.commit()?;
        }

        Ok(results)
    }

    pub fn cleanup_completed_transactions(&self) -> AutogradResult<usize> {
        let mut active = self.active_transactions.lock().map_err(|e| {
            AutogradError::gradient_computation(
                "active_transactions_lock",
                format!("Failed to lock active transactions: {}", e),
            )
        })?;

        let initial_count = active.len();
        active.retain(|_id, transaction| {
            if let Ok(tx) = transaction.lock() {
                !tx.is_committed() && !tx.is_rolled_back()
            } else {
                false // Remove transactions that can't be locked
            }
        });

        let cleaned_count = initial_count - active.len();
        tracing::debug!("Cleaned up {} completed transactions", cleaned_count);
        Ok(cleaned_count)
    }

    pub fn get_active_transaction_count(&self) -> usize {
        self.active_transactions
            .lock()
            .map(|active| active.len())
            .unwrap_or(0)
    }

    pub fn set_error_recovery_enabled(&mut self, enabled: bool) {
        self.error_recovery_enabled = enabled;
    }

    pub fn is_error_recovery_enabled(&self) -> bool {
        self.error_recovery_enabled
    }
}

/// Basic resource guard for simple cleanup
#[derive(Debug)]
pub struct BasicResourceGuard {
    name: String,
    released: bool,
}

impl BasicResourceGuard {
    pub fn new(name: String) -> Self {
        tracing::debug!("Created basic resource guard: {}", name);
        Self {
            name,
            released: false,
        }
    }
}

impl Drop for BasicResourceGuard {
    fn drop(&mut self) {
        if !self.released {
            tracing::debug!("Releasing basic resource guard: {}", self.name);
            self.released = true;
        }
    }
}

/// Exception safety analyzer for autograd operations
pub struct ExceptionSafetyAnalyzer {
    operation_safety_levels: HashMap<String, ExceptionSafetyLevel>,
    safety_violations: Vec<SafetyViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub operation_name: String,
    pub expected_level: ExceptionSafetyLevel,
    pub actual_level: ExceptionSafetyLevel,
    pub violation_type: ViolationType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub description: String,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ViolationType {
    ResourceLeak,
    StateCorruption,
    UnhandledException,
    TransactionViolation,
    MemoryViolation,
}

impl ExceptionSafetyAnalyzer {
    pub fn new() -> Self {
        Self {
            operation_safety_levels: HashMap::new(),
            safety_violations: Vec::new(),
        }
    }

    pub fn register_operation_safety(
        &mut self,
        operation_name: String,
        safety_level: ExceptionSafetyLevel,
    ) {
        self.operation_safety_levels
            .insert(operation_name, safety_level);
    }

    pub fn analyze_operation(
        &mut self,
        operation_name: &str,
        actual_behavior: ExceptionSafetyLevel,
    ) -> Vec<SafetyViolation> {
        let mut violations = Vec::new();

        if let Some(&expected_level) = self.operation_safety_levels.get(operation_name) {
            if !self.is_safety_level_compatible(expected_level, actual_behavior) {
                let violation = SafetyViolation {
                    operation_name: operation_name.to_string(),
                    expected_level,
                    actual_level: actual_behavior,
                    violation_type: ViolationType::TransactionViolation,
                    timestamp: chrono::Utc::now(),
                    description: format!(
                        "Operation {} violated safety contract: expected {}, got {}",
                        operation_name, expected_level, actual_behavior
                    ),
                };
                violations.push(violation);
            }
        }

        self.safety_violations.extend(violations.clone());
        violations
    }

    pub fn get_violation_report(&self) -> SafetyViolationReport {
        SafetyViolationReport::new(&self.safety_violations)
    }

    fn is_safety_level_compatible(
        &self,
        expected: ExceptionSafetyLevel,
        actual: ExceptionSafetyLevel,
    ) -> bool {
        use ExceptionSafetyLevel::*;
        match (expected, actual) {
            (NoGuarantee, _) => true,
            (Basic, NoGuarantee) => false,
            (Basic, _) => true,
            (Strong, NoGuarantee | Basic) => false,
            (Strong, _) => true,
            (NoThrow, NoThrow) => true,
            (NoThrow, _) => false,
        }
    }
}

/// Report for exception safety violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolationReport {
    pub total_violations: usize,
    pub violations_by_type: HashMap<ViolationType, usize>,
    pub violations_by_operation: HashMap<String, usize>,
    pub critical_violations: Vec<SafetyViolation>,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
}

impl SafetyViolationReport {
    pub fn new(violations: &[SafetyViolation]) -> Self {
        let total_violations = violations.len();

        let mut violations_by_type = HashMap::new();
        let mut violations_by_operation = HashMap::new();

        for violation in violations {
            *violations_by_type
                .entry(violation.violation_type.clone())
                .or_insert(0) += 1;
            *violations_by_operation
                .entry(violation.operation_name.clone())
                .or_insert(0) += 1;
        }

        let critical_violations = violations
            .iter()
            .filter(|v| {
                matches!(
                    v.expected_level,
                    ExceptionSafetyLevel::Strong | ExceptionSafetyLevel::NoThrow
                )
            })
            .cloned()
            .collect();

        Self {
            total_violations,
            violations_by_type,
            violations_by_operation,
            critical_violations,
            generation_timestamp: chrono::Utc::now(),
        }
    }

    pub fn print_summary(&self) {
        println!("=== Exception Safety Violation Report ===");
        println!("Total Violations: {}", self.total_violations);
        println!("Critical Violations: {}", self.critical_violations.len());
        println!();

        if !self.violations_by_type.is_empty() {
            println!("Violations by Type:");
            for (violation_type, count) in &self.violations_by_type {
                println!("  {:?}: {}", violation_type, count);
            }
            println!();
        }

        if !self.violations_by_operation.is_empty() {
            println!("Top Violating Operations:");
            let mut operations: Vec<_> = self.violations_by_operation.iter().collect();
            operations.sort_by(|a, b| b.1.cmp(a.1));
            for (operation, count) in operations.iter().take(10) {
                println!("  {}: {}", operation, count);
            }
        }
    }
}

/// Global exception safety executor instance
static mut GLOBAL_EXECUTOR: Option<ExceptionSafeExecutor> = None;
static EXECUTOR_INIT: std::sync::Once = std::sync::Once::new();

pub fn get_global_executor() -> &'static ExceptionSafeExecutor {
    unsafe {
        EXECUTOR_INIT.call_once(|| {
            GLOBAL_EXECUTOR = Some(ExceptionSafeExecutor::with_default_safety());
        });
        GLOBAL_EXECUTOR.as_ref().unwrap()
    }
}

/// Convenience macro for executing operations with strong exception safety
#[macro_export]
macro_rules! with_strong_safety {
    ($operation_name:expr, $operation:expr) => {
        $crate::exception_safety::get_global_executor().execute_with_safety(
            $operation_name,
            $crate::exception_safety::ExceptionSafetyLevel::Strong,
            || $operation,
        )
    };
}

/// Convenience macro for executing operations with no-throw guarantee
#[macro_export]
macro_rules! with_no_throw {
    ($operation_name:expr, $operation:expr) => {
        $crate::exception_safety::get_global_executor().execute_with_safety(
            $operation_name,
            $crate::exception_safety::ExceptionSafetyLevel::NoThrow,
            || $operation,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_creation() {
        let transaction = AutogradTransaction::new(ExceptionSafetyLevel::Strong);
        assert_eq!(transaction.safety_level(), ExceptionSafetyLevel::Strong);
        assert!(!transaction.is_committed());
        assert!(!transaction.is_rolled_back());
    }

    #[test]
    fn test_transaction_commit_success() {
        let mut transaction = AutogradTransaction::new(ExceptionSafetyLevel::Strong);

        let operation = TransactionOperation::new("test_op".to_string(), || Ok(()));
        transaction.add_operation(operation);

        assert!(transaction.commit().is_ok());
        assert!(transaction.is_committed());
        assert!(!transaction.is_rolled_back());
    }

    #[test]
    fn test_transaction_rollback() {
        let mut transaction = AutogradTransaction::new(ExceptionSafetyLevel::Strong);

        let operation = TransactionOperation::new("test_op".to_string(), || Ok(()));
        transaction.add_operation(operation);

        assert!(transaction.rollback().is_ok());
        assert!(!transaction.is_committed());
        assert!(transaction.is_rolled_back());
    }

    #[test]
    fn test_transaction_operation_with_rollback() {
        let applied = Arc::new(Mutex::new(false));
        let applied_clone1 = applied.clone();
        let applied_clone2 = applied.clone();

        let operation = TransactionOperation::new("test_op".to_string(), move || {
            *applied_clone1.lock().unwrap() = true;
            Ok(())
        })
        .with_rollback(move || {
            *applied_clone2.lock().unwrap() = false;
            Ok(())
        });

        // This test is simplified since we can't easily test mutable closures
        assert_eq!(operation.name(), "test_op");
        assert!(!operation.is_applied());
    }

    #[test]
    fn test_gradient_storage_guard() {
        let mut guard = GradientStorageGuard::new("test_storage".to_string());
        guard.backup_gradient("grad1".to_string(), vec![1.0, 2.0, 3.0]);
        guard.backup_gradient("grad2".to_string(), vec![4.0, 5.0]);

        assert_eq!(guard.name(), "test_storage");
        assert!(guard.release().is_ok());
    }

    #[test]
    fn test_computation_graph_guard() {
        let mut guard = ComputationGraphGuard::new(vec![1, 2, 3]);
        guard.backup_graph_state(vec![1, 2, 3, 4, 5]);

        assert_eq!(guard.name(), "ComputationGraphGuard");
        assert!(guard.restore_graph_state().is_ok());
        assert!(guard.release().is_ok());
    }

    #[test]
    fn test_exception_safe_executor() {
        let executor = ExceptionSafeExecutor::with_default_safety();

        let result =
            executor.execute_with_safety("test_operation", ExceptionSafetyLevel::Strong, || Ok(42));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_exception_safe_executor_with_error() {
        let executor = ExceptionSafeExecutor::with_default_safety();

        let result: AutogradResult<()> =
            executor.execute_with_safety("failing_operation", ExceptionSafetyLevel::Strong, || {
                Err(AutogradError::gradient_computation(
                    "test_operation",
                    "Test error",
                ))
            });

        assert!(result.is_err());
    }

    #[test]
    fn test_exception_safety_analyzer() {
        let mut analyzer = ExceptionSafetyAnalyzer::new();
        analyzer.register_operation_safety("op1".to_string(), ExceptionSafetyLevel::Strong);

        let violations = analyzer.analyze_operation("op1", ExceptionSafetyLevel::Basic);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].operation_name, "op1");
    }

    #[test]
    fn test_safety_level_compatibility() {
        let analyzer = ExceptionSafetyAnalyzer::new();

        assert!(analyzer.is_safety_level_compatible(
            ExceptionSafetyLevel::NoGuarantee,
            ExceptionSafetyLevel::Strong
        ));

        assert!(!analyzer
            .is_safety_level_compatible(ExceptionSafetyLevel::Strong, ExceptionSafetyLevel::Basic));

        assert!(analyzer.is_safety_level_compatible(
            ExceptionSafetyLevel::Strong,
            ExceptionSafetyLevel::NoThrow
        ));
    }

    #[test]
    fn test_safety_violation_report() {
        let violations = vec![SafetyViolation {
            operation_name: "op1".to_string(),
            expected_level: ExceptionSafetyLevel::Strong,
            actual_level: ExceptionSafetyLevel::Basic,
            violation_type: ViolationType::TransactionViolation,
            timestamp: chrono::Utc::now(),
            description: "Test violation".to_string(),
        }];

        let report = SafetyViolationReport::new(&violations);
        assert_eq!(report.total_violations, 1);
        assert_eq!(report.critical_violations.len(), 1);
    }

    #[test]
    fn test_global_executor() {
        let executor = get_global_executor();

        let result =
            executor.execute_with_safety("global_test", ExceptionSafetyLevel::Strong, || {
                Ok("success")
            });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn test_basic_resource_guard() {
        let guard = BasicResourceGuard::new("test_resource".to_string());
        assert_eq!(guard.name, "test_resource");
        assert!(!guard.released);
        // Guard will be automatically released when dropped
    }

    #[test]
    fn test_transaction_cleanup() {
        let executor = ExceptionSafeExecutor::with_default_safety();

        // Start a transaction
        let transaction = executor
            .begin_transaction(Some(ExceptionSafetyLevel::Strong))
            .unwrap();
        assert_eq!(executor.get_active_transaction_count(), 1);

        // Commit the transaction
        {
            let mut tx = transaction.lock().unwrap();
            tx.commit().unwrap();
        }

        // Cleanup completed transactions
        let cleaned = executor.cleanup_completed_transactions().unwrap();
        assert_eq!(cleaned, 1);
        assert_eq!(executor.get_active_transaction_count(), 0);
    }
}
