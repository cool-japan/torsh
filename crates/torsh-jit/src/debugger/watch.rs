//! Watch expression management for JIT debugging
//!
//! This module provides comprehensive watch expression management capabilities
//! including adding, removing, and evaluating watch expressions during debugging.

use super::core::{DebugValue, Watch, WatchId, WatchUpdate};
use crate::{JitError, JitResult};
use std::collections::HashMap;

/// Watch manager for watching expressions
pub struct WatchManager {
    watches: HashMap<WatchId, Watch>,
    next_id: WatchId,
}

impl WatchManager {
    /// Create a new watch manager
    pub fn new() -> Self {
        Self {
            watches: HashMap::new(),
            next_id: WatchId(0),
        }
    }

    /// Add a watch expression
    ///
    /// # Arguments
    /// * `expression` - The expression to watch
    ///
    /// # Returns
    /// The ID of the newly created watch
    ///
    /// # Examples
    /// ```rust
    /// use torsh_jit::debugger::WatchManager;
    ///
    /// let mut manager = WatchManager::new();
    /// let id = manager.add_watch("variable_name".to_string()).unwrap();
    /// ```
    pub fn add_watch(&mut self, expression: String) -> JitResult<WatchId> {
        let id = self.next_id;
        self.next_id = WatchId(self.next_id.0 + 1);

        let watch = Watch {
            id,
            expression,
            enabled: true,
            last_value: None,
        };

        self.watches.insert(id, watch);
        Ok(id)
    }

    /// Remove a watch expression
    ///
    /// # Arguments
    /// * `id` - The ID of the watch to remove
    ///
    /// # Returns
    /// `Ok(())` if the watch was removed, error if not found
    pub fn remove_watch(&mut self, id: WatchId) -> JitResult<()> {
        if self.watches.remove(&id).is_some() {
            Ok(())
        } else {
            Err(JitError::RuntimeError(format!("Watch {} not found", id.0)))
        }
    }

    /// Enable a watch expression
    ///
    /// # Arguments
    /// * `id` - The ID of the watch to enable
    pub fn enable_watch(&mut self, id: WatchId) -> JitResult<()> {
        if let Some(watch) = self.watches.get_mut(&id) {
            watch.enabled = true;
            Ok(())
        } else {
            Err(JitError::RuntimeError(format!("Watch {} not found", id.0)))
        }
    }

    /// Disable a watch expression
    ///
    /// # Arguments
    /// * `id` - The ID of the watch to disable
    pub fn disable_watch(&mut self, id: WatchId) -> JitResult<()> {
        if let Some(watch) = self.watches.get_mut(&id) {
            watch.enabled = false;
            Ok(())
        } else {
            Err(JitError::RuntimeError(format!("Watch {} not found", id.0)))
        }
    }

    /// Get a list of all watch expressions
    ///
    /// # Returns
    /// A vector of references to all watches
    pub fn list_watches(&self) -> Vec<&Watch> {
        self.watches.values().collect()
    }

    /// Get a specific watch by ID
    ///
    /// # Arguments
    /// * `id` - The ID of the watch to retrieve
    ///
    /// # Returns
    /// An optional reference to the watch
    pub fn get_watch(&self, id: WatchId) -> Option<&Watch> {
        self.watches.get(&id)
    }

    /// Update all watch expressions with current values
    ///
    /// # Arguments
    /// * `session` - The debug session to evaluate expressions against
    ///
    /// # Returns
    /// A vector of watch updates for expressions that changed
    pub fn update_watches<S>(&mut self, session: &S) -> JitResult<Vec<WatchUpdate>>
    where
        S: ExpressionEvaluator,
    {
        let mut updates = Vec::new();
        let mut watch_evaluations = Vec::new();

        // First pass: collect evaluations without borrowing self mutably
        for (watch_id, watch) in &self.watches {
            if watch.enabled {
                match session.evaluate_expression(&watch.expression) {
                    Ok(result) if result.success => {
                        let changed = match &watch.last_value {
                            Some(last) => !self.values_equal(last, &result.result),
                            None => true,
                        };

                        if changed {
                            watch_evaluations.push((
                                *watch_id,
                                watch.last_value.clone(),
                                result.result.clone(),
                            ));
                        }
                    }
                    Ok(_) => {
                        // Evaluation failed - could create an update with error
                    }
                    Err(_) => {
                        // Expression error - could create an update with error
                    }
                }
            }
        }

        // Second pass: update watches and build updates
        for (watch_id, old_value, new_value) in watch_evaluations {
            if let Some(watch) = self.watches.get_mut(&watch_id) {
                updates.push(WatchUpdate {
                    watch_id,
                    old_value,
                    new_value: new_value.clone(),
                });
                watch.last_value = Some(new_value);
            }
        }

        Ok(updates)
    }

    /// Check if two debug values are equal
    fn values_equal(&self, a: &DebugValue, b: &DebugValue) -> bool {
        match (a, b) {
            (DebugValue::Scalar(a), DebugValue::Scalar(b)) => (a - b).abs() < 1e-10,
            (DebugValue::Integer(a), DebugValue::Integer(b)) => a == b,
            (DebugValue::Boolean(a), DebugValue::Boolean(b)) => a == b,
            (
                DebugValue::Tensor {
                    data: data_a,
                    shape: shape_a,
                    dtype: dtype_a,
                },
                DebugValue::Tensor {
                    data: data_b,
                    shape: shape_b,
                    dtype: dtype_b,
                },
            ) => {
                dtype_a == dtype_b
                    && shape_a == shape_b
                    && data_a.len() == data_b.len()
                    && data_a
                        .iter()
                        .zip(data_b.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-6)
            }
            _ => false,
        }
    }

    /// Clear all watch expressions
    pub fn clear_all_watches(&mut self) {
        self.watches.clear();
    }

    /// Get the number of watch expressions
    pub fn count(&self) -> usize {
        self.watches.len()
    }

    /// Get the number of enabled watch expressions
    pub fn enabled_count(&self) -> usize {
        self.watches.values().filter(|w| w.enabled).count()
    }

    /// Find watches by expression pattern
    ///
    /// # Arguments
    /// * `pattern` - The pattern to search for in watch expressions
    ///
    /// # Returns
    /// A vector of references to watches matching the pattern
    pub fn find_watches_by_pattern(&self, pattern: &str) -> Vec<&Watch> {
        self.watches
            .values()
            .filter(|watch| watch.expression.contains(pattern))
            .collect()
    }

    /// Get watch statistics
    ///
    /// # Returns
    /// A tuple containing (total_watches, enabled_watches, watches_with_values)
    pub fn get_statistics(&self) -> (usize, usize, usize) {
        let total = self.watches.len();
        let enabled = self.watches.values().filter(|w| w.enabled).count();
        let with_values = self
            .watches
            .values()
            .filter(|w| w.last_value.is_some())
            .count();

        (total, enabled, with_values)
    }

    /// Reset all watch values
    ///
    /// This clears the last_value for all watches, forcing them to be re-evaluated
    pub fn reset_watch_values(&mut self) {
        for watch in self.watches.values_mut() {
            watch.last_value = None;
        }
    }

    /// Get all watches that have changed values
    ///
    /// # Returns
    /// A vector of references to watches that have last_value set
    pub fn get_watches_with_values(&self) -> Vec<&Watch> {
        self.watches
            .values()
            .filter(|watch| watch.last_value.is_some())
            .collect()
    }
}

/// Trait for types that can evaluate expressions
pub trait ExpressionEvaluator {
    /// Evaluate an expression and return the result
    fn evaluate_expression(&self, expression: &str) -> JitResult<super::core::EvaluationResult>;
}

impl Default for WatchManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debugger::core::{DebugValue, EvaluationResult};

    // Mock expression evaluator for testing
    struct MockEvaluator {
        values: HashMap<String, DebugValue>,
    }

    impl MockEvaluator {
        fn new() -> Self {
            let mut values = HashMap::new();
            values.insert("x".to_string(), DebugValue::Scalar(42.0));
            values.insert("y".to_string(), DebugValue::Integer(100));
            values.insert("flag".to_string(), DebugValue::Boolean(true));

            Self { values }
        }
    }

    impl ExpressionEvaluator for MockEvaluator {
        fn evaluate_expression(&self, expression: &str) -> JitResult<EvaluationResult> {
            if let Some(value) = self.values.get(expression) {
                Ok(EvaluationResult {
                    expression: expression.to_string(),
                    result: value.clone(),
                    success: true,
                    error_message: None,
                })
            } else {
                Ok(EvaluationResult {
                    expression: expression.to_string(),
                    result: DebugValue::Scalar(0.0),
                    success: false,
                    error_message: Some("Variable not found".to_string()),
                })
            }
        }
    }

    #[test]
    fn test_watch_manager_creation() {
        let manager = WatchManager::new();
        assert_eq!(manager.count(), 0);
        assert_eq!(manager.enabled_count(), 0);
    }

    #[test]
    fn test_add_and_remove_watch() {
        let mut manager = WatchManager::new();

        let id = manager.add_watch("test_expression".to_string()).unwrap();
        assert_eq!(manager.count(), 1);
        assert_eq!(manager.enabled_count(), 1);

        assert!(manager.remove_watch(id).is_ok());
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_enable_disable_watch() {
        let mut manager = WatchManager::new();

        let id = manager.add_watch("test_expression".to_string()).unwrap();
        assert_eq!(manager.enabled_count(), 1);

        manager.disable_watch(id).unwrap();
        assert_eq!(manager.enabled_count(), 0);

        manager.enable_watch(id).unwrap();
        assert_eq!(manager.enabled_count(), 1);
    }

    #[test]
    fn test_update_watches() {
        let mut manager = WatchManager::new();
        let evaluator = MockEvaluator::new();

        let id1 = manager.add_watch("x".to_string()).unwrap();
        let id2 = manager.add_watch("y".to_string()).unwrap();

        let updates = manager.update_watches(&evaluator).unwrap();
        assert_eq!(updates.len(), 2); // Both watches should have initial values

        // Update again - should not produce changes since values are the same
        let updates = manager.update_watches(&evaluator).unwrap();
        assert_eq!(updates.len(), 0);
    }

    #[test]
    fn test_values_equal() {
        let manager = WatchManager::new();

        assert!(manager.values_equal(&DebugValue::Scalar(42.0), &DebugValue::Scalar(42.0)));
        assert!(!manager.values_equal(&DebugValue::Scalar(42.0), &DebugValue::Scalar(43.0)));

        assert!(manager.values_equal(&DebugValue::Integer(100), &DebugValue::Integer(100)));
        assert!(!manager.values_equal(&DebugValue::Integer(100), &DebugValue::Integer(101)));

        assert!(manager.values_equal(&DebugValue::Boolean(true), &DebugValue::Boolean(true)));
        assert!(!manager.values_equal(&DebugValue::Boolean(true), &DebugValue::Boolean(false)));
    }

    #[test]
    fn test_find_watches_by_pattern() {
        let mut manager = WatchManager::new();

        manager.add_watch("variable_x".to_string()).unwrap();
        manager.add_watch("variable_y".to_string()).unwrap();
        manager.add_watch("other_var".to_string()).unwrap();

        let matches = manager.find_watches_by_pattern("variable");
        assert_eq!(matches.len(), 2);

        let matches = manager.find_watches_by_pattern("other");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_watch_statistics() {
        let mut manager = WatchManager::new();
        let evaluator = MockEvaluator::new();

        manager.add_watch("x".to_string()).unwrap();
        let id2 = manager.add_watch("y".to_string()).unwrap();
        manager.add_watch("z".to_string()).unwrap(); // This will fail evaluation

        manager.disable_watch(id2).unwrap();

        let (total, enabled, _) = manager.get_statistics();
        assert_eq!(total, 3);
        assert_eq!(enabled, 2); // One is disabled

        // Update watches to get values
        manager.update_watches(&evaluator).unwrap();

        let (_, _, with_values) = manager.get_statistics();
        assert_eq!(with_values, 1); // Only 'x' will have a value (y is disabled, z fails)
    }

    #[test]
    fn test_clear_all_watches() {
        let mut manager = WatchManager::new();

        manager.add_watch("watch1".to_string()).unwrap();
        manager.add_watch("watch2".to_string()).unwrap();

        assert_eq!(manager.count(), 2);
        manager.clear_all_watches();
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_reset_watch_values() {
        let mut manager = WatchManager::new();
        let evaluator = MockEvaluator::new();

        manager.add_watch("x".to_string()).unwrap();

        // Update to get initial value
        manager.update_watches(&evaluator).unwrap();
        let (_, _, with_values) = manager.get_statistics();
        assert_eq!(with_values, 1);

        // Reset values
        manager.reset_watch_values();
        let (_, _, with_values) = manager.get_statistics();
        assert_eq!(with_values, 0);
    }
}
