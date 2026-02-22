//! Gradient Hooks System
//!
//! This module provides a comprehensive hook system for monitoring, modifying,
//! and debugging gradients during backpropagation. Hooks can be attached to
//! tensors or parameters to intercept gradients as they flow through the
//! computation graph.
//!
//! ## Features
//!
//! - **Pre-backward Hooks**: Execute before gradient computation
//! - **Post-backward Hooks**: Execute after gradient computation
//! - **Gradient Modification**: Modify gradients on-the-fly
//! - **Gradient Monitoring**: Track gradient statistics
//! - **Conditional Hooks**: Execute hooks based on conditions
//! - **Hook Composition**: Chain multiple hooks together
//!
//! ## Usage
//!
//! ```rust,no_run
//! use torsh_autograd::gradient_hooks::{GradientHookManager, HookType};
//!
//! # fn example() -> torsh_core::error::Result<()> {
//! // Create hook manager
//! let mut manager = GradientHookManager::new();
//!
//! // Register a hook to monitor gradients
//! manager.register_hook("layer1", HookType::PostBackward, |grad| {
//!     println!("Gradient norm: {}", grad.iter().map(|x| x * x).sum::<f32>().sqrt());
//!     Ok(grad.to_vec())
//! })?;
//! # Ok(())
//! # }
//! ```

use crate::error_handling::{AutogradError, AutogradResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Type of gradient hook
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookType {
    /// Execute before backward pass
    PreBackward,
    /// Execute after backward pass
    PostBackward,
    /// Execute on gradient accumulation
    OnAccumulation,
    /// Execute on zero_grad
    OnZeroGrad,
}

impl HookType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::PreBackward => "Pre-Backward",
            Self::PostBackward => "Post-Backward",
            Self::OnAccumulation => "On-Accumulation",
            Self::OnZeroGrad => "On-ZeroGrad",
        }
    }
}

/// Hook priority (higher values execute first)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HookPriority(pub i32);

impl Default for HookPriority {
    fn default() -> Self {
        Self(0)
    }
}

/// Hook execution context
#[derive(Debug, Clone)]
pub struct HookContext {
    /// Name of the parameter/tensor
    pub name: String,
    /// Current gradient norm
    pub grad_norm: f32,
    /// Iteration/step number
    pub iteration: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl HookContext {
    /// Create a new hook context
    pub fn new(name: String) -> Self {
        Self {
            name,
            grad_norm: 0.0,
            iteration: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set gradient norm
    pub fn with_grad_norm(mut self, norm: f32) -> Self {
        self.grad_norm = norm;
        self
    }

    /// Set iteration
    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = iteration;
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Gradient hook function signature
pub type HookFn = dyn Fn(&[f32]) -> AutogradResult<Vec<f32>> + Send + Sync;

/// Gradient hook with metadata
pub struct GradientHook {
    /// Unique hook identifier
    pub id: String,
    /// Hook type
    pub hook_type: HookType,
    /// Hook priority
    pub priority: HookPriority,
    /// Hook function
    pub function: Arc<HookFn>,
    /// Whether this hook is enabled
    pub enabled: bool,
    /// Execution count
    pub execution_count: usize,
}

impl GradientHook {
    /// Create a new gradient hook
    pub fn new<F>(id: String, hook_type: HookType, function: F) -> Self
    where
        F: Fn(&[f32]) -> AutogradResult<Vec<f32>> + Send + Sync + 'static,
    {
        Self {
            id,
            hook_type,
            priority: HookPriority::default(),
            function: Arc::new(function),
            enabled: true,
            execution_count: 0,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: HookPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Enable the hook
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the hook
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Execute the hook
    pub fn execute(&mut self, gradient: &[f32]) -> AutogradResult<Vec<f32>> {
        if !self.enabled {
            return Ok(gradient.to_vec());
        }

        self.execution_count += 1;
        (self.function)(gradient)
    }
}

/// Statistics about hook executions
#[derive(Debug, Clone, Default)]
pub struct HookStats {
    /// Total hook executions
    pub total_executions: usize,
    /// Executions per hook type
    pub executions_by_type: HashMap<HookType, usize>,
    /// Average execution time (ms)
    pub avg_execution_time_ms: f64,
    /// Total time spent in hooks (ms)
    pub total_time_ms: f64,
}

/// Gradient hook manager
pub struct GradientHookManager {
    /// Registered hooks by parameter name
    hooks: HashMap<String, Vec<GradientHook>>,
    /// Global hooks (apply to all parameters)
    global_hooks: Vec<GradientHook>,
    /// Statistics
    stats: HookStats,
    /// Current iteration
    current_iteration: usize,
}

impl GradientHookManager {
    /// Create a new gradient hook manager
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
            global_hooks: Vec::new(),
            stats: HookStats::default(),
            current_iteration: 0,
        }
    }

    /// Register a hook for a specific parameter
    pub fn register_hook<F>(
        &mut self,
        param_name: &str,
        hook_type: HookType,
        function: F,
    ) -> AutogradResult<String>
    where
        F: Fn(&[f32]) -> AutogradResult<Vec<f32>> + Send + Sync + 'static,
    {
        let hook_id = format!("{}_{:?}_{}", param_name, hook_type, self.hooks.len());
        let hook = GradientHook::new(hook_id.clone(), hook_type, function);

        self.hooks
            .entry(param_name.to_string())
            .or_insert_with(Vec::new)
            .push(hook);

        Ok(hook_id)
    }

    /// Register a global hook (applies to all parameters)
    pub fn register_global_hook<F>(
        &mut self,
        hook_type: HookType,
        function: F,
    ) -> AutogradResult<String>
    where
        F: Fn(&[f32]) -> AutogradResult<Vec<f32>> + Send + Sync + 'static,
    {
        let hook_id = format!("global_{:?}_{}", hook_type, self.global_hooks.len());
        let hook = GradientHook::new(hook_id.clone(), hook_type, function);

        self.global_hooks.push(hook);

        Ok(hook_id)
    }

    /// Remove a hook by ID
    pub fn remove_hook(&mut self, hook_id: &str) -> AutogradResult<()> {
        // Try to remove from parameter-specific hooks
        for hooks in self.hooks.values_mut() {
            hooks.retain(|h| h.id != hook_id);
        }

        // Try to remove from global hooks
        self.global_hooks.retain(|h| h.id != hook_id);

        Ok(())
    }

    /// Enable a hook
    pub fn enable_hook(&mut self, hook_id: &str) -> AutogradResult<()> {
        for hooks in self.hooks.values_mut() {
            for hook in hooks.iter_mut() {
                if hook.id == hook_id {
                    hook.enable();
                    return Ok(());
                }
            }
        }

        for hook in self.global_hooks.iter_mut() {
            if hook.id == hook_id {
                hook.enable();
                return Ok(());
            }
        }

        Err(AutogradError::gradient_computation(
            "enable_hook",
            format!("Hook not found: {}", hook_id),
        )
        .into())
    }

    /// Disable a hook
    pub fn disable_hook(&mut self, hook_id: &str) -> AutogradResult<()> {
        for hooks in self.hooks.values_mut() {
            for hook in hooks.iter_mut() {
                if hook.id == hook_id {
                    hook.disable();
                    return Ok(());
                }
            }
        }

        for hook in self.global_hooks.iter_mut() {
            if hook.id == hook_id {
                hook.disable();
                return Ok(());
            }
        }

        Err(AutogradError::gradient_computation(
            "disable_hook",
            format!("Hook not found: {}", hook_id),
        )
        .into())
    }

    /// Execute hooks for a parameter
    pub fn execute_hooks(
        &mut self,
        param_name: &str,
        hook_type: HookType,
        gradient: &[f32],
    ) -> AutogradResult<Vec<f32>> {
        use std::time::Instant;
        let start = Instant::now();

        let mut result = gradient.to_vec();

        // Execute global hooks first
        for hook in self.global_hooks.iter_mut() {
            if hook.hook_type == hook_type && hook.enabled {
                result = hook.execute(&result)?;
            }
        }

        // Execute parameter-specific hooks
        if let Some(hooks) = self.hooks.get_mut(param_name) {
            // Sort by priority (higher first)
            hooks.sort_by(|a, b| b.priority.cmp(&a.priority));

            for hook in hooks.iter_mut() {
                if hook.hook_type == hook_type && hook.enabled {
                    result = hook.execute(&result)?;
                }
            }
        }

        // Update statistics
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_executions += 1;
        *self.stats.executions_by_type.entry(hook_type).or_insert(0) += 1;
        self.stats.total_time_ms += elapsed;
        self.stats.avg_execution_time_ms =
            self.stats.total_time_ms / self.stats.total_executions as f64;

        Ok(result)
    }

    /// Clear all hooks
    pub fn clear_all_hooks(&mut self) {
        self.hooks.clear();
        self.global_hooks.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &HookStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = HookStats::default();
        self.current_iteration = 0;
    }

    /// Increment iteration counter
    pub fn next_iteration(&mut self) {
        self.current_iteration += 1;
    }

    /// Get current iteration
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Generate performance report
    pub fn report(&self) -> String {
        let mut report = String::from("Gradient Hooks Statistics:\n");
        report.push_str(&format!(
            "- Total executions: {}\n",
            self.stats.total_executions
        ));
        report.push_str(&format!(
            "- Average execution time: {:.4}ms\n",
            self.stats.avg_execution_time_ms
        ));
        report.push_str(&format!(
            "- Total time in hooks: {:.2}ms\n",
            self.stats.total_time_ms
        ));
        report.push_str("- Executions by type:\n");

        for (hook_type, count) in &self.stats.executions_by_type {
            report.push_str(&format!("  - {}: {}\n", hook_type.name(), count));
        }

        report.push_str(&format!("- Registered hooks: {}\n", self.hooks.len()));
        report.push_str(&format!("- Global hooks: {}\n", self.global_hooks.len()));

        report
    }
}

impl Default for GradientHookManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global gradient hook manager
static GLOBAL_HOOK_MANAGER: once_cell::sync::Lazy<parking_lot::RwLock<GradientHookManager>> =
    once_cell::sync::Lazy::new(|| parking_lot::RwLock::new(GradientHookManager::new()));

/// Get the global gradient hook manager
pub fn get_global_hook_manager() -> parking_lot::RwLockReadGuard<'static, GradientHookManager> {
    GLOBAL_HOOK_MANAGER.read()
}

/// Get mutable access to the global gradient hook manager
pub fn get_global_hook_manager_mut() -> parking_lot::RwLockWriteGuard<'static, GradientHookManager>
{
    GLOBAL_HOOK_MANAGER.write()
}

/// Register a global hook
pub fn register_global_hook<F>(hook_type: HookType, function: F) -> AutogradResult<String>
where
    F: Fn(&[f32]) -> AutogradResult<Vec<f32>> + Send + Sync + 'static,
{
    let mut manager = GLOBAL_HOOK_MANAGER.write();
    manager.register_global_hook(hook_type, function)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_types() {
        assert_eq!(HookType::PreBackward.name(), "Pre-Backward");
        assert_eq!(HookType::PostBackward.name(), "Post-Backward");
    }

    #[test]
    fn test_hook_context() {
        let ctx = HookContext::new("layer1".to_string())
            .with_grad_norm(1.5)
            .with_iteration(10);

        assert_eq!(ctx.name, "layer1");
        assert_eq!(ctx.grad_norm, 1.5);
        assert_eq!(ctx.iteration, 10);
    }

    #[test]
    fn test_register_hook() {
        let mut manager = GradientHookManager::new();

        let hook_id = manager
            .register_hook("layer1", HookType::PostBackward, |grad| Ok(grad.to_vec()))
            .unwrap();

        assert!(hook_id.contains("layer1"));
    }

    #[test]
    fn test_execute_hooks() {
        let mut manager = GradientHookManager::new();

        // Register a hook that doubles gradients
        manager
            .register_hook("layer1", HookType::PostBackward, |grad| {
                Ok(grad.iter().map(|&x| x * 2.0).collect())
            })
            .unwrap();

        let gradient = vec![1.0, 2.0, 3.0];
        let result = manager
            .execute_hooks("layer1", HookType::PostBackward, &gradient)
            .unwrap();

        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_global_hooks() {
        let mut manager = GradientHookManager::new();

        // Register a global hook
        manager
            .register_global_hook(HookType::PostBackward, |grad| {
                Ok(grad.iter().map(|&x| x + 1.0).collect())
            })
            .unwrap();

        let gradient = vec![1.0, 2.0, 3.0];
        let result = manager
            .execute_hooks("any_param", HookType::PostBackward, &gradient)
            .unwrap();

        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_hook_enable_disable() {
        let mut manager = GradientHookManager::new();

        let hook_id = manager
            .register_hook("layer1", HookType::PostBackward, |grad| {
                Ok(grad.iter().map(|&x| x * 2.0).collect())
            })
            .unwrap();

        // Disable the hook
        manager.disable_hook(&hook_id).unwrap();

        let gradient = vec![1.0, 2.0, 3.0];
        let result = manager
            .execute_hooks("layer1", HookType::PostBackward, &gradient)
            .unwrap();

        // Should return original gradient since hook is disabled
        assert_eq!(result, gradient);
    }

    #[test]
    fn test_hook_stats() {
        let mut manager = GradientHookManager::new();

        manager
            .register_hook("layer1", HookType::PostBackward, |grad| Ok(grad.to_vec()))
            .unwrap();

        let gradient = vec![1.0, 2.0, 3.0];
        manager
            .execute_hooks("layer1", HookType::PostBackward, &gradient)
            .unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_executions, 1);
    }

    #[test]
    fn test_report() {
        let manager = GradientHookManager::new();
        let report = manager.report();

        assert!(report.contains("Gradient Hooks Statistics"));
        assert!(report.contains("Total executions:"));
    }
}
