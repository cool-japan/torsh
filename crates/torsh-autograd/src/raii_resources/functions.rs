//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error_handling::AutogradResult;
use std::sync::{Arc, Mutex};

use super::types::{AutogradResourceFactory, AutogradResourceManager, ResourceStats};

/// Trait for resources that can be automatically managed
pub trait AutogradResource: Send + Sync {
    /// Get the resource type name
    fn resource_type(&self) -> &'static str;
    /// Get the size/impact of this resource
    fn resource_size(&self) -> usize;
    /// Clean up the resource (called automatically on drop)
    fn cleanup(&mut self) -> AutogradResult<()>;
    /// Check if the resource is still valid/needed
    fn is_valid(&self) -> bool;
    /// Get resource statistics
    fn get_stats(&self) -> ResourceStats;
}
/// Global resource manager instance
static GLOBAL_RESOURCE_MANAGER: std::sync::OnceLock<Arc<Mutex<AutogradResourceManager>>> =
    std::sync::OnceLock::new();
/// Get the global resource manager
pub fn get_global_resource_manager() -> Arc<Mutex<AutogradResourceManager>> {
    GLOBAL_RESOURCE_MANAGER
        .get_or_init(|| Arc::new(Mutex::new(AutogradResourceManager::new())))
        .clone()
}
/// Convenience macro for creating RAII scoped autograd operations
#[macro_export]
macro_rules! autograd_scope {
    ($scope_name:ident, $body:block) => {
        let mut $scope_name = $crate::raii_resources::AutogradScope::new();
        $body
    };
}
/// Global RAII factory instance
static GLOBAL_FACTORY: once_cell::sync::Lazy<AutogradResourceFactory> =
    once_cell::sync::Lazy::new(|| AutogradResourceFactory::new());
/// Get the global RAII factory
pub fn get_global_factory() -> &'static AutogradResourceFactory {
    &GLOBAL_FACTORY
}
/// Convenience macro for creating RAII guards
#[macro_export]
macro_rules! autograd_guard {
    (tensor_grad, $requires_grad:expr) => {
        crate::raii_resources::get_global_factory().create_tensor_grad_guard($requires_grad)
    };
    (checkpoint, $data:expr) => {
        crate::raii_resources::get_global_factory().create_checkpoint_guard($data)
    };
    (distributed, $rank:expr, $world_size:expr, $is_coord:expr) => {
        crate::raii_resources::get_global_factory().create_distributed_context_guard(
            $rank,
            $world_size,
            $is_coord,
        )
    };
    (profile, $name:expr) => {
        crate::raii_resources::get_global_factory().create_profile_session_guard($name.to_string())
    };
    (var_env) => {
        crate::raii_resources::get_global_factory().create_variable_environment_guard()
    };
}
#[cfg(test)]
mod tests {
    use super::super::types::{
        AutogradContextGuard, AutogradScope, ComputationGraphGuard, ComputationGraphManager,
        GradientStorageGuard, GradientStorageManager, MemoryBufferGuard, MemoryBufferManager,
    };
    use super::*;
    use std::time::Duration;
    #[test]
    fn test_computation_graph_guard() {
        let manager = Arc::new(Mutex::new(ComputationGraphManager::new()));
        let guard = ComputationGraphGuard::new(1, manager.clone());
        assert_eq!(guard.node_id(), 1);
        assert_eq!(guard.resource_type(), "ComputationGraphNode");
        assert!(guard.is_valid());
    }
    #[test]
    fn test_gradient_storage_guard() {
        let manager = Arc::new(Mutex::new(GradientStorageManager::new()));
        let guard = GradientStorageGuard::new(1, 1024, manager);
        assert_eq!(guard.tensor_id(), 1);
        assert_eq!(guard.resource_type(), "GradientStorage");
        assert_eq!(guard.resource_size(), 1024);
    }
    #[test]
    fn test_memory_buffer_guard() {
        let manager = Arc::new(Mutex::new(MemoryBufferManager::new()));
        let guard = MemoryBufferGuard::new(1, 2048, manager);
        assert_eq!(guard.resource_type(), "MemoryBuffer");
        assert_eq!(guard.resource_size(), 2048);
    }
    #[test]
    fn test_autograd_context_guard() {
        let manager = Arc::new(Mutex::new(AutogradResourceManager::new()));
        let guard = AutogradContextGuard::new(1, manager);
        assert_eq!(guard.resource_type(), "AutogradContext");
        assert!(guard.is_valid());
    }
    #[test]
    fn test_resource_manager_stats() {
        let manager = AutogradResourceManager::new();
        let stats = manager.get_resource_stats();
        assert_eq!(stats.graph_nodes, 0);
        assert_eq!(stats.gradient_count, 0);
        assert_eq!(stats.buffer_count, 0);
        assert_eq!(stats.context_count, 0);
        assert_eq!(stats.total_memory, 0);
    }
    #[test]
    fn test_autograd_scope() {
        let scope = AutogradScope::new();
        assert_eq!(scope.total_size(), 0);
        assert!(scope.all_resources_valid());
        let counts = scope.resource_count_by_type();
        assert!(counts.is_empty());
    }
    #[test]
    fn test_computation_graph_manager() {
        let mut manager = ComputationGraphManager::new();
        assert_eq!(manager.node_count(), 0);
        assert_eq!(manager.total_memory_usage(), 0);
        let cleaned = manager
            .cleanup_old_nodes(Duration::from_secs(1))
            .expect("operation should succeed");
        assert_eq!(cleaned, 0);
    }
    #[test]
    fn test_gradient_storage_manager() {
        let mut manager = GradientStorageManager::new();
        assert_eq!(manager.gradient_count(), 0);
        assert_eq!(manager.total_memory_usage(), 0);
        let gradient = vec![1.0, 2.0, 3.0];
        manager
            .set_gradient(1, gradient.clone())
            .expect("operation should succeed");
        assert_eq!(manager.gradient_count(), 1);
        let retrieved = manager
            .get_gradient(1)
            .expect("gradient retrieval should succeed");
        assert_eq!(retrieved, gradient);
        manager
            .release_gradient(1)
            .expect("gradient release should succeed");
        assert_eq!(manager.gradient_count(), 0);
    }
    #[test]
    fn test_memory_buffer_manager() {
        let mut manager = MemoryBufferManager::new();
        assert_eq!(manager.buffer_count(), 0);
        assert_eq!(manager.total_allocated(), 0);
        let cleaned = manager
            .cleanup_old_buffers(Duration::from_secs(1))
            .expect("operation should succeed");
        assert_eq!(cleaned, 0);
    }
    #[test]
    fn test_global_resource_manager() {
        let manager1 = get_global_resource_manager();
        let manager2 = get_global_resource_manager();
        assert!(Arc::ptr_eq(&manager1, &manager2));
    }
}
#[cfg(test)]
mod enhanced_tests {
    use super::super::types::{
        CleanupAction, EnhancedResourceManager, MemoryPressureLevel, MemoryPressureMonitor,
        ResourceLeakDetector,
    };
    use std::time::Duration;
    #[test]
    fn test_resource_leak_detector() {
        let mut detector = ResourceLeakDetector::new();
        let resource_id = detector.register_resource("TestResource", 1024, "test_location");
        assert!(resource_id > 0);
        let leaks = detector.detect_leaks();
        assert!(leaks.is_empty());
        let stats = detector.get_tracking_stats();
        assert_eq!(stats.total_resources, 1);
        assert_eq!(stats.total_memory, 1024);
    }
    #[test]
    fn test_memory_pressure_monitor() {
        let mut monitor = MemoryPressureMonitor::new();
        monitor.check_interval = Duration::from_secs(0);
        let pressure = monitor.check_pressure(1024 * 1024);
        assert_eq!(pressure, MemoryPressureLevel::Low);
        let stats = monitor.get_pressure_stats();
        assert_eq!(stats.readings_count, 1);
    }
    #[test]
    fn test_enhanced_resource_manager() {
        let mut manager = EnhancedResourceManager::new();
        let result = manager.perform_maintenance();
        assert!(result.leaks_detected.is_empty());
        assert_eq!(result.pressure_level, MemoryPressureLevel::Low);
        let stats = manager.get_comprehensive_stats();
        assert_eq!(stats.performance_stats.cleanup_operations, 1);
    }
    #[test]
    fn test_cleanup_action_suggestions() {
        let monitor = MemoryPressureMonitor::new();
        let low_actions = monitor.suggest_cleanup_actions(MemoryPressureLevel::Low);
        assert!(low_actions.is_empty());
        let critical_actions = monitor.suggest_cleanup_actions(MemoryPressureLevel::Critical);
        assert!(critical_actions.contains(&CleanupAction::EmergencyCleanup));
    }
}
