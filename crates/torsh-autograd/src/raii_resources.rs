//! RAII (Resource Acquisition Is Initialization) for autograd resources
//!
//! This module provides automatic resource management for autograd operations,
//! ensuring that computation graphs, gradient storage, memory buffers, and
//! other resources are properly cleaned up when they go out of scope.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

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

/// Statistics about a resource
#[derive(Debug, Clone, Default)]
pub struct ResourceStats {
    pub creation_time: Option<Instant>,
    pub last_access_time: Option<Instant>,
    pub access_count: usize,
    pub memory_usage: usize,
    pub is_active: bool,
}

/// RAII wrapper for computation graph nodes
#[derive(Debug)]
pub struct ComputationGraphGuard {
    node_id: usize,
    graph_manager: Arc<Mutex<ComputationGraphManager>>,
    stats: ResourceStats,
}

impl ComputationGraphGuard {
    /// Create a new computation graph guard
    pub fn new(node_id: usize, graph_manager: Arc<Mutex<ComputationGraphManager>>) -> Self {
        Self {
            node_id,
            graph_manager,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: 0,
                is_active: true,
            },
        }
    }

    /// Get the node ID
    pub fn node_id(&self) -> usize {
        self.node_id
    }

    /// Mark the node as accessed
    pub fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for ComputationGraphGuard {
    fn resource_type(&self) -> &'static str {
        "ComputationGraphNode"
    }

    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.graph_manager.lock() {
            manager.remove_node(self.node_id)?;
            self.stats.is_active = false;
        }
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for ComputationGraphGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup computation graph node {}: {}",
                self.node_id, e
            );
        }
    }
}

/// RAII wrapper for gradient storage
#[derive(Debug)]
pub struct GradientStorageGuard {
    tensor_id: usize,
    storage_size: usize,
    gradient_manager: Arc<Mutex<GradientStorageManager>>,
    stats: ResourceStats,
}

impl GradientStorageGuard {
    /// Create a new gradient storage guard
    pub fn new(
        tensor_id: usize,
        storage_size: usize,
        gradient_manager: Arc<Mutex<GradientStorageManager>>,
    ) -> Self {
        Self {
            tensor_id,
            storage_size,
            gradient_manager,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: storage_size,
                is_active: true,
            },
        }
    }

    /// Get the tensor ID
    pub fn tensor_id(&self) -> usize {
        self.tensor_id
    }

    /// Get gradient data (mock implementation)
    pub fn get_gradient(&mut self) -> Option<Vec<f32>> {
        self.mark_accessed();
        if let Ok(manager) = self.gradient_manager.lock() {
            manager.get_gradient(self.tensor_id)
        } else {
            None
        }
    }

    /// Set gradient data (mock implementation)
    pub fn set_gradient(&mut self, gradient: Vec<f32>) -> AutogradResult<()> {
        self.mark_accessed();
        if let Ok(mut manager) = self.gradient_manager.lock() {
            manager.set_gradient(self.tensor_id, gradient)?;
        }
        Ok(())
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for GradientStorageGuard {
    fn resource_type(&self) -> &'static str {
        "GradientStorage"
    }

    fn resource_size(&self) -> usize {
        self.storage_size
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.gradient_manager.lock() {
            manager.release_gradient(self.tensor_id)?;
            self.stats.is_active = false;
        }
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for GradientStorageGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup gradient storage for tensor {}: {}",
                self.tensor_id, e
            );
        }
    }
}

/// RAII wrapper for memory buffers
#[derive(Debug)]
pub struct MemoryBufferGuard {
    buffer_id: usize,
    size: usize,
    buffer_manager: Arc<Mutex<MemoryBufferManager>>,
    stats: ResourceStats,
}

impl MemoryBufferGuard {
    /// Create a new memory buffer guard
    pub fn new(
        buffer_id: usize,
        size: usize,
        buffer_manager: Arc<Mutex<MemoryBufferManager>>,
    ) -> Self {
        Self {
            buffer_id,
            size,
            buffer_manager,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: size,
                is_active: true,
            },
        }
    }

    /// Get buffer data (mock implementation)
    pub fn as_slice(&mut self) -> Option<&[u8]> {
        self.mark_accessed();
        // In a real implementation, this would return actual buffer data
        None
    }

    /// Get mutable buffer data (mock implementation)
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.mark_accessed();
        // In a real implementation, this would return actual buffer data
        None
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for MemoryBufferGuard {
    fn resource_type(&self) -> &'static str {
        "MemoryBuffer"
    }

    fn resource_size(&self) -> usize {
        self.size
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.buffer_manager.lock() {
            manager.release_buffer(self.buffer_id)?;
            self.stats.is_active = false;
        }
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for MemoryBufferGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!("Failed to cleanup memory buffer {}: {}", self.buffer_id, e);
        }
    }
}

/// RAII wrapper for autograd context
#[derive(Debug)]
pub struct AutogradContextGuard {
    context_id: usize,
    resource_manager: Arc<Mutex<AutogradResourceManager>>,
    stats: ResourceStats,
}

impl AutogradContextGuard {
    /// Create a new autograd context guard
    pub fn new(context_id: usize, resource_manager: Arc<Mutex<AutogradResourceManager>>) -> Self {
        Self {
            context_id,
            resource_manager,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: 0,
                is_active: true,
            },
        }
    }

    /// Enter gradient computation mode
    pub fn enable_grad(&mut self) -> AutogradResult<()> {
        self.mark_accessed();
        if let Ok(mut manager) = self.resource_manager.lock() {
            manager.set_grad_enabled(self.context_id, true)?;
        }
        Ok(())
    }

    /// Exit gradient computation mode
    pub fn disable_grad(&mut self) -> AutogradResult<()> {
        self.mark_accessed();
        if let Ok(mut manager) = self.resource_manager.lock() {
            manager.set_grad_enabled(self.context_id, false)?;
        }
        Ok(())
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for AutogradContextGuard {
    fn resource_type(&self) -> &'static str {
        "AutogradContext"
    }

    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        if let Ok(mut manager) = self.resource_manager.lock() {
            manager.cleanup_context(self.context_id)?;
            self.stats.is_active = false;
        }
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for AutogradContextGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup autograd context {}: {}",
                self.context_id, e
            );
        }
    }
}

/// Manager for computation graph resources
#[derive(Debug)]
pub struct ComputationGraphManager {
    nodes: HashMap<usize, NodeInfo>,
    next_id: usize,
}

#[derive(Debug, Clone)]
struct NodeInfo {
    operation: String,
    memory_usage: usize,
    creation_time: Instant,
}

impl ComputationGraphManager {
    /// Create a new computation graph manager
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create a new node and return its guard
    pub fn create_node(
        &mut self,
        operation: String,
        memory_usage: usize,
        manager_ref: Arc<Mutex<ComputationGraphManager>>,
    ) -> ComputationGraphGuard {
        let node_id = self.next_id;
        self.next_id += 1;

        let node_info = NodeInfo {
            operation,
            memory_usage,
            creation_time: Instant::now(),
        };

        self.nodes.insert(node_id, node_info);

        let mut guard = ComputationGraphGuard::new(node_id, manager_ref);
        guard.stats.memory_usage = memory_usage;
        guard
    }

    /// Remove a node
    pub fn remove_node(&mut self, node_id: usize) -> AutogradResult<()> {
        self.nodes.remove(&node_id);
        Ok(())
    }

    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.nodes.values().map(|n| n.memory_usage).sum()
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Cleanup old nodes
    pub fn cleanup_old_nodes(&mut self, max_age: Duration) -> AutogradResult<usize> {
        let now = Instant::now();
        let initial_count = self.nodes.len();

        self.nodes
            .retain(|_, node| now.duration_since(node.creation_time) < max_age);

        Ok(initial_count - self.nodes.len())
    }
}

impl Default for ComputationGraphManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Manager for gradient storage resources
#[derive(Debug)]
pub struct GradientStorageManager {
    gradients: HashMap<usize, Vec<f32>>,
    total_size: usize,
}

impl GradientStorageManager {
    /// Create a new gradient storage manager
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
            total_size: 0,
        }
    }

    /// Create gradient storage and return its guard
    pub fn create_gradient_storage(
        &mut self,
        tensor_id: usize,
        size: usize,
        manager_ref: Arc<Mutex<GradientStorageManager>>,
    ) -> GradientStorageGuard {
        // Initialize empty gradient storage
        let gradient = vec![0.0; size];
        let storage_size = size * std::mem::size_of::<f32>();
        self.gradients.insert(tensor_id, gradient);
        self.total_size += storage_size;

        GradientStorageGuard::new(tensor_id, storage_size, manager_ref)
    }

    /// Get gradient data
    pub fn get_gradient(&self, tensor_id: usize) -> Option<Vec<f32>> {
        self.gradients.get(&tensor_id).cloned()
    }

    /// Set gradient data
    pub fn set_gradient(&mut self, tensor_id: usize, gradient: Vec<f32>) -> AutogradResult<()> {
        let new_size = gradient.len() * std::mem::size_of::<f32>();

        if let Some(old_gradient) = self.gradients.get(&tensor_id) {
            let old_size = old_gradient.len() * std::mem::size_of::<f32>();
            self.total_size = self.total_size - old_size + new_size;
        } else {
            self.total_size += new_size;
        }

        self.gradients.insert(tensor_id, gradient);
        Ok(())
    }

    /// Release gradient storage
    pub fn release_gradient(&mut self, tensor_id: usize) -> AutogradResult<()> {
        if let Some(gradient) = self.gradients.remove(&tensor_id) {
            let size = gradient.len() * std::mem::size_of::<f32>();
            self.total_size -= size;
        }
        Ok(())
    }

    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.total_size
    }

    /// Get gradient count
    pub fn gradient_count(&self) -> usize {
        self.gradients.len()
    }
}

impl Default for GradientStorageManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Manager for memory buffer resources
#[derive(Debug)]
pub struct MemoryBufferManager {
    buffers: HashMap<usize, BufferInfo>,
    next_id: usize,
    total_allocated: usize,
}

#[derive(Debug)]
struct BufferInfo {
    size: usize,
    creation_time: Instant,
    // In a real implementation, this would contain actual buffer data
}

impl MemoryBufferManager {
    /// Create a new memory buffer manager
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_id: 0,
            total_allocated: 0,
        }
    }

    /// Create a memory buffer and return its guard
    pub fn create_buffer(
        &mut self,
        size: usize,
        manager_ref: Arc<Mutex<MemoryBufferManager>>,
    ) -> AutogradResult<MemoryBufferGuard> {
        let buffer_id = self.next_id;
        self.next_id += 1;

        let buffer_info = BufferInfo {
            size,
            creation_time: Instant::now(),
        };

        self.buffers.insert(buffer_id, buffer_info);
        self.total_allocated += size;

        Ok(MemoryBufferGuard::new(buffer_id, size, manager_ref))
    }

    /// Release a buffer
    pub fn release_buffer(&mut self, buffer_id: usize) -> AutogradResult<()> {
        if let Some(buffer_info) = self.buffers.remove(&buffer_id) {
            self.total_allocated -= buffer_info.size;
        }
        Ok(())
    }

    /// Get total allocated memory
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get buffer count
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Cleanup old buffers
    pub fn cleanup_old_buffers(&mut self, max_age: Duration) -> AutogradResult<usize> {
        let now = Instant::now();
        let initial_count = self.buffers.len();
        let mut freed_memory = 0;

        self.buffers.retain(|_, buffer| {
            let should_keep = now.duration_since(buffer.creation_time) < max_age;
            if !should_keep {
                freed_memory += buffer.size;
            }
            should_keep
        });

        self.total_allocated -= freed_memory;
        Ok(initial_count - self.buffers.len())
    }
}

impl Default for MemoryBufferManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Central resource manager for autograd operations
#[derive(Debug)]
pub struct AutogradResourceManager {
    computation_graph_manager: Arc<Mutex<ComputationGraphManager>>,
    gradient_storage_manager: Arc<Mutex<GradientStorageManager>>,
    memory_buffer_manager: Arc<Mutex<MemoryBufferManager>>,
    contexts: HashMap<usize, ContextInfo>,
    next_context_id: usize,
}

#[derive(Debug, Clone)]
struct ContextInfo {
    grad_enabled: bool,
    creation_time: Instant,
}

impl AutogradResourceManager {
    /// Create a new autograd resource manager
    pub fn new() -> Self {
        Self {
            computation_graph_manager: Arc::new(Mutex::new(ComputationGraphManager::new())),
            gradient_storage_manager: Arc::new(Mutex::new(GradientStorageManager::new())),
            memory_buffer_manager: Arc::new(Mutex::new(MemoryBufferManager::new())),
            contexts: HashMap::new(),
            next_context_id: 0,
        }
    }

    /// Create a new autograd context
    pub fn create_context(
        &mut self,
        manager_ref: Arc<Mutex<AutogradResourceManager>>,
    ) -> AutogradContextGuard {
        let context_id = self.next_context_id;
        self.next_context_id += 1;

        let context_info = ContextInfo {
            grad_enabled: true,
            creation_time: Instant::now(),
        };

        self.contexts.insert(context_id, context_info);
        AutogradContextGuard::new(context_id, manager_ref)
    }

    /// Create a computation graph node
    pub fn create_graph_node(
        &self,
        operation: String,
        memory_usage: usize,
    ) -> AutogradResult<ComputationGraphGuard> {
        let mut manager = self.computation_graph_manager.lock().map_err(|_| {
            AutogradError::gradient_computation(
                "create_graph_node",
                "Failed to lock computation graph manager",
            )
        })?;

        Ok(manager.create_node(
            operation,
            memory_usage,
            self.computation_graph_manager.clone(),
        ))
    }

    /// Create gradient storage
    pub fn create_gradient_storage(
        &self,
        tensor_id: usize,
        size: usize,
    ) -> AutogradResult<GradientStorageGuard> {
        let mut manager = self.gradient_storage_manager.lock().map_err(|_| {
            AutogradError::gradient_computation(
                "create_gradient_storage",
                "Failed to lock gradient storage manager",
            )
        })?;

        Ok(manager.create_gradient_storage(tensor_id, size, self.gradient_storage_manager.clone()))
    }

    /// Create memory buffer
    pub fn create_memory_buffer(&self, size: usize) -> AutogradResult<MemoryBufferGuard> {
        let mut manager = self.memory_buffer_manager.lock().map_err(|_| {
            AutogradError::gradient_computation(
                "create_memory_buffer",
                "Failed to lock memory buffer manager",
            )
        })?;

        manager.create_buffer(size, self.memory_buffer_manager.clone())
    }

    /// Set gradient enabled status for a context
    pub fn set_grad_enabled(&mut self, context_id: usize, enabled: bool) -> AutogradResult<()> {
        if let Some(context_info) = self.contexts.get_mut(&context_id) {
            context_info.grad_enabled = enabled;
            Ok(())
        } else {
            Err(AutogradError::gradient_computation(
                "set_grad_enabled",
                format!("Context {} not found", context_id),
            ))
        }
    }

    /// Cleanup context
    pub fn cleanup_context(&mut self, context_id: usize) -> AutogradResult<()> {
        self.contexts.remove(&context_id);
        Ok(())
    }

    /// Get total resource statistics
    pub fn get_resource_stats(&self) -> ResourceManagerStats {
        let graph_stats = self
            .computation_graph_manager
            .lock()
            .map(|manager| (manager.node_count(), manager.total_memory_usage()))
            .unwrap_or((0, 0));

        let gradient_stats = self
            .gradient_storage_manager
            .lock()
            .map(|manager| (manager.gradient_count(), manager.total_memory_usage()))
            .unwrap_or((0, 0));

        let buffer_stats = self
            .memory_buffer_manager
            .lock()
            .map(|manager| (manager.buffer_count(), manager.total_allocated()))
            .unwrap_or((0, 0));

        ResourceManagerStats {
            graph_nodes: graph_stats.0,
            graph_memory: graph_stats.1,
            gradient_count: gradient_stats.0,
            gradient_memory: gradient_stats.1,
            buffer_count: buffer_stats.0,
            buffer_memory: buffer_stats.1,
            context_count: self.contexts.len(),
            total_memory: graph_stats.1 + gradient_stats.1 + buffer_stats.1,
        }
    }

    /// Cleanup old resources
    pub fn cleanup_old_resources(&self, max_age: Duration) -> AutogradResult<CleanupStats> {
        let graph_cleaned = self
            .computation_graph_manager
            .lock()
            .map_err(|_| {
                AutogradError::gradient_computation(
                    "cleanup_old_resources",
                    "Failed to lock computation graph manager",
                )
            })?
            .cleanup_old_nodes(max_age)?;

        let buffers_cleaned = self
            .memory_buffer_manager
            .lock()
            .map_err(|_| {
                AutogradError::gradient_computation(
                    "cleanup_old_resources",
                    "Failed to lock memory buffer manager",
                )
            })?
            .cleanup_old_buffers(max_age)?;

        Ok(CleanupStats {
            nodes_cleaned: graph_cleaned,
            buffers_cleaned,
        })
    }
}

impl Default for AutogradResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about resource manager state
#[derive(Debug, Clone)]
pub struct ResourceManagerStats {
    pub graph_nodes: usize,
    pub graph_memory: usize,
    pub gradient_count: usize,
    pub gradient_memory: usize,
    pub buffer_count: usize,
    pub buffer_memory: usize,
    pub context_count: usize,
    pub total_memory: usize,
}

/// Statistics about cleanup operations
#[derive(Debug, Clone)]
pub struct CleanupStats {
    pub nodes_cleaned: usize,
    pub buffers_cleaned: usize,
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

/// Scoped RAII helper for multiple resources
pub struct AutogradScope {
    resources: Vec<Box<dyn AutogradResource>>,
    start_time: Instant,
}

impl AutogradScope {
    /// Create a new autograd scope
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Add a resource to this scope
    pub fn add_resource(&mut self, resource: Box<dyn AutogradResource>) {
        self.resources.push(resource);
    }

    /// Get scope duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get total resource size
    pub fn total_size(&self) -> usize {
        self.resources.iter().map(|r| r.resource_size()).sum()
    }

    /// Get resource count by type
    pub fn resource_count_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for resource in &self.resources {
            *counts
                .entry(resource.resource_type().to_string())
                .or_insert(0) += 1;
        }
        counts
    }

    /// Check if all resources are valid
    pub fn all_resources_valid(&self) -> bool {
        self.resources.iter().all(|r| r.is_valid())
    }
}

impl Default for AutogradScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutogradScope {
    fn drop(&mut self) {
        // Resources are automatically cleaned up when their individual Drop implementations are called
        if !self.resources.is_empty() {
            println!(
                "AutogradScope dropping {} resources after {:?}",
                self.resources.len(),
                self.duration()
            );
        }
    }
}

/// Convenience macro for creating RAII scoped autograd operations
#[macro_export]
macro_rules! autograd_scope {
    ($scope_name:ident, $body:block) => {
        let mut $scope_name = $crate::raii_resources::AutogradScope::new();
        $body
    };
}

/// RAII wrapper for tensors requiring gradient computation
#[derive(Debug)]
pub struct TensorGradGuard {
    tensor_id: usize,
    gradient_enabled: bool,
    requires_grad_original: bool,
    stats: ResourceStats,
}

impl TensorGradGuard {
    /// Create a new tensor gradient guard
    pub fn new(tensor_id: usize, requires_grad: bool) -> Self {
        Self {
            tensor_id,
            gradient_enabled: requires_grad,
            requires_grad_original: requires_grad,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: 0, // Will be updated based on gradient size
                is_active: true,
            },
        }
    }

    /// Enable gradient computation for this tensor
    pub fn enable_grad(&mut self) {
        self.gradient_enabled = true;
        self.mark_accessed();
    }

    /// Disable gradient computation for this tensor
    pub fn disable_grad(&mut self) {
        self.gradient_enabled = false;
        self.mark_accessed();
    }

    /// Check if gradients are enabled
    pub fn requires_grad(&self) -> bool {
        self.gradient_enabled
    }

    /// Get tensor ID
    pub fn tensor_id(&self) -> usize {
        self.tensor_id
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for TensorGradGuard {
    fn resource_type(&self) -> &'static str {
        "TensorGradient"
    }

    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        // Restore original gradient state
        self.gradient_enabled = self.requires_grad_original;
        self.stats.is_active = false;
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for TensorGradGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup tensor gradient guard for tensor {}: {}",
                self.tensor_id, e
            );
        }
    }
}

/// RAII wrapper for gradient checkpointing sessions
#[derive(Debug)]
pub struct CheckpointGuard {
    checkpoint_id: usize,
    checkpoint_data: Option<Vec<u8>>,
    memory_usage: usize,
    stats: ResourceStats,
}

impl CheckpointGuard {
    /// Create a new checkpoint guard
    pub fn new(checkpoint_id: usize, data: Vec<u8>) -> Self {
        let memory_usage = data.len();
        Self {
            checkpoint_id,
            checkpoint_data: Some(data),
            memory_usage,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage,
                is_active: true,
            },
        }
    }

    /// Get checkpoint data
    pub fn get_data(&mut self) -> Option<&Vec<u8>> {
        self.mark_accessed();
        self.checkpoint_data.as_ref()
    }

    /// Release checkpoint data early to free memory
    pub fn release_data(&mut self) {
        self.checkpoint_data = None;
        self.memory_usage = 0;
        self.stats.memory_usage = 0;
        self.mark_accessed();
    }

    /// Get checkpoint ID
    pub fn checkpoint_id(&self) -> usize {
        self.checkpoint_id
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for CheckpointGuard {
    fn resource_type(&self) -> &'static str {
        "GradientCheckpoint"
    }

    fn resource_size(&self) -> usize {
        self.memory_usage
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        self.checkpoint_data = None;
        self.memory_usage = 0;
        self.stats.is_active = false;
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for CheckpointGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup checkpoint guard {}: {}",
                self.checkpoint_id, e
            );
        }
    }
}

/// RAII wrapper for distributed training contexts
#[derive(Debug)]
pub struct DistributedContextGuard {
    context_id: usize,
    rank: i32,
    world_size: i32,
    communication_buffers: Vec<Vec<u8>>,
    is_coordinator: bool,
    stats: ResourceStats,
}

impl DistributedContextGuard {
    /// Create a new distributed context guard
    pub fn new(context_id: usize, rank: i32, world_size: i32, is_coordinator: bool) -> Self {
        Self {
            context_id,
            rank,
            world_size,
            communication_buffers: Vec::new(),
            is_coordinator,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: 0,
                is_active: true,
            },
        }
    }

    /// Add a communication buffer
    pub fn add_communication_buffer(&mut self, buffer: Vec<u8>) {
        self.stats.memory_usage += buffer.len();
        self.communication_buffers.push(buffer);
        self.mark_accessed();
    }

    /// Get rank
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get world size
    pub fn world_size(&self) -> i32 {
        self.world_size
    }

    /// Check if this is the coordinator
    pub fn is_coordinator(&self) -> bool {
        self.is_coordinator
    }

    /// Clean up all communication buffers
    pub fn cleanup_buffers(&mut self) {
        self.communication_buffers.clear();
        self.stats.memory_usage = 0;
        self.mark_accessed();
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for DistributedContextGuard {
    fn resource_type(&self) -> &'static str {
        "DistributedContext"
    }

    fn resource_size(&self) -> usize {
        self.stats.memory_usage
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        self.communication_buffers.clear();
        self.stats.memory_usage = 0;
        self.stats.is_active = false;
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for DistributedContextGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup distributed context guard {}: {}",
                self.context_id, e
            );
        }
    }
}

/// RAII wrapper for profiling sessions
#[derive(Debug)]
pub struct ProfileSessionGuard {
    session_id: usize,
    session_name: String,
    start_time: Instant,
    is_active: AtomicBool,
    collected_samples: AtomicUsize,
    stats: ResourceStats,
}

impl ProfileSessionGuard {
    /// Create a new profile session guard
    pub fn new(session_id: usize, session_name: String) -> Self {
        Self {
            session_id,
            session_name,
            start_time: Instant::now(),
            is_active: AtomicBool::new(true),
            collected_samples: AtomicUsize::new(0),
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: 0,
                is_active: true,
            },
        }
    }

    /// Record a profiling sample
    pub fn record_sample(&mut self) {
        if self.is_active.load(Ordering::Relaxed) {
            self.collected_samples.fetch_add(1, Ordering::Relaxed);
            self.mark_accessed();
        }
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get number of collected samples
    pub fn sample_count(&self) -> usize {
        self.collected_samples.load(Ordering::Relaxed)
    }

    /// Stop profiling session
    pub fn stop_session(&mut self) {
        self.is_active.store(false, Ordering::Relaxed);
        self.mark_accessed();
    }

    /// Check if session is active
    pub fn is_session_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for ProfileSessionGuard {
    fn resource_type(&self) -> &'static str {
        "ProfileSession"
    }

    fn resource_size(&self) -> usize {
        self.stats.memory_usage + self.collected_samples.load(Ordering::Relaxed) * 64
        // Estimate 64 bytes per sample
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        self.is_active.store(false, Ordering::Relaxed);
        self.stats.is_active = false;
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for ProfileSessionGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup profile session guard '{}': {}",
                self.session_name, e
            );
        }
    }
}

/// RAII wrapper for thread-local variable environments
#[derive(Debug)]
pub struct VariableEnvironmentGuard {
    thread_id: ThreadId,
    environment_id: usize,
    variables_count: usize,
    memory_usage: usize,
    stats: ResourceStats,
}

impl VariableEnvironmentGuard {
    /// Create a new variable environment guard
    pub fn new(environment_id: usize) -> Self {
        Self {
            thread_id: thread::current().id(),
            environment_id,
            variables_count: 0,
            memory_usage: 0,
            stats: ResourceStats {
                creation_time: Some(Instant::now()),
                last_access_time: Some(Instant::now()),
                access_count: 0,
                memory_usage: 0,
                is_active: true,
            },
        }
    }

    /// Register a variable in the environment
    pub fn register_variable(&mut self, variable_size: usize) {
        self.variables_count += 1;
        self.memory_usage += variable_size;
        self.stats.memory_usage = self.memory_usage;
        self.mark_accessed();
    }

    /// Unregister a variable from the environment
    pub fn unregister_variable(&mut self, variable_size: usize) {
        self.variables_count = self.variables_count.saturating_sub(1);
        self.memory_usage = self.memory_usage.saturating_sub(variable_size);
        self.stats.memory_usage = self.memory_usage;
        self.mark_accessed();
    }

    /// Get number of variables in environment
    pub fn variable_count(&self) -> usize {
        self.variables_count
    }

    /// Get thread ID this environment belongs to
    pub fn thread_id(&self) -> ThreadId {
        self.thread_id
    }

    fn mark_accessed(&mut self) {
        self.stats.last_access_time = Some(Instant::now());
        self.stats.access_count += 1;
    }
}

impl AutogradResource for VariableEnvironmentGuard {
    fn resource_type(&self) -> &'static str {
        "VariableEnvironment"
    }

    fn resource_size(&self) -> usize {
        self.memory_usage
    }

    fn cleanup(&mut self) -> AutogradResult<()> {
        self.variables_count = 0;
        self.memory_usage = 0;
        self.stats.memory_usage = 0;
        self.stats.is_active = false;
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.stats.is_active
    }

    fn get_stats(&self) -> ResourceStats {
        self.stats.clone()
    }
}

impl Drop for VariableEnvironmentGuard {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!(
                "Failed to cleanup variable environment guard {}: {}",
                self.environment_id, e
            );
        }
    }
}

/// Comprehensive RAII factory for creating and managing autograd resource guards
#[derive(Debug, Default)]
pub struct AutogradResourceFactory {
    next_id: AtomicUsize,
}

impl AutogradResourceFactory {
    /// Create a new factory
    pub fn new() -> Self {
        Self {
            next_id: AtomicUsize::new(1),
        }
    }

    /// Create a tensor gradient guard
    pub fn create_tensor_grad_guard(&self, requires_grad: bool) -> TensorGradGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        TensorGradGuard::new(id, requires_grad)
    }

    /// Create a checkpoint guard
    pub fn create_checkpoint_guard(&self, data: Vec<u8>) -> CheckpointGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        CheckpointGuard::new(id, data)
    }

    /// Create a distributed context guard
    pub fn create_distributed_context_guard(
        &self,
        rank: i32,
        world_size: i32,
        is_coordinator: bool,
    ) -> DistributedContextGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        DistributedContextGuard::new(id, rank, world_size, is_coordinator)
    }

    /// Create a profile session guard
    pub fn create_profile_session_guard(&self, session_name: String) -> ProfileSessionGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        ProfileSessionGuard::new(id, session_name)
    }

    /// Create a variable environment guard
    pub fn create_variable_environment_guard(&self) -> VariableEnvironmentGuard {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        VariableEnvironmentGuard::new(id)
    }
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
    use super::*;

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

        // Test cleanup of old nodes
        let cleaned = manager.cleanup_old_nodes(Duration::from_secs(1)).unwrap();
        assert_eq!(cleaned, 0);
    }

    #[test]
    fn test_gradient_storage_manager() {
        let mut manager = GradientStorageManager::new();
        assert_eq!(manager.gradient_count(), 0);
        assert_eq!(manager.total_memory_usage(), 0);

        // Test setting and getting gradients
        let gradient = vec![1.0, 2.0, 3.0];
        manager.set_gradient(1, gradient.clone()).unwrap();
        assert_eq!(manager.gradient_count(), 1);

        let retrieved = manager.get_gradient(1).unwrap();
        assert_eq!(retrieved, gradient);

        manager.release_gradient(1).unwrap();
        assert_eq!(manager.gradient_count(), 0);
    }

    #[test]
    fn test_memory_buffer_manager() {
        let mut manager = MemoryBufferManager::new();
        assert_eq!(manager.buffer_count(), 0);
        assert_eq!(manager.total_allocated(), 0);

        // Test cleanup of old buffers
        let cleaned = manager.cleanup_old_buffers(Duration::from_secs(1)).unwrap();
        assert_eq!(cleaned, 0);
    }

    #[test]
    fn test_global_resource_manager() {
        let manager1 = get_global_resource_manager();
        let manager2 = get_global_resource_manager();

        // Should return the same instance
        assert!(Arc::ptr_eq(&manager1, &manager2));
    }
}

// Additional enhancements for advanced resource management

/// Advanced resource leak detector
#[derive(Debug)]
pub struct ResourceLeakDetector {
    tracking_enabled: bool,
    resource_history: HashMap<usize, ResourceTrackingInfo>,
    leak_threshold: Duration,
    next_resource_id: usize,
}

#[derive(Debug, Clone)]
struct ResourceTrackingInfo {
    resource_type: String,
    creation_time: Instant,
    creation_location: String,
    size: usize,
    last_access: Instant,
}

impl ResourceLeakDetector {
    /// Create a new resource leak detector
    pub fn new() -> Self {
        Self {
            tracking_enabled: true,
            resource_history: HashMap::new(),
            leak_threshold: Duration::from_secs(300), // 5 minutes
            next_resource_id: 1,
        }
    }

    /// Register a new resource for tracking
    pub fn register_resource(&mut self, resource_type: &str, size: usize, location: &str) -> usize {
        if !self.tracking_enabled {
            return 0;
        }

        let resource_id = self.next_resource_id;
        self.next_resource_id += 1;

        let tracking_info = ResourceTrackingInfo {
            resource_type: resource_type.to_string(),
            creation_time: Instant::now(),
            creation_location: location.to_string(),
            size,
            last_access: Instant::now(),
        };

        self.resource_history.insert(resource_id, tracking_info);
        resource_id
    }

    /// Unregister a resource when it's cleaned up
    pub fn unregister_resource(&mut self, resource_id: usize) {
        self.resource_history.remove(&resource_id);
    }

    /// Record access to a resource
    pub fn record_access(&mut self, resource_id: usize) {
        if let Some(info) = self.resource_history.get_mut(&resource_id) {
            info.last_access = Instant::now();
        }
    }

    /// Detect potential resource leaks
    pub fn detect_leaks(&self) -> Vec<ResourceLeak> {
        let now = Instant::now();
        let mut leaks = Vec::new();

        for (resource_id, info) in &self.resource_history {
            let age = now.duration_since(info.creation_time);
            let idle_time = now.duration_since(info.last_access);

            if age > self.leak_threshold && idle_time > self.leak_threshold {
                leaks.push(ResourceLeak {
                    resource_id: *resource_id,
                    resource_type: info.resource_type.clone(),
                    age,
                    idle_time,
                    size: info.size,
                    creation_location: info.creation_location.clone(),
                });
            }
        }

        leaks
    }

    /// Get statistics about tracked resources
    pub fn get_tracking_stats(&self) -> ResourceTrackingStats {
        let total_resources = self.resource_history.len();
        let total_memory: usize = self.resource_history.values().map(|info| info.size).sum();

        let mut type_counts = HashMap::new();
        for info in self.resource_history.values() {
            *type_counts.entry(info.resource_type.clone()).or_insert(0) += 1;
        }

        ResourceTrackingStats {
            total_resources,
            total_memory,
            type_counts,
            leak_threshold: self.leak_threshold,
        }
    }

    /// Enable or disable resource tracking
    pub fn set_tracking_enabled(&mut self, enabled: bool) {
        self.tracking_enabled = enabled;
        if !enabled {
            self.resource_history.clear();
        }
    }

    /// Set the threshold for considering a resource as potentially leaked
    pub fn set_leak_threshold(&mut self, threshold: Duration) {
        self.leak_threshold = threshold;
    }
}

impl Default for ResourceLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a detected resource leak
#[derive(Debug, Clone)]
pub struct ResourceLeak {
    pub resource_id: usize,
    pub resource_type: String,
    pub age: Duration,
    pub idle_time: Duration,
    pub size: usize,
    pub creation_location: String,
}

/// Statistics about resource tracking
#[derive(Debug, Clone)]
pub struct ResourceTrackingStats {
    pub total_resources: usize,
    pub total_memory: usize,
    pub type_counts: HashMap<String, usize>,
    pub leak_threshold: Duration,
}

/// Memory pressure monitor for autograd resources
#[derive(Debug)]
pub struct MemoryPressureMonitor {
    pressure_threshold: f64, // 0.0 to 1.0
    warning_threshold: f64,  // 0.0 to 1.0
    critical_threshold: f64, // 0.0 to 1.0
    pub check_interval: Duration,
    last_check: Instant,
    pressure_history: Vec<MemoryPressureReading>,
    max_history_size: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryPressureReading {
    timestamp: Instant,
    pressure_level: f64,
    total_memory: usize,
    available_memory: usize,
    autograd_memory: usize,
}

impl MemoryPressureMonitor {
    /// Create a new memory pressure monitor
    pub fn new() -> Self {
        Self {
            pressure_threshold: 0.8,  // 80% memory usage
            warning_threshold: 0.85,  // 85% warning
            critical_threshold: 0.95, // 95% critical
            check_interval: Duration::from_secs(5),
            last_check: Instant::now(),
            pressure_history: Vec::new(),
            max_history_size: 100,
        }
    }

    /// Check current memory pressure
    pub fn check_pressure(&mut self, autograd_memory: usize) -> MemoryPressureLevel {
        let now = Instant::now();
        if now.duration_since(self.last_check) < self.check_interval {
            return self.get_current_pressure_level();
        }

        self.last_check = now;

        // Get system memory info (simplified for this implementation)
        let (total_memory, available_memory) = self.get_system_memory_info();
        let used_memory = total_memory - available_memory;
        let pressure_level = used_memory as f64 / total_memory as f64;

        let reading = MemoryPressureReading {
            timestamp: now,
            pressure_level,
            total_memory,
            available_memory,
            autograd_memory,
        };

        self.pressure_history.push(reading);
        if self.pressure_history.len() > self.max_history_size {
            self.pressure_history.remove(0);
        }

        self.classify_pressure_level(pressure_level)
    }

    /// Get current pressure level without performing a new check
    pub fn get_current_pressure_level(&self) -> MemoryPressureLevel {
        if let Some(last_reading) = self.pressure_history.last() {
            self.classify_pressure_level(last_reading.pressure_level)
        } else {
            MemoryPressureLevel::Low
        }
    }

    /// Classify pressure level based on thresholds
    fn classify_pressure_level(&self, pressure: f64) -> MemoryPressureLevel {
        if pressure >= self.critical_threshold {
            MemoryPressureLevel::Critical
        } else if pressure >= self.warning_threshold {
            MemoryPressureLevel::High
        } else if pressure >= self.pressure_threshold {
            MemoryPressureLevel::Medium
        } else {
            MemoryPressureLevel::Low
        }
    }

    /// Get simplified system memory info
    fn get_system_memory_info(&self) -> (usize, usize) {
        // Simplified implementation - in production this would use system APIs
        let total_memory = 8 * 1024 * 1024 * 1024; // 8GB
        let available_memory = total_memory / 2; // Assume 50% available
        (total_memory, available_memory)
    }

    /// Get memory pressure statistics
    pub fn get_pressure_stats(&self) -> MemoryPressureStats {
        if self.pressure_history.is_empty() {
            return MemoryPressureStats::default();
        }

        let current_pressure = self.pressure_history.last().unwrap().pressure_level;
        let average_pressure = self
            .pressure_history
            .iter()
            .map(|r| r.pressure_level)
            .sum::<f64>()
            / self.pressure_history.len() as f64;

        let max_pressure = self
            .pressure_history
            .iter()
            .map(|r| r.pressure_level)
            .fold(0.0f64, f64::max);

        let critical_events = self
            .pressure_history
            .iter()
            .filter(|r| r.pressure_level >= self.critical_threshold)
            .count();

        MemoryPressureStats {
            current_pressure,
            average_pressure,
            max_pressure,
            critical_events,
            readings_count: self.pressure_history.len(),
            pressure_threshold: self.pressure_threshold,
            warning_threshold: self.warning_threshold,
            critical_threshold: self.critical_threshold,
        }
    }

    /// Suggest cleanup actions based on pressure level
    pub fn suggest_cleanup_actions(
        &self,
        pressure_level: MemoryPressureLevel,
    ) -> Vec<CleanupAction> {
        match pressure_level {
            MemoryPressureLevel::Low => vec![],
            MemoryPressureLevel::Medium => vec![
                CleanupAction::RunGarbageCollection,
                CleanupAction::CleanOldGradients,
            ],
            MemoryPressureLevel::High => vec![
                CleanupAction::RunGarbageCollection,
                CleanupAction::CleanOldGradients,
                CleanupAction::CompactMemoryBuffers,
                CleanupAction::ReduceCheckpointFrequency,
            ],
            MemoryPressureLevel::Critical => vec![
                CleanupAction::EmergencyCleanup,
                CleanupAction::RunGarbageCollection,
                CleanupAction::CleanOldGradients,
                CleanupAction::CompactMemoryBuffers,
                CleanupAction::ReduceCheckpointFrequency,
                CleanupAction::DropOldComputationNodes,
            ],
        }
    }
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressureLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Memory pressure statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryPressureStats {
    pub current_pressure: f64,
    pub average_pressure: f64,
    pub max_pressure: f64,
    pub critical_events: usize,
    pub readings_count: usize,
    pub pressure_threshold: f64,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
}

/// Suggested cleanup actions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanupAction {
    RunGarbageCollection,
    CleanOldGradients,
    CompactMemoryBuffers,
    ReduceCheckpointFrequency,
    DropOldComputationNodes,
    EmergencyCleanup,
}

/// Enhanced resource manager with leak detection and pressure monitoring
#[derive(Debug)]
pub struct EnhancedResourceManager {
    base_manager: AutogradResourceManager,
    leak_detector: ResourceLeakDetector,
    pressure_monitor: MemoryPressureMonitor,
    auto_cleanup_enabled: bool,
    performance_stats: PerformanceStats,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub cleanup_operations: usize,
    pub resources_cleaned: usize,
    pub memory_freed: usize,
    pub leak_detections: usize,
    pub pressure_checks: usize,
}

impl EnhancedResourceManager {
    /// Create a new enhanced resource manager
    pub fn new() -> Self {
        Self {
            base_manager: AutogradResourceManager::new(),
            leak_detector: ResourceLeakDetector::new(),
            pressure_monitor: MemoryPressureMonitor::new(),
            auto_cleanup_enabled: true,
            performance_stats: PerformanceStats::default(),
        }
    }

    /// Register a resource with leak detection
    pub fn register_resource_tracked<T: AutogradResource>(
        &mut self,
        resource: T,
        location: &str,
    ) -> usize {
        let resource_id = self.leak_detector.register_resource(
            resource.resource_type(),
            resource.resource_size(),
            location,
        );

        // Note: Base manager registration would be handled separately
        // in a real implementation based on resource type

        resource_id
    }

    /// Perform comprehensive resource maintenance
    pub fn perform_maintenance(&mut self) -> MaintenanceResult {
        self.performance_stats.cleanup_operations += 1;

        // Check for leaks
        let leaks = self.leak_detector.detect_leaks();
        self.performance_stats.leak_detections += leaks.len();

        // Check memory pressure (simplified - use stats from base manager)
        let stats = self.base_manager.get_resource_stats();
        let autograd_memory = stats.total_memory;
        let pressure_level = self.pressure_monitor.check_pressure(autograd_memory);
        self.performance_stats.pressure_checks += 1;

        // Perform cleanup based on pressure and leaks
        let mut actions_taken = Vec::new();
        let mut memory_freed = 0usize;

        if self.auto_cleanup_enabled {
            let suggested_actions = self
                .pressure_monitor
                .suggest_cleanup_actions(pressure_level);
            for action in &suggested_actions {
                match self.perform_cleanup_action(action) {
                    Ok(freed) => {
                        memory_freed += freed;
                        actions_taken.push(action.clone());
                    }
                    Err(_) => {} // Log error in production
                }
            }
        }

        self.performance_stats.memory_freed += memory_freed;
        self.performance_stats.resources_cleaned += actions_taken.len();

        MaintenanceResult {
            leaks_detected: leaks,
            pressure_level,
            actions_taken,
            memory_freed,
        }
    }

    /// Perform a specific cleanup action
    fn perform_cleanup_action(&mut self, action: &CleanupAction) -> AutogradResult<usize> {
        match action {
            CleanupAction::RunGarbageCollection => {
                // Use the available cleanup_old_resources method
                let stats = self
                    .base_manager
                    .cleanup_old_resources(Duration::from_secs(60))?;
                // Estimate memory freed based on nodes and buffers cleaned
                Ok(stats.nodes_cleaned * 1024 + stats.buffers_cleaned * 512)
            }
            CleanupAction::CleanOldGradients => {
                // Use the available cleanup_old_resources method with shorter duration
                let stats = self
                    .base_manager
                    .cleanup_old_resources(Duration::from_secs(30))?;
                // Estimate memory freed based on cleaned resources
                Ok(stats.nodes_cleaned * 1024 + stats.buffers_cleaned * 512)
            }
            CleanupAction::CompactMemoryBuffers => {
                // Simulate buffer compaction (in real implementation this would compact buffers)
                Ok(1024) // Return simulated freed memory
            }
            CleanupAction::ReduceCheckpointFrequency => {
                // This would typically reduce checkpoint frequency in the scheduler
                Ok(0)
            }
            CleanupAction::DropOldComputationNodes => {
                // Use the available cleanup_old_resources method with longer duration
                let stats = self
                    .base_manager
                    .cleanup_old_resources(Duration::from_secs(120))?;
                // Estimate memory freed based on nodes cleaned
                Ok(stats.nodes_cleaned * 1024 + stats.buffers_cleaned * 512)
            }
            CleanupAction::EmergencyCleanup => {
                // Perform aggressive cleanup using available methods
                let stats = self
                    .base_manager
                    .cleanup_old_resources(Duration::from_secs(10))?;
                // Estimate memory freed based on emergency cleanup
                Ok(stats.nodes_cleaned * 2048 + stats.buffers_cleaned * 1024) // Higher estimates for emergency
            }
        }
    }

    /// Enable or disable automatic cleanup
    pub fn set_auto_cleanup(&mut self, enabled: bool) {
        self.auto_cleanup_enabled = enabled;
    }

    /// Get comprehensive resource statistics
    pub fn get_comprehensive_stats(&self) -> ComprehensiveResourceStats {
        ComprehensiveResourceStats {
            base_stats: self.base_manager.get_resource_stats(),
            tracking_stats: self.leak_detector.get_tracking_stats(),
            pressure_stats: self.pressure_monitor.get_pressure_stats(),
            performance_stats: self.performance_stats.clone(),
        }
    }
}

impl Default for EnhancedResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of maintenance operation
#[derive(Debug, Clone)]
pub struct MaintenanceResult {
    pub leaks_detected: Vec<ResourceLeak>,
    pub pressure_level: MemoryPressureLevel,
    pub actions_taken: Vec<CleanupAction>,
    pub memory_freed: usize,
}

/// Comprehensive resource statistics
#[derive(Debug, Clone)]
pub struct ComprehensiveResourceStats {
    pub base_stats: ResourceManagerStats,
    pub tracking_stats: ResourceTrackingStats,
    pub pressure_stats: MemoryPressureStats,
    pub performance_stats: PerformanceStats,
}

#[cfg(test)]
mod enhanced_tests {
    use super::*;

    #[test]
    fn test_resource_leak_detector() {
        let mut detector = ResourceLeakDetector::new();

        // Register a resource
        let resource_id = detector.register_resource("TestResource", 1024, "test_location");
        assert!(resource_id > 0);

        // Should not detect leak immediately
        let leaks = detector.detect_leaks();
        assert!(leaks.is_empty());

        // Get stats
        let stats = detector.get_tracking_stats();
        assert_eq!(stats.total_resources, 1);
        assert_eq!(stats.total_memory, 1024);
    }

    #[test]
    fn test_memory_pressure_monitor() {
        let mut monitor = MemoryPressureMonitor::new();

        // Force a reading by setting check interval to 0
        monitor.check_interval = Duration::from_secs(0);

        // Check pressure with low memory usage
        let pressure = monitor.check_pressure(1024 * 1024); // 1MB
        assert_eq!(pressure, MemoryPressureLevel::Low);

        // Get stats
        let stats = monitor.get_pressure_stats();
        assert_eq!(stats.readings_count, 1);
    }

    #[test]
    fn test_enhanced_resource_manager() {
        let mut manager = EnhancedResourceManager::new();

        // Perform maintenance
        let result = manager.perform_maintenance();
        assert!(result.leaks_detected.is_empty());
        assert_eq!(result.pressure_level, MemoryPressureLevel::Low);

        // Get comprehensive stats
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
