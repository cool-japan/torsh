//! CUDA Cooperative Groups implementation for enhanced thread cooperation
//!
//! This module provides advanced thread cooperation mechanisms using CUDA Cooperative Groups,
//! enabling more efficient synchronization and communication patterns within GPU kernels.

use crate::error::{BackendError, BackendResult};
use cust::stream::Stream;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

/// CUDA Cooperative Groups capability detection
#[derive(Debug, Clone)]
pub struct CooperativeGroupsCapabilities {
    /// Whether cooperative groups are supported
    pub supported: bool,
    /// Maximum number of thread blocks that can cooperate
    pub max_cooperative_blocks: u32,
    /// Whether grid-wide synchronization is supported
    pub grid_sync_supported: bool,
    /// Whether cluster groups are supported (compute capability 9.0+)
    pub cluster_groups_supported: bool,
    /// Maximum cluster size
    pub max_cluster_size: u32,
    /// Whether device-wide barriers are supported
    pub device_barriers_supported: bool,
}

/// Types of cooperative groups
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CooperativeGroupType {
    /// Thread block group - all threads in a block
    ThreadBlock,
    /// Warp group - all threads in a warp (32 threads)
    Warp,
    /// Grid group - all thread blocks in a grid (requires cooperative launch)
    Grid,
    /// Cluster group - group of thread blocks (compute capability 9.0+)
    Cluster,
    /// Custom sized group within a warp
    CoalescedThreads,
}

/// Cooperative group descriptor
#[derive(Debug, Clone)]
pub struct CooperativeGroupDescriptor {
    /// Type of cooperative group
    pub group_type: CooperativeGroupType,
    /// Group size (for custom groups)
    pub size: Option<u32>,
    /// Thread mask (for coalesced thread groups)
    pub thread_mask: Option<u32>,
    /// Synchronization requirements
    pub sync_requirements: SynchronizationRequirements,
}

/// Synchronization requirements for cooperative groups
#[derive(Debug, Clone)]
pub struct SynchronizationRequirements {
    /// Whether barrier synchronization is needed
    pub needs_barrier: bool,
    /// Whether memory fence is needed
    pub needs_memory_fence: bool,
    /// Memory scope for synchronization
    pub memory_scope: MemoryScope,
    /// Synchronization frequency (high/medium/low)
    pub sync_frequency: SyncFrequency,
}

/// Memory scope for synchronization operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryScope {
    /// Thread scope (no synchronization)
    Thread,
    /// Warp scope (within warp)
    Warp,
    /// Block scope (within thread block)
    Block,
    /// Grid scope (across grid)
    Grid,
    /// Device scope (device-wide)
    Device,
    /// System scope (across all devices)
    System,
}

/// Synchronization frequency hints
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncFrequency {
    /// High frequency synchronization (every few operations)
    High,
    /// Medium frequency synchronization (periodic)
    Medium,
    /// Low frequency synchronization (rare)
    Low,
}

/// Cooperative kernel launch configuration
#[derive(Debug, Clone)]
pub struct CooperativeKernelConfig {
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Shared memory size per block
    pub shared_memory_size: usize,
    /// Stream for kernel execution
    pub stream: Option<Stream>,
    /// Whether to use grid-wide cooperation
    pub grid_cooperation: bool,
    /// Cluster dimensions (for cluster groups)
    pub cluster_dim: Option<(u32, u32, u32)>,
    /// Cooperative group descriptors
    pub cooperative_groups: Vec<CooperativeGroupDescriptor>,
}

/// Cooperative groups context for managing cooperative operations
pub struct CooperativeGroupsContext {
    /// Device capabilities
    capabilities: CooperativeGroupsCapabilities,
    /// Active cooperative kernels
    active_kernels: Arc<Mutex<HashMap<u64, CooperativeKernelState>>>,
    /// Performance statistics
    performance_stats: Arc<Mutex<CooperativeGroupsStats>>,
    /// Next kernel ID
    next_kernel_id: Arc<Mutex<u64>>,
}

/// State tracking for cooperative kernels
#[derive(Debug)]
struct CooperativeKernelState {
    /// Kernel ID
    kernel_id: u64,
    /// Configuration used
    config: CooperativeKernelConfig,
    /// Launch timestamp
    launched_at: std::time::Instant,
    /// Number of synchronization events
    sync_events: u32,
    /// Memory usage
    memory_usage: usize,
    /// Performance metrics
    performance_metrics: KernelPerformanceMetrics,
}

/// Performance metrics for cooperative kernels
#[derive(Debug, Default)]
pub struct KernelPerformanceMetrics {
    /// Total execution time
    pub execution_time_us: u64,
    /// Number of barrier synchronizations
    pub barrier_syncs: u32,
    /// Number of memory fence operations
    pub memory_fences: u32,
    /// Average synchronization overhead
    pub sync_overhead_us: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f32,
    /// Compute utilization
    pub compute_utilization: f32,
    /// Warp efficiency
    pub warp_efficiency: f32,
}

/// Performance statistics for cooperative groups
#[derive(Debug, Default)]
pub struct CooperativeGroupsStats {
    /// Total kernels launched
    pub total_kernels_launched: u64,
    /// Total grid-cooperative kernels
    pub grid_cooperative_kernels: u64,
    /// Total cluster-cooperative kernels
    pub cluster_cooperative_kernels: u64,
    /// Average kernel execution time
    pub avg_kernel_execution_time_us: f64,
    /// Total synchronization events
    pub total_sync_events: u64,
    /// Average synchronization overhead
    pub avg_sync_overhead_us: f64,
    /// Memory efficiency metrics
    pub memory_efficiency: MemoryEfficiencyStats,
}

/// Memory efficiency statistics
#[derive(Debug, Default)]
pub struct MemoryEfficiencyStats {
    /// Average memory bandwidth utilization
    pub avg_bandwidth_utilization: f32,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Memory access patterns efficiency
    pub access_patterns_efficiency: f32,
    /// Shared memory bank conflicts
    pub bank_conflicts_per_kernel: f32,
}

impl CooperativeGroupsContext {
    /// Create a new cooperative groups context
    pub fn new(device_id: usize) -> BackendResult<Self> {
        let capabilities = Self::detect_capabilities(device_id)?;

        Ok(Self {
            capabilities,
            active_kernels: Arc::new(Mutex::new(HashMap::new())),
            performance_stats: Arc::new(Mutex::new(CooperativeGroupsStats::default())),
            next_kernel_id: Arc::new(Mutex::new(0)),
        })
    }

    /// Detect cooperative groups capabilities for a device
    fn detect_capabilities(device_id: usize) -> BackendResult<CooperativeGroupsCapabilities> {
        use cust::device::Device;

        let device = Device::get_device(device_id as u32).map_err(|e| {
            BackendError::InitializationError(format!("Failed to get device: {}", e))
        })?;

        // Check compute capability
        let major = device
            .get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMajor)
            .map_err(|e| {
                BackendError::InitializationError(format!(
                    "Failed to get compute capability: {}",
                    e
                ))
            })?;
        let minor = device
            .get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMinor)
            .map_err(|e| {
                BackendError::InitializationError(format!(
                    "Failed to get compute capability: {}",
                    e
                ))
            })?;

        let compute_capability = major as f32 + (minor as f32 / 10.0);

        // Cooperative groups require compute capability 6.0+
        let supported = compute_capability >= 6.0;

        // Grid synchronization requires compute capability 6.0+
        let grid_sync_supported = compute_capability >= 6.0;

        // Cluster groups require compute capability 9.0+
        let cluster_groups_supported = compute_capability >= 9.0;

        // Device barriers require compute capability 7.0+
        let device_barriers_supported = compute_capability >= 7.0;

        // Get maximum cooperative blocks (if supported)
        let max_cooperative_blocks = if supported {
            // This would typically be queried from the device
            // For now, use a conservative estimate
            let max_blocks_per_sm = 32u32; // Conservative estimate
            let multiprocessor_count = device
                .get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)
                .unwrap_or(1) as u32;
            max_blocks_per_sm * multiprocessor_count
        } else {
            0
        };

        let max_cluster_size = if cluster_groups_supported {
            8 // Typical maximum cluster size
        } else {
            0
        };

        Ok(CooperativeGroupsCapabilities {
            supported,
            max_cooperative_blocks,
            grid_sync_supported,
            cluster_groups_supported,
            max_cluster_size,
            device_barriers_supported,
        })
    }

    /// Check if cooperative groups are supported
    pub fn is_supported(&self) -> bool {
        self.capabilities.supported
    }

    /// Get cooperative groups capabilities
    pub fn capabilities(&self) -> &CooperativeGroupsCapabilities {
        &self.capabilities
    }

    /// Validate cooperative kernel configuration
    pub fn validate_config(&self, config: &CooperativeKernelConfig) -> BackendResult<()> {
        if !self.capabilities.supported {
            return Err(BackendError::UnsupportedOperation(
                "Cooperative groups not supported on this device".to_string(),
            ));
        }

        // Check if grid cooperation is supported
        if config.grid_cooperation && !self.capabilities.grid_sync_supported {
            return Err(BackendError::UnsupportedOperation(
                "Grid-wide cooperation not supported on this device".to_string(),
            ));
        }

        // Check cluster dimensions
        if let Some(cluster_dim) = &config.cluster_dim {
            if !self.capabilities.cluster_groups_supported {
                return Err(BackendError::UnsupportedOperation(
                    "Cluster groups not supported on this device".to_string(),
                ));
            }

            let cluster_size = cluster_dim.0 * cluster_dim.1 * cluster_dim.2;
            if cluster_size > self.capabilities.max_cluster_size {
                return Err(BackendError::InvalidArgument(format!(
                    "Cluster size {} exceeds maximum {}",
                    cluster_size, self.capabilities.max_cluster_size
                )));
            }
        }

        // Validate cooperative groups
        for group in &config.cooperative_groups {
            self.validate_group_descriptor(group)?;
        }

        // Check total blocks vs. maximum cooperative blocks
        let total_blocks = config.grid_dim.0 * config.grid_dim.1 * config.grid_dim.2;
        if config.grid_cooperation && total_blocks > self.capabilities.max_cooperative_blocks {
            return Err(BackendError::InvalidArgument(format!(
                "Grid size {} exceeds maximum cooperative blocks {}",
                total_blocks, self.capabilities.max_cooperative_blocks
            )));
        }

        Ok(())
    }

    /// Validate cooperative group descriptor
    fn validate_group_descriptor(&self, desc: &CooperativeGroupDescriptor) -> BackendResult<()> {
        match desc.group_type {
            CooperativeGroupType::Grid => {
                if !self.capabilities.grid_sync_supported {
                    return Err(BackendError::UnsupportedOperation(
                        "Grid groups not supported on this device".to_string(),
                    ));
                }
            }
            CooperativeGroupType::Cluster => {
                if !self.capabilities.cluster_groups_supported {
                    return Err(BackendError::UnsupportedOperation(
                        "Cluster groups not supported on this device".to_string(),
                    ));
                }
            }
            CooperativeGroupType::CoalescedThreads => {
                if desc.thread_mask.is_none() {
                    return Err(BackendError::InvalidArgument(
                        "Thread mask required for coalesced thread groups".to_string(),
                    ));
                }
            }
            _ => {} // ThreadBlock and Warp are always supported
        }

        // Validate synchronization requirements
        if desc.sync_requirements.memory_scope == MemoryScope::System
            && !self.capabilities.device_barriers_supported
        {
            return Err(BackendError::UnsupportedOperation(
                "System-wide memory scope not supported on this device".to_string(),
            ));
        }

        Ok(())
    }

    /// Launch a cooperative kernel
    pub unsafe fn launch_cooperative_kernel(
        &self,
        kernel_func: *const c_void,
        config: &CooperativeKernelConfig,
        kernel_params: &[*mut c_void],
    ) -> BackendResult<u64> {
        // Validate configuration
        self.validate_config(config)?;

        let start_time = std::time::Instant::now();

        // Generate kernel ID
        let kernel_id = {
            let mut next_id = self.next_kernel_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Launch the kernel using CUDA driver API
        let result = if config.grid_cooperation {
            self.launch_cooperative_kernel_grid(kernel_func, config, kernel_params)
        } else {
            self.launch_cooperative_kernel_regular(kernel_func, config, kernel_params)
        };

        match result {
            Ok(_) => {
                // Record kernel state
                let kernel_state = CooperativeKernelState {
                    kernel_id,
                    config: config.clone(),
                    launched_at: start_time,
                    sync_events: 0,
                    memory_usage: config.shared_memory_size,
                    performance_metrics: KernelPerformanceMetrics::default(),
                };

                {
                    let mut active_kernels = self.active_kernels.lock().unwrap();
                    active_kernels.insert(kernel_id, kernel_state);
                }

                // Update statistics
                {
                    let mut stats = self.performance_stats.lock().unwrap();
                    stats.total_kernels_launched += 1;
                    if config.grid_cooperation {
                        stats.grid_cooperative_kernels += 1;
                    }
                    if config.cluster_dim.is_some() {
                        stats.cluster_cooperative_kernels += 1;
                    }
                }

                Ok(kernel_id)
            }
            Err(e) => Err(e),
        }
    }

    /// Launch cooperative kernel with grid-wide cooperation
    unsafe fn launch_cooperative_kernel_grid(
        &self,
        kernel_func: *const c_void,
        config: &CooperativeKernelConfig,
        kernel_params: &[*mut c_void],
    ) -> BackendResult<()> {
        use cust::sys as cuda_sys;

        // Prepare launch parameters
        let mut launch_params = cuda_sys::CUDA_LAUNCH_PARAMS {
            function: kernel_func as *mut c_void,
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_memory_size,
            hStream: config
                .stream
                .as_ref()
                .map(|s| s.as_inner() as *mut c_void)
                .unwrap_or(std::ptr::null_mut()),
            kernelParams: kernel_params.as_ptr() as *mut *mut c_void,
        };

        // Launch cooperative kernel
        let result = cuda_sys::cuLaunchCooperativeKernel(&mut launch_params as *mut _);

        if result != cuda_sys::CUDA_SUCCESS {
            return Err(BackendError::ComputeError {
                reason: format!("Failed to launch cooperative kernel: {:?}", result),
            });
        }

        Ok(())
    }

    /// Launch regular cooperative kernel (block-level cooperation only)
    unsafe fn launch_cooperative_kernel_regular(
        &self,
        kernel_func: *const c_void,
        config: &CooperativeKernelConfig,
        kernel_params: &[*mut c_void],
    ) -> BackendResult<()> {
        use cust::sys as cuda_sys;

        // For regular kernels, use standard launch
        let result = cuda_sys::cuLaunchKernel(
            kernel_func as *mut c_void,
            config.grid_dim.0,
            config.grid_dim.1,
            config.grid_dim.2,
            config.block_dim.0,
            config.block_dim.1,
            config.block_dim.2,
            config.shared_memory_size as u32,
            config
                .stream
                .as_ref()
                .map(|s| s.as_inner() as *mut c_void)
                .unwrap_or(std::ptr::null_mut()),
            kernel_params.as_ptr() as *mut *mut c_void,
            std::ptr::null_mut(),
        );

        if result != cuda_sys::CUDA_SUCCESS {
            return Err(BackendError::ComputeError {
                reason: format!("Failed to launch kernel: {:?}", result),
            });
        }

        Ok(())
    }

    /// Record synchronization event for a kernel
    pub fn record_sync_event(
        &self,
        kernel_id: u64,
        sync_type: SynchronizationType,
    ) -> BackendResult<()> {
        let mut active_kernels = self.active_kernels.lock().unwrap();

        if let Some(kernel_state) = active_kernels.get_mut(&kernel_id) {
            kernel_state.sync_events += 1;

            match sync_type {
                SynchronizationType::Barrier => {
                    kernel_state.performance_metrics.barrier_syncs += 1;
                }
                SynchronizationType::MemoryFence => {
                    kernel_state.performance_metrics.memory_fences += 1;
                }
            }
        }

        Ok(())
    }

    /// Finish kernel execution and collect performance metrics
    pub fn finish_kernel(&self, kernel_id: u64) -> BackendResult<KernelPerformanceMetrics> {
        let mut active_kernels = self.active_kernels.lock().unwrap();

        if let Some(kernel_state) = active_kernels.remove(&kernel_id) {
            let execution_time = kernel_state.launched_at.elapsed();

            let mut metrics = kernel_state.performance_metrics;
            metrics.execution_time_us = execution_time.as_micros() as u64;

            // Calculate synchronization overhead
            if kernel_state.sync_events > 0 {
                metrics.sync_overhead_us =
                    (execution_time.as_micros() as f64) / (kernel_state.sync_events as f64 * 10.0);
                // Estimate
            }

            // Update global statistics
            {
                let mut stats = self.performance_stats.lock().unwrap();
                stats.total_sync_events += kernel_state.sync_events as u64;

                // Update averages
                let total_kernels = stats.total_kernels_launched as f64;
                stats.avg_kernel_execution_time_us = (stats.avg_kernel_execution_time_us
                    * (total_kernels - 1.0)
                    + metrics.execution_time_us as f64)
                    / total_kernels;

                stats.avg_sync_overhead_us = (stats.avg_sync_overhead_us * (total_kernels - 1.0)
                    + metrics.sync_overhead_us)
                    / total_kernels;
            }

            Ok(metrics)
        } else {
            Err(BackendError::InvalidArgument(format!(
                "Kernel ID {} not found",
                kernel_id
            )))
        }
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> CooperativeGroupsStats {
        self.performance_stats.lock().unwrap().clone()
    }

    /// Clear performance statistics
    pub fn clear_stats(&self) {
        let mut stats = self.performance_stats.lock().unwrap();
        *stats = CooperativeGroupsStats::default();
    }

    /// Get optimal configuration for a given workload
    pub fn suggest_optimal_config(
        &self,
        workload: &CooperativeWorkload,
    ) -> BackendResult<CooperativeKernelConfig> {
        if !self.capabilities.supported {
            return Err(BackendError::UnsupportedOperation(
                "Cooperative groups not supported".to_string(),
            ));
        }

        let mut config = CooperativeKernelConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1), // Default block size
            shared_memory_size: 0,
            stream: None,
            grid_cooperation: false,
            cluster_dim: None,
            cooperative_groups: vec![],
        };

        // Adjust based on workload characteristics
        match workload.cooperation_pattern {
            CooperationPattern::WarpLevel => {
                config.cooperative_groups.push(CooperativeGroupDescriptor {
                    group_type: CooperativeGroupType::Warp,
                    size: Some(32),
                    thread_mask: None,
                    sync_requirements: SynchronizationRequirements {
                        needs_barrier: workload.needs_synchronization,
                        needs_memory_fence: workload.memory_intensive,
                        memory_scope: MemoryScope::Warp,
                        sync_frequency: workload.sync_frequency,
                    },
                });
            }
            CooperationPattern::BlockLevel => {
                config.cooperative_groups.push(CooperativeGroupDescriptor {
                    group_type: CooperativeGroupType::ThreadBlock,
                    size: None,
                    thread_mask: None,
                    sync_requirements: SynchronizationRequirements {
                        needs_barrier: workload.needs_synchronization,
                        needs_memory_fence: workload.memory_intensive,
                        memory_scope: MemoryScope::Block,
                        sync_frequency: workload.sync_frequency,
                    },
                });
            }
            CooperationPattern::GridLevel => {
                if self.capabilities.grid_sync_supported {
                    config.grid_cooperation = true;
                    config.cooperative_groups.push(CooperativeGroupDescriptor {
                        group_type: CooperativeGroupType::Grid,
                        size: None,
                        thread_mask: None,
                        sync_requirements: SynchronizationRequirements {
                            needs_barrier: workload.needs_synchronization,
                            needs_memory_fence: workload.memory_intensive,
                            memory_scope: MemoryScope::Grid,
                            sync_frequency: workload.sync_frequency,
                        },
                    });
                }
            }
        }

        // Calculate optimal grid and block dimensions
        let total_threads = workload.problem_size;
        let threads_per_block = config.block_dim.0;
        let num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

        config.grid_dim.0 = num_blocks.min(self.capabilities.max_cooperative_blocks);

        // Set shared memory if needed
        if workload.shared_memory_per_block > 0 {
            config.shared_memory_size = workload.shared_memory_per_block;
        }

        Ok(config)
    }
}

/// Types of synchronization operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SynchronizationType {
    /// Barrier synchronization
    Barrier,
    /// Memory fence operation
    MemoryFence,
}

/// Cooperation patterns for workloads
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CooperationPattern {
    /// Cooperation within warps
    WarpLevel,
    /// Cooperation within thread blocks
    BlockLevel,
    /// Cooperation across the entire grid
    GridLevel,
}

/// Workload characteristics for optimization
#[derive(Debug, Clone)]
pub struct CooperativeWorkload {
    /// Total problem size (number of elements)
    pub problem_size: u32,
    /// Pattern of cooperation needed
    pub cooperation_pattern: CooperationPattern,
    /// Whether synchronization is needed
    pub needs_synchronization: bool,
    /// Whether the workload is memory-intensive
    pub memory_intensive: bool,
    /// Frequency of synchronization
    pub sync_frequency: SyncFrequency,
    /// Shared memory required per block
    pub shared_memory_per_block: usize,
    /// Memory access pattern
    pub memory_access_pattern: MemoryAccessPattern,
}

/// Memory access patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryAccessPattern {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Strided access pattern
    Strided,
    /// Coalesced access pattern
    Coalesced,
}

/// Builder for cooperative kernel configurations
pub struct CooperativeKernelConfigBuilder {
    config: CooperativeKernelConfig,
}

impl CooperativeKernelConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: CooperativeKernelConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_memory_size: 0,
                stream: None,
                grid_cooperation: false,
                cluster_dim: None,
                cooperative_groups: vec![],
            },
        }
    }

    /// Set grid dimensions
    pub fn grid_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.grid_dim = (x, y, z);
        self
    }

    /// Set block dimensions
    pub fn block_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.block_dim = (x, y, z);
        self
    }

    /// Set shared memory size
    pub fn shared_memory(mut self, size: usize) -> Self {
        self.config.shared_memory_size = size;
        self
    }

    /// Enable grid-wide cooperation
    pub fn grid_cooperation(mut self, enable: bool) -> Self {
        self.config.grid_cooperation = enable;
        self
    }

    /// Set cluster dimensions
    pub fn cluster_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.config.cluster_dim = Some((x, y, z));
        self
    }

    /// Add a cooperative group
    pub fn add_group(mut self, group: CooperativeGroupDescriptor) -> Self {
        self.config.cooperative_groups.push(group);
        self
    }

    /// Build the configuration
    pub fn build(self) -> CooperativeKernelConfig {
        self.config
    }
}

impl Default for CooperativeKernelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooperative_groups_detection() {
        // This test requires a CUDA device
        if crate::cuda::is_available() {
            if let Ok(context) = CooperativeGroupsContext::new(0) {
                let capabilities = context.capabilities();
                println!("Cooperative groups supported: {}", capabilities.supported);
                println!("Grid sync supported: {}", capabilities.grid_sync_supported);
                println!(
                    "Max cooperative blocks: {}",
                    capabilities.max_cooperative_blocks
                );
            }
        }
    }

    #[test]
    fn test_config_builder() {
        let config = CooperativeKernelConfigBuilder::new()
            .grid_dim(10, 1, 1)
            .block_dim(256, 1, 1)
            .shared_memory(1024)
            .grid_cooperation(true)
            .build();

        assert_eq!(config.grid_dim, (10, 1, 1));
        assert_eq!(config.block_dim, (256, 1, 1));
        assert_eq!(config.shared_memory_size, 1024);
        assert!(config.grid_cooperation);
    }

    #[test]
    fn test_workload_config_suggestion() {
        if crate::cuda::is_available() {
            if let Ok(context) = CooperativeGroupsContext::new(0) {
                let workload = CooperativeWorkload {
                    problem_size: 1000000,
                    cooperation_pattern: CooperationPattern::BlockLevel,
                    needs_synchronization: true,
                    memory_intensive: false,
                    sync_frequency: SyncFrequency::Medium,
                    shared_memory_per_block: 1024,
                    memory_access_pattern: MemoryAccessPattern::Coalesced,
                };

                if let Ok(config) = context.suggest_optimal_config(&workload) {
                    assert!(config.cooperative_groups.len() > 0);
                    assert_eq!(config.shared_memory_size, 1024);
                }
            }
        }
    }
}
