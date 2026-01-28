//! NUMA (Non-Uniform Memory Access) support for memory allocation
//!
//! This module provides NUMA topology detection and NUMA-aware memory allocation
//! to optimize memory access patterns on multi-socket systems.

use crate::error::Result;
use crate::storage::allocation::{BackendAllocator, RawMemoryHandle};
use crate::storage::memory_info::{AllocationStrategy, MemoryInfo};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};

/// NUMA (Non-Uniform Memory Access) topology information
///
/// This structure represents the NUMA topology of the system, including
/// the number of nodes, CPU cores per node, memory sizes, and inter-node
/// distances for optimal allocation decisions.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::NumaTopology;
///
/// // Detect system NUMA topology
/// let topology = NumaTopology::detect()?;
/// println!("NUMA nodes: {}", topology.node_count);
///
/// // Create single-node topology for testing
/// let single_node = NumaTopology::single_node();
/// ```
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// CPU cores per NUMA node
    pub cores_per_node: Vec<Vec<usize>>,
    /// Memory size per NUMA node in bytes
    pub memory_per_node: Vec<usize>,
    /// Distance matrix between NUMA nodes (lower is better)
    pub distance_matrix: Vec<Vec<u32>>,
    /// Available memory per NUMA node in bytes
    pub available_memory: Vec<usize>,
}

impl NumaTopology {
    /// Create a new empty NUMA topology
    pub fn new() -> Self {
        Self {
            node_count: 0,
            cores_per_node: Vec::new(),
            memory_per_node: Vec::new(),
            distance_matrix: Vec::new(),
            available_memory: Vec::new(),
        }
    }

    /// Detect NUMA topology from system
    ///
    /// This method attempts to detect the actual NUMA topology of the system
    /// by reading system information. Falls back to single-node topology if
    /// NUMA is not available or detection fails.
    pub fn detect() -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }
        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            // Fallback: single node
            Ok(Self::single_node())
        }
    }

    /// Create a single-node topology (fallback)
    ///
    /// This method creates a simple topology with a single NUMA node
    /// containing all CPU cores and system memory.
    pub fn single_node() -> Self {
        let cpu_count = num_cpus::get();
        let total_memory = Self::get_total_system_memory();

        Self {
            node_count: 1,
            cores_per_node: vec![(0..cpu_count).collect()],
            memory_per_node: vec![total_memory],
            distance_matrix: vec![vec![10]], // Standard local distance
            available_memory: vec![total_memory],
        }
    }

    /// Get total system memory
    fn get_total_system_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|line| line.starts_with("MemTotal:"))
                        .and_then(|line| {
                            line.split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse::<usize>().ok())
                                .map(|kb| kb * 1024) // Convert KB to bytes
                        })
                })
                .unwrap_or(8 * 1024 * 1024 * 1024) // 8GB fallback
        }
        #[cfg(not(target_os = "linux"))]
        {
            8 * 1024 * 1024 * 1024 // 8GB fallback
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Result<Self> {
        use std::fs;
        use std::path::Path;

        let numa_path = Path::new("/sys/devices/system/node");
        if !numa_path.exists() {
            return Ok(Self::single_node());
        }

        let mut topology = Self::new();

        // Count NUMA nodes
        let entries = fs::read_dir(numa_path).map_err(|e| {
            crate::error::TorshError::Other(format!("Failed to read NUMA directory: {e}"))
        })?;

        let mut node_ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                crate::error::TorshError::Other(format!("Failed to read NUMA entry: {e}"))
            })?;
            let name = entry.file_name();
            if let Some(name_str) = name.to_str() {
                if let Some(stripped) = name_str.strip_prefix("node") {
                    if let Ok(id) = stripped.parse::<usize>() {
                        node_ids.push(id);
                    }
                }
            }
        }

        node_ids.sort_unstable();
        topology.node_count = node_ids.len();

        if topology.node_count == 0 {
            return Ok(Self::single_node());
        }

        // Read CPU and memory information for each node
        for &node_id in &node_ids {
            let node_path = numa_path.join(format!("node{node_id}"));

            // Read CPU list
            let cpulist_path = node_path.join("cpulist");
            let cpus = if cpulist_path.exists() {
                fs::read_to_string(&cpulist_path)
                    .map_err(|e| {
                        crate::error::TorshError::Other(format!("Failed to read cpulist: {e}"))
                    })?
                    .trim()
                    .split(',')
                    .flat_map(|range| {
                        if range.contains('-') {
                            let parts: Vec<&str> = range.split('-').collect();
                            if parts.len() == 2 {
                                if let (Ok(start), Ok(end)) =
                                    (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                                {
                                    return (start..=end).collect();
                                }
                            }
                        } else if let Ok(cpu) = range.parse::<usize>() {
                            return vec![cpu];
                        }
                        Vec::new()
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

            topology.cores_per_node.push(cpus);

            // Read memory info
            let meminfo_path = node_path.join("meminfo");
            let (total_memory, available_memory) = if meminfo_path.exists() {
                let content = fs::read_to_string(&meminfo_path).map_err(|e| {
                    crate::error::TorshError::Other(format!("Failed to read meminfo: {e}"))
                })?;

                let mut total = 0;
                let mut available = 0;

                for line in content.lines() {
                    if line.contains("MemTotal:") {
                        if let Some(value) = line.split_whitespace().nth(3) {
                            total = value.parse::<usize>().unwrap_or(0) * 1024; // Convert KB to bytes
                        }
                    } else if line.contains("MemFree:") {
                        if let Some(value) = line.split_whitespace().nth(3) {
                            available = value.parse::<usize>().unwrap_or(0) * 1024;
                            // Convert KB to bytes
                        }
                    }
                }

                (total, available)
            } else {
                (0, 0)
            };

            topology.memory_per_node.push(total_memory);
            topology.available_memory.push(available_memory);
        }

        // Build distance matrix
        topology.distance_matrix = vec![vec![255; topology.node_count]; topology.node_count];
        for (i, &node_i) in node_ids.iter().enumerate() {
            for (j, &_node_j) in node_ids.iter().enumerate() {
                if i == j {
                    topology.distance_matrix[i][j] = 10; // Local distance
                } else {
                    let distance_path = numa_path.join(format!("node{node_i}/distance"));
                    if distance_path.exists() {
                        if let Ok(content) = fs::read_to_string(&distance_path) {
                            let distances: Vec<u32> = content
                                .split_whitespace()
                                .filter_map(|s| s.parse().ok())
                                .collect();
                            if j < distances.len() {
                                topology.distance_matrix[i][j] = distances[j];
                            }
                        }
                    }

                    // Fallback: assume remote distance
                    if topology.distance_matrix[i][j] == 255 {
                        topology.distance_matrix[i][j] = 20;
                    }
                }
            }
        }

        Ok(topology)
    }

    #[cfg(target_os = "windows")]
    fn detect_windows() -> Result<Self> {
        // TODO: Implement Windows NUMA detection using GetNumaNodeProcessorMask
        // For now, return single node
        Ok(Self::single_node())
    }

    #[cfg(target_os = "macos")]
    fn detect_macos() -> Result<Self> {
        // macOS doesn't have traditional NUMA, but we can detect processor groups
        // For now, return single node
        Ok(Self::single_node())
    }

    /// Get the optimal NUMA node for allocation based on current thread
    pub fn optimal_node(&self) -> usize {
        self.optimal_node_for_thread(std::thread::current().id())
    }

    /// Get the optimal NUMA node for a specific thread
    pub fn optimal_node_for_thread(&self, _thread_id: std::thread::ThreadId) -> usize {
        // TODO: Implement CPU affinity detection
        // For now, return the node with the most available memory
        self.available_memory
            .iter()
            .enumerate()
            .max_by_key(|(_, &mem)| mem)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get the distance between two NUMA nodes
    pub fn distance(&self, node1: usize, node2: usize) -> u32 {
        if node1 < self.node_count && node2 < self.node_count {
            self.distance_matrix[node1][node2]
        } else {
            255 // Maximum distance for invalid nodes
        }
    }

    /// Check if allocation is possible on a specific node
    pub fn can_allocate_on_node(&self, node: usize, size_bytes: usize) -> bool {
        if node >= self.node_count {
            return false;
        }

        self.available_memory[node] >= size_bytes
    }

    /// Update available memory after allocation
    pub fn update_allocation(&mut self, node: usize, size_bytes: usize, is_allocation: bool) {
        if node < self.node_count {
            if is_allocation {
                self.available_memory[node] =
                    self.available_memory[node].saturating_sub(size_bytes);
            } else {
                self.available_memory[node] = std::cmp::min(
                    self.available_memory[node] + size_bytes,
                    self.memory_per_node[node],
                );
            }
        }
    }

    /// Get statistics about the NUMA topology
    pub fn statistics(&self) -> NumaTopologyStats {
        let total_memory = self.memory_per_node.iter().sum::<usize>();
        let total_available = self.available_memory.iter().sum::<usize>();
        let total_cores = self
            .cores_per_node
            .iter()
            .map(|cores| cores.len())
            .sum::<usize>();

        let avg_distance = if self.node_count > 1 {
            let mut total_distance = 0;
            let mut count = 0;
            for i in 0..self.node_count {
                for j in 0..self.node_count {
                    if i != j {
                        total_distance += self.distance_matrix[i][j] as usize;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                total_distance as f64 / count as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        NumaTopologyStats {
            node_count: self.node_count,
            total_memory,
            total_available,
            total_cores,
            avg_distance,
            utilization: if total_memory > 0 {
                (total_memory - total_available) as f64 / total_memory as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// NUMA-aware memory allocation policy
///
/// This enum defines different strategies for choosing NUMA nodes
/// when allocating memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaPolicy {
    /// Use local node if possible, fallback to any node
    LocalPreferred,
    /// Strict local allocation only
    LocalOnly,
    /// Interleave allocations across all nodes
    Interleave,
    /// Use specific node
    Bind(usize),
    /// Use first available node
    FirstAvailable,
}

impl Default for NumaPolicy {
    fn default() -> Self {
        Self::LocalPreferred
    }
}

/// NUMA-aware memory allocator
///
/// This allocator wraps another allocator and adds NUMA awareness,
/// allowing for optimal memory placement based on the system topology.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::{NumaAllocator, NumaPolicy};
///
/// let topology = NumaTopology::detect()?;
/// let mut numa_allocator = NumaAllocator::new(base_allocator, topology);
/// numa_allocator.set_policy(NumaPolicy::LocalPreferred);
///
/// let handle = numa_allocator.allocate_raw(&device, 1024, 8)?;
/// ```
#[derive(Debug)]
pub struct NumaAllocator<A: BackendAllocator> {
    inner: A,
    topology: Arc<RwLock<NumaTopology>>,
    policy: NumaPolicy,
    interleave_counter: AtomicUsize,
}

impl<A: BackendAllocator> NumaAllocator<A> {
    /// Create a new NUMA-aware allocator
    pub fn new(inner: A, topology: NumaTopology) -> Self {
        Self {
            inner,
            topology: Arc::new(RwLock::new(topology)),
            policy: NumaPolicy::default(),
            interleave_counter: AtomicUsize::new(0),
        }
    }

    /// Set NUMA allocation policy
    pub fn set_policy(&mut self, policy: NumaPolicy) {
        self.policy = policy;
    }

    /// Get current NUMA policy
    pub fn policy(&self) -> NumaPolicy {
        self.policy
    }

    /// Get NUMA topology
    pub fn topology(&self) -> Arc<RwLock<NumaTopology>> {
        self.topology.clone()
    }

    /// Choose the optimal NUMA node for allocation
    fn choose_numa_node(&self, size_bytes: usize) -> Option<usize> {
        let topology = self.topology.read().expect("lock should not be poisoned");

        // Skip NUMA allocation for single-node systems
        if topology.node_count <= 1 {
            return None;
        }

        match self.policy {
            NumaPolicy::LocalPreferred => {
                let local_node = topology.optimal_node();
                if topology.can_allocate_on_node(local_node, size_bytes) {
                    Some(local_node)
                } else {
                    // Find any available node
                    (0..topology.node_count)
                        .find(|&node| topology.can_allocate_on_node(node, size_bytes))
                }
            }
            NumaPolicy::LocalOnly => {
                let local_node = topology.optimal_node();
                if topology.can_allocate_on_node(local_node, size_bytes) {
                    Some(local_node)
                } else {
                    None
                }
            }
            NumaPolicy::Interleave => {
                let start_node = self
                    .interleave_counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    % topology.node_count;

                // Try each node starting from the calculated position
                for i in 0..topology.node_count {
                    let node = (start_node + i) % topology.node_count;
                    if topology.can_allocate_on_node(node, size_bytes) {
                        return Some(node);
                    }
                }
                None
            }
            NumaPolicy::Bind(node) => {
                if topology.can_allocate_on_node(node, size_bytes) {
                    Some(node)
                } else {
                    None
                }
            }
            NumaPolicy::FirstAvailable => (0..topology.node_count)
                .find(|&node| topology.can_allocate_on_node(node, size_bytes)),
        }
    }

    /// Allocate memory on a specific NUMA node
    pub fn allocate_on_node(
        &self,
        device: &A::Device,
        size_bytes: usize,
        alignment: usize,
        numa_node: usize,
    ) -> std::result::Result<NumaMemoryHandle, A::Error> {
        // For now, we'll use the inner allocator and track the NUMA node
        // In a full implementation, this would use libnuma or similar
        let raw_handle = self.inner.allocate_raw(device, size_bytes, alignment)?;

        // Update topology
        {
            let mut topology = self.topology.write().expect("lock should not be poisoned");
            topology.update_allocation(numa_node, size_bytes, true);
        }

        Ok(NumaMemoryHandle::new(raw_handle, numa_node))
    }

    /// Migrate memory to a different NUMA node
    pub fn migrate_memory(
        &self,
        handle: &NumaMemoryHandle,
        target_node: usize,
        device: &A::Device,
    ) -> std::result::Result<NumaMemoryHandle, A::Error> {
        // Allocate on target node
        let new_handle =
            self.allocate_on_node(device, handle.size_bytes(), handle.alignment(), target_node)?;

        // Copy data (in a real implementation, this would be more efficient)
        unsafe {
            std::ptr::copy_nonoverlapping(handle.ptr(), new_handle.ptr(), handle.size_bytes());
        }

        Ok(new_handle)
    }
}

impl<A: BackendAllocator> BackendAllocator for NumaAllocator<A> {
    type Device = A::Device;
    type Error = A::Error;

    fn allocate_raw(
        &self,
        device: &Self::Device,
        size_bytes: usize,
        alignment: usize,
    ) -> std::result::Result<RawMemoryHandle, Self::Error> {
        // Choose optimal NUMA node
        if let Some(numa_node) = self.choose_numa_node(size_bytes) {
            let numa_handle = self.allocate_on_node(device, size_bytes, alignment, numa_node)?;
            Ok(numa_handle.into_raw())
        } else {
            // Fallback to regular allocation
            self.inner.allocate_raw(device, size_bytes, alignment)
        }
    }

    unsafe fn deallocate_raw(
        &self,
        handle: RawMemoryHandle,
    ) -> std::result::Result<(), Self::Error> {
        // Check if this is a NUMA handle
        if let Some(numa_data) = handle.backend_data.downcast_ref::<NumaMetadata>() {
            let mut topology = self.topology.write().expect("lock should not be poisoned");
            topology.update_allocation(numa_data.node, handle.size_bytes, false);
        }

        self.inner.deallocate_raw(handle)
    }

    fn memory_info(&self, device: &Self::Device) -> std::result::Result<MemoryInfo, Self::Error> {
        // Aggregate memory info across all NUMA nodes
        let mut info = self.inner.memory_info(device)?;

        let topology = self.topology.read().expect("lock should not be poisoned");
        info.total_memory = topology.memory_per_node.iter().sum();
        info.free_memory = topology.available_memory.iter().sum();
        info.used_memory = info.total_memory - info.free_memory;

        Ok(info)
    }

    fn set_strategy(
        &mut self,
        strategy: AllocationStrategy,
    ) -> std::result::Result<(), Self::Error> {
        self.inner.set_strategy(strategy)
    }

    fn strategy(&self) -> AllocationStrategy {
        self.inner.strategy()
    }
}

/// NUMA-specific memory handle
///
/// This handle extends the basic memory handle with NUMA node information,
/// allowing for topology-aware operations.
#[derive(Debug)]
pub struct NumaMemoryHandle {
    raw: RawMemoryHandle,
    numa_node: usize,
}

impl NumaMemoryHandle {
    /// Create a new NUMA memory handle
    pub fn new(mut raw: RawMemoryHandle, numa_node: usize) -> Self {
        // Store NUMA metadata in the backend data
        raw.backend_data = Box::new(NumaMetadata { node: numa_node });

        Self { raw, numa_node }
    }

    /// Get the NUMA node this memory is allocated on
    pub fn numa_node(&self) -> usize {
        self.numa_node
    }

    /// Get the raw memory handle
    pub fn raw(&self) -> &RawMemoryHandle {
        &self.raw
    }

    /// Convert to raw memory handle
    pub fn into_raw(self) -> RawMemoryHandle {
        self.raw
    }

    /// Get the memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.raw.size_bytes
    }

    /// Get the alignment
    pub fn alignment(&self) -> usize {
        self.raw.alignment
    }

    /// Get the pointer
    pub fn ptr(&self) -> *mut u8 {
        self.raw.ptr
    }

    /// Check if this handle is on the same NUMA node as another
    pub fn same_numa_node(&self, other: &NumaMemoryHandle) -> bool {
        self.numa_node == other.numa_node
    }

    /// Get the distance to another NUMA node
    pub fn distance_to_node(&self, topology: &NumaTopology, other_node: usize) -> u32 {
        topology.distance(self.numa_node, other_node)
    }
}

/// NUMA metadata stored in memory handles
#[derive(Debug, Clone)]
pub struct NumaMetadata {
    pub node: usize,
}

/// Statistics about NUMA topology
#[derive(Debug, Clone)]
pub struct NumaTopologyStats {
    /// Number of NUMA nodes
    pub node_count: usize,
    /// Total memory across all nodes
    pub total_memory: usize,
    /// Total available memory across all nodes
    pub total_available: usize,
    /// Total CPU cores across all nodes
    pub total_cores: usize,
    /// Average distance between nodes
    pub avg_distance: f64,
    /// Memory utilization (0.0 to 1.0)
    pub utilization: f64,
}

impl std::fmt::Display for NumaTopologyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NUMA Stats(nodes={}, memory={}/{} MB, cores={}, avg_dist={:.1}, util={:.1}%)",
            self.node_count,
            (self.total_memory - self.total_available) / (1024 * 1024),
            self.total_memory / (1024 * 1024),
            self.total_cores,
            self.avg_distance,
            self.utilization * 100.0
        )
    }
}

/// Utility functions for NUMA operations
pub mod utils {
    use super::*;

    /// Get the optimal NUMA policy for a given workload pattern
    pub fn recommend_numa_policy(
        workload_type: WorkloadType,
        memory_pattern: MemoryAccessPattern,
    ) -> NumaPolicy {
        match (workload_type, memory_pattern) {
            (WorkloadType::Compute, MemoryAccessPattern::Local) => NumaPolicy::LocalOnly,
            (WorkloadType::Compute, MemoryAccessPattern::Scattered) => NumaPolicy::Interleave,
            (WorkloadType::DataProcessing, MemoryAccessPattern::Sequential) => {
                NumaPolicy::LocalPreferred
            }
            (WorkloadType::DataProcessing, MemoryAccessPattern::Random) => NumaPolicy::Interleave,
            (WorkloadType::NetworkIO, _) => NumaPolicy::FirstAvailable,
            _ => NumaPolicy::LocalPreferred,
        }
    }

    /// Calculate memory locality score for a set of allocations
    pub fn calculate_locality_score(handles: &[NumaMemoryHandle], topology: &NumaTopology) -> f64 {
        if handles.len() <= 1 {
            return 1.0; // Perfect locality for single allocation
        }

        let total_distance: u32 = handles
            .iter()
            .enumerate()
            .flat_map(|(i, h1)| {
                handles
                    .iter()
                    .skip(i + 1)
                    .map(move |h2| topology.distance(h1.numa_node(), h2.numa_node()))
            })
            .sum();

        let num_pairs = handles.len() * (handles.len() - 1) / 2;
        let avg_distance = total_distance as f64 / num_pairs as f64;
        let max_distance = 255.0; // Maximum NUMA distance

        // Convert to score (lower distance = higher score)
        1.0 - (avg_distance / max_distance)
    }

    /// Find the optimal NUMA nodes for a set of allocations
    pub fn find_optimal_nodes(
        sizes: &[usize],
        topology: &NumaTopology,
        minimize_distance: bool,
    ) -> Vec<Option<usize>> {
        let mut result = vec![None; sizes.len()];

        if minimize_distance {
            // Try to place all allocations on the same node if possible
            for node in 0..topology.node_count {
                let total_size: usize = sizes.iter().sum();
                if topology.can_allocate_on_node(node, total_size) {
                    // All allocations can fit on one node
                    result.iter_mut().for_each(|r| *r = Some(node));
                    return result;
                }
            }
        }

        // Greedy assignment to available nodes
        for (i, &size) in sizes.iter().enumerate() {
            result[i] =
                (0..topology.node_count).find(|&node| topology.can_allocate_on_node(node, size));
        }

        result
    }

    /// Check if the system has meaningful NUMA topology
    pub fn has_numa_topology(topology: &NumaTopology) -> bool {
        topology.node_count > 1
            && topology
                .distance_matrix
                .iter()
                .any(|row| row.iter().any(|&dist| dist > 10))
    }
}

/// Workload type for NUMA policy recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// CPU-intensive computation
    Compute,
    /// Data processing and transformation
    DataProcessing,
    /// Network I/O bound operations
    NetworkIO,
    /// Mixed workload
    Mixed,
}

/// Memory access pattern for NUMA optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccessPattern {
    /// Local memory access within threads
    Local,
    /// Sequential memory access patterns
    Sequential,
    /// Random memory access patterns
    Random,
    /// Scattered access across memory regions
    Scattered,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuDevice;
    use crate::error::TorshError;

    // Mock allocator for testing
    #[derive(Debug)]
    struct MockAllocator;

    impl BackendAllocator for MockAllocator {
        type Device = CpuDevice;
        type Error = TorshError;

        fn allocate_raw(
            &self,
            _device: &Self::Device,
            size_bytes: usize,
            alignment: usize,
        ) -> std::result::Result<RawMemoryHandle, Self::Error> {
            let layout = std::alloc::Layout::from_size_align(size_bytes, alignment)
                .map_err(|_| TorshError::InvalidArgument("Invalid layout".to_string()))?;

            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(TorshError::AllocationError(
                    "Failed to allocate memory".to_string(),
                ));
            }

            Ok(RawMemoryHandle::new(
                ptr,
                size_bytes,
                alignment,
                Box::new(layout),
            ))
        }

        unsafe fn deallocate_raw(
            &self,
            handle: RawMemoryHandle,
        ) -> std::result::Result<(), Self::Error> {
            if let Some(layout) = handle.backend_data.downcast_ref::<std::alloc::Layout>() {
                std::alloc::dealloc(handle.ptr, *layout);
                Ok(())
            } else {
                Err(TorshError::InvalidArgument(
                    "Invalid backend data".to_string(),
                ))
            }
        }

        fn memory_info(
            &self,
            _device: &Self::Device,
        ) -> std::result::Result<MemoryInfo, Self::Error> {
            Ok(MemoryInfo {
                total_memory: 1024 * 1024 * 1024,
                free_memory: 512 * 1024 * 1024,
                used_memory: 512 * 1024 * 1024,
                max_allocation_size: 128 * 1024 * 1024,
                bandwidth: None,
                is_unified: true,
                supported_alignments: vec![1, 2, 4, 8, 16, 32, 64, 128],
            })
        }

        fn set_strategy(
            &mut self,
            _strategy: AllocationStrategy,
        ) -> std::result::Result<(), Self::Error> {
            Ok(())
        }

        fn strategy(&self) -> AllocationStrategy {
            AllocationStrategy::Immediate
        }
    }

    #[test]
    fn test_numa_topology_single_node() {
        let topology = NumaTopology::single_node();

        assert_eq!(topology.node_count, 1);
        assert!(!topology.cores_per_node.is_empty());
        assert!(!topology.memory_per_node.is_empty());
        assert_eq!(topology.distance_matrix.len(), 1);
        assert_eq!(topology.distance_matrix[0][0], 10); // Local distance

        // Test distance calculation
        assert_eq!(topology.distance(0, 0), 10);
        assert_eq!(topology.distance(1, 0), 255); // Invalid node

        // Test allocation check
        let total_memory = topology.memory_per_node[0];
        assert!(topology.can_allocate_on_node(0, total_memory / 2));
        assert!(!topology.can_allocate_on_node(1, 100)); // Invalid node
    }

    #[test]
    fn test_numa_topology_update() {
        let mut topology = NumaTopology::single_node();
        let initial_available = topology.available_memory[0];
        let allocation_size = 1024;

        // Test allocation update
        topology.update_allocation(0, allocation_size, true);
        assert_eq!(
            topology.available_memory[0],
            initial_available - allocation_size
        );

        // Test deallocation update
        topology.update_allocation(0, allocation_size, false);
        assert_eq!(topology.available_memory[0], initial_available);

        // Test invalid node
        topology.update_allocation(10, allocation_size, true);
        // Should not panic and should not affect valid nodes
        assert_eq!(topology.available_memory[0], initial_available);
    }

    #[test]
    fn test_numa_policy() {
        let policy = NumaPolicy::default();
        assert_eq!(policy, NumaPolicy::LocalPreferred);

        let bind_policy = NumaPolicy::Bind(2);
        if let NumaPolicy::Bind(node) = bind_policy {
            assert_eq!(node, 2);
        } else {
            panic!("Expected Bind policy");
        }
    }

    #[test]
    fn test_numa_allocator() {
        let mock_allocator = MockAllocator;
        let topology = NumaTopology::single_node();
        let mut numa_allocator = NumaAllocator::new(mock_allocator, topology);

        // Test policy setting
        assert_eq!(numa_allocator.policy(), NumaPolicy::LocalPreferred);
        numa_allocator.set_policy(NumaPolicy::Interleave);
        assert_eq!(numa_allocator.policy(), NumaPolicy::Interleave);

        // Test topology access
        let topology_ref = numa_allocator.topology();
        let topology_guard = topology_ref.read().expect("lock should not be poisoned");
        assert_eq!(topology_guard.node_count, 1);
        drop(topology_guard);

        // Test allocation through NUMA allocator
        let device = CpuDevice::new();
        let handle = numa_allocator.allocate_raw(&device, 100, 8).unwrap();
        assert_eq!(handle.size_bytes, 100);
        assert_eq!(handle.alignment, 8);

        // Clean up
        unsafe {
            numa_allocator.deallocate_raw(handle).unwrap();
        }
    }

    #[test]
    fn test_numa_memory_handle() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let ptr = data.as_ptr() as *mut u8;

        let raw_handle = RawMemoryHandle::new(ptr, 8, 4, Box::new("test".to_string()));

        let numa_handle = NumaMemoryHandle::new(raw_handle, 0);

        assert_eq!(numa_handle.numa_node(), 0);
        assert_eq!(numa_handle.size_bytes(), 8);
        assert_eq!(numa_handle.alignment(), 4);
        assert_eq!(numa_handle.ptr(), ptr);

        // Test conversion
        let raw_back = numa_handle.into_raw();
        assert_eq!(raw_back.size_bytes, 8);
        assert_eq!(raw_back.alignment, 4);

        // Check NUMA metadata
        assert!(raw_back
            .backend_data
            .downcast_ref::<NumaMetadata>()
            .is_some());
    }

    #[test]
    fn test_numa_policy_recommendations() {
        use utils::*;

        assert_eq!(
            recommend_numa_policy(WorkloadType::Compute, MemoryAccessPattern::Local),
            NumaPolicy::LocalOnly
        );

        assert_eq!(
            recommend_numa_policy(WorkloadType::DataProcessing, MemoryAccessPattern::Random),
            NumaPolicy::Interleave
        );

        assert_eq!(
            recommend_numa_policy(WorkloadType::NetworkIO, MemoryAccessPattern::Sequential),
            NumaPolicy::FirstAvailable
        );
    }

    #[test]
    fn test_numa_topology_statistics() {
        let topology = NumaTopology::single_node();
        let stats = topology.statistics();

        assert_eq!(stats.node_count, 1);
        assert!(stats.total_memory > 0);
        assert!(stats.total_cores > 0);
        assert_eq!(stats.avg_distance, 0.0); // Single node has no inter-node distances
    }

    #[test]
    fn test_locality_score() {
        use utils::*;

        let topology = NumaTopology::single_node();

        // Create some test handles
        let data = [1u8, 2, 3, 4];
        let raw_handle = RawMemoryHandle::simple(data.as_ptr() as *mut u8, 4, 1);
        let numa_handle = NumaMemoryHandle::new(raw_handle, 0);

        let handles = vec![numa_handle];
        let score = calculate_locality_score(&handles, &topology);
        assert_eq!(score, 1.0); // Single allocation has perfect locality
    }
}
