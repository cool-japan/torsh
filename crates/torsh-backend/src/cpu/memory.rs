//! CPU Memory Management with NUMA awareness

use crate::buffer::{generate_buffer_id, BufferHandle};
use crate::memory::{MemoryManager, MemoryPool, MemoryStats, PoolStats};
use crate::{Buffer, BufferDescriptor, Device};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::sync::{Arc, Mutex};

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, sync::Arc};
#[cfg(not(feature = "std"))]
use spin::Mutex;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// NUMA node information
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: u32,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f64,
    /// CPU cores associated with this node
    pub cpu_cores: Vec<u32>,
    /// Distance to other NUMA nodes
    pub distances: HashMap<u32, u32>,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Available NUMA nodes
    pub nodes: HashMap<u32, NumaNode>,
    /// Current thread's preferred NUMA node
    pub current_node: u32,
    /// Whether NUMA is available on this system
    pub numa_available: bool,
}

/// Memory access pattern tracking
#[derive(Debug)]
pub struct MemoryAccessPattern {
    /// Access frequency counter
    pub access_count: AtomicU64,
    /// Last access time
    pub last_access: Instant,
    /// Access pattern (sequential, random, etc.)
    pub pattern_type: AccessPatternType,
    /// Preferred NUMA node for this allocation
    pub preferred_node: u32,
}

impl Clone for MemoryAccessPattern {
    fn clone(&self) -> Self {
        Self {
            access_count: AtomicU64::new(self.access_count.load(Ordering::Relaxed)),
            last_access: self.last_access,
            pattern_type: self.pattern_type,
            preferred_node: self.preferred_node,
        }
    }
}

/// Types of memory access patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPatternType {
    Sequential,
    Random,
    Strided,
    Temporal,
    Unknown,
}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaAllocationStrategy {
    /// Allocate on local NUMA node
    Local,
    /// Allocate on preferred node
    Preferred(u32),
    /// Interleave across all nodes
    Interleaved,
    /// Best fit based on available memory
    BestFit,
    /// Round-robin across nodes
    RoundRobin,
}

/// CPU memory manager implementation
#[derive(Debug, Clone)]
pub struct CpuMemoryManager {
    pools: Arc<Mutex<HashMap<usize, CpuMemoryPool>>>,
    stats: Arc<Mutex<MemoryStats>>,
    numa_topology: Arc<Mutex<NumaTopology>>,
    #[allow(dead_code)]
    numa_pools: Arc<Mutex<HashMap<(usize, u32), CpuMemoryPool>>>, // (size_class, numa_node) -> pool
    allocation_strategy: Arc<Mutex<NumaAllocationStrategy>>,
    access_patterns: Arc<Mutex<HashMap<usize, MemoryAccessPattern>>>,
    round_robin_counter: Arc<Mutex<u32>>,
}

impl CpuMemoryManager {
    /// Create a new CPU memory manager
    pub fn new() -> Self {
        let numa_topology = Self::detect_numa_topology();
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MemoryStats::default())),
            numa_topology: Arc::new(Mutex::new(numa_topology)),
            numa_pools: Arc::new(Mutex::new(HashMap::new())),
            allocation_strategy: Arc::new(Mutex::new(NumaAllocationStrategy::Local)),
            access_patterns: Arc::new(Mutex::new(HashMap::new())),
            round_robin_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Detect NUMA topology of the system
    fn detect_numa_topology() -> NumaTopology {
        let mut numa_topology = NumaTopology {
            nodes: HashMap::new(),
            current_node: 0,
            numa_available: false,
        };

        // Try to detect NUMA nodes
        #[cfg(target_os = "linux")]
        {
            numa_topology.numa_available = Self::detect_linux_numa(&mut numa_topology);
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback: create a single node
            numa_topology.nodes.insert(
                0,
                NumaNode {
                    id: 0,
                    available_memory: 8 * 1024 * 1024 * 1024, // 8GB
                    memory_bandwidth: 50.0,
                    cpu_cores: (0..num_cpus::get() as u32).collect(),
                    distances: HashMap::new(),
                },
            );
        }

        numa_topology
    }

    /// Detect NUMA topology on Linux systems
    #[cfg(target_os = "linux")]
    fn detect_linux_numa(numa_topology: &mut NumaTopology) -> bool {
        use std::fs;
        use std::path::Path;

        let numa_path = Path::new("/sys/devices/system/node");
        if !numa_path.exists() {
            return false;
        }

        let mut numa_available = false;

        // Read NUMA nodes
        if let Ok(entries) = fs::read_dir(numa_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                if let Some(stripped) = name_str.strip_prefix("node") {
                    if let Ok(node_id) = stripped.parse::<u32>() {
                        numa_available = true;

                        // Get memory information
                        let meminfo_path = entry.path().join("meminfo");
                        let available_memory =
                            if let Ok(content) = fs::read_to_string(&meminfo_path) {
                                Self::parse_numa_meminfo(&content)
                            } else {
                                1024 * 1024 * 1024 // 1GB default
                            };

                        // Get CPU list
                        let cpulist_path = entry.path().join("cpulist");
                        let cpu_cores = if let Ok(content) = fs::read_to_string(&cpulist_path) {
                            Self::parse_cpu_list(&content)
                        } else {
                            Vec::new()
                        };

                        // Get distances
                        let distance_path = entry.path().join("distance");
                        let distances = if let Ok(content) = fs::read_to_string(&distance_path) {
                            Self::parse_numa_distances(&content)
                        } else {
                            HashMap::new()
                        };

                        numa_topology.nodes.insert(
                            node_id,
                            NumaNode {
                                id: node_id,
                                available_memory,
                                memory_bandwidth: 50.0, // Default bandwidth
                                cpu_cores,
                                distances,
                            },
                        );
                    }
                }
            }
        }

        // Set current node
        numa_topology.current_node = Self::get_current_numa_node();

        numa_available
    }

    /// Parse NUMA memory information
    #[cfg(target_os = "linux")]
    fn parse_numa_meminfo(content: &str) -> u64 {
        for line in content.lines() {
            if line.starts_with("Node") && line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    if let Ok(kb) = parts[2].parse::<u64>() {
                        return kb * 1024; // Convert KB to bytes
                    }
                }
            }
        }
        1024 * 1024 * 1024 // 1GB default
    }

    /// Parse CPU list from NUMA node
    #[cfg(target_os = "linux")]
    fn parse_cpu_list(content: &str) -> Vec<u32> {
        let mut cpu_cores = Vec::new();
        let content = content.trim();

        for range in content.split(',') {
            if let Some(dash_pos) = range.find('-') {
                // Range format: "0-3"
                let start = range[..dash_pos].parse::<u32>().unwrap_or(0);
                let end = range[dash_pos + 1..].parse::<u32>().unwrap_or(0);
                for cpu in start..=end {
                    cpu_cores.push(cpu);
                }
            } else {
                // Single CPU
                if let Ok(cpu) = range.parse::<u32>() {
                    cpu_cores.push(cpu);
                }
            }
        }

        cpu_cores
    }

    /// Parse NUMA distance information
    #[cfg(target_os = "linux")]
    fn parse_numa_distances(content: &str) -> HashMap<u32, u32> {
        let mut distances = HashMap::new();
        let content = content.trim();

        for (node_id, distance_str) in content.split_whitespace().enumerate() {
            if let Ok(distance) = distance_str.parse::<u32>() {
                distances.insert(node_id as u32, distance);
            }
        }

        distances
    }

    /// Get current thread's NUMA node
    fn get_current_numa_node() -> u32 {
        #[cfg(target_os = "linux")]
        {
            // Try to get current CPU and map it to NUMA node
            if let Ok(cpu) = std::fs::read_to_string("/proc/self/stat") {
                // Parse the CPU field from /proc/self/stat
                let fields: Vec<&str> = cpu.split_whitespace().collect();
                if fields.len() > 38 {
                    if let Ok(cpu_id) = fields[38].parse::<u32>() {
                        return Self::cpu_to_numa_node(cpu_id);
                    }
                }
            }
        }

        0 // Default to node 0
    }

    /// Map CPU ID to NUMA node
    #[cfg(target_os = "linux")]
    fn cpu_to_numa_node(cpu_id: u32) -> u32 {
        use std::fs;
        use std::path::Path;

        let cpu_path_str = format!("/sys/devices/system/cpu/cpu{}/node", cpu_id);
        let cpu_path = Path::new(&cpu_path_str);
        if let Ok(content) = fs::read_to_string(cpu_path) {
            content.trim().parse::<u32>().unwrap_or(0)
        } else {
            0
        }
    }

    /// Set NUMA allocation strategy
    pub fn set_allocation_strategy(&self, strategy: NumaAllocationStrategy) -> Result<()> {
        let mut allocation_strategy = self.allocation_strategy.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocation strategy lock".to_string())
        })?;
        *allocation_strategy = strategy;
        Ok(())
    }

    /// Get current allocation strategy
    pub fn get_allocation_strategy(&self) -> Result<NumaAllocationStrategy> {
        let allocation_strategy = self.allocation_strategy.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocation strategy lock".to_string())
        })?;
        Ok(*allocation_strategy)
    }

    /// Select NUMA node for allocation based on strategy
    fn select_numa_node(&self, size: usize) -> Result<u32> {
        let strategy = self.get_allocation_strategy()?;
        let numa_topology = self.numa_topology.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire NUMA topology lock".to_string())
        })?;

        if !numa_topology.numa_available {
            return Ok(0); // Default to node 0 if NUMA is not available
        }

        match strategy {
            NumaAllocationStrategy::Local => Ok(numa_topology.current_node),
            NumaAllocationStrategy::Preferred(node_id) => {
                if numa_topology.nodes.contains_key(&node_id) {
                    Ok(node_id)
                } else {
                    Ok(numa_topology.current_node)
                }
            }
            NumaAllocationStrategy::Interleaved => {
                // Select node based on address interleaving
                let node_count = numa_topology.nodes.len() as u32;
                Ok((size / 4096) as u32 % node_count) // Page-based interleaving
            }
            NumaAllocationStrategy::BestFit => {
                // Find node with most available memory
                let mut best_node = numa_topology.current_node;
                let mut best_memory = 0;

                for (node_id, node) in &numa_topology.nodes {
                    if node.available_memory > best_memory {
                        best_memory = node.available_memory;
                        best_node = *node_id;
                    }
                }

                Ok(best_node)
            }
            NumaAllocationStrategy::RoundRobin => {
                let mut counter = self.round_robin_counter.lock().map_err(|_| {
                    TorshError::AllocationError(
                        "Failed to acquire round-robin counter lock".to_string(),
                    )
                })?;

                let node_count = numa_topology.nodes.len() as u32;
                let selected_node = *counter % node_count;
                *counter = (*counter + 1) % node_count;

                Ok(selected_node)
            }
        }
    }

    /// Allocate memory on specific NUMA node
    fn allocate_on_numa_node(
        &self,
        size: usize,
        alignment: usize,
        numa_node: u32,
    ) -> Result<*mut u8> {
        #[cfg(target_os = "linux")]
        {
            // NUMA allocation is disabled to avoid linking dependencies
            // In a production version, this would check for libnuma availability
            // and use numa_alloc_onnode if available
            let _ = numa_node; // Acknowledge the parameter
        }

        // Fallback to regular allocation
        let layout = std::alloc::Layout::from_size_align(size, alignment)
            .map_err(|e| TorshError::AllocationError(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(TorshError::AllocationError(format!(
                "Failed to allocate {} bytes on NUMA node {}",
                size, numa_node
            )));
        }

        Ok(ptr)
    }

    /// Prefetch memory to improve access patterns
    pub fn prefetch_memory(
        &self,
        ptr: *mut u8,
        size: usize,
        pattern: AccessPatternType,
    ) -> Result<()> {
        match pattern {
            AccessPatternType::Sequential => {
                // Prefetch sequential pages
                self.prefetch_sequential(ptr, size)
            }
            AccessPatternType::Random => {
                // Prefetch random access patterns
                self.prefetch_random(ptr, size)
            }
            AccessPatternType::Strided => {
                // Prefetch strided access patterns
                self.prefetch_strided(ptr, size)
            }
            AccessPatternType::Temporal => {
                // Prefetch for temporal locality
                self.prefetch_temporal(ptr, size)
            }
            AccessPatternType::Unknown => {
                // Use default prefetching strategy
                self.prefetch_default(ptr, size)
            }
        }
    }

    /// Prefetch sequential memory pattern
    fn prefetch_sequential(&self, ptr: *mut u8, size: usize) -> Result<()> {
        let page_size = 4096;
        let pages = (size + page_size - 1) / page_size;

        for i in 0..pages {
            let page_ptr = unsafe { ptr.add(i * page_size) };
            self.prefetch_page(page_ptr);
        }

        Ok(())
    }

    /// Prefetch random memory pattern
    fn prefetch_random(&self, ptr: *mut u8, size: usize) -> Result<()> {
        // For random access, prefetch a few strategic pages
        let page_size = 4096;
        let pages = (size + page_size - 1) / page_size;

        // Prefetch first, middle, and last pages
        self.prefetch_page(ptr);
        if pages > 1 {
            let mid_ptr = unsafe { ptr.add((pages / 2) * page_size) };
            self.prefetch_page(mid_ptr);
        }
        if pages > 2 {
            let last_ptr = unsafe { ptr.add((pages - 1) * page_size) };
            self.prefetch_page(last_ptr);
        }

        Ok(())
    }

    /// Prefetch strided memory pattern
    fn prefetch_strided(&self, ptr: *mut u8, size: usize) -> Result<()> {
        let page_size = 4096;
        let stride = page_size * 2; // Every other page
        let mut offset = 0;

        while offset < size {
            let page_ptr = unsafe { ptr.add(offset) };
            self.prefetch_page(page_ptr);
            offset += stride;
        }

        Ok(())
    }

    /// Prefetch temporal memory pattern
    fn prefetch_temporal(&self, ptr: *mut u8, size: usize) -> Result<()> {
        // For temporal locality, prefetch the entire region
        self.prefetch_sequential(ptr, size)
    }

    /// Default prefetch strategy
    fn prefetch_default(&self, ptr: *mut u8, size: usize) -> Result<()> {
        // Use sequential prefetching as default
        self.prefetch_sequential(ptr, size)
    }

    /// Prefetch a single page
    fn prefetch_page(&self, ptr: *mut u8) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                // Use Intel prefetch instructions
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Prefetch instructions are currently unstable in Rust
            // TODO: Re-enable when stabilized (see issue #117217)
            // unsafe {
            //     std::arch::aarch64::_prefetch(
            //         ptr as *const i8,
            //         std::arch::aarch64::_PREFETCH_READ,
            //         std::arch::aarch64::_PREFETCH_LOCALITY3,
            //     );
            // }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // No prefetch instruction available
            let _ = ptr;
        }
    }

    /// Track memory access pattern
    pub fn track_access(&self, ptr: *mut u8, access_type: AccessPatternType) -> Result<()> {
        let mut patterns = self.access_patterns.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire access patterns lock".to_string())
        })?;

        if let Some(pattern) = patterns.get_mut(&(ptr as usize)) {
            pattern.access_count.fetch_add(1, Ordering::Relaxed);
            pattern.last_access = Instant::now();
            pattern.pattern_type = access_type;
        }

        Ok(())
    }

    /// Get memory access statistics
    pub fn get_access_stats(&self, ptr: *mut u8) -> Result<Option<MemoryAccessPattern>> {
        let patterns = self.access_patterns.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire access patterns lock".to_string())
        })?;

        Ok(patterns.get(&(ptr as usize)).cloned())
    }

    /// Optimize memory layout based on access patterns
    pub fn optimize_layout(&self) -> Result<()> {
        let patterns = self.access_patterns.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire access patterns lock".to_string())
        })?;

        // Analyze access patterns and suggest NUMA node migrations
        for (ptr, pattern) in patterns.iter() {
            let access_count = pattern.access_count.load(Ordering::Relaxed);
            let current_node = Self::get_current_numa_node();

            // If access count is high and preferred node is different, consider migration
            if access_count > 1000 && pattern.preferred_node != current_node {
                // Log suggestion for migration (implementation would depend on specific needs)
                #[cfg(feature = "tracing")]
                tracing::info!(
                    "High access count {} for allocation {:?}, consider migrating to node {}",
                    access_count,
                    ptr,
                    pattern.preferred_node
                );
            }
        }

        Ok(())
    }

    /// Create a new CPU memory manager with config
    pub fn with_config(config: crate::MemoryPoolConfig) -> Self {
        let numa_topology = Self::detect_numa_topology();
        let manager = Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MemoryStats::default())),
            numa_topology: Arc::new(Mutex::new(numa_topology)),
            numa_pools: Arc::new(Mutex::new(HashMap::new())),
            allocation_strategy: Arc::new(Mutex::new(NumaAllocationStrategy::Local)),
            access_patterns: Arc::new(Mutex::new(HashMap::new())),
            round_robin_counter: Arc::new(Mutex::new(0)),
        };

        // Configure NUMA strategy based on config
        if let Some(strategy) = config.numa_strategy {
            let _ = manager.set_allocation_strategy(strategy);
        }

        manager
    }

    /// Calculate size class for a given size (power of 2 rounding)
    fn calculate_size_class(size: usize) -> usize {
        if size <= 64 {
            64
        } else {
            size.next_power_of_two()
        }
    }
}

impl MemoryManager for CpuMemoryManager {
    fn allocate(&mut self, descriptor: &BufferDescriptor) -> Result<Buffer> {
        let size = descriptor.size;
        let alignment = descriptor.alignment.unwrap_or(std::mem::align_of::<u8>());
        let size_class = Self::calculate_size_class(size);

        // Select NUMA node for allocation
        let numa_node = self.select_numa_node(size)?;

        // Try NUMA-aware allocation first
        let ptr = if let Ok(numa_ptr) = self.allocate_on_numa_node(size, alignment, numa_node) {
            // Track access pattern for NUMA-allocated memory
            let mut patterns = self.access_patterns.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire access patterns lock".to_string())
            })?;

            let pattern = MemoryAccessPattern {
                access_count: AtomicU64::new(0),
                last_access: Instant::now(),
                pattern_type: AccessPatternType::Unknown,
                preferred_node: numa_node,
            };

            patterns.insert(numa_ptr as usize, pattern);
            numa_ptr
        } else {
            // Fallback to regular pool allocation
            {
                let mut pools = self.pools.lock().map_err(|_| {
                    TorshError::AllocationError("Failed to acquire pools lock".to_string())
                })?;

                pools
                    .entry(size_class)
                    .or_insert_with(|| CpuMemoryPool::new(size_class));
            }

            let mut pools = self.pools.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire pools lock".to_string())
            })?;

            let pool = pools.get_mut(&size_class).unwrap();
            pool.allocate(size, alignment)
                .map_err(|e| TorshError::AllocationError(e.to_string()))?
        };

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| TorshError::AllocationError("Failed to acquire stats lock".to_string()))?;
        stats.allocated_memory += size;
        stats.peak_memory = stats.peak_memory.max(stats.allocated_memory);
        stats.total_allocations += 1;
        stats.active_allocations += 1;

        Ok(Buffer::new(
            generate_buffer_id(),
            self.device().clone(),
            size,
            descriptor.usage,
            descriptor.clone(),
            BufferHandle::Cpu { ptr, size },
        ))
    }

    fn deallocate(&mut self, buffer: &Buffer) -> Result<()> {
        let size_class = Self::calculate_size_class(buffer.size);

        // Extract pointer from buffer handle
        let ptr = match &buffer.handle {
            BufferHandle::Cpu { ptr, .. } => *ptr,
            _ => {
                return Err(TorshError::AllocationError(
                    "Invalid buffer handle type for CPU backend".to_string(),
                ))
            }
        };

        // Deallocate from the pool
        {
            let mut pools = self.pools.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire pools lock".to_string())
            })?;

            if let Some(pool) = pools.get_mut(&size_class) {
                pool.deallocate(ptr, buffer.size)
                    .map_err(|e| TorshError::AllocationError(e.to_string()))?;
            } else {
                return Err(TorshError::InvalidArgument(
                    "No pool found for deallocating buffer".to_string(),
                ));
            }
        }

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| TorshError::AllocationError("Failed to acquire stats lock".to_string()))?;
        stats.allocated_memory = stats.allocated_memory.saturating_sub(buffer.size);
        stats.total_deallocations += 1;
        stats.active_allocations = stats.active_allocations.saturating_sub(1);

        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    fn garbage_collect(&mut self) -> Result<usize> {
        // For CPU memory, we don't need explicit garbage collection
        Ok(0)
    }

    fn set_pool(&mut self, _pool: Box<dyn MemoryPool>) -> Result<()> {
        // Simple implementation - we manage our own pools
        Ok(())
    }

    fn device(&self) -> &Device {
        use crate::device::DeviceInfo;
        use std::sync::OnceLock;
        static CPU_DEVICE: OnceLock<Device> = OnceLock::new();
        CPU_DEVICE.get_or_init(|| Device {
            id: 0,
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            info: DeviceInfo {
                vendor: "CPU".to_string(),
                driver_version: "CPU Backend 1.0".to_string(),
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB typical
                available_memory: 8 * 1024 * 1024 * 1024,
                compute_units: 1,
                max_work_group_size: u32::MAX as usize,
                max_work_group_dimensions: vec![u32::MAX as usize, 1, 1],
                clock_frequency_mhz: 3000, // 3GHz typical
                memory_bandwidth_gbps: 50.0,
                peak_gflops: 100.0,
                features: Vec::new(),
                properties: Vec::new(),
            },
        })
    }

    fn allocate_raw(&mut self, size: usize, alignment: usize) -> Result<*mut u8> {
        let size_class = Self::calculate_size_class(size);

        // Select NUMA node for allocation
        let numa_node = self.select_numa_node(size)?;

        // Try NUMA-aware allocation first
        if let Ok(numa_ptr) = self.allocate_on_numa_node(size, alignment, numa_node) {
            // Track access pattern for NUMA-allocated memory
            let mut patterns = self.access_patterns.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire access patterns lock".to_string())
            })?;

            let pattern = MemoryAccessPattern {
                access_count: AtomicU64::new(0),
                last_access: Instant::now(),
                pattern_type: AccessPatternType::Unknown,
                preferred_node: numa_node,
            };

            patterns.insert(numa_ptr as usize, pattern);
            return Ok(numa_ptr);
        }

        // Fallback to regular pool allocation
        {
            let mut pools = self.pools.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire pools lock".to_string())
            })?;

            pools
                .entry(size_class)
                .or_insert_with(|| CpuMemoryPool::new(size_class));
        }

        let mut pools = self
            .pools
            .lock()
            .map_err(|_| TorshError::AllocationError("Failed to acquire pools lock".to_string()))?;

        let pool = pools.get_mut(&size_class).unwrap();
        pool.allocate(size, alignment)
            .map_err(|e| TorshError::AllocationError(e.to_string()))
    }

    fn deallocate_raw(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        let size_class = Self::calculate_size_class(size);

        // Clean up access pattern tracking
        {
            let mut patterns = self.access_patterns.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire access patterns lock".to_string())
            })?;
            patterns.remove(&(ptr as usize));
        }

        let mut pools = self
            .pools
            .lock()
            .map_err(|_| TorshError::AllocationError("Failed to acquire pools lock".to_string()))?;

        if let Some(pool) = pools.get_mut(&size_class) {
            pool.deallocate(ptr, size)
                .map_err(|e| TorshError::AllocationError(e.to_string()))
        } else {
            // If no pool found, this might be NUMA-allocated memory
            // Use std::alloc::dealloc as fallback (same as allocate_on_numa_node)
            let alignment = if size <= 8 { 8 } else { 16 }; // Default alignment
            if let Ok(layout) = std::alloc::Layout::from_size_align(size, alignment) {
                unsafe {
                    std::alloc::dealloc(ptr, layout);
                }
                Ok(())
            } else {
                Err(TorshError::InvalidArgument(
                    "Invalid layout for deallocating memory".to_string(),
                ))
            }
        }
    }

    fn supports_unified_memory(&self) -> bool {
        // CPU memory is essentially "unified" since it's all system memory
        true
    }

    fn allocate_unified(&mut self, size: usize) -> Result<*mut u8> {
        // For CPU, unified memory is just regular allocation
        self.allocate_raw(size, std::mem::align_of::<u8>())
    }

    fn deallocate_unified(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        // For CPU, unified memory deallocation is just regular deallocation
        self.deallocate_raw(ptr, size)
    }

    fn prefetch_to_device(&self, _ptr: *mut u8, _size: usize) -> Result<()> {
        // No-op for CPU since all memory is already on the "device"
        Ok(())
    }

    fn prefetch_to_host(&self, _ptr: *mut u8, _size: usize) -> Result<()> {
        // No-op for CPU since all memory is already on the host
        Ok(())
    }

    fn set_memory_advice(
        &self,
        _ptr: *mut u8,
        _size: usize,
        _advice: crate::memory::MemoryAdvice,
    ) -> Result<()> {
        // No-op for CPU since memory advice doesn't apply
        Ok(())
    }

    fn available_memory(&self) -> Result<usize> {
        // Get system memory info
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return Ok(kb * 1024); // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback estimate
        Ok(4 * 1024 * 1024 * 1024) // 4GB
    }

    fn total_memory(&self) -> Result<usize> {
        // Get total system memory
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return Ok(kb * 1024); // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Fallback estimate
        Ok(8 * 1024 * 1024 * 1024) // 8GB
    }

    fn synchronize(&self) -> Result<()> {
        // CPU operations are synchronous, so nothing to do
        Ok(())
    }

    fn defragment(&mut self) -> Result<crate::memory::DefragmentationResult> {
        // Simple stub implementation for CPU
        Ok(crate::memory::DefragmentationResult {
            blocks_moved: 0,
            memory_compacted: 0,
            duration_ms: 0.0,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
            efficiency_improvement: 0.0,
            success: true,
        })
    }

    fn needs_defragmentation(&self) -> bool {
        // CPU memory pools typically don't need defragmentation
        false
    }

    fn fragmentation_info(&self) -> crate::memory::FragmentationInfo {
        // Simple implementation returning basic fragmentation info
        crate::memory::FragmentationInfo {
            overall_fragmentation: 0.1,
            external_fragmentation: 0.05,
            internal_fragmentation: 0.05,
            free_blocks: 1,
            allocated_blocks: 0,
            largest_free_block: 1024 * 1024,
            smallest_free_block: 1024,
            average_free_block: 512 * 1024,
            total_free_memory: 1024 * 1024,
            total_allocated_memory: 0,
            utilization_efficiency: 0.9,
            allocation_efficiency: 0.9,
        }
    }

    fn compact_memory(&mut self) -> Result<crate::memory::CompactionResult> {
        // Simple stub implementation for CPU
        Ok(crate::memory::CompactionResult {
            allocations_moved: 0,
            bytes_moved: 0,
            duration_ms: 0.0,
            largest_free_before: 1024 * 1024,
            largest_free_after: 1024 * 1024,
            free_blocks_before: 1,
            free_blocks_after: 1,
            success: true,
        })
    }

    fn set_defragmentation_policy(&mut self, _policy: crate::memory::DefragmentationPolicy) {
        // No-op for CPU since it doesn't need complex defragmentation
    }
}

impl Default for CpuMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU memory manager factory
pub struct CpuMemoryManagerFactory;

impl crate::memory::MemoryManagerFactory for CpuMemoryManagerFactory {
    fn create_manager(&self, _device: &Device) -> Result<Box<dyn crate::memory::MemoryManager>> {
        Ok(Box::new(CpuMemoryManager::new()))
    }

    fn backend_type(&self) -> crate::BackendType {
        crate::BackendType::Cpu
    }

    fn supports_device(&self, device: &Device) -> bool {
        device.device_type == DeviceType::Cpu
    }
}

/// CPU memory pool implementation
#[derive(Debug)]
pub struct CpuMemoryPool {
    size_class: usize,
    #[allow(clippy::arc_with_non_send_sync)]
    free_blocks: Arc<Mutex<Vec<*mut u8>>>,
    #[allow(clippy::arc_with_non_send_sync)]
    allocated_blocks: Arc<Mutex<HashMap<*mut u8, (usize, usize)>>>, // (size, alignment)
}

// SAFETY: We ensure thread safety by using Mutex and proper synchronization
unsafe impl Send for CpuMemoryPool {}
unsafe impl Sync for CpuMemoryPool {}

impl Clone for CpuMemoryPool {
    fn clone(&self) -> Self {
        Self {
            size_class: self.size_class,
            free_blocks: Arc::clone(&self.free_blocks),
            allocated_blocks: Arc::clone(&self.allocated_blocks),
        }
    }
}

impl CpuMemoryPool {
    /// Create a new CPU memory pool
    pub fn new(size_class: usize) -> Self {
        #[allow(clippy::arc_with_non_send_sync)]
        let free_blocks = Arc::new(Mutex::new(Vec::new()));
        #[allow(clippy::arc_with_non_send_sync)]
        let allocated_blocks = Arc::new(Mutex::new(HashMap::new()));

        Self {
            size_class,
            free_blocks,
            allocated_blocks,
        }
    }
}

impl MemoryPool for CpuMemoryPool {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8> {
        let mut free_blocks = self.free_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire free_blocks lock".to_string())
        })?;

        // Try to reuse a free block
        if let Some(ptr) = free_blocks.pop() {
            let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
            })?;
            allocated_blocks.insert(ptr, (size, alignment));
            return Ok(ptr);
        }

        // Allocate new block
        let layout = std::alloc::Layout::from_size_align(self.size_class, alignment)
            .map_err(|e| TorshError::AllocationError(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            return Err(TorshError::AllocationError(format!(
                "Failed to allocate {} bytes",
                size
            )));
        }

        let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
        })?;
        allocated_blocks.insert(ptr, (size, alignment));

        Ok(ptr)
    }

    fn deallocate(&mut self, ptr: *mut u8, _size: usize) -> Result<()> {
        let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
        })?;

        if allocated_blocks.remove(&ptr).is_some() {
            let mut free_blocks = self.free_blocks.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire free_blocks lock".to_string())
            })?;

            // Add to free list for reuse
            free_blocks.push(ptr);
            Ok(())
        } else {
            Err(TorshError::InvalidArgument(
                "Attempted to deallocate unknown pointer".to_string(),
            ))
        }
    }

    fn stats(&self) -> PoolStats {
        let allocated_blocks = self
            .allocated_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        let free_blocks = self
            .free_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));

        let allocated_bytes: usize = allocated_blocks.values().map(|(size, _)| *size).sum();
        let total_capacity = (allocated_blocks.len() + free_blocks.len()) * self.size_class;

        PoolStats {
            capacity: total_capacity,
            allocated: allocated_bytes,
            available: total_capacity - allocated_bytes,
            free_blocks: free_blocks.len(),
            allocated_blocks: allocated_blocks.len(),
            largest_free_block: if free_blocks.is_empty() {
                0
            } else {
                self.size_class
            },
            smallest_free_block: if free_blocks.is_empty() {
                0
            } else {
                self.size_class
            },
            average_free_block: if free_blocks.is_empty() {
                0
            } else {
                self.size_class
            },
        }
    }

    fn reset(&mut self) -> Result<()> {
        let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
        })?;
        let mut free_blocks = self.free_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire free_blocks lock".to_string())
        })?;

        // Deallocate all blocks
        for (&ptr, &(size, alignment)) in allocated_blocks.iter() {
            let layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
            unsafe {
                std::alloc::dealloc(ptr, layout);
            }
        }

        for &ptr in free_blocks.iter() {
            let layout = std::alloc::Layout::from_size_align(self.size_class, 1).unwrap();
            unsafe {
                std::alloc::dealloc(ptr, layout);
            }
        }

        allocated_blocks.clear();
        free_blocks.clear();

        Ok(())
    }

    fn capacity(&self) -> usize {
        let allocated_blocks = self
            .allocated_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        let free_blocks = self
            .free_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        (allocated_blocks.len() + free_blocks.len()) * self.size_class
    }

    fn available(&self) -> usize {
        let free_blocks = self
            .free_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        free_blocks.len() * self.size_class
    }

    fn defragment(&mut self) -> Result<crate::memory::DefragmentationResult> {
        // Simple stub implementation for CPU memory pool
        Ok(crate::memory::DefragmentationResult {
            blocks_moved: 0,
            memory_compacted: 0,
            duration_ms: 0.0,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
            efficiency_improvement: 0.0,
            success: true,
        })
    }

    fn needs_defragmentation(&self) -> bool {
        // CPU memory pools typically don't need defragmentation
        false
    }

    fn fragmentation_info(&self) -> crate::memory::FragmentationInfo {
        let allocated_count = self.allocated_blocks.lock().unwrap().len();
        let free_count = self.free_blocks.lock().unwrap().len();

        crate::memory::FragmentationInfo {
            overall_fragmentation: 0.1,
            external_fragmentation: 0.05,
            internal_fragmentation: 0.05,
            free_blocks: free_count,
            allocated_blocks: allocated_count,
            largest_free_block: self.size_class,
            smallest_free_block: if free_count > 0 { self.size_class } else { 0 },
            average_free_block: self.size_class,
            total_free_memory: free_count * self.size_class,
            total_allocated_memory: allocated_count * self.size_class,
            utilization_efficiency: 0.9,
            allocation_efficiency: 0.9,
        }
    }

    fn compact(&mut self) -> Result<crate::memory::CompactionResult> {
        // Simple stub implementation for CPU memory pool
        Ok(crate::memory::CompactionResult {
            allocations_moved: 0,
            bytes_moved: 0,
            duration_ms: 0.0,
            largest_free_before: self.size_class,
            largest_free_after: self.size_class,
            free_blocks_before: self.free_blocks.lock().unwrap().len(),
            free_blocks_after: self.free_blocks.lock().unwrap().len(),
            success: true,
        })
    }
}

// Implement Drop for CpuMemoryPool to clean up allocated memory
impl Drop for CpuMemoryPool {
    fn drop(&mut self) {
        // Clean up any remaining allocated blocks (still in use)
        if let Ok(allocated_blocks) = self.allocated_blocks.lock() {
            for (&ptr, &(size, alignment)) in allocated_blocks.iter() {
                let layout = std::alloc::Layout::from_size_align(size, alignment).unwrap();
                unsafe {
                    std::alloc::dealloc(ptr, layout);
                }
            }
        }

        // Clean up free blocks (available for reuse)
        if let Ok(free_blocks) = self.free_blocks.lock() {
            for &ptr in free_blocks.iter() {
                // Use consistent alignment for size_class allocations
                let layout = std::alloc::Layout::from_size_align(self.size_class, 8).unwrap();
                unsafe {
                    std::alloc::dealloc(ptr, layout);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Removed unused import

    #[test]
    fn test_size_class_calculation() {
        assert_eq!(CpuMemoryManager::calculate_size_class(32), 64);
        assert_eq!(CpuMemoryManager::calculate_size_class(64), 64);
        assert_eq!(CpuMemoryManager::calculate_size_class(65), 128);
        assert_eq!(CpuMemoryManager::calculate_size_class(1000), 1024);
        assert_eq!(CpuMemoryManager::calculate_size_class(2048), 2048);
    }

    #[test]
    fn test_memory_manager_creation() {
        let manager = CpuMemoryManager::new();
        let stats = manager.stats();
        assert_eq!(stats.allocated_memory, 0);
    }

    #[test]
    fn test_unified_memory_allocation() {
        let mut manager = CpuMemoryManager::new();

        // Test unified memory support
        assert!(manager.supports_unified_memory());

        // Test unified memory allocation
        let ptr = manager.allocate_unified(1024).unwrap();
        assert!(!ptr.is_null());

        // Test deallocation
        let result = manager.deallocate_unified(ptr, 1024);
        assert!(result.is_ok());
    }

    #[test]
    fn test_raw_memory_allocation() {
        let mut manager = CpuMemoryManager::new();

        // Test raw memory allocation with alignment
        let ptr = manager.allocate_raw(256, 16).unwrap();
        assert!(!ptr.is_null());

        // Check alignment
        assert_eq!(ptr as usize % 16, 0);

        // Test deallocation
        let result = manager.deallocate_raw(ptr, 256);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_info_queries() {
        let manager = CpuMemoryManager::new();

        // Test memory queries
        let total = manager.total_memory().unwrap();
        let available = manager.available_memory().unwrap();

        assert!(total > 0);
        assert!(available > 0);
        assert!(available <= total);
    }

    #[test]
    fn test_memory_operations() {
        let mut manager = CpuMemoryManager::new();

        // Test prefetch operations (should be no-ops for CPU)
        let ptr = manager.allocate_raw(64, 8).unwrap();

        assert!(manager.prefetch_to_device(ptr, 64).is_ok());
        assert!(manager.prefetch_to_host(ptr, 64).is_ok());

        // Test memory advice (should be no-op for CPU)
        assert!(manager
            .set_memory_advice(ptr, 64, crate::memory::MemoryAdvice::SetReadMostly)
            .is_ok());

        // Test synchronization (should be no-op for CPU)
        assert!(manager.synchronize().is_ok());

        // Cleanup
        manager.deallocate_raw(ptr, 64).unwrap();
    }

    #[test]
    fn test_memory_manager_factory() {
        use crate::device::{Device, DeviceInfo};
        use crate::memory::MemoryManagerFactory;

        let factory = CpuMemoryManagerFactory;
        let device = Device::new(
            0,
            DeviceType::Cpu,
            "Test CPU".to_string(),
            DeviceInfo::default(),
        );

        // Test device support
        assert!(factory.supports_device(&device));
        assert_eq!(factory.backend_type(), crate::BackendType::Cpu);

        // Test manager creation
        let manager = factory.create_manager(&device);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(manager.supports_unified_memory());
    }
}
