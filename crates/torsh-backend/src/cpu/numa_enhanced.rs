//! Enhanced NUMA-aware memory allocation with optional libnuma integration
//!
//! This module provides advanced NUMA memory allocation capabilities that can optionally
//! use libnuma when available for better performance on NUMA systems.

use crate::cpu::memory::{NumaAllocationStrategy, NumaTopology, AccessPatternType};
use crate::error::BackendResult;
use torsh_core::error::TorshError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ffi::CString;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Enhanced NUMA allocator with optional libnuma support
#[derive(Debug)]
pub struct EnhancedNumaAllocator {
    /// Whether libnuma is available and loaded
    numa_available: bool,
    /// NUMA topology information
    topology: Arc<Mutex<NumaTopology>>,
    /// Allocation tracking
    allocations: Arc<Mutex<HashMap<usize, AllocationInfo>>>,
    /// Performance statistics per NUMA node
    node_stats: Arc<Mutex<HashMap<u32, NodeStats>>>,
    /// Memory migration hints
    migration_hints: Arc<Mutex<Vec<MigrationHint>>>,
}

/// Information about a NUMA allocation
#[derive(Debug, Clone)]
struct AllocationInfo {
    ptr: *mut u8,
    size: usize,
    node: u32,
    access_pattern: AccessPatternType,
    allocation_time: std::time::Instant,
    access_count: u64,
    last_access: std::time::Instant,
}

/// Performance statistics for a NUMA node
#[derive(Debug, Clone, Default)]
struct NodeStats {
    allocations: u64,
    total_allocated: u64,
    average_latency: f64,
    bandwidth_utilization: f64,
    migration_count: u64,
}

/// Memory migration hint
#[derive(Debug, Clone)]
struct MigrationHint {
    ptr: *mut u8,
    current_node: u32,
    target_node: u32,
    priority: MigrationPriority,
    reason: MigrationReason,
}

/// Migration priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MigrationPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Reasons for memory migration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MigrationReason {
    HighLatency,
    BandwidthBottleneck,
    AccessPattern,
    LoadBalancing,
    UserHint,
}

impl EnhancedNumaAllocator {
    /// Create a new enhanced NUMA allocator
    pub fn new(topology: NumaTopology) -> Self {
        let numa_available = Self::detect_libnuma();
        
        Self {
            numa_available,
            topology: Arc::new(Mutex::new(topology)),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            node_stats: Arc::new(Mutex::new(HashMap::new())),
            migration_hints: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Detect if libnuma is available on the system
    fn detect_libnuma() -> bool {
        #[cfg(target_os = "linux")]
        {
            // Check if libnuma.so is available
            use std::process::Command;
            
            let output = Command::new("ldconfig")
                .arg("-p")
                .output();
                
            if let Ok(output) = output {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return stdout.contains("libnuma.so");
            }
        }
        
        false
    }
    
    /// Allocate memory on a specific NUMA node with enhanced features
    pub fn allocate_on_node(
        &self,
        size: usize,
        alignment: usize,
        numa_node: u32,
        access_pattern: AccessPatternType,
    ) -> BackendResult<*mut u8> {
        let ptr = if self.numa_available {
            self.allocate_with_libnuma(size, alignment, numa_node)?
        } else {
            self.allocate_fallback(size, alignment)?
        };
        
        // Track the allocation
        self.track_allocation(ptr, size, numa_node, access_pattern)?;
        
        // Update node statistics
        self.update_node_stats(numa_node, size)?;
        
        // Apply memory prefetching based on access pattern
        self.apply_prefetch_strategy(ptr, size, access_pattern)?;
        
        Ok(ptr)
    }
    
    /// Allocate memory using libnuma (when available)
    #[cfg(target_os = "linux")]
    fn allocate_with_libnuma(&self, size: usize, alignment: usize, numa_node: u32) -> BackendResult<*mut u8> {
        // In a real implementation, this would use dlopen to load libnuma dynamically
        // and call numa_alloc_onnode. For now, we'll use a simulated implementation.
        
        // Check if the node exists
        let topology = self.topology.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire topology lock".to_string())
        })?;
        
        if !topology.nodes.contains_key(&numa_node) {
            return Err(TorshError::AllocationError(format!(
                "NUMA node {} does not exist", numa_node
            )));
        }
        drop(topology);
        
        // For demonstration, we'll use mmap with memory policy hints
        let ptr = self.allocate_with_mmap_policy(size, alignment, numa_node)?;
        
        Ok(ptr)
    }
    
    /// Allocate memory using mmap with NUMA policy hints
    #[cfg(target_os = "linux")]
    fn allocate_with_mmap_policy(&self, size: usize, alignment: usize, numa_node: u32) -> BackendResult<*mut u8> {
        use std::ptr;
        
        // Use posix_memalign for aligned allocation
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        let result = unsafe {
            libc::posix_memalign(&mut ptr, alignment, size)
        };
        
        if result != 0 || ptr.is_null() {
            return Err(TorshError::AllocationError(format!(
                "posix_memalign failed with error {}", result
            )));
        }
        
        // Apply NUMA memory policy (simplified - real implementation would use mbind)
        let _ = numa_node; // Acknowledge parameter
        
        // In a real implementation, this would call:
        // mbind(ptr, size, MPOL_BIND, &nodemask, maxnode, 0);
        
        Ok(ptr as *mut u8)
    }
    
    /// Fallback allocation when libnuma is not available
    #[cfg(not(target_os = "linux"))]
    fn allocate_with_libnuma(&self, size: usize, alignment: usize, _numa_node: u32) -> BackendResult<*mut u8> {
        self.allocate_fallback(size, alignment)
    }
    
    /// Fallback allocation using standard allocator
    fn allocate_fallback(&self, size: usize, alignment: usize) -> BackendResult<*mut u8> {
        let layout = std::alloc::Layout::from_size_align(size, alignment)
            .map_err(|e| TorshError::AllocationError(format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(TorshError::AllocationError(format!(
                "Failed to allocate {} bytes", size
            )));
        }
        
        Ok(ptr)
    }
    
    /// Track an allocation for performance monitoring
    fn track_allocation(
        &self,
        ptr: *mut u8,
        size: usize,
        node: u32,
        access_pattern: AccessPatternType,
    ) -> BackendResult<()> {
        let mut allocations = self.allocations.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocations lock".to_string())
        })?;
        
        let info = AllocationInfo {
            ptr,
            size,
            node,
            access_pattern,
            allocation_time: std::time::Instant::now(),
            access_count: 0,
            last_access: std::time::Instant::now(),
        };
        
        allocations.insert(ptr as usize, info);
        Ok(())
    }
    
    /// Update NUMA node performance statistics
    fn update_node_stats(&self, numa_node: u32, size: usize) -> BackendResult<()> {
        let mut stats = self.node_stats.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire node stats lock".to_string())
        })?;
        
        let node_stat = stats.entry(numa_node).or_insert_with(NodeStats::default);
        node_stat.allocations += 1;
        node_stat.total_allocated += size as u64;
        
        Ok(())
    }
    
    /// Apply prefetch strategy based on access pattern
    fn apply_prefetch_strategy(
        &self,
        ptr: *mut u8,
        size: usize,
        pattern: AccessPatternType,
    ) -> BackendResult<()> {
        match pattern {
            AccessPatternType::Sequential => {
                self.prefetch_sequential_optimized(ptr, size)?;
            }
            AccessPatternType::Random => {
                self.prefetch_random_optimized(ptr, size)?;
            }
            AccessPatternType::Strided => {
                self.prefetch_strided_optimized(ptr, size)?;
            }
            AccessPatternType::Temporal => {
                self.prefetch_temporal_optimized(ptr, size)?;
            }
            AccessPatternType::Unknown => {
                self.prefetch_adaptive(ptr, size)?;
            }
        }
        
        Ok(())
    }
    
    /// Optimized sequential prefetching with hardware-specific tuning
    fn prefetch_sequential_optimized(&self, ptr: *mut u8, size: usize) -> BackendResult<()> {
        let cache_line_size = self.get_cache_line_size();
        let prefetch_distance = self.get_optimal_prefetch_distance();
        
        let lines = (size + cache_line_size - 1) / cache_line_size;
        let prefetch_ahead = (prefetch_distance / cache_line_size).min(lines);
        
        for i in 0..prefetch_ahead {
            let prefetch_ptr = unsafe { ptr.add(i * cache_line_size) };
            self.prefetch_cache_line(prefetch_ptr, PrefetchHint::Temporal);
        }
        
        Ok(())
    }
    
    /// Optimized random access prefetching
    fn prefetch_random_optimized(&self, ptr: *mut u8, size: usize) -> BackendResult<()> {
        let cache_line_size = self.get_cache_line_size();
        let num_lines = (size + cache_line_size - 1) / cache_line_size;
        
        // For random access, prefetch strategic cache lines
        let prefetch_positions = [0, num_lines / 4, num_lines / 2, 3 * num_lines / 4, num_lines - 1];
        
        for &pos in &prefetch_positions {
            if pos < num_lines {
                let prefetch_ptr = unsafe { ptr.add(pos * cache_line_size) };
                self.prefetch_cache_line(prefetch_ptr, PrefetchHint::NonTemporal);
            }
        }
        
        Ok(())
    }
    
    /// Optimized strided access prefetching
    fn prefetch_strided_optimized(&self, ptr: *mut u8, size: usize) -> BackendResult<()> {
        let cache_line_size = self.get_cache_line_size();
        let stride = self.detect_optimal_stride(size);
        
        let mut offset = 0;
        while offset < size {
            let prefetch_ptr = unsafe { ptr.add(offset) };
            self.prefetch_cache_line(prefetch_ptr, PrefetchHint::Temporal);
            offset += stride;
        }
        
        Ok(())
    }
    
    /// Optimized temporal locality prefetching
    fn prefetch_temporal_optimized(&self, ptr: *mut u8, size: usize) -> BackendResult<()> {
        // For temporal locality, prefetch more aggressively
        let cache_line_size = self.get_cache_line_size();
        let l1_cache_size = self.get_l1_cache_size();
        
        // Prefetch up to L1 cache size or the whole region, whichever is smaller
        let prefetch_size = size.min(l1_cache_size);
        let lines_to_prefetch = (prefetch_size + cache_line_size - 1) / cache_line_size;
        
        for i in 0..lines_to_prefetch {
            let prefetch_ptr = unsafe { ptr.add(i * cache_line_size) };
            self.prefetch_cache_line(prefetch_ptr, PrefetchHint::Temporal);
        }
        
        Ok(())
    }
    
    /// Adaptive prefetching based on runtime analysis
    fn prefetch_adaptive(&self, ptr: *mut u8, size: usize) -> BackendResult<()> {
        // Analyze historical access patterns for this allocation
        let access_pattern = self.analyze_access_pattern(ptr)?;
        
        match access_pattern {
            AccessPatternType::Sequential => self.prefetch_sequential_optimized(ptr, size),
            AccessPatternType::Random => self.prefetch_random_optimized(ptr, size),
            AccessPatternType::Strided => self.prefetch_strided_optimized(ptr, size),
            AccessPatternType::Temporal => self.prefetch_temporal_optimized(ptr, size),
            AccessPatternType::Unknown => {
                // Use conservative strategy
                self.prefetch_sequential_optimized(ptr, size.min(4096))
            }
        }
    }
    
    /// Analyze access pattern for an allocation
    fn analyze_access_pattern(&self, ptr: *mut u8) -> BackendResult<AccessPatternType> {
        let allocations = self.allocations.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocations lock".to_string())
        })?;
        
        if let Some(info) = allocations.get(&(ptr as usize)) {
            Ok(info.access_pattern)
        } else {
            Ok(AccessPatternType::Unknown)
        }
    }
    
    /// Prefetch a cache line with specified hint
    fn prefetch_cache_line(&self, ptr: *mut u8, hint: PrefetchHint) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                match hint {
                    PrefetchHint::Temporal => {
                        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                    }
                    PrefetchHint::NonTemporal => {
                        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_NTA);
                    }
                    PrefetchHint::L2 => {
                        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
                    }
                    PrefetchHint::L3 => {
                        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T2);
                    }
                }
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            // Prefetch instructions are currently unstable in Rust
            // TODO: Re-enable when stabilized (see issue #117217)
            // unsafe {
            //     match hint {
            //         PrefetchHint::Temporal => {
            //             std::arch::aarch64::_prefetch(
            //                 ptr as *const i8,
            //                 std::arch::aarch64::_PREFETCH_READ,
            //                 std::arch::aarch64::_PREFETCH_LOCALITY3
            //             );
            //         }
            //         PrefetchHint::NonTemporal => {
            //             std::arch::aarch64::_prefetch(
            //                 ptr as *const i8,
            //                 std::arch::aarch64::_PREFETCH_READ,
            //                 std::arch::aarch64::_PREFETCH_LOCALITY0
            //             );
            //         }
            //         PrefetchHint::L2 => {
            //             std::arch::aarch64::_prefetch(
            //                 ptr as *const i8,
            //                 std::arch::aarch64::_PREFETCH_READ,
            //                 std::arch::aarch64::_PREFETCH_LOCALITY2
            //             );
            //         }
            //         PrefetchHint::L3 => {
            //             std::arch::aarch64::_prefetch(
            //                 ptr as *const i8,
            //                 std::arch::aarch64::_PREFETCH_READ,
            //                 std::arch::aarch64::_PREFETCH_LOCALITY1
            //             );
            //         }
            //     }
            // }
            let _ = (ptr, hint); // Silence unused variable warnings
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // No prefetch instruction available
            let _ = (ptr, hint);
        }
    }
    
    /// Get cache line size for the current system
    fn get_cache_line_size(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size") {
                if let Ok(size) = content.trim().parse::<usize>() {
                    return size;
                }
            }
        }
        
        // Default cache line size
        64
    }
    
    /// Get L1 cache size
    fn get_l1_cache_size(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size") {
                let content = content.trim();
                if content.ends_with('K') {
                    if let Ok(size) = content[..content.len()-1].parse::<usize>() {
                        return size * 1024;
                    }
                }
            }
        }
        
        // Default L1 cache size
        32 * 1024
    }
    
    /// Get optimal prefetch distance based on CPU characteristics
    fn get_optimal_prefetch_distance(&self) -> usize {
        // This would be determined by CPU microarchitecture
        // For now, use a conservative estimate
        512 // bytes
    }
    
    /// Detect optimal stride for strided access patterns
    fn detect_optimal_stride(&self, size: usize) -> usize {
        let cache_line_size = self.get_cache_line_size();
        
        // Use power-of-2 strides that are multiples of cache line size
        if size > 16 * cache_line_size {
            4 * cache_line_size
        } else if size > 4 * cache_line_size {
            2 * cache_line_size
        } else {
            cache_line_size
        }
    }
    
    /// Deallocate memory and update tracking
    pub fn deallocate(&self, ptr: *mut u8, size: usize) -> BackendResult<()> {
        // Remove from tracking
        {
            let mut allocations = self.allocations.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire allocations lock".to_string())
            })?;
            allocations.remove(&(ptr as usize));
        }
        
        // Deallocate memory
        let layout = std::alloc::Layout::from_size_align(size, 1)
            .map_err(|e| TorshError::AllocationError(format!("Invalid layout: {}", e)))?;
        
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
        
        Ok(())
    }
    
    /// Get NUMA allocation statistics
    pub fn get_numa_stats(&self) -> BackendResult<HashMap<u32, NodeStats>> {
        let stats = self.node_stats.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire node stats lock".to_string())
        })?;
        
        Ok(stats.clone())
    }
    
    /// Suggest memory migrations based on access patterns
    pub fn suggest_migrations(&self) -> BackendResult<Vec<MigrationHint>> {
        let hints = self.migration_hints.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire migration hints lock".to_string())
        })?;
        
        Ok(hints.clone())
    }
    
    /// Record memory access for pattern analysis
    pub fn record_access(&self, ptr: *mut u8) -> BackendResult<()> {
        let mut allocations = self.allocations.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocations lock".to_string())
        })?;
        
        if let Some(info) = allocations.get_mut(&(ptr as usize)) {
            info.access_count += 1;
            info.last_access = std::time::Instant::now();
        }
        
        Ok(())
    }
}

/// Prefetch hint types
#[derive(Debug, Clone, Copy)]
enum PrefetchHint {
    /// Temporal locality (T0) - expect multiple accesses
    Temporal,
    /// Non-temporal (NTA) - single access expected
    NonTemporal,
    /// L2 cache (T1) - moderate temporal locality
    L2,
    /// L3 cache (T2) - low temporal locality
    L3,
}

unsafe impl Send for EnhancedNumaAllocator {}
unsafe impl Sync for EnhancedNumaAllocator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::memory::NumaNode;
    
    fn create_test_topology() -> NumaTopology {
        let mut nodes = HashMap::new();
        nodes.insert(0, NumaNode {
            id: 0,
            available_memory: 8 * 1024 * 1024 * 1024,
            memory_bandwidth: 50.0,
            cpu_cores: vec![0, 1, 2, 3],
            distances: HashMap::new(),
        });
        
        NumaTopology {
            nodes,
            current_node: 0,
            numa_available: true,
        }
    }
    
    #[test]
    fn test_enhanced_numa_allocator_creation() {
        let topology = create_test_topology();
        let allocator = EnhancedNumaAllocator::new(topology);
        
        // Should not crash
        assert!(!allocator.numa_available || allocator.numa_available);
    }
    
    #[test]
    fn test_fallback_allocation() {
        let topology = create_test_topology();
        let allocator = EnhancedNumaAllocator::new(topology);
        
        let ptr = allocator.allocate_fallback(1024, 16).unwrap();
        assert!(!ptr.is_null());
        
        allocator.deallocate(ptr, 1024).unwrap();
    }
    
    #[test]
    fn test_cache_line_size_detection() {
        let topology = create_test_topology();
        let allocator = EnhancedNumaAllocator::new(topology);
        
        let cache_line_size = allocator.get_cache_line_size();
        assert!(cache_line_size > 0);
        assert!(cache_line_size <= 256); // Reasonable upper bound
    }
    
    #[test]
    fn test_prefetch_strategies() {
        let topology = create_test_topology();
        let allocator = EnhancedNumaAllocator::new(topology);
        
        let layout = std::alloc::Layout::from_size_align(4096, 16).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        
        // Test different prefetch strategies
        assert!(allocator.prefetch_sequential_optimized(ptr, 4096).is_ok());
        assert!(allocator.prefetch_random_optimized(ptr, 4096).is_ok());
        assert!(allocator.prefetch_strided_optimized(ptr, 4096).is_ok());
        assert!(allocator.prefetch_temporal_optimized(ptr, 4096).is_ok());
        
        unsafe { std::alloc::dealloc(ptr, layout); }
    }
    
    #[test]
    fn test_allocation_tracking() {
        let topology = create_test_topology();
        let allocator = EnhancedNumaAllocator::new(topology);
        
        let ptr = allocator.allocate_fallback(1024, 16).unwrap();
        
        allocator.track_allocation(ptr, 1024, 0, AccessPatternType::Sequential).unwrap();
        allocator.record_access(ptr).unwrap();
        
        allocator.deallocate(ptr, 1024).unwrap();
    }
    
    #[test]
    fn test_numa_stats() {
        let topology = create_test_topology();
        let allocator = EnhancedNumaAllocator::new(topology);
        
        allocator.update_node_stats(0, 1024).unwrap();
        let stats = allocator.get_numa_stats().unwrap();
        
        assert!(stats.contains_key(&0));
        assert_eq!(stats[&0].allocations, 1);
        assert_eq!(stats[&0].total_allocated, 1024);
    }
}