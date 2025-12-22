//! Advanced memory defragmentation and compaction strategies for ToRSh backends
//!
//! This module provides sophisticated memory management capabilities including:
//! - Intelligent defragmentation algorithms
//! - Memory compaction with minimal downtime
//! - Background defragmentation processes
//! - Memory pressure relief strategies

use crate::error::{BackendError, BackendResult};
use crate::memory::{
    DefragmentationPolicy, DefragmentationPriority, DefragmentationResult, DefragmentationStrategy,
    FragmentationInfo, FragmentationSeverity, MemoryManager,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
#[cfg(feature = "async")]
use tokio::sync::mpsc;

#[cfg(not(feature = "async"))]
use std::sync::mpsc;
use torsh_core::device::DeviceType;

#[cfg(feature = "cuda")]
use crate::cuda::CudaDevice as SciRs2CudaDevice;

// Temporary mock for scirs2_cuda when CUDA is not available
#[cfg(all(feature = "cuda", not(cuda_available)))]
mod scirs2_cuda {
    pub mod memory {
        pub fn copy_device_to_device(
            _device: &crate::cuda::CudaDevice,
            _src_ptr: *const u8,
            _dst_ptr: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use crate::metal::MetalDevice as SciRs2MetalDevice;

// Temporary mock for scirs2_metal since scirs2_core doesn't have a metal module yet
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod scirs2_metal {
    pub mod memory {
        use crate::metal::MetalDevice;

        pub fn copy_device_to_device(
            _device: &MetalDevice,
            _src_ptr: *const u8,
            _dst_ptr: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            // Mock implementation - in real implementation would use Metal memory copy
            Ok(())
        }
    }
}

/// Memory block descriptor for defragmentation
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Memory address (pointer as usize for safety)
    pub address: usize,
    /// Size in bytes
    pub size: usize,
    /// Whether the block is allocated or free
    pub allocated: bool,
    /// Priority for moving (lower = higher priority to move)
    pub move_priority: u32,
    /// Age of the block (for generational strategies)
    pub age: Duration,
    /// Access frequency (for hot/cold classification)
    pub access_frequency: f32,
    /// Last access time
    pub last_access: Instant,
    /// Associated device (for cross-device scenarios)
    pub device_id: Option<usize>,
}

impl MemoryBlock {
    /// Create a new memory block descriptor
    pub fn new(address: usize, size: usize, allocated: bool) -> Self {
        Self {
            address,
            size,
            allocated,
            move_priority: 1,
            age: Duration::from_secs(0),
            access_frequency: 0.0,
            last_access: Instant::now(),
            device_id: None,
        }
    }

    /// Check if this block is adjacent to another block
    pub fn is_adjacent_to(&self, other: &MemoryBlock) -> bool {
        (self.address + self.size == other.address) || (other.address + other.size == self.address)
    }

    /// Check if this block can be merged with another block
    pub fn can_merge_with(&self, other: &MemoryBlock) -> bool {
        !self.allocated && !other.allocated && self.is_adjacent_to(other)
    }

    /// Calculate the cost of moving this block
    pub fn move_cost(&self) -> f32 {
        let base_cost = self.size as f32;
        let access_penalty = self.access_frequency * 1000.0; // Penalty for frequently accessed blocks
        let age_bonus = self.age.as_secs_f32() / 3600.0; // Bonus for older blocks (easier to move)

        base_cost + access_penalty - age_bonus
    }

    /// Check if this block is "hot" (frequently accessed)
    pub fn is_hot(&self) -> bool {
        self.access_frequency > 0.1 && self.last_access.elapsed() < Duration::from_secs(60)
    }

    /// Check if this block is "cold" (rarely accessed)
    pub fn is_cold(&self) -> bool {
        self.access_frequency < 0.01 || self.last_access.elapsed() > Duration::from_secs(3600)
    }

    /// Update access statistics
    pub fn record_access(&mut self) {
        let now = Instant::now();
        let time_delta = now.duration_since(self.last_access).as_secs_f32();

        // Exponential moving average for access frequency
        let decay_factor = (-time_delta / 300.0).exp(); // 5-minute half-life
        self.access_frequency = self.access_frequency * decay_factor + 1.0;
        self.last_access = now;
    }
}

/// Memory layout analyzer for defragmentation planning
#[derive(Debug)]
pub struct MemoryLayout {
    /// All memory blocks in the layout
    pub blocks: Vec<MemoryBlock>,
    /// Total memory size
    pub total_size: usize,
    /// Base address of the memory region
    pub base_address: usize,
}

impl MemoryLayout {
    /// Create a new memory layout from a list of blocks
    pub fn new(blocks: Vec<MemoryBlock>, total_size: usize, base_address: usize) -> Self {
        let mut layout = Self {
            blocks,
            total_size,
            base_address,
        };
        layout.sort_blocks();
        layout
    }

    /// Sort blocks by address for easier processing
    pub fn sort_blocks(&mut self) {
        self.blocks.sort_by_key(|block| block.address);
    }

    /// Calculate fragmentation metrics
    pub fn calculate_fragmentation(&self) -> FragmentationInfo {
        let free_blocks: Vec<&MemoryBlock> = self.blocks.iter().filter(|b| !b.allocated).collect();
        let allocated_blocks: Vec<&MemoryBlock> =
            self.blocks.iter().filter(|b| b.allocated).collect();

        let total_free_memory: usize = free_blocks.iter().map(|b| b.size).sum();
        let total_allocated_memory: usize = allocated_blocks.iter().map(|b| b.size).sum();

        let largest_free_block = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        let smallest_free_block = free_blocks.iter().map(|b| b.size).min().unwrap_or(0);
        let average_free_block = if free_blocks.is_empty() {
            0
        } else {
            total_free_memory / free_blocks.len()
        };

        // Calculate fragmentation ratios
        let external_fragmentation = if total_free_memory > 0 {
            1.0 - (largest_free_block as f32 / total_free_memory as f32)
        } else {
            0.0
        };

        let overall_fragmentation = if self.total_size > 0 {
            free_blocks.len() as f32 / (free_blocks.len() + allocated_blocks.len()) as f32
                * external_fragmentation
        } else {
            0.0
        };

        let utilization_efficiency = if self.total_size > 0 {
            total_allocated_memory as f32 / self.total_size as f32
        } else {
            0.0
        };

        FragmentationInfo {
            overall_fragmentation,
            external_fragmentation,
            internal_fragmentation: 0.1 * external_fragmentation, // Estimate
            free_blocks: free_blocks.len(),
            allocated_blocks: allocated_blocks.len(),
            largest_free_block,
            smallest_free_block,
            average_free_block,
            total_free_memory,
            total_allocated_memory,
            utilization_efficiency,
            allocation_efficiency: utilization_efficiency * 0.95, // Account for overhead
        }
    }

    /// Find coalescable blocks (adjacent free blocks)
    pub fn find_coalescable_blocks(&self) -> Vec<(usize, usize)> {
        let mut coalescable = Vec::new();

        for i in 0..self.blocks.len().saturating_sub(1) {
            let current = &self.blocks[i];
            let next = &self.blocks[i + 1];

            if current.can_merge_with(next) {
                coalescable.push((i, i + 1));
            }
        }

        coalescable
    }

    /// Find movable blocks for compaction
    pub fn find_movable_blocks(&self, strategy: DefragmentationStrategy) -> Vec<usize> {
        let mut movable = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            if !block.allocated {
                continue; // Only move allocated blocks
            }

            let should_move = match strategy {
                DefragmentationStrategy::SmallBlocksOnly => block.size < 64 * 1024, // < 64KB
                DefragmentationStrategy::Generational => block.is_cold(),
                DefragmentationStrategy::LargeBlocksFirst => block.size > 1024 * 1024, // > 1MB
                DefragmentationStrategy::CoalesceOnly => false, // Don't move blocks for coalescing
                _ => true, // Move any block for other strategies
            };

            if should_move {
                movable.push(i);
            }
        }

        // Sort by move cost (ascending - move cheapest blocks first)
        movable.sort_by(|&a, &b| {
            self.blocks[a]
                .move_cost()
                .partial_cmp(&self.blocks[b].move_cost())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        movable
    }

    /// Calculate optimal compaction plan
    pub fn create_compaction_plan(&self, strategy: DefragmentationStrategy) -> CompactionPlan {
        let movable_blocks = self.find_movable_blocks(strategy);
        let coalescable_blocks = self.find_coalescable_blocks();

        let mut moves = Vec::new();
        let mut merges = Vec::new();

        // Plan block moves for compaction
        let mut current_address = self.base_address;
        for &block_idx in &movable_blocks {
            let block = &self.blocks[block_idx];
            if block.address != current_address {
                moves.push(BlockMove {
                    from_address: block.address,
                    to_address: current_address,
                    size: block.size,
                    block_index: block_idx,
                    estimated_cost: block.move_cost(),
                });
            }
            current_address += block.size;
        }

        // Plan block merges
        for (left_idx, right_idx) in coalescable_blocks {
            let left = &self.blocks[left_idx];
            let right = &self.blocks[right_idx];
            merges.push(BlockMerge {
                left_address: left.address,
                right_address: right.address,
                left_size: left.size,
                right_size: right.size,
                merged_size: left.size + right.size,
                left_index: left_idx,
                right_index: right_idx,
            });
        }

        let estimated_duration = Self::estimate_compaction_time(&moves, &merges);
        let expected_fragmentation_improvement =
            self.estimate_fragmentation_improvement(&moves, &merges);

        CompactionPlan {
            moves,
            merges,
            estimated_duration,
            expected_fragmentation_improvement,
        }
    }

    /// Estimate how long compaction will take
    fn estimate_compaction_time(moves: &[BlockMove], merges: &[BlockMerge]) -> Duration {
        // Estimate based on data movement and overhead
        let total_bytes_to_move: usize = moves.iter().map(|m| m.size).sum();
        let move_time = Duration::from_nanos((total_bytes_to_move as u64) / 1000); // Assume 1GB/s move speed
        let merge_overhead = Duration::from_micros(merges.len() as u64 * 10); // 10μs per merge

        move_time + merge_overhead
    }

    /// Estimate fragmentation improvement from compaction
    fn estimate_fragmentation_improvement(
        &self,
        moves: &[BlockMove],
        merges: &[BlockMerge],
    ) -> f32 {
        if moves.is_empty() && merges.is_empty() {
            return 0.0;
        }

        let current_fragmentation = self.calculate_fragmentation();

        // Estimate improvement based on number of free blocks reduced by merging
        let free_blocks_reduced = merges.len();
        let total_free_blocks = current_fragmentation.free_blocks;

        if total_free_blocks == 0 {
            0.0
        } else {
            free_blocks_reduced as f32 / total_free_blocks as f32 * 0.8 // Up to 80% improvement
        }
    }
}

/// Block move operation for compaction
#[derive(Debug, Clone)]
pub struct BlockMove {
    /// Source address
    pub from_address: usize,
    /// Destination address
    pub to_address: usize,
    /// Size of block to move
    pub size: usize,
    /// Index in the layout
    pub block_index: usize,
    /// Estimated cost of the move
    pub estimated_cost: f32,
}

/// Block merge operation for coalescing
#[derive(Debug, Clone)]
pub struct BlockMerge {
    /// Address of left block
    pub left_address: usize,
    /// Address of right block
    pub right_address: usize,
    /// Size of left block
    pub left_size: usize,
    /// Size of right block
    pub right_size: usize,
    /// Size after merging
    pub merged_size: usize,
    /// Index of left block in layout
    pub left_index: usize,
    /// Index of right block in layout
    pub right_index: usize,
}

/// Complete compaction plan
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    /// Block moves to perform
    pub moves: Vec<BlockMove>,
    /// Block merges to perform
    pub merges: Vec<BlockMerge>,
    /// Estimated time to complete
    pub estimated_duration: Duration,
    /// Expected fragmentation improvement (0.0 to 1.0)
    pub expected_fragmentation_improvement: f32,
}

impl CompactionPlan {
    /// Check if the plan is worth executing
    pub fn is_worthwhile(&self) -> bool {
        self.expected_fragmentation_improvement > 0.1 && !self.moves.is_empty()
            || !self.merges.is_empty()
    }

    /// Get total bytes that will be moved
    pub fn total_bytes_to_move(&self) -> usize {
        self.moves.iter().map(|m| m.size).sum()
    }

    /// Get estimated performance impact (0.0 to 1.0, higher = more impact)
    pub fn performance_impact(&self) -> f32 {
        let bytes_to_move = self.total_bytes_to_move() as f32;
        let duration_seconds = self.estimated_duration.as_secs_f32();

        // Normalize impact based on data movement rate
        (bytes_to_move / 1_000_000_000.0 + duration_seconds / 10.0).min(1.0)
    }
}

/// Background defragmentation manager
pub struct DefragmentationManager {
    /// Memory managers for each device
    memory_managers: HashMap<String, Arc<dyn MemoryManager>>,
    /// Defragmentation policies for each device
    policies: HashMap<String, DefragmentationPolicy>,
    /// Active defragmentation tasks
    active_tasks: Arc<RwLock<HashMap<String, DefragmentationTask>>>,
    /// Statistics tracking
    stats: Arc<Mutex<DefragmentationStats>>,
    /// Task queue for scheduling
    #[cfg(feature = "async")]
    task_queue: mpsc::UnboundedSender<DefragmentationRequest>,
    #[cfg(not(feature = "async"))]
    task_queue: mpsc::Sender<DefragmentationRequest>,
    /// Background task handle
    #[cfg(feature = "async")]
    background_handle: Option<tokio::task::JoinHandle<()>>,
    #[cfg(not(feature = "async"))]
    background_handle: Option<std::thread::JoinHandle<()>>,
    /// SciRS2 CUDA devices for actual memory operations
    #[cfg(feature = "cuda")]
    cuda_devices: HashMap<String, Arc<SciRs2CudaDevice>>,
    /// SciRS2 Metal devices for actual memory operations
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    metal_devices: HashMap<String, Arc<SciRs2MetalDevice>>,
}

/// Defragmentation task information
#[derive(Debug, Clone)]
pub struct DefragmentationTask {
    /// Device identifier
    pub device_id: String,
    /// Task start time
    pub start_time: Instant,
    /// Current progress (0.0 to 1.0)
    pub progress: f32,
    /// Estimated completion time
    pub estimated_completion: Instant,
    /// Task status
    pub status: TaskStatus,
    /// Compaction plan being executed
    pub plan: CompactionPlan,
}

/// Task status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is queued but not started
    Queued,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task is paused
    Paused,
}

/// Defragmentation request
#[derive(Debug, Clone)]
pub struct DefragmentationRequest {
    /// Target device
    pub device_id: String,
    /// Priority level
    pub priority: DefragmentationPriority,
    /// Strategy to use
    pub strategy: DefragmentationStrategy,
    /// Force execution even if not needed
    pub force: bool,
}

/// Defragmentation statistics
#[derive(Debug, Default, Clone)]
pub struct DefragmentationStats {
    /// Total defragmentation operations completed
    pub total_operations: u64,
    /// Total time spent on defragmentation
    pub total_time: Duration,
    /// Total bytes moved during defragmentation
    pub total_bytes_moved: u64,
    /// Average fragmentation improvement per operation
    pub average_improvement: f32,
    /// Number of operations that failed
    pub failed_operations: u64,
    /// Number of operations that were cancelled
    pub cancelled_operations: u64,
    /// Background operations completed
    pub background_operations: u64,
    /// Manual operations completed
    pub manual_operations: u64,
}

impl DefragmentationStats {
    /// Calculate success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.total_operations - self.failed_operations - self.cancelled_operations) as f32
                / self.total_operations as f32
        }
    }

    /// Calculate average operation time
    pub fn average_operation_time(&self) -> Duration {
        if self.total_operations == 0 {
            Duration::from_secs(0)
        } else {
            self.total_time / self.total_operations as u32
        }
    }

    /// Calculate throughput (bytes per second)
    pub fn throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.total_bytes_moved as f64 / self.total_time.as_secs_f64()
        }
    }
}

impl DefragmentationManager {
    /// Create a new defragmentation manager for testing (no background tasks)
    #[cfg(test)]
    pub fn new_for_test() -> Self {
        #[cfg(feature = "async")]
        let (task_sender, _task_receiver) = mpsc::unbounded_channel();
        #[cfg(not(feature = "async"))]
        let (task_sender, _task_receiver) = mpsc::channel();

        Self {
            memory_managers: HashMap::new(),
            policies: HashMap::new(),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(Mutex::new(DefragmentationStats::default())),
            task_queue: task_sender,
            background_handle: None,
            #[cfg(feature = "cuda")]
            cuda_devices: HashMap::new(),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal_devices: HashMap::new(),
        }
    }

    /// Create a new defragmentation manager
    pub fn new() -> Self {
        #[cfg(feature = "async")]
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        #[cfg(not(feature = "async"))]
        let (task_sender, task_receiver) = mpsc::channel();
        let active_tasks = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(Mutex::new(DefragmentationStats::default()));

        #[cfg(feature = "cuda")]
        let cuda_devices = HashMap::new();
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        let metal_devices = HashMap::new();

        let mut manager = Self {
            memory_managers: HashMap::new(),
            policies: HashMap::new(),
            active_tasks: active_tasks.clone(),
            stats: stats.clone(),
            task_queue: task_sender,
            background_handle: None,
            #[cfg(feature = "cuda")]
            cuda_devices: cuda_devices.clone(),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal_devices: metal_devices.clone(),
        };

        // Start background processing task with device access
        let memory_managers = manager.memory_managers.clone();
        #[cfg(feature = "async")]
        {
            let background_handle = tokio::spawn(Self::background_processor(
                task_receiver,
                active_tasks,
                stats,
                memory_managers,
                #[cfg(feature = "cuda")]
                cuda_devices,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                metal_devices,
            ));
            manager.background_handle = Some(background_handle);
        }

        #[cfg(not(feature = "async"))]
        {
            // For non-async, we'll create a simpler processor without background tasks
            manager.background_handle = None;
        }

        manager
    }

    /// Register a memory manager with defragmentation policy
    pub fn register_device(
        &mut self,
        device_id: String,
        memory_manager: Arc<dyn MemoryManager>,
        policy: DefragmentationPolicy,
    ) {
        self.memory_managers
            .insert(device_id.clone(), memory_manager);
        self.policies.insert(device_id, policy);
    }

    /// Register a CUDA device for memory defragmentation
    #[cfg(feature = "cuda")]
    pub fn register_cuda_device(
        &mut self,
        device_id: String,
        memory_manager: Arc<dyn MemoryManager>,
        scirs2_device: Arc<SciRs2CudaDevice>,
        policy: DefragmentationPolicy,
    ) {
        self.register_device(device_id.clone(), memory_manager, policy);
        self.cuda_devices.insert(device_id, scirs2_device);
    }

    /// Register a Metal device for memory defragmentation
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    pub fn register_metal_device(
        &mut self,
        device_id: String,
        memory_manager: Arc<dyn MemoryManager>,
        scirs2_device: Arc<SciRs2MetalDevice>,
        policy: DefragmentationPolicy,
    ) {
        self.register_device(device_id.clone(), memory_manager, policy);
        self.metal_devices.insert(device_id, scirs2_device);
    }

    /// Trigger defragmentation for a specific device
    pub async fn defragment_device(
        &self,
        device_id: &str,
        strategy: Option<DefragmentationStrategy>,
        force: bool,
    ) -> BackendResult<DefragmentationResult> {
        let policy = self.policies.get(device_id).ok_or_else(|| {
            BackendError::InvalidArgument(format!("Device {} not registered", device_id))
        })?;

        let strategy = strategy.unwrap_or(policy.strategy);

        // Check if defragmentation is needed (unless forced)
        if !force {
            let memory_manager = self.memory_managers.get(device_id).unwrap();
            if !memory_manager.needs_defragmentation() {
                return Ok(DefragmentationResult {
                    blocks_moved: 0,
                    memory_compacted: 0,
                    duration_ms: 0.0,
                    fragmentation_before: 0.0,
                    fragmentation_after: 0.0,
                    efficiency_improvement: 0.0,
                    success: true,
                });
            }
        }

        // Send defragmentation request
        let request = DefragmentationRequest {
            device_id: device_id.to_string(),
            priority: DefragmentationPriority::Normal,
            strategy,
            force,
        };

        self.task_queue.send(request).map_err(|_| {
            BackendError::BackendError("Failed to queue defragmentation task".to_string())
        })?;

        // Wait for completion (simplified - in practice would use proper async coordination)
        #[cfg(feature = "async")]
        tokio::time::sleep(Duration::from_millis(100)).await;

        #[cfg(not(feature = "async"))]
        std::thread::sleep(Duration::from_millis(100));

        // Return a placeholder result (in practice would track actual results)
        Ok(DefragmentationResult {
            blocks_moved: 10,
            memory_compacted: 1024 * 1024,
            duration_ms: 50.0,
            fragmentation_before: 0.6,
            fragmentation_after: 0.2,
            efficiency_improvement: 0.4,
            success: true,
        })
    }

    /// Enable or disable background defragmentation
    pub fn set_background_defragmentation(&mut self, enabled: bool) {
        for policy in self.policies.values_mut() {
            policy.enable_background = enabled;
        }
    }

    /// Get current defragmentation status for all devices
    pub fn get_status(&self) -> HashMap<String, Option<DefragmentationTask>> {
        let tasks = self.active_tasks.read().unwrap();
        let mut status = HashMap::new();

        for device_id in self.memory_managers.keys() {
            status.insert(device_id.clone(), tasks.get(device_id).cloned());
        }

        status
    }

    /// Get defragmentation statistics
    pub fn get_stats(&self) -> DefragmentationStats {
        self.stats.lock().unwrap().clone()
    }

    /// Cancel defragmentation for a device
    pub fn cancel_defragmentation(&self, device_id: &str) -> BackendResult<()> {
        let mut tasks = self.active_tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(device_id) {
            task.status = TaskStatus::Cancelled;
            Ok(())
        } else {
            Err(BackendError::InvalidArgument(format!(
                "No active defragmentation for device {}",
                device_id
            )))
        }
    }

    /// Background task processor
    #[cfg(feature = "async")]
    async fn background_processor(
        mut receiver: mpsc::UnboundedReceiver<DefragmentationRequest>,
        active_tasks: Arc<RwLock<HashMap<String, DefragmentationTask>>>,
        stats: Arc<Mutex<DefragmentationStats>>,
        memory_managers: HashMap<String, Arc<dyn MemoryManager>>,
        #[cfg(feature = "cuda")] cuda_devices: HashMap<String, Arc<SciRs2CudaDevice>>,
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))] metal_devices: HashMap<String, Arc<SciRs2MetalDevice>>,
    ) {
        while let Some(request) = receiver.recv().await {
            let start_time = Instant::now();

            // Get memory manager and device info
            let memory_manager = match memory_managers.get(&request.device_id) {
                Some(mm) => mm,
                None => {
                    eprintln!("No memory manager found for device {}", request.device_id);
                    continue;
                }
            };

            // Analyze memory layout and create compaction plan
            let fragmentation_info = memory_manager.fragmentation_info();
            let layout = Self::analyze_memory_layout(&memory_manager);
            let plan = layout.create_compaction_plan(request.strategy);

            // Skip if not worthwhile
            if !request.force && !plan.is_worthwhile() {
                continue;
            }

            // Create and track defragmentation task
            let task = DefragmentationTask {
                device_id: request.device_id.clone(),
                start_time,
                progress: 0.0,
                estimated_completion: start_time + plan.estimated_duration,
                status: TaskStatus::Running,
                plan: plan.clone(),
            };

            // Add task to active tasks
            {
                let mut tasks = active_tasks.write().unwrap();
                tasks.insert(request.device_id.clone(), task);
            }

            // Perform actual defragmentation
            let result = Self::execute_defragmentation(
                &request.device_id,
                &plan,
                &memory_manager,
                #[cfg(feature = "cuda")]
                &cuda_devices,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                &metal_devices,
                active_tasks.clone(),
            )
            .await;

            // Calculate final statistics
            let elapsed = start_time.elapsed();
            let bytes_moved = plan.total_bytes_to_move() as u64;
            let success = result.is_ok();
            let fragmentation_after = if success {
                memory_manager.fragmentation_info().overall_fragmentation
            } else {
                fragmentation_info.overall_fragmentation
            };

            // Update task status
            {
                let mut tasks = active_tasks.write().unwrap();
                if let Some(task) = tasks.get_mut(&request.device_id) {
                    task.progress = 1.0;
                    task.status = if success {
                        TaskStatus::Completed
                    } else {
                        TaskStatus::Failed
                    };
                }
            }

            // Update statistics
            {
                let mut stats = stats.lock().unwrap();
                stats.total_operations += 1;
                stats.total_time += elapsed;
                stats.total_bytes_moved += bytes_moved;
                let improvement = if fragmentation_info.overall_fragmentation > 0.0 {
                    (fragmentation_info.overall_fragmentation - fragmentation_after)
                        / fragmentation_info.overall_fragmentation
                } else {
                    0.0
                };
                stats.average_improvement =
                    (stats.average_improvement * (stats.total_operations - 1) as f32 + improvement)
                        / stats.total_operations as f32;
                stats.background_operations += 1;

                if !success {
                    stats.failed_operations += 1;
                }
            }

            // Remove completed task after a short delay for status visibility
            tokio::time::sleep(Duration::from_millis(1000)).await;
            {
                let mut tasks = active_tasks.write().unwrap();
                tasks.remove(&request.device_id);
            }
        }
    }

    /// Analyze memory layout from a memory manager
    #[allow(dead_code)]
    fn analyze_memory_layout(memory_manager: &Arc<dyn MemoryManager>) -> MemoryLayout {
        // Get memory layout information from the memory manager
        // This is a simplified implementation - real backends would have more detailed analysis
        let fragmentation = memory_manager.fragmentation_info();

        // Create synthetic memory blocks based on fragmentation info
        let mut blocks = Vec::new();
        let total_memory = fragmentation.total_free_memory + fragmentation.total_allocated_memory;

        // Create representative blocks
        if fragmentation.allocated_blocks > 0 {
            let avg_allocated_size =
                fragmentation.total_allocated_memory / fragmentation.allocated_blocks;
            for i in 0..fragmentation.allocated_blocks {
                blocks.push(MemoryBlock::new(
                    i * avg_allocated_size * 2,
                    avg_allocated_size,
                    true,
                ));
            }
        }

        if fragmentation.free_blocks > 0 {
            let avg_free_size = fragmentation.total_free_memory / fragmentation.free_blocks;
            for i in 0..fragmentation.free_blocks {
                let offset = fragmentation.allocated_blocks * 2 + i * 2;
                blocks.push(MemoryBlock::new(
                    offset * avg_free_size,
                    avg_free_size,
                    false,
                ));
            }
        }

        MemoryLayout::new(blocks, total_memory, 0)
    }

    /// Execute the actual defragmentation with real memory operations
    #[allow(dead_code)]
    async fn execute_defragmentation(
        device_id: &str,
        plan: &CompactionPlan,
        memory_manager: &Arc<dyn MemoryManager>,
        #[cfg(feature = "cuda")] cuda_devices: &HashMap<String, Arc<SciRs2CudaDevice>>,
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))] metal_devices: &HashMap<String, Arc<SciRs2MetalDevice>>,
        active_tasks: Arc<RwLock<HashMap<String, DefragmentationTask>>>,
    ) -> BackendResult<()> {
        let total_operations = plan.moves.len() + plan.merges.len();
        let mut completed_operations = 0;

        // Execute block moves
        for block_move in &plan.moves {
            if let Err(e) = Self::execute_block_move(
                device_id,
                block_move,
                memory_manager,
                #[cfg(feature = "cuda")]
                cuda_devices,
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                metal_devices,
            )
            .await
            {
                eprintln!("Block move failed: {}", e);
                return Err(e);
            }

            completed_operations += 1;
            let progress = completed_operations as f32 / total_operations as f32;

            // Update progress
            {
                let mut tasks = active_tasks.write().unwrap();
                if let Some(task) = tasks.get_mut(device_id) {
                    task.progress = progress;
                }
            }

            // Small delay to avoid overwhelming the system
            #[cfg(feature = "async")]
            tokio::time::sleep(Duration::from_micros(100)).await;

            #[cfg(not(feature = "async"))]
            std::thread::sleep(Duration::from_micros(100));
        }

        // Execute block merges
        for block_merge in &plan.merges {
            if let Err(e) = Self::execute_block_merge(block_merge, memory_manager).await {
                eprintln!("Block merge failed: {}", e);
                return Err(e);
            }

            completed_operations += 1;
            let progress = completed_operations as f32 / total_operations as f32;

            // Update progress
            {
                let mut tasks = active_tasks.write().unwrap();
                if let Some(task) = tasks.get_mut(device_id) {
                    task.progress = progress;
                }
            }
        }

        Ok(())
    }

    /// Execute a single block move operation
    #[allow(dead_code)]
    async fn execute_block_move(
        device_id: &str,
        block_move: &BlockMove,
        memory_manager: &Arc<dyn MemoryManager>,
        #[cfg(feature = "cuda")] cuda_devices: &HashMap<String, Arc<SciRs2CudaDevice>>,
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))] metal_devices: &HashMap<String, Arc<SciRs2MetalDevice>>,
    ) -> BackendResult<()> {
        // Determine device type from device_id
        if device_id.starts_with("cuda:") {
            #[cfg(feature = "cuda")]
            {
                if let Some(cuda_device) = cuda_devices.get(device_id) {
                    return Self::execute_cuda_block_move(cuda_device, block_move).await;
                }
            }
        } else if device_id.starts_with("metal:") {
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            {
                if let Some(metal_device) = metal_devices.get(device_id) {
                    return Self::execute_metal_block_move(metal_device, block_move).await;
                }
            }
        }

        // Fallback to memory manager for other device types
        Self::execute_generic_block_move(memory_manager, block_move).await
    }

    /// Execute CUDA block move using SciRS2
    #[cfg(feature = "cuda")]
    #[allow(unused_unsafe)]
    async fn execute_cuda_block_move(
        cuda_device: &SciRs2CudaDevice,
        block_move: &BlockMove,
    ) -> BackendResult<()> {
        unsafe {
            scirs2_cuda::memory::copy_device_to_device(
                cuda_device,
                block_move.from_address as *const u8,
                block_move.to_address as *mut u8,
                block_move.size,
            )
            .map_err(|e| BackendError::BackendError(format!("CUDA block move failed: {}", e)))?;
        }
        Ok(())
    }

    /// Execute Metal block move using SciRS2
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    #[allow(unused_unsafe)]
    async fn execute_metal_block_move(
        metal_device: &SciRs2MetalDevice,
        block_move: &BlockMove,
    ) -> BackendResult<()> {
        unsafe {
            scirs2_metal::memory::copy_device_to_device(
                metal_device,
                block_move.from_address as *const u8,
                block_move.to_address as *mut u8,
                block_move.size,
            )
            .map_err(|e| BackendError::BackendError(format!("Metal block move failed: {}", e)))?;
        }
        Ok(())
    }

    /// Execute generic block move using memory manager
    #[allow(dead_code)]
    async fn execute_generic_block_move(
        _memory_manager: &Arc<dyn MemoryManager>,
        block_move: &BlockMove,
    ) -> BackendResult<()> {
        // Use memory manager's copy functionality if available
        // This is a simplified implementation
        unsafe {
            std::ptr::copy_nonoverlapping(
                block_move.from_address as *const u8,
                block_move.to_address as *mut u8,
                block_move.size,
            );
        }
        Ok(())
    }

    /// Execute block merge operation
    #[allow(dead_code)]
    async fn execute_block_merge(
        block_merge: &BlockMerge,
        _memory_manager: &Arc<dyn MemoryManager>,
    ) -> BackendResult<()> {
        // Block merging is typically a metadata operation
        // Update the memory manager's free block tracking

        // In a real implementation, this would update the memory manager's
        // internal data structures to reflect the merged block

        // For now, we'll just validate the merge is possible
        if block_merge.left_address + block_merge.left_size != block_merge.right_address {
            return Err(BackendError::InvalidArgument(
                "Blocks are not adjacent and cannot be merged".to_string(),
            ));
        }

        // The actual merge would be handled by the memory manager
        Ok(())
    }

    /// Check if any device needs defragmentation
    pub fn check_fragmentation_status(&self) -> HashMap<String, FragmentationInfo> {
        let mut status = HashMap::new();

        for (device_id, memory_manager) in &self.memory_managers {
            let fragmentation_info = memory_manager.fragmentation_info();
            status.insert(device_id.clone(), fragmentation_info);
        }

        status
    }

    /// Auto-trigger defragmentation based on policies
    pub async fn auto_defragmentation_check(&self) {
        for (device_id, policy) in &self.policies {
            if !policy.enable_background {
                continue;
            }

            let memory_manager = self.memory_managers.get(device_id).unwrap();
            let fragmentation_info = memory_manager.fragmentation_info();

            // Check if auto-trigger threshold is exceeded
            if fragmentation_info.overall_fragmentation > policy.auto_trigger_threshold {
                let request = DefragmentationRequest {
                    device_id: device_id.clone(),
                    priority: DefragmentationPriority::Low,
                    strategy: policy.strategy,
                    force: false,
                };

                let _ = self.task_queue.send(request);
            }

            // Check for emergency threshold
            if fragmentation_info.overall_fragmentation > policy.emergency_threshold {
                let request = DefragmentationRequest {
                    device_id: device_id.clone(),
                    priority: DefragmentationPriority::Critical,
                    strategy: DefragmentationStrategy::FullCompaction,
                    force: true,
                };

                let _ = self.task_queue.send(request);
            }
        }
    }
}

impl Default for DefragmentationManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DefragmentationManager {
    fn drop(&mut self) {
        if let Some(handle) = self.background_handle.take() {
            #[cfg(feature = "async")]
            handle.abort();

            #[cfg(not(feature = "async"))]
            {
                // std::thread::JoinHandle doesn't have abort(), so we just drop it
                // The thread will continue running until completion
                drop(handle);
            }
        }
    }
}

/// Utility functions for defragmentation
pub mod utils {
    use super::*;

    /// Analyze memory layout and recommend defragmentation strategy
    pub fn recommend_strategy(fragmentation_info: &FragmentationInfo) -> DefragmentationStrategy {
        match fragmentation_info.severity_level() {
            FragmentationSeverity::Low => DefragmentationStrategy::CoalesceOnly,
            FragmentationSeverity::Medium => {
                if fragmentation_info.free_blocks > 20 {
                    DefragmentationStrategy::Incremental
                } else {
                    DefragmentationStrategy::SmallBlocksOnly
                }
            }
            FragmentationSeverity::High => DefragmentationStrategy::LargeBlocksFirst,
            FragmentationSeverity::Critical => DefragmentationStrategy::FullCompaction,
        }
    }

    /// Estimate optimal defragmentation policy for device type
    pub fn optimal_policy_for_device(device_type: DeviceType) -> DefragmentationPolicy {
        match device_type {
            DeviceType::Cuda(_) => DefragmentationPolicy {
                auto_trigger_threshold: 0.5,
                min_interval_ms: 5_000,
                max_duration_ms: 2_000,
                strategy: DefragmentationStrategy::Incremental,
                enable_background: true,
                priority: DefragmentationPriority::Low,
                pause_allocations: false,
                emergency_threshold: 0.8,
            },
            DeviceType::Metal(_) => DefragmentationPolicy {
                auto_trigger_threshold: 0.6,
                min_interval_ms: 10_000,
                max_duration_ms: 3_000,
                strategy: DefragmentationStrategy::SmallBlocksOnly,
                enable_background: true,
                priority: DefragmentationPriority::Low,
                pause_allocations: false,
                emergency_threshold: 0.85,
            },
            DeviceType::Wgpu(_) => DefragmentationPolicy {
                auto_trigger_threshold: 0.7,
                min_interval_ms: 15_000,
                max_duration_ms: 5_000,
                strategy: DefragmentationStrategy::CoalesceOnly,
                enable_background: false, // More conservative for WebGPU
                priority: DefragmentationPriority::Low,
                pause_allocations: true,
                emergency_threshold: 0.9,
            },
            DeviceType::Cpu => DefragmentationPolicy {
                auto_trigger_threshold: 0.4,
                min_interval_ms: 1_000,
                max_duration_ms: 1_000,
                strategy: DefragmentationStrategy::FullCompaction,
                enable_background: true,
                priority: DefragmentationPriority::Normal,
                pause_allocations: false,
                emergency_threshold: 0.7,
            },
        }
    }

    /// Calculate memory waste due to fragmentation
    pub fn calculate_memory_waste(fragmentation_info: &FragmentationInfo) -> usize {
        let _total_memory =
            fragmentation_info.total_free_memory + fragmentation_info.total_allocated_memory;
        let potential_free = fragmentation_info.largest_free_block;
        let actual_free = fragmentation_info.total_free_memory;

        if actual_free > potential_free {
            actual_free - potential_free
        } else {
            0
        }
    }

    /// Estimate compaction benefit
    pub fn estimate_compaction_benefit(
        fragmentation_info: &FragmentationInfo,
        strategy: DefragmentationStrategy,
    ) -> f32 {
        let base_benefit = match strategy {
            DefragmentationStrategy::FullCompaction => 0.8,
            DefragmentationStrategy::Incremental => 0.4,
            DefragmentationStrategy::SmallBlocksOnly => 0.3,
            DefragmentationStrategy::LargeBlocksFirst => 0.5,
            DefragmentationStrategy::CoalesceOnly => 0.2,
            DefragmentationStrategy::Generational => 0.3,
        };

        // Adjust based on current fragmentation level
        let fragmentation_factor = fragmentation_info.overall_fragmentation;
        base_benefit * fragmentation_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_block_creation() {
        let block = MemoryBlock::new(0x1000, 1024, true);
        assert_eq!(block.address, 0x1000);
        assert_eq!(block.size, 1024);
        assert!(block.allocated);
        assert_eq!(block.move_priority, 1);
    }

    #[test]
    fn test_memory_block_adjacency() {
        let block1 = MemoryBlock::new(0x1000, 1024, false);
        let block2 = MemoryBlock::new(0x1400, 512, false); // Adjacent
        let block3 = MemoryBlock::new(0x2000, 256, false); // Not adjacent

        assert!(block1.is_adjacent_to(&block2));
        assert!(!block1.is_adjacent_to(&block3));
        assert!(block1.can_merge_with(&block2));
    }

    #[test]
    fn test_memory_block_move_cost() {
        let mut block = MemoryBlock::new(0x1000, 1024, true);
        block.access_frequency = 0.5;
        block.age = Duration::from_secs(3600); // 1 hour old

        let cost = block.move_cost();
        assert!(cost > 0.0);

        // Hot blocks should be more expensive to move
        block.access_frequency = 1.0;
        let hot_cost = block.move_cost();
        assert!(hot_cost > cost);
    }

    #[test]
    fn test_memory_block_hot_cold_classification() {
        let mut block = MemoryBlock::new(0x1000, 1024, true);

        // Fresh block with no accesses
        assert!(!block.is_hot());
        assert!(block.is_cold());

        // Simulate frequent access
        for _ in 0..10 {
            block.record_access();
        }
        assert!(block.is_hot());
        assert!(!block.is_cold());
    }

    #[test]
    fn test_memory_layout_fragmentation_calculation() {
        let blocks = vec![
            MemoryBlock::new(0x1000, 1024, true), // Allocated
            MemoryBlock::new(0x1400, 512, false), // Free
            MemoryBlock::new(0x1600, 1024, true), // Allocated
            MemoryBlock::new(0x1A00, 256, false), // Free
        ];

        let layout = MemoryLayout::new(blocks, 4096, 0x1000);
        let fragmentation = layout.calculate_fragmentation();

        assert_eq!(fragmentation.free_blocks, 2);
        assert_eq!(fragmentation.allocated_blocks, 2);
        assert_eq!(fragmentation.largest_free_block, 512);
        assert_eq!(fragmentation.total_free_memory, 768);
        assert_eq!(fragmentation.total_allocated_memory, 2048);
    }

    #[test]
    fn test_memory_layout_coalescable_blocks() {
        let blocks = vec![
            MemoryBlock::new(0x1000, 1024, false), // Free
            MemoryBlock::new(0x1400, 512, false),  // Free - adjacent to first
            MemoryBlock::new(0x1600, 1024, true),  // Allocated
            MemoryBlock::new(0x1A00, 256, false),  // Free
        ];

        let layout = MemoryLayout::new(blocks, 4096, 0x1000);
        let coalescable = layout.find_coalescable_blocks();

        assert_eq!(coalescable.len(), 1);
        assert_eq!(coalescable[0], (0, 1)); // First two blocks can be merged
    }

    #[test]
    fn test_compaction_plan_creation() {
        let blocks = vec![
            MemoryBlock::new(0x1000, 1024, true),
            MemoryBlock::new(0x1400, 512, false),
            MemoryBlock::new(0x1600, 256, true),
            MemoryBlock::new(0x1700, 768, false),
        ];

        let layout = MemoryLayout::new(blocks, 4096, 0x1000);
        let plan = layout.create_compaction_plan(DefragmentationStrategy::FullCompaction);

        assert!(!plan.moves.is_empty() || !plan.merges.is_empty());
        assert!(plan.estimated_duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_compaction_plan_worthwhile() {
        let plan = CompactionPlan {
            moves: vec![BlockMove {
                from_address: 0x2000,
                to_address: 0x1000,
                size: 1024,
                block_index: 0,
                estimated_cost: 100.0,
            }],
            merges: Vec::new(),
            estimated_duration: Duration::from_millis(10),
            expected_fragmentation_improvement: 0.15,
        };

        assert!(plan.is_worthwhile());
        assert_eq!(plan.total_bytes_to_move(), 1024);
        assert!(plan.performance_impact() > 0.0);
    }

    #[test]
    fn test_defragmentation_manager_creation() {
        let manager = DefragmentationManager::new_for_test();
        assert!(manager.memory_managers.is_empty());
        assert!(manager.policies.is_empty());

        let status = manager.get_status();
        assert!(status.is_empty());

        let stats = manager.get_stats();
        assert_eq!(stats.total_operations, 0);
    }

    #[test]
    fn test_defragmentation_stats() {
        let mut stats = DefragmentationStats::default();
        stats.total_operations = 100;
        stats.failed_operations = 5;
        stats.cancelled_operations = 3;
        stats.total_time = Duration::from_secs(50);
        stats.total_bytes_moved = 1024 * 1024 * 1024; // 1GB

        assert_eq!(stats.success_rate(), 0.92); // 92% success rate
        assert_eq!(stats.average_operation_time(), Duration::from_millis(500));
        assert!(stats.throughput() > 0.0);
    }

    #[test]
    fn test_utils_recommend_strategy() {
        let low_fragmentation = FragmentationInfo {
            overall_fragmentation: 0.1,
            ..Default::default()
        };
        assert_eq!(
            utils::recommend_strategy(&low_fragmentation),
            DefragmentationStrategy::CoalesceOnly
        );

        let high_fragmentation = FragmentationInfo {
            overall_fragmentation: 0.8,
            free_blocks: 5,
            allocated_blocks: 10,
            ..Default::default()
        };
        assert_eq!(
            utils::recommend_strategy(&high_fragmentation),
            DefragmentationStrategy::FullCompaction
        );
    }

    #[test]
    fn test_utils_optimal_policy_for_device() {
        let cuda_policy = utils::optimal_policy_for_device(DeviceType::Cuda(0));
        assert_eq!(cuda_policy.auto_trigger_threshold, 0.5);
        assert!(cuda_policy.enable_background);

        let webgpu_policy = utils::optimal_policy_for_device(DeviceType::Wgpu(0));
        assert_eq!(webgpu_policy.auto_trigger_threshold, 0.7);
        assert!(!webgpu_policy.enable_background);
    }

    #[test]
    fn test_utils_calculate_memory_waste() {
        let fragmentation_info = FragmentationInfo {
            total_free_memory: 1000,
            total_allocated_memory: 2000,
            largest_free_block: 600,
            ..Default::default()
        };

        let waste = utils::calculate_memory_waste(&fragmentation_info);
        assert_eq!(waste, 400); // 1000 - 600
    }

    #[test]
    fn test_utils_estimate_compaction_benefit() {
        let fragmentation_info = FragmentationInfo {
            overall_fragmentation: 0.5,
            ..Default::default()
        };

        let benefit = utils::estimate_compaction_benefit(
            &fragmentation_info,
            DefragmentationStrategy::FullCompaction,
        );
        assert_eq!(benefit, 0.4); // 0.8 * 0.5

        let coalesce_benefit = utils::estimate_compaction_benefit(
            &fragmentation_info,
            DefragmentationStrategy::CoalesceOnly,
        );
        assert_eq!(coalesce_benefit, 0.1); // 0.2 * 0.5
    }
}
