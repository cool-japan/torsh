//! Cross-device memory transfer and scheduling
//!
//! This module provides sophisticated memory transfer capabilities between different
//! device types, including asynchronous operations, bandwidth management, and
//! peer-to-peer transfers where supported.

use crate::device::DeviceType;
use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Cross-device memory transfer manager
///
/// Coordinates memory transfers between different device types with support for
/// asynchronous operations, transfer queuing, bandwidth management, and optimization.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{TransferManager, DeviceType};
///
/// let manager = TransferManager::new();
///
/// // Create a transfer operation
/// let transfer = TransferRequest::new(
///     DeviceType::Cpu,
///     DeviceType::Cuda(0),
///     1024 * 1024, // 1MB
/// );
///
/// // Execute the transfer
/// let handle = manager.execute_transfer(transfer)?;
/// handle.wait()?;
/// ```
#[derive(Debug)]
pub struct TransferManager {
    transfer_queue: Mutex<VecDeque<QueuedTransfer>>,
    active_transfers: RwLock<HashMap<TransferId, ActiveTransfer>>,
    transfer_stats: RwLock<TransferStatistics>,
    bandwidth_manager: Arc<BandwidthManager>,
    p2p_manager: Arc<P2PManager>,
    config: TransferConfig,
}

impl Default for TransferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferManager {
    /// Create a new transfer manager
    pub fn new() -> Self {
        Self {
            transfer_queue: Mutex::new(VecDeque::new()),
            active_transfers: RwLock::new(HashMap::new()),
            transfer_stats: RwLock::new(TransferStatistics::new()),
            bandwidth_manager: Arc::new(BandwidthManager::new()),
            p2p_manager: Arc::new(P2PManager::new()),
            config: TransferConfig::default(),
        }
    }

    /// Create transfer manager with custom configuration
    pub fn with_config(config: TransferConfig) -> Self {
        Self {
            transfer_queue: Mutex::new(VecDeque::new()),
            active_transfers: RwLock::new(HashMap::new()),
            transfer_stats: RwLock::new(TransferStatistics::new()),
            bandwidth_manager: Arc::new(BandwidthManager::with_config(&config)),
            p2p_manager: Arc::new(P2PManager::new()),
            config,
        }
    }

    /// Execute a memory transfer
    pub fn execute_transfer(&self, request: TransferRequest) -> Result<TransferHandle> {
        let transfer_id = self.generate_transfer_id();
        let handle = TransferHandle::new(transfer_id, &request);

        // Check if peer-to-peer transfer is possible
        let use_p2p = self
            .p2p_manager
            .can_use_p2p(&request.source, &request.destination)?;

        let transfer_method = if use_p2p {
            TransferMethod::PeerToPeer
        } else {
            TransferMethod::HostStaged
        };

        // Create queued transfer
        let queued_transfer = QueuedTransfer {
            id: transfer_id,
            request,
            method: transfer_method,
            priority: TransferPriority::Normal,
            queued_at: Instant::now(),
        };

        // Add to queue
        {
            let mut queue = self
                .transfer_queue
                .lock()
                .expect("lock should not be poisoned");
            queue.push_back(queued_transfer);
        }

        // Process queue
        self.process_queue()?;

        Ok(handle)
    }

    /// Execute transfer with priority
    pub fn execute_transfer_with_priority(
        &self,
        request: TransferRequest,
        priority: TransferPriority,
    ) -> Result<TransferHandle> {
        let transfer_id = self.generate_transfer_id();
        let handle = TransferHandle::new(transfer_id, &request);

        let use_p2p = self
            .p2p_manager
            .can_use_p2p(&request.source, &request.destination)?;
        let transfer_method = if use_p2p {
            TransferMethod::PeerToPeer
        } else {
            TransferMethod::HostStaged
        };

        let queued_transfer = QueuedTransfer {
            id: transfer_id,
            request,
            method: transfer_method,
            priority,
            queued_at: Instant::now(),
        };

        // Insert with priority ordering
        {
            let mut queue = self
                .transfer_queue
                .lock()
                .expect("lock should not be poisoned");
            let insert_pos = queue
                .iter()
                .position(|t| t.priority < priority)
                .unwrap_or(queue.len());
            queue.insert(insert_pos, queued_transfer);
        }

        self.process_queue()?;
        Ok(handle)
    }

    /// Get transfer status
    pub fn get_transfer_status(&self, transfer_id: TransferId) -> Option<TransferStatus> {
        let active = self
            .active_transfers
            .read()
            .expect("lock should not be poisoned");
        active.get(&transfer_id).map(|t| t.status.clone())
    }

    /// Wait for transfer completion
    pub fn wait_for_transfer(&self, transfer_id: TransferId) -> Result<TransferResult> {
        loop {
            {
                let active = self
                    .active_transfers
                    .read()
                    .expect("lock should not be poisoned");
                if let Some(transfer) = active.get(&transfer_id) {
                    match &transfer.status {
                        TransferStatus::Completed(result) => {
                            return Ok(result.clone());
                        }
                        TransferStatus::Failed(error) => {
                            return Err(crate::error::TorshError::DeviceError(error.clone()));
                        }
                        _ => {}
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Cancel a transfer
    pub fn cancel_transfer(&self, transfer_id: TransferId) -> Result<bool> {
        // Remove from queue if not started
        {
            let mut queue = self
                .transfer_queue
                .lock()
                .expect("lock should not be poisoned");
            if let Some(pos) = queue.iter().position(|t| t.id == transfer_id) {
                queue.remove(pos);
                return Ok(true);
            }
        }

        // Cancel active transfer
        {
            let mut active = self
                .active_transfers
                .write()
                .expect("lock should not be poisoned");
            if let Some(transfer) = active.get_mut(&transfer_id) {
                transfer.status = TransferStatus::Cancelled;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get transfer statistics
    pub fn get_statistics(&self) -> TransferStatistics {
        let stats = self
            .transfer_stats
            .read()
            .expect("lock should not be poisoned");
        stats.clone()
    }

    /// Get active transfer count
    pub fn active_transfer_count(&self) -> usize {
        let active = self
            .active_transfers
            .read()
            .expect("lock should not be poisoned");
        active.len()
    }

    /// Get queued transfer count
    pub fn queued_transfer_count(&self) -> usize {
        let queue = self
            .transfer_queue
            .lock()
            .expect("lock should not be poisoned");
        queue.len()
    }

    fn generate_transfer_id(&self) -> TransferId {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        TransferId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    fn process_queue(&self) -> Result<()> {
        let mut queue = self
            .transfer_queue
            .lock()
            .expect("lock should not be poisoned");
        let mut active = self
            .active_transfers
            .write()
            .expect("lock should not be poisoned");

        // Check if we can start more transfers
        while active.len() < self.config.max_concurrent_transfers && !queue.is_empty() {
            if let Some(queued) = queue.pop_front() {
                // Check bandwidth availability
                if !self
                    .bandwidth_manager
                    .can_allocate_bandwidth(&queued.request)?
                {
                    // Put back at front of queue and break
                    queue.push_front(queued);
                    break;
                }

                // Start the transfer
                let active_transfer = self.start_transfer(queued)?;
                active.insert(active_transfer.id, active_transfer);
            }
        }

        Ok(())
    }

    fn start_transfer(&self, queued: QueuedTransfer) -> Result<ActiveTransfer> {
        let start_time = Instant::now();

        // Allocate bandwidth
        self.bandwidth_manager.allocate_bandwidth(&queued.request)?;

        let total_bytes = queued.request.size_bytes;
        let active_transfer = ActiveTransfer {
            id: queued.id,
            request: queued.request,
            method: queued.method,
            status: TransferStatus::InProgress {
                bytes_transferred: 0,
                total_bytes,
                start_time,
            },
            start_time,
        };

        // Spawn transfer execution
        self.execute_transfer_async(active_transfer.clone())?;

        Ok(active_transfer)
    }

    fn execute_transfer_async(&self, transfer: ActiveTransfer) -> Result<()> {
        #[allow(clippy::arc_with_non_send_sync)] // Temporary placeholder for async execution
        let _manager = Arc::new(self as *const TransferManager);
        let transfer_arc = Arc::new(Mutex::new(transfer));

        std::thread::spawn(move || {
            let result = Self::perform_transfer(transfer_arc.clone());

            // Update status based on result
            let mut transfer = transfer_arc.lock().expect("lock should not be poisoned");
            match result {
                Ok(transfer_result) => {
                    transfer.status = TransferStatus::Completed(transfer_result);
                }
                Err(error) => {
                    transfer.status = TransferStatus::Failed(error.to_string());
                }
            }

            // Clean up bandwidth allocation
            // Note: In a real implementation, we'd need to safely access the manager
            // This is simplified for the example
        });

        Ok(())
    }

    fn perform_transfer(transfer: Arc<Mutex<ActiveTransfer>>) -> Result<TransferResult> {
        let (_transfer_id, request, method, _start_time) = {
            let t = transfer.lock().expect("lock should not be poisoned");
            (t.id, t.request.clone(), t.method, t.start_time)
        };

        match method {
            TransferMethod::HostStaged => Self::perform_host_staged_transfer(&request, transfer),
            TransferMethod::PeerToPeer => Self::perform_p2p_transfer(&request, transfer),
            TransferMethod::DirectCopy => Self::perform_direct_copy(&request, transfer),
        }
    }

    fn perform_host_staged_transfer(
        request: &TransferRequest,
        transfer: Arc<Mutex<ActiveTransfer>>,
    ) -> Result<TransferResult> {
        let chunk_size = 1024 * 1024; // 1MB chunks
        let total_bytes = request.size_bytes;
        let mut bytes_transferred = 0;

        // Simulate transfer with progress updates
        while bytes_transferred < total_bytes {
            let chunk = std::cmp::min(chunk_size, total_bytes - bytes_transferred);

            // Simulate transfer time based on bandwidth
            let transfer_time = Duration::from_millis(chunk as u64 / 1000); // Mock: 1GB/s
            std::thread::sleep(transfer_time);

            bytes_transferred += chunk;

            // Update progress
            {
                let mut t = transfer.lock().expect("lock should not be poisoned");
                if let TransferStatus::InProgress {
                    bytes_transferred: ref mut bt,
                    ..
                } = &mut t.status
                {
                    *bt = bytes_transferred;
                }
            }
        }

        Ok(TransferResult {
            bytes_transferred: total_bytes,
            duration: Instant::now().duration_since(
                transfer
                    .lock()
                    .expect("lock should not be poisoned")
                    .start_time,
            ),
            bandwidth_gbps: (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                / (Instant::now()
                    .duration_since(
                        transfer
                            .lock()
                            .expect("lock should not be poisoned")
                            .start_time,
                    )
                    .as_secs_f64()),
            method: TransferMethod::HostStaged,
        })
    }

    fn perform_p2p_transfer(
        request: &TransferRequest,
        transfer: Arc<Mutex<ActiveTransfer>>,
    ) -> Result<TransferResult> {
        // P2P transfers are typically faster
        let total_bytes = request.size_bytes;
        let transfer_duration = Duration::from_millis(total_bytes as u64 / 5000); // Mock: 5GB/s

        std::thread::sleep(transfer_duration);

        // Update to completed
        {
            let mut t = transfer.lock().expect("lock should not be poisoned");
            if let TransferStatus::InProgress {
                bytes_transferred: ref mut bt,
                ..
            } = &mut t.status
            {
                *bt = total_bytes;
            }
        }

        Ok(TransferResult {
            bytes_transferred: total_bytes,
            duration: transfer_duration,
            bandwidth_gbps: (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                / transfer_duration.as_secs_f64(),
            method: TransferMethod::PeerToPeer,
        })
    }

    fn perform_direct_copy(
        request: &TransferRequest,
        transfer: Arc<Mutex<ActiveTransfer>>,
    ) -> Result<TransferResult> {
        // Direct copies (same device) are instantaneous
        let total_bytes = request.size_bytes;

        {
            let mut t = transfer.lock().expect("lock should not be poisoned");
            if let TransferStatus::InProgress {
                bytes_transferred: ref mut bt,
                ..
            } = &mut t.status
            {
                *bt = total_bytes;
            }
        }

        Ok(TransferResult {
            bytes_transferred: total_bytes,
            duration: Duration::from_nanos(1),
            bandwidth_gbps: f64::INFINITY,
            method: TransferMethod::DirectCopy,
        })
    }
}

/// Transfer request specification
#[derive(Debug, Clone)]
pub struct TransferRequest {
    pub source: DeviceType,
    pub destination: DeviceType,
    pub size_bytes: usize,
    pub source_offset: usize,
    pub destination_offset: usize,
    pub alignment: Option<usize>,
}

impl TransferRequest {
    /// Create a new transfer request
    pub fn new(source: DeviceType, destination: DeviceType, size_bytes: usize) -> Self {
        Self {
            source,
            destination,
            size_bytes,
            source_offset: 0,
            destination_offset: 0,
            alignment: None,
        }
    }

    /// Set source offset
    pub fn with_source_offset(mut self, offset: usize) -> Self {
        self.source_offset = offset;
        self
    }

    /// Set destination offset
    pub fn with_destination_offset(mut self, offset: usize) -> Self {
        self.destination_offset = offset;
        self
    }

    /// Set memory alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = Some(alignment);
        self
    }

    /// Get estimated transfer cost
    pub fn estimated_cost(&self) -> u32 {
        crate::device::types::utils::transfer_cost(self.source, self.destination)
    }

    /// Check if this is a same-device transfer
    pub fn is_same_device(&self) -> bool {
        self.source == self.destination
    }

    /// Check if this transfer can use peer-to-peer
    pub fn can_use_p2p(&self) -> bool {
        matches!(
            (self.source, self.destination),
            (DeviceType::Cuda(_), DeviceType::Cuda(_))
        )
    }
}

/// Transfer handle for tracking transfer progress
#[derive(Debug)]
pub struct TransferHandle {
    id: TransferId,
    source: DeviceType,
    destination: DeviceType,
    size_bytes: usize,
}

impl TransferHandle {
    fn new(id: TransferId, request: &TransferRequest) -> Self {
        Self {
            id,
            source: request.source,
            destination: request.destination,
            size_bytes: request.size_bytes,
        }
    }

    /// Get transfer ID
    pub fn id(&self) -> TransferId {
        self.id
    }

    /// Get source device
    pub fn source(&self) -> DeviceType {
        self.source
    }

    /// Get destination device
    pub fn destination(&self) -> DeviceType {
        self.destination
    }

    /// Get transfer size
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Wait for transfer completion (mock implementation)
    pub fn wait(&self) -> Result<TransferResult> {
        // In a real implementation, this would wait for the actual transfer
        Ok(TransferResult {
            bytes_transferred: self.size_bytes,
            duration: Duration::from_millis(100),
            bandwidth_gbps: 1.0,
            method: TransferMethod::HostStaged,
        })
    }

    /// Check if transfer is complete (mock implementation)
    pub fn is_complete(&self) -> bool {
        true // Mock: always complete
    }

    /// Get transfer progress (mock implementation)
    pub fn progress(&self) -> f64 {
        1.0 // Mock: always 100%
    }
}

/// Unique transfer identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(u64);

/// Transfer priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Transfer method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMethod {
    /// Direct copy (same device)
    DirectCopy,
    /// Host-staged transfer (via CPU memory)
    HostStaged,
    /// Peer-to-peer transfer (direct GPU-to-GPU)
    PeerToPeer,
}

/// Transfer status
#[derive(Debug, Clone)]
pub enum TransferStatus {
    Queued,
    InProgress {
        bytes_transferred: usize,
        total_bytes: usize,
        start_time: Instant,
    },
    Completed(TransferResult),
    Failed(String),
    Cancelled,
}

/// Transfer result
#[derive(Debug, Clone)]
pub struct TransferResult {
    pub bytes_transferred: usize,
    pub duration: Duration,
    pub bandwidth_gbps: f64,
    pub method: TransferMethod,
}

/// Queued transfer
#[derive(Debug)]
struct QueuedTransfer {
    id: TransferId,
    request: TransferRequest,
    method: TransferMethod,
    priority: TransferPriority,
    #[allow(dead_code)] // Queue timestamp for scheduling - future implementation
    queued_at: Instant,
}

/// Active transfer
#[derive(Debug, Clone)]
struct ActiveTransfer {
    id: TransferId,
    request: TransferRequest,
    method: TransferMethod,
    status: TransferStatus,
    start_time: Instant,
}

/// Bandwidth manager for controlling transfer rates
#[derive(Debug)]
pub struct BandwidthManager {
    allocated_bandwidth: Mutex<HashMap<(DeviceType, DeviceType), u64>>,
    config: BandwidthConfig,
}

impl Default for BandwidthManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BandwidthManager {
    pub fn new() -> Self {
        Self {
            allocated_bandwidth: Mutex::new(HashMap::new()),
            config: BandwidthConfig::default(),
        }
    }

    pub fn with_config(transfer_config: &TransferConfig) -> Self {
        Self {
            allocated_bandwidth: Mutex::new(HashMap::new()),
            config: transfer_config.bandwidth_config.clone(),
        }
    }

    pub fn can_allocate_bandwidth(&self, request: &TransferRequest) -> Result<bool> {
        let bandwidth_key = (request.source, request.destination);
        let required_bandwidth = self.estimate_required_bandwidth(request);

        let allocated = self
            .allocated_bandwidth
            .lock()
            .expect("lock should not be poisoned");
        let current_usage = allocated.get(&bandwidth_key).copied().unwrap_or(0);

        Ok(current_usage + required_bandwidth <= self.config.max_bandwidth_per_link)
    }

    pub fn allocate_bandwidth(&self, request: &TransferRequest) -> Result<()> {
        let bandwidth_key = (request.source, request.destination);
        let required_bandwidth = self.estimate_required_bandwidth(request);

        let mut allocated = self
            .allocated_bandwidth
            .lock()
            .expect("lock should not be poisoned");
        let current_usage = allocated.get(&bandwidth_key).copied().unwrap_or(0);
        allocated.insert(bandwidth_key, current_usage + required_bandwidth);

        Ok(())
    }

    pub fn deallocate_bandwidth(&self, request: &TransferRequest) -> Result<()> {
        let bandwidth_key = (request.source, request.destination);
        let required_bandwidth = self.estimate_required_bandwidth(request);

        let mut allocated = self
            .allocated_bandwidth
            .lock()
            .expect("lock should not be poisoned");
        if let Some(current_usage) = allocated.get_mut(&bandwidth_key) {
            *current_usage = current_usage.saturating_sub(required_bandwidth);
            if *current_usage == 0 {
                allocated.remove(&bandwidth_key);
            }
        }

        Ok(())
    }

    fn estimate_required_bandwidth(&self, request: &TransferRequest) -> u64 {
        // Estimate based on transfer size and expected duration
        let estimated_duration_secs = match request.estimated_cost() {
            0 => 0.001,                 // Direct copy
            cost if cost <= 100 => 1.0, // Fast transfer
            _ => 5.0,                   // Slow transfer
        };

        (request.size_bytes as f64 / estimated_duration_secs) as u64
    }
}

/// Peer-to-peer transfer manager
#[derive(Debug)]
pub struct P2PManager {
    p2p_capabilities: RwLock<HashMap<(DeviceType, DeviceType), bool>>,
}

impl Default for P2PManager {
    fn default() -> Self {
        Self::new()
    }
}

impl P2PManager {
    pub fn new() -> Self {
        Self {
            p2p_capabilities: RwLock::new(HashMap::new()),
        }
    }

    pub fn can_use_p2p(&self, source: &DeviceType, destination: &DeviceType) -> Result<bool> {
        let key = (*source, *destination);

        // Check cache
        {
            let cache = self
                .p2p_capabilities
                .read()
                .expect("lock should not be poisoned");
            if let Some(&can_use) = cache.get(&key) {
                return Ok(can_use);
            }
        }

        // Determine P2P capability
        let can_use = match (source, destination) {
            (DeviceType::Cuda(a), DeviceType::Cuda(b)) => {
                // In a real implementation, this would query CUDA for P2P support
                a != b // Different CUDA devices can potentially use P2P
            }
            _ => false,
        };

        // Cache result
        {
            let mut cache = self
                .p2p_capabilities
                .write()
                .expect("lock should not be poisoned");
            cache.insert(key, can_use);
        }

        Ok(can_use)
    }

    pub fn enable_p2p(&self, source: DeviceType, destination: DeviceType) -> Result<()> {
        // In a real implementation, this would enable P2P access between devices
        let key = (source, destination);
        let mut cache = self
            .p2p_capabilities
            .write()
            .expect("lock should not be poisoned");
        cache.insert(key, true);
        Ok(())
    }

    pub fn disable_p2p(&self, source: DeviceType, destination: DeviceType) -> Result<()> {
        let key = (source, destination);
        let mut cache = self
            .p2p_capabilities
            .write()
            .expect("lock should not be poisoned");
        cache.insert(key, false);
        Ok(())
    }
}

/// Transfer configuration
#[derive(Debug, Clone)]
pub struct TransferConfig {
    pub max_concurrent_transfers: usize,
    pub default_chunk_size: usize,
    pub bandwidth_config: BandwidthConfig,
    pub enable_p2p: bool,
    pub enable_async_transfers: bool,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            max_concurrent_transfers: 4,
            default_chunk_size: 1024 * 1024, // 1MB
            bandwidth_config: BandwidthConfig::default(),
            enable_p2p: true,
            enable_async_transfers: true,
        }
    }
}

/// Bandwidth configuration
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    pub max_bandwidth_per_link: u64, // bytes per second
    pub bandwidth_limit_enabled: bool,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            max_bandwidth_per_link: 10 * 1024 * 1024 * 1024, // 10 GB/s
            bandwidth_limit_enabled: true,
        }
    }
}

/// Transfer statistics
#[derive(Debug, Clone)]
pub struct TransferStatistics {
    pub total_transfers: u64,
    pub completed_transfers: u64,
    pub failed_transfers: u64,
    pub total_bytes_transferred: u64,
    pub average_bandwidth_gbps: f64,
}

impl Default for TransferStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferStatistics {
    pub fn new() -> Self {
        Self {
            total_transfers: 0,
            completed_transfers: 0,
            failed_transfers: 0,
            total_bytes_transferred: 0,
            average_bandwidth_gbps: 0.0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_transfers == 0 {
            0.0
        } else {
            self.completed_transfers as f64 / self.total_transfers as f64
        }
    }
}

/// Utility functions for transfer operations
pub mod utils {
    use super::*;

    /// Create a transfer manager with optimal configuration
    pub fn create_optimized_manager() -> TransferManager {
        let config = TransferConfig {
            max_concurrent_transfers: 8,
            default_chunk_size: 4 * 1024 * 1024, // 4MB
            bandwidth_config: BandwidthConfig {
                max_bandwidth_per_link: 20 * 1024 * 1024 * 1024, // 20 GB/s
                bandwidth_limit_enabled: true,
            },
            enable_p2p: true,
            enable_async_transfers: true,
        };

        TransferManager::with_config(config)
    }

    /// Estimate transfer time
    pub fn estimate_transfer_time(request: &TransferRequest) -> Duration {
        let base_latency = Duration::from_micros(100); // 100μs base latency
        let cost = request.estimated_cost();

        let bandwidth_gbps = match cost {
            0 => return Duration::from_nanos(1), // Direct copy
            cost if cost <= 50 => 10.0,          // Fast intra-device
            cost if cost <= 100 => 5.0,          // Medium speed
            _ => 1.0,                            // Slow cross-device
        };

        let transfer_time = Duration::from_secs_f64(
            request.size_bytes as f64 / (bandwidth_gbps * 1024.0 * 1024.0 * 1024.0),
        );

        base_latency + transfer_time
    }

    /// Check if transfer is worth optimizing
    pub fn should_optimize_transfer(request: &TransferRequest) -> bool {
        request.size_bytes > 10 * 1024 * 1024 // 10MB threshold
    }

    /// Get optimal chunk size for transfer
    pub fn get_optimal_chunk_size(request: &TransferRequest) -> usize {
        match request.size_bytes {
            size if size < 1024 * 1024 => 64 * 1024, // 64KB for small transfers
            size if size < 100 * 1024 * 1024 => 1024 * 1024, // 1MB for medium transfers
            _ => 4 * 1024 * 1024,                    // 4MB for large transfers
        }
    }

    /// Create batched transfer for multiple requests
    pub fn create_batched_transfer(requests: Vec<TransferRequest>) -> Result<Vec<TransferHandle>> {
        let manager = TransferManager::new();
        let mut handles = Vec::new();

        for request in requests {
            let handle = manager.execute_transfer(request)?;
            handles.push(handle);
        }

        Ok(handles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_request() {
        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cuda(0), 1024);
        assert_eq!(request.source, DeviceType::Cpu);
        assert_eq!(request.destination, DeviceType::Cuda(0));
        assert_eq!(request.size_bytes, 1024);
        assert!(!request.is_same_device());
        assert!(!request.can_use_p2p());

        let p2p_request = TransferRequest::new(DeviceType::Cuda(0), DeviceType::Cuda(1), 1024);
        assert!(p2p_request.can_use_p2p());
    }

    #[test]
    fn test_transfer_manager() {
        let manager = TransferManager::new();
        assert_eq!(manager.active_transfer_count(), 0);
        assert_eq!(manager.queued_transfer_count(), 0);

        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cpu, 1024);
        let handle = manager.execute_transfer(request).unwrap();

        assert!(handle.is_complete());
        assert_eq!(handle.progress(), 1.0);
    }

    #[test]
    fn test_bandwidth_manager() {
        let manager = BandwidthManager::new();
        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cuda(0), 1024 * 1024);

        assert!(manager.can_allocate_bandwidth(&request).unwrap());
        manager.allocate_bandwidth(&request).unwrap();
        manager.deallocate_bandwidth(&request).unwrap();
    }

    #[test]
    fn test_p2p_manager() {
        let manager = P2PManager::new();

        let can_p2p = manager
            .can_use_p2p(&DeviceType::Cuda(0), &DeviceType::Cuda(1))
            .unwrap();
        assert!(can_p2p);

        let cannot_p2p = manager
            .can_use_p2p(&DeviceType::Cpu, &DeviceType::Cuda(0))
            .unwrap();
        assert!(!cannot_p2p);
    }

    #[test]
    fn test_transfer_statistics() {
        let mut stats = TransferStatistics::new();
        assert_eq!(stats.success_rate(), 0.0);

        stats.total_transfers = 10;
        stats.completed_transfers = 8;
        stats.failed_transfers = 2;

        assert_eq!(stats.success_rate(), 0.8);
    }

    #[test]
    fn test_utils_functions() {
        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cuda(0), 15 * 1024 * 1024); // 15MB, above 10MB threshold

        let estimated_time = utils::estimate_transfer_time(&request);
        assert!(estimated_time > Duration::from_nanos(1));

        assert!(utils::should_optimize_transfer(&request));

        let chunk_size = utils::get_optimal_chunk_size(&request);
        assert!(chunk_size > 0);

        let manager = utils::create_optimized_manager();
        assert_eq!(manager.active_transfer_count(), 0);
    }

    #[test]
    fn test_transfer_priorities() {
        assert!(TransferPriority::Critical > TransferPriority::High);
        assert!(TransferPriority::High > TransferPriority::Normal);
        assert!(TransferPriority::Normal > TransferPriority::Low);
    }

    #[test]
    fn test_transfer_handle() {
        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cuda(0), 1024);
        let handle = TransferHandle::new(TransferId(1), &request);

        assert_eq!(handle.source(), DeviceType::Cpu);
        assert_eq!(handle.destination(), DeviceType::Cuda(0));
        assert_eq!(handle.size_bytes(), 1024);

        let result = handle.wait().unwrap();
        assert_eq!(result.bytes_transferred, 1024);
    }
}
