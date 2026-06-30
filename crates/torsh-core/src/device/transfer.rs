//! Cross-device memory transfer and scheduling
//!
//! This module provides sophisticated memory transfer capabilities between different
//! device types, including asynchronous operations, bandwidth management, and
//! peer-to-peer transfers where supported.

use crate::device::DeviceType;
use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock, Weak};
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
    /// Shared transfer bookkeeping, also referenced (weakly) by every
    /// [`TransferHandle`] this manager hands out so that handle queries observe
    /// the live transfer state rather than fabricated constants.
    state: Arc<TransferState>,
    bandwidth_manager: Arc<BandwidthManager>,
    p2p_manager: Arc<P2PManager>,
    config: TransferConfig,
}

/// Shared, reference-counted transfer bookkeeping.
///
/// Held by the [`TransferManager`] and weakly referenced by each
/// [`TransferHandle`]. Worker threads update the [`ActiveTransfer`] entries in
/// place through this shared state, so a handle can report the genuine status,
/// progress, and result of its transfer.
#[derive(Debug)]
struct TransferState {
    active_transfers: RwLock<HashMap<TransferId, Arc<Mutex<ActiveTransfer>>>>,
    transfer_stats: RwLock<TransferStatistics>,
}

impl TransferState {
    fn new() -> Self {
        Self {
            active_transfers: RwLock::new(HashMap::new()),
            transfer_stats: RwLock::new(TransferStatistics::new()),
        }
    }

    /// Read the current status of a transfer, if it is still tracked.
    fn status_of(&self, id: TransferId) -> Option<TransferStatus> {
        let active = self
            .active_transfers
            .read()
            .expect("lock should not be poisoned");
        active.get(&id).map(|t| {
            t.lock()
                .expect("lock should not be poisoned")
                .status
                .clone()
        })
    }
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
            state: Arc::new(TransferState::new()),
            bandwidth_manager: Arc::new(BandwidthManager::new()),
            p2p_manager: Arc::new(P2PManager::new()),
            config: TransferConfig::default(),
        }
    }

    /// Create transfer manager with custom configuration
    pub fn with_config(config: TransferConfig) -> Self {
        Self {
            transfer_queue: Mutex::new(VecDeque::new()),
            state: Arc::new(TransferState::new()),
            bandwidth_manager: Arc::new(BandwidthManager::with_config(&config)),
            p2p_manager: Arc::new(P2PManager::new()),
            config,
        }
    }

    /// Execute a memory transfer
    pub fn execute_transfer(&self, request: TransferRequest) -> Result<TransferHandle> {
        let transfer_id = self.generate_transfer_id();
        let handle = TransferHandle::new(transfer_id, &request, Arc::downgrade(&self.state));

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
        let handle = TransferHandle::new(transfer_id, &request, Arc::downgrade(&self.state));

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
        self.state.status_of(transfer_id)
    }

    /// Wait for transfer completion
    pub fn wait_for_transfer(&self, transfer_id: TransferId) -> Result<TransferResult> {
        loop {
            match self.state.status_of(transfer_id) {
                Some(TransferStatus::Completed(result)) => return Ok(result),
                Some(TransferStatus::Failed(error)) => {
                    return Err(crate::error::TorshError::DeviceError(error));
                }
                Some(TransferStatus::Cancelled) => {
                    return Err(crate::error::TorshError::DeviceError(format!(
                        "transfer {transfer_id:?} was cancelled"
                    )));
                }
                // Still queued / in progress, or no longer tracked.
                Some(_) => {}
                None => {
                    return Err(crate::error::TorshError::DeviceError(format!(
                        "transfer {transfer_id:?} is not tracked by this manager"
                    )));
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
            let active = self
                .state
                .active_transfers
                .read()
                .expect("lock should not be poisoned");
            if let Some(transfer) = active.get(&transfer_id) {
                let mut transfer = transfer.lock().expect("lock should not be poisoned");
                transfer.status = TransferStatus::Cancelled;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get transfer statistics
    pub fn get_statistics(&self) -> TransferStatistics {
        let stats = self
            .state
            .transfer_stats
            .read()
            .expect("lock should not be poisoned");
        stats.clone()
    }

    /// Get active transfer count
    pub fn active_transfer_count(&self) -> usize {
        let active = self
            .state
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
            .state
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
                let id = active_transfer
                    .lock()
                    .expect("lock should not be poisoned")
                    .id;
                active.insert(id, active_transfer);
            }
        }

        // Refresh aggregate statistics so `get_statistics` reflects reality.
        {
            let mut stats = self
                .state
                .transfer_stats
                .write()
                .expect("lock should not be poisoned");
            stats.total_transfers = stats.total_transfers.max(active.len() as u64);
        }

        Ok(())
    }

    fn start_transfer(&self, queued: QueuedTransfer) -> Result<Arc<Mutex<ActiveTransfer>>> {
        let start_time = Instant::now();

        // Allocate bandwidth
        self.bandwidth_manager.allocate_bandwidth(&queued.request)?;

        let total_bytes = queued.request.size_bytes;
        let active_transfer = Arc::new(Mutex::new(ActiveTransfer {
            id: queued.id,
            request: queued.request,
            method: queued.method,
            status: TransferStatus::InProgress {
                bytes_transferred: 0,
                total_bytes,
                start_time,
            },
            start_time,
        }));

        // Spawn transfer execution against the *same* shared entry, so progress
        // and completion are observable through the manager and any handle.
        self.execute_transfer_async(Arc::clone(&active_transfer))?;

        Ok(active_transfer)
    }

    fn execute_transfer_async(&self, transfer_arc: Arc<Mutex<ActiveTransfer>>) -> Result<()> {
        // The worker only needs the bandwidth manager (to release its
        // reservation) and the shared statistics; both are cheaply clonable
        // `Arc`s, so no raw pointer to the manager is required.
        let bandwidth_manager = Arc::clone(&self.bandwidth_manager);
        let state = Arc::clone(&self.state);

        std::thread::spawn(move || {
            let request = transfer_arc
                .lock()
                .expect("lock should not be poisoned")
                .request
                .clone();

            let result = Self::perform_transfer(Arc::clone(&transfer_arc));

            // Update status based on result, in place on the shared entry.
            let completed = {
                let mut transfer = transfer_arc.lock().expect("lock should not be poisoned");
                match result {
                    Ok(transfer_result) => {
                        transfer.status = TransferStatus::Completed(transfer_result);
                        true
                    }
                    Err(error) => {
                        transfer.status = TransferStatus::Failed(error.to_string());
                        false
                    }
                }
            };

            // Release the bandwidth reservation regardless of outcome.
            let _ = bandwidth_manager.deallocate_bandwidth(&request);

            // Record the outcome in the aggregate statistics.
            let mut stats = state
                .transfer_stats
                .write()
                .expect("lock should not be poisoned");
            if completed {
                stats.completed_transfers += 1;
                stats.total_bytes_transferred += request.size_bytes as u64;
            } else {
                stats.failed_transfers += 1;
            }
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
        // This is a chunked transfer model: it advances the byte counters in
        // real time at a fixed per-link throughput so that progress reporting,
        // bandwidth accounting, and scheduling behave realistically. It does not
        // itself copy device memory; an actual host-staged copy (pinned host
        // buffer plus per-backend memcpy) is performed by the backend layer.
        const MODELED_HOST_STAGED_BYTES_PER_SEC: u64 = 1024 * 1024 * 1024; // 1 GiB/s
        let chunk_size = 1024 * 1024; // 1 MiB chunks
        let total_bytes = request.size_bytes;
        let mut bytes_transferred = 0;

        while bytes_transferred < total_bytes {
            let chunk = std::cmp::min(chunk_size, total_bytes - bytes_transferred);

            // Advance time according to the modeled link throughput.
            let transfer_time =
                Duration::from_secs_f64(chunk as f64 / MODELED_HOST_STAGED_BYTES_PER_SEC as f64);
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

        // Capture the start time with a single lock acquisition. Locking the
        // same non-reentrant mutex more than once inside a single expression
        // would self-deadlock, so the duration and bandwidth are derived here.
        let start_time = {
            transfer
                .lock()
                .expect("lock should not be poisoned")
                .start_time
        };
        let duration = Instant::now().duration_since(start_time);
        let bandwidth_gbps = Self::bandwidth_gbps(total_bytes, duration);

        Ok(TransferResult {
            bytes_transferred: total_bytes,
            duration,
            bandwidth_gbps,
            method: TransferMethod::HostStaged,
        })
    }

    /// Compute throughput in GiB/s from a byte count and elapsed duration,
    /// guarding against a zero-length interval (which would otherwise yield a
    /// non-finite value).
    fn bandwidth_gbps(total_bytes: usize, duration: Duration) -> f64 {
        let seconds = duration.as_secs_f64();
        if seconds <= 0.0 {
            return f64::INFINITY;
        }
        (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / seconds
    }

    fn perform_p2p_transfer(
        request: &TransferRequest,
        transfer: Arc<Mutex<ActiveTransfer>>,
    ) -> Result<TransferResult> {
        // Peer-to-peer transfer model. P2P links (e.g. NVLink / PCIe P2P) are
        // modeled at a higher throughput than host-staged transfers. As with
        // the host-staged path, this advances timing and byte counters for
        // scheduling and reporting; the actual GPU-to-GPU copy is issued by the
        // backend layer when real device handles are available.
        const MODELED_P2P_BYTES_PER_SEC: u64 = 5 * 1024 * 1024 * 1024; // 5 GiB/s
        let total_bytes = request.size_bytes;
        let transfer_duration =
            Duration::from_secs_f64(total_bytes as f64 / MODELED_P2P_BYTES_PER_SEC as f64);

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
            bandwidth_gbps: Self::bandwidth_gbps(total_bytes, transfer_duration),
            method: TransferMethod::PeerToPeer,
        })
    }

    fn perform_direct_copy(
        request: &TransferRequest,
        transfer: Arc<Mutex<ActiveTransfer>>,
    ) -> Result<TransferResult> {
        // Same-device "transfer": there is no cross-device movement, so the
        // model completes immediately. Any genuine intra-device copy is handled
        // by the backend without going through this scheduler.
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

/// Transfer handle for tracking transfer progress.
///
/// A handle holds a weak reference to its issuing [`TransferManager`]'s shared
/// state, so [`Self::wait`], [`Self::is_complete`], and [`Self::progress`]
/// report the genuine, live status of the underlying transfer rather than
/// fabricated constants. If the manager has been dropped, those methods report
/// the transfer as no longer trackable.
#[derive(Debug)]
pub struct TransferHandle {
    id: TransferId,
    source: DeviceType,
    destination: DeviceType,
    size_bytes: usize,
    state: Weak<TransferState>,
}

impl TransferHandle {
    fn new(id: TransferId, request: &TransferRequest, state: Weak<TransferState>) -> Self {
        Self {
            id,
            source: request.source,
            destination: request.destination,
            size_bytes: request.size_bytes,
            state,
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

    /// Block until the underlying transfer finishes, returning its result.
    ///
    /// Polls the issuing manager's live transfer state. Returns an error if the
    /// transfer failed, was cancelled, the issuing manager has been dropped, or
    /// the transfer is no longer tracked.
    pub fn wait(&self) -> Result<TransferResult> {
        let state = self.state.upgrade().ok_or_else(|| {
            crate::error::TorshError::DeviceError(
                "transfer manager has been dropped; transfer state is unavailable".to_string(),
            )
        })?;

        loop {
            match state.status_of(self.id) {
                Some(TransferStatus::Completed(result)) => return Ok(result),
                Some(TransferStatus::Failed(error)) => {
                    return Err(crate::error::TorshError::DeviceError(error));
                }
                Some(TransferStatus::Cancelled) => {
                    return Err(crate::error::TorshError::DeviceError(format!(
                        "transfer {:?} was cancelled",
                        self.id
                    )));
                }
                // Queued or in progress: keep polling.
                Some(_) => {}
                None => {
                    return Err(crate::error::TorshError::DeviceError(format!(
                        "transfer {:?} is no longer tracked by its manager",
                        self.id
                    )));
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Report whether the underlying transfer has finished (completed, failed,
    /// or cancelled).
    ///
    /// Reflects the live transfer state. If the issuing manager has been dropped
    /// or the transfer is no longer tracked, it is reported as finished, since
    /// no further progress is possible.
    pub fn is_complete(&self) -> bool {
        match self.state.upgrade() {
            Some(state) => match state.status_of(self.id) {
                Some(TransferStatus::Queued) | Some(TransferStatus::InProgress { .. }) => false,
                // Completed / Failed / Cancelled, or no longer tracked.
                _ => true,
            },
            // Manager gone: nothing can advance the transfer further.
            None => true,
        }
    }

    /// Report the fraction of the transfer that has completed, in `[0.0, 1.0]`.
    ///
    /// Computed from the live byte counters. A finished transfer reports `1.0`;
    /// a transfer that is no longer tracked or whose manager has been dropped
    /// also reports `1.0`, since no further progress is observable.
    pub fn progress(&self) -> f64 {
        let Some(state) = self.state.upgrade() else {
            return 1.0;
        };

        match state.status_of(self.id) {
            Some(TransferStatus::InProgress {
                bytes_transferred,
                total_bytes,
                ..
            }) => {
                if total_bytes == 0 {
                    1.0
                } else {
                    (bytes_transferred as f64 / total_bytes as f64).clamp(0.0, 1.0)
                }
            }
            Some(TransferStatus::Queued) => 0.0,
            Some(TransferStatus::Completed(_)) => 1.0,
            // Failed / Cancelled / no longer tracked: no further progress.
            _ => 1.0,
        }
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
        let handle = manager
            .execute_transfer(request)
            .expect("execute_transfer should succeed");

        // The transfer runs asynchronously; block on the handle until the
        // worker reports genuine completion, then verify the live state.
        let result = handle.wait().expect("transfer should complete");
        assert_eq!(result.bytes_transferred, 1024);
        assert!(handle.is_complete());
        assert_eq!(handle.progress(), 1.0);
    }

    #[test]
    fn test_bandwidth_manager() {
        let manager = BandwidthManager::new();
        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cuda(0), 1024 * 1024);

        assert!(manager
            .can_allocate_bandwidth(&request)
            .expect("can_allocate_bandwidth should succeed"));
        manager
            .allocate_bandwidth(&request)
            .expect("allocate_bandwidth should succeed");
        manager
            .deallocate_bandwidth(&request)
            .expect("deallocate_bandwidth should succeed");
    }

    #[test]
    fn test_p2p_manager() {
        let manager = P2PManager::new();

        let can_p2p = manager
            .can_use_p2p(&DeviceType::Cuda(0), &DeviceType::Cuda(1))
            .expect("can_use_p2p should succeed");
        assert!(can_p2p);

        let cannot_p2p = manager
            .can_use_p2p(&DeviceType::Cpu, &DeviceType::Cuda(0))
            .expect("can_use_p2p should succeed");
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
        let manager = TransferManager::new();
        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cuda(0), 1024);
        let handle = manager
            .execute_transfer(request)
            .expect("execute_transfer should succeed");

        assert_eq!(handle.source(), DeviceType::Cpu);
        assert_eq!(handle.destination(), DeviceType::Cuda(0));
        assert_eq!(handle.size_bytes(), 1024);

        // `wait` blocks on, and returns, the real transfer result.
        let result = handle.wait().expect("wait should succeed");
        assert_eq!(result.bytes_transferred, 1024);
        assert!(handle.is_complete());
    }

    #[test]
    fn test_handle_reports_completion_after_manager_dropped() {
        // A handle whose manager has been dropped honestly reports the transfer
        // as no longer trackable rather than fabricating a success result.
        let request = TransferRequest::new(DeviceType::Cpu, DeviceType::Cpu, 1024);
        let handle = {
            let manager = TransferManager::new();
            let handle = manager
                .execute_transfer(request)
                .expect("execute_transfer should succeed");
            // Let the worker finish so it releases its `Arc<TransferState>`.
            let _ = handle.wait();
            handle
            // `manager` dropped here.
        };

        // With both the manager and any worker gone, the state Arc is released.
        assert!(handle.is_complete());
        assert_eq!(handle.progress(), 1.0);
        assert!(handle.wait().is_err());
    }
}
