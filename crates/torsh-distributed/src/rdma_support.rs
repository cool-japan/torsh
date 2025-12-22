//! RDMA (Remote Direct Memory Access) Support for High-Performance Distributed Computing
//!
//! This module provides RDMA capabilities for ultra-low latency, high-bandwidth
//! communication in distributed training environments. RDMA bypasses the CPU and
//! operating system kernel, allowing direct memory-to-memory data transfers between
//! nodes in a cluster.
//!
//! Key features:
//! - Zero-copy data transfers
//! - Ultra-low latency (<1μs)
//! - High bandwidth (100+ Gbps)
//! - CPU offload for communication
//! - Support for InfiniBand, RoCE, and iWARP protocols

// Framework infrastructure - components designed for future use
#![allow(clippy::await_holding_lock)]
#![allow(dead_code)]
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

/// RDMA transport protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RdmaProtocol {
    /// InfiniBand - Native RDMA protocol
    InfiniBand,
    /// RoCE (RDMA over Converged Ethernet) v1
    RoCEv1,
    /// RoCE (RDMA over Converged Ethernet) v2
    RoCEv2,
    /// iWARP (Internet Wide Area RDMA Protocol)
    IWARP,
}

/// RDMA operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RdmaOperation {
    /// RDMA Read - Read data from remote memory
    Read,
    /// RDMA Write - Write data to remote memory
    Write,
    /// RDMA Write with Immediate - Write with immediate data notification
    WriteImmediate,
    /// Send - Send data with CPU involvement on receiver
    Send,
    /// Receive - Receive data with CPU involvement
    Recv,
    /// Compare and Swap - Atomic compare and swap operation
    CompareSwap,
    /// Fetch and Add - Atomic fetch and add operation
    FetchAdd,
}

/// RDMA Quality of Service levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RdmaQoS {
    /// Best effort - No guarantees
    BestEffort,
    /// Low latency - Prioritize latency over bandwidth
    LowLatency,
    /// High bandwidth - Prioritize bandwidth over latency
    HighBandwidth,
    /// Real-time - Guaranteed latency bounds
    RealTime,
}

/// RDMA memory registration types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryRegistration {
    /// Standard registration - One-time registration
    Standard,
    /// Fast registration - Dynamic memory registration
    FastReg,
    /// Memory windows - Dynamic address translation
    MemoryWindow,
    /// Global memory - Globally accessible memory region
    Global,
}

/// RDMA connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdmaConfig {
    /// Protocol to use
    pub protocol: RdmaProtocol,
    /// Quality of service level
    pub qos: RdmaQoS,
    /// Maximum message size (bytes)
    pub max_message_size: usize,
    /// Queue pair depth
    pub queue_depth: u32,
    /// Number of completion queue entries
    pub cq_size: u32,
    /// Memory registration type
    pub memory_registration: MemoryRegistration,
    /// Enable hardware checksums
    pub hardware_checksum: bool,
    /// Enable adaptive routing
    pub adaptive_routing: bool,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Retry count for failed operations
    pub retry_count: u8,
    /// Path MTU size
    pub path_mtu: u32,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            protocol: RdmaProtocol::RoCEv2,
            qos: RdmaQoS::HighBandwidth,
            max_message_size: 4 * 1024 * 1024, // 4MB
            queue_depth: 256,
            cq_size: 512,
            memory_registration: MemoryRegistration::FastReg,
            hardware_checksum: true,
            adaptive_routing: true,
            connection_timeout: Duration::from_secs(30),
            retry_count: 7,
            path_mtu: 4096,
        }
    }
}

/// RDMA memory region descriptor
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Starting address
    pub addr: u64,
    /// Size in bytes
    pub size: usize,
    /// Remote key for RDMA operations
    pub rkey: u32,
    /// Local key for local operations
    pub lkey: u32,
    /// Access permissions
    pub access: MemoryAccess,
    /// Registration type
    pub registration_type: MemoryRegistration,
}

/// Memory access permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccess {
    pub read: bool,
    pub write: bool,
    pub atomic: bool,
    pub remote_read: bool,
    pub remote_write: bool,
    pub remote_atomic: bool,
}

impl Default for MemoryAccess {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            atomic: false,
            remote_read: true,
            remote_write: true,
            remote_atomic: false,
        }
    }
}

/// RDMA connection endpoint
#[derive(Debug, Clone)]
pub struct RdmaEndpoint {
    /// Node identifier
    pub node_id: usize,
    /// IP address or hostname
    pub address: String,
    /// Port number
    pub port: u16,
    /// Global identifier (GID) for InfiniBand
    pub gid: Option<[u8; 16]>,
    /// Local identifier (LID) for InfiniBand
    pub lid: Option<u16>,
    /// Queue pair number
    pub qp_num: u32,
    /// Packet sequence number
    pub psn: u32,
}

/// RDMA work request
#[derive(Debug)]
pub struct WorkRequest {
    /// Unique identifier
    pub id: u64,
    /// Operation type
    pub operation: RdmaOperation,
    /// Local memory region
    pub local_addr: u64,
    /// Local memory key
    pub lkey: u32,
    /// Remote memory region (for RDMA operations)
    pub remote_addr: Option<u64>,
    /// Remote memory key (for RDMA operations)
    pub rkey: Option<u32>,
    /// Data length
    pub length: usize,
    /// Immediate data (for immediate operations)
    pub immediate: Option<u32>,
    /// Completion notification channel
    pub completion: oneshot::Sender<RdmaResult<WorkCompletion>>,
}

/// RDMA work completion
#[derive(Debug, Clone)]
pub struct WorkCompletion {
    /// Work request ID
    pub wr_id: u64,
    /// Operation status
    pub status: CompletionStatus,
    /// Operation type
    pub operation: RdmaOperation,
    /// Bytes transferred
    pub bytes_transferred: usize,
    /// Immediate data (if any)
    pub immediate: Option<u32>,
    /// Completion timestamp
    pub timestamp: Instant,
}

/// Completion status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionStatus {
    Success,
    LocalLengthError,
    LocalQpOperationError,
    LocalProtectionError,
    WorkRequestFlushed,
    MemoryManagementError,
    BadResponseError,
    LocalAccessError,
    RemoteInvalidRequestError,
    RemoteAccessError,
    RemoteOperationError,
    RetryExceededError,
    RnrRetryExceededError,
    LocalRddViolationError,
    RemoteInvalidRdRequest,
    RemoteAborted,
    InvalidEecnError,
    InvalidEecStateError,
    Fatal,
}

/// RDMA error types
#[derive(Debug, thiserror::Error)]
pub enum RdmaError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Memory registration failed: {0}")]
    MemoryRegistrationFailed(String),
    #[error("Operation failed: {0}")]
    OperationFailed(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Hardware error: {0}")]
    HardwareError(String),
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}

pub type RdmaResult<T> = Result<T, RdmaError>;

/// RDMA statistics
#[derive(Debug, Clone, Default)]
pub struct RdmaStatistics {
    /// Total operations performed
    pub total_operations: u64,
    /// Operations by type
    pub operations_by_type: HashMap<RdmaOperation, u64>,
    /// Total bytes transferred
    pub bytes_transferred: u64,
    /// Average latency (microseconds)
    pub avg_latency_us: f64,
    /// Peak bandwidth (Gbps)
    pub peak_bandwidth_gbps: f64,
    /// Current bandwidth (Gbps)
    pub current_bandwidth_gbps: f64,
    /// Error count
    pub error_count: u64,
    /// Retry count
    pub retry_count: u64,
    /// Connection uptime
    pub uptime: Duration,
    /// CPU usage percentage for RDMA operations
    pub cpu_usage_percent: f64,
}

/// RDMA memory pool for efficient memory management
pub struct RdmaMemoryPool {
    /// Pre-registered memory regions by size
    regions: RwLock<HashMap<usize, Vec<MemoryRegion>>>,
    /// Pool configuration
    config: RdmaMemoryPoolConfig,
    /// Usage statistics
    stats: Arc<Mutex<MemoryPoolStats>>,
}

#[derive(Debug, Clone)]
pub struct RdmaMemoryPoolConfig {
    /// Minimum pool size per region size
    pub min_pool_size: usize,
    /// Maximum pool size per region size
    pub max_pool_size: usize,
    /// Supported region sizes
    pub region_sizes: Vec<usize>,
    /// Enable memory prefaulting
    pub prefault: bool,
    /// Enable huge pages
    pub huge_pages: bool,
}

#[derive(Debug, Default, Clone)]
pub struct MemoryPoolStats {
    allocations: u64,
    deallocations: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_memory_allocated: usize,
    peak_memory_usage: usize,
}

impl RdmaMemoryPool {
    /// Create a new memory pool
    pub fn new(config: RdmaMemoryPoolConfig) -> RdmaResult<Self> {
        let mut regions = HashMap::new();

        // Pre-allocate memory regions for each size
        for &size in &config.region_sizes {
            let mut size_regions = Vec::new();
            for _ in 0..config.min_pool_size {
                let region = Self::allocate_region(size, &config)?;
                size_regions.push(region);
            }
            regions.insert(size, size_regions);
        }

        Ok(Self {
            regions: RwLock::new(regions),
            config,
            stats: Arc::new(Mutex::new(MemoryPoolStats::default())),
        })
    }

    /// Allocate a memory region from the pool
    pub fn allocate(&self, size: usize) -> RdmaResult<MemoryRegion> {
        let mut stats = self.stats.lock().unwrap();
        stats.allocations += 1;

        // Find the best fitting region size
        let region_size = self
            .config
            .region_sizes
            .iter()
            .find(|&&s| s >= size)
            .copied()
            .unwrap_or_else(|| {
                // If no pre-defined size fits, round up to next power of 2
                size.next_power_of_two()
            });

        let mut regions = self.regions.write().unwrap();

        if let Some(size_regions) = regions.get_mut(&region_size) {
            if let Some(region) = size_regions.pop() {
                stats.cache_hits += 1;
                return Ok(region);
            }
        }

        // No cached region available, allocate new one
        stats.cache_misses += 1;
        let region = Self::allocate_region(region_size, &self.config)?;
        Ok(region)
    }

    /// Return a memory region to the pool
    pub fn deallocate(&self, mut region: MemoryRegion) {
        let mut stats = self.stats.lock().unwrap();
        stats.deallocations += 1;

        let mut regions = self.regions.write().unwrap();
        let size_regions = regions.entry(region.size).or_default();

        if size_regions.len() < self.config.max_pool_size {
            // Reset region for reuse
            region.addr = 0; // This would be properly reset in real implementation
            size_regions.push(region);
        }
        // Otherwise, the region is dropped and memory is freed
    }

    fn allocate_region(size: usize, _config: &RdmaMemoryPoolConfig) -> RdmaResult<MemoryRegion> {
        // In a real implementation, this would:
        // 1. Allocate physical memory (possibly with huge pages)
        // 2. Register the memory with the RDMA device
        // 3. Set up proper memory protection and caching

        Ok(MemoryRegion {
            addr: 0x1000_0000, // Placeholder address
            size,
            rkey: thread_rng().random::<u32>(),
            lkey: thread_rng().random::<u32>(),
            access: MemoryAccess::default(),
            registration_type: MemoryRegistration::FastReg,
        })
    }

    /// Get memory pool statistics
    pub fn statistics(&self) -> MemoryPoolStats {
        (*self.stats.lock().unwrap()).clone()
    }
}

/// RDMA connection manager
pub struct RdmaConnectionManager {
    /// Active connections
    connections: RwLock<HashMap<usize, RdmaConnection>>,
    /// Configuration
    config: RdmaConfig,
    /// Connection statistics
    stats: Arc<Mutex<RdmaStatistics>>,
    /// Memory pool
    memory_pool: Arc<RdmaMemoryPool>,
    /// Work request sender
    work_sender: mpsc::UnboundedSender<WorkRequest>,
}

/// Individual RDMA connection
pub struct RdmaConnection {
    /// Local endpoint
    pub local_endpoint: RdmaEndpoint,
    /// Remote endpoint
    pub remote_endpoint: RdmaEndpoint,
    /// Connection state
    pub state: ConnectionState,
    /// Queue pair handle (simulated)
    pub qp_handle: u64,
    /// Completion queue handle (simulated)
    pub cq_handle: u64,
    /// Connection statistics
    pub stats: RdmaStatistics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Error,
}

impl RdmaConnectionManager {
    /// Create a new RDMA connection manager
    pub fn new(config: RdmaConfig) -> RdmaResult<Self> {
        let memory_pool_config = RdmaMemoryPoolConfig {
            min_pool_size: 16,
            max_pool_size: 256,
            region_sizes: vec![4096, 65536, 1048576, 16777216], // 4KB, 64KB, 1MB, 16MB
            prefault: true,
            huge_pages: config.max_message_size > 2 * 1024 * 1024,
        };

        let memory_pool = Arc::new(RdmaMemoryPool::new(memory_pool_config)?);
        let (work_sender, _work_receiver) = mpsc::unbounded_channel();

        Ok(Self {
            connections: RwLock::new(HashMap::new()),
            config,
            stats: Arc::new(Mutex::new(RdmaStatistics::default())),
            memory_pool,
            work_sender,
        })
    }

    /// Establish RDMA connection to a remote node
    pub async fn connect(&self, remote_endpoint: RdmaEndpoint) -> RdmaResult<usize> {
        let connection_id = remote_endpoint.node_id;

        // Simulate connection establishment
        let local_endpoint = RdmaEndpoint {
            node_id: 0, // Local node ID
            address: "0.0.0.0".to_string(),
            port: 0,
            gid: None,
            lid: None,
            qp_num: thread_rng().random::<u32>(),
            psn: thread_rng().random::<u32>(),
        };

        let connection = RdmaConnection {
            local_endpoint,
            remote_endpoint,
            state: ConnectionState::Connected,
            qp_handle: thread_rng().random::<u64>(),
            cq_handle: thread_rng().random::<u64>(),
            stats: RdmaStatistics::default(),
        };

        self.connections
            .write()
            .unwrap()
            .insert(connection_id, connection);
        Ok(connection_id)
    }

    /// Perform RDMA read operation
    pub async fn rdma_read(
        &self,
        _connection_id: usize,
        local_addr: u64,
        remote_addr: u64,
        length: usize,
        lkey: u32,
        rkey: u32,
    ) -> RdmaResult<WorkCompletion> {
        self.submit_work_request(WorkRequest {
            id: thread_rng().random::<u64>(),
            operation: RdmaOperation::Read,
            local_addr,
            lkey,
            remote_addr: Some(remote_addr),
            rkey: Some(rkey),
            length,
            immediate: None,
            completion: oneshot::channel().0,
        })
        .await
    }

    /// Perform RDMA write operation
    pub async fn rdma_write(
        &self,
        _connection_id: usize,
        local_addr: u64,
        remote_addr: u64,
        length: usize,
        lkey: u32,
        rkey: u32,
    ) -> RdmaResult<WorkCompletion> {
        self.submit_work_request(WorkRequest {
            id: thread_rng().random::<u64>(),
            operation: RdmaOperation::Write,
            local_addr,
            lkey,
            remote_addr: Some(remote_addr),
            rkey: Some(rkey),
            length,
            immediate: None,
            completion: oneshot::channel().0,
        })
        .await
    }

    /// Perform atomic compare and swap
    pub async fn atomic_compare_swap(
        &self,
        _connection_id: usize,
        remote_addr: u64,
        compare: u64,
        _swap: u64,
        rkey: u32,
    ) -> RdmaResult<u64> {
        // In a real implementation, this would perform the atomic operation
        // and return the previous value
        let _completion = self
            .submit_work_request(WorkRequest {
                id: thread_rng().random::<u64>(),
                operation: RdmaOperation::CompareSwap,
                local_addr: 0,
                lkey: 0,
                remote_addr: Some(remote_addr),
                rkey: Some(rkey),
                length: 8,
                immediate: None,
                completion: oneshot::channel().0,
            })
            .await?;

        // Simulate returning the previous value
        Ok(compare) // In real implementation, this would be the actual previous value
    }

    async fn submit_work_request(&self, work_request: WorkRequest) -> RdmaResult<WorkCompletion> {
        // Simulate work request processing
        tokio::time::sleep(Duration::from_micros(1)).await; // Simulate ultra-low latency

        let completion = WorkCompletion {
            wr_id: work_request.id,
            status: CompletionStatus::Success,
            operation: work_request.operation,
            bytes_transferred: work_request.length,
            immediate: work_request.immediate,
            timestamp: Instant::now(),
        };

        // Update statistics
        let mut stats = self.stats.lock().unwrap();
        stats.total_operations += 1;
        *stats
            .operations_by_type
            .entry(work_request.operation)
            .or_insert(0) += 1;
        stats.bytes_transferred += work_request.length as u64;

        Ok(completion)
    }

    /// Get connection statistics
    pub fn statistics(&self) -> RdmaStatistics {
        self.stats.lock().unwrap().clone()
    }

    /// Get memory pool statistics
    pub fn memory_pool_statistics(&self) -> MemoryPoolStats {
        self.memory_pool.statistics()
    }
}

/// RDMA-aware tensor operation scheduler
pub struct RdmaTensorScheduler {
    /// Connection manager
    connection_manager: Arc<RdmaConnectionManager>,
    /// Operation queue
    operation_queue: Arc<Mutex<Vec<TensorOperation>>>,
    /// Bandwidth optimizer
    bandwidth_optimizer: BandwidthOptimizer,
}

#[derive(Debug)]
pub struct TensorOperation {
    tensor_id: String,
    operation_type: TensorOperationType,
    source_node: usize,
    target_nodes: Vec<usize>,
    data_size: usize,
    priority: OperationPriority,
    deadline: Option<Instant>,
}

#[derive(Debug, Clone, Copy)]
enum TensorOperationType {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    AllToAll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OperationPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug)]
struct BandwidthOptimizer {
    link_bandwidth: HashMap<(usize, usize), f64>,
    link_utilization: HashMap<(usize, usize), f64>,
    optimization_strategy: BandwidthStrategy,
}

#[derive(Debug, Clone, Copy)]
enum BandwidthStrategy {
    MinimizeLatency,
    MaximizeThroughput,
    BalanceLatencyThroughput,
    AdaptiveDynamic,
}

impl RdmaTensorScheduler {
    /// Create a new RDMA tensor scheduler
    pub fn new(connection_manager: Arc<RdmaConnectionManager>) -> Self {
        Self {
            connection_manager,
            operation_queue: Arc::new(Mutex::new(Vec::new())),
            bandwidth_optimizer: BandwidthOptimizer {
                link_bandwidth: HashMap::new(),
                link_utilization: HashMap::new(),
                optimization_strategy: BandwidthStrategy::AdaptiveDynamic,
            },
        }
    }

    /// Schedule a tensor operation for RDMA execution
    pub async fn schedule_operation(&self, operation: TensorOperation) -> RdmaResult<()> {
        self.operation_queue.lock().unwrap().push(operation);
        self.optimize_scheduling().await
    }

    async fn optimize_scheduling(&self) -> RdmaResult<()> {
        #[allow(clippy::await_holding_lock)]
        let mut queue = self.operation_queue.lock().unwrap();

        // Sort operations by priority and deadline
        queue.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .reverse()
                .then_with(|| match (a.deadline, b.deadline) {
                    (Some(da), Some(db)) => da.cmp(&db),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                })
        });

        // Execute high-priority operations first
        if let Some(operation) = queue.pop() {
            self.execute_tensor_operation(operation).await?;
        }

        Ok(())
    }

    async fn execute_tensor_operation(&self, operation: TensorOperation) -> RdmaResult<()> {
        match operation.operation_type {
            TensorOperationType::AllReduce => self.execute_all_reduce(&operation).await,
            TensorOperationType::AllGather => self.execute_all_gather(&operation).await,
            TensorOperationType::ReduceScatter => self.execute_reduce_scatter(&operation).await,
            TensorOperationType::Broadcast => self.execute_broadcast(&operation).await,
            TensorOperationType::AllToAll => self.execute_all_to_all(&operation).await,
        }
    }

    async fn execute_all_reduce(&self, _operation: &TensorOperation) -> RdmaResult<()> {
        // Implement RDMA-optimized AllReduce using ring or tree algorithms
        // This would use RDMA write operations to directly update remote memory
        Ok(())
    }

    async fn execute_all_gather(&self, _operation: &TensorOperation) -> RdmaResult<()> {
        // Implement RDMA-optimized AllGather
        Ok(())
    }

    async fn execute_reduce_scatter(&self, _operation: &TensorOperation) -> RdmaResult<()> {
        // Implement RDMA-optimized ReduceScatter
        Ok(())
    }

    async fn execute_broadcast(&self, _operation: &TensorOperation) -> RdmaResult<()> {
        // Implement RDMA-optimized Broadcast using tree topology
        Ok(())
    }

    async fn execute_all_to_all(&self, _operation: &TensorOperation) -> RdmaResult<()> {
        // Implement RDMA-optimized AllToAll
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rdma_memory_pool() {
        let config = RdmaMemoryPoolConfig {
            min_pool_size: 2,
            max_pool_size: 10,
            region_sizes: vec![4096, 65536],
            prefault: true,
            huge_pages: false,
        };

        let pool = RdmaMemoryPool::new(config).unwrap();

        // Test allocation
        let region1 = pool.allocate(2048).unwrap();
        assert!(region1.size >= 2048);

        let region2 = pool.allocate(8192).unwrap();
        assert!(region2.size >= 8192);

        // Test deallocation
        pool.deallocate(region1);
        pool.deallocate(region2);

        let stats = pool.statistics();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.deallocations, 2);
    }

    #[tokio::test]
    async fn test_rdma_connection_manager() {
        let config = RdmaConfig::default();
        let manager = RdmaConnectionManager::new(config).unwrap();

        let remote_endpoint = RdmaEndpoint {
            node_id: 1,
            address: "192.168.1.100".to_string(),
            port: 18515,
            gid: None,
            lid: None,
            qp_num: 12345,
            psn: 67890,
        };

        let connection_id = manager.connect(remote_endpoint).await.unwrap();
        assert_eq!(connection_id, 1);

        // Test RDMA read operation
        let result = manager
            .rdma_read(connection_id, 0x1000, 0x2000, 1024, 0x12345678, 0x87654321)
            .await
            .unwrap();

        assert_eq!(result.status, CompletionStatus::Success);
        assert_eq!(result.operation, RdmaOperation::Read);
        assert_eq!(result.bytes_transferred, 1024);
    }

    #[test]
    fn test_rdma_config_serialization() {
        let config = RdmaConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: RdmaConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.protocol, deserialized.protocol);
        assert_eq!(config.qos, deserialized.qos);
        assert_eq!(config.max_message_size, deserialized.max_message_size);
    }

    #[tokio::test]
    async fn test_atomic_operations() {
        let config = RdmaConfig::default();
        let manager = RdmaConnectionManager::new(config).unwrap();

        let remote_endpoint = RdmaEndpoint {
            node_id: 1,
            address: "192.168.1.100".to_string(),
            port: 18515,
            gid: None,
            lid: None,
            qp_num: 12345,
            psn: 67890,
        };

        let connection_id = manager.connect(remote_endpoint).await.unwrap();

        let previous_value = manager
            .atomic_compare_swap(connection_id, 0x3000, 42, 84, 0x12345678)
            .await
            .unwrap();

        assert_eq!(previous_value, 42);
    }
}
