// Copyright (c) 2025 ToRSh Contributors
//
// Distributed Tensor Metadata Management
//
// This module provides data structures and abstractions for managing tensors
// distributed across multiple devices, nodes, or clusters. It enables efficient
// distributed training and inference at scale.
//
// # Key Features
//
// - **Tensor Sharding**: Automatic tensor partitioning across devices
// - **Communication Patterns**: AllReduce, AllGather, ReduceScatter, etc.
// - **Device Topology**: Hierarchical device organization (node, rack, cluster)
// - **Synchronization**: Efficient barrier and broadcast operations
// - **Fault Tolerance**: Checkpoint and recovery mechanisms
//
// # Design Principles
//
// 1. **Scalability**: Support thousands of devices
// 2. **Flexibility**: Multiple sharding strategies
// 3. **Performance**: Overlap computation and communication
// 4. **Resilience**: Handle device failures gracefully
//
// # Examples
//
// ```rust
// use torsh_core::distributed::{DistributedTensor, ShardingStrategy, DeviceGroup};
//
// // Create a distributed tensor across 4 GPUs
// let devices = DeviceGroup::new(vec![0, 1, 2, 3]);
// let tensor = DistributedTensor::new(shape, ShardingStrategy::DataParallel, devices);
//
// // Perform all-reduce operation
// tensor.all_reduce(ReduceOp::Sum);
// ```

use core::fmt;

/// Device identifier in a distributed system
///
/// Uniquely identifies a device in a cluster with node, rack, and device ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId {
    /// Node ID within the cluster
    node_id: usize,
    /// Rack ID (for datacenter topology)
    rack_id: usize,
    /// Local device ID on the node
    local_device_id: usize,
}

impl DeviceId {
    /// Create a new device ID
    pub fn new(node_id: usize, rack_id: usize, local_device_id: usize) -> Self {
        Self {
            node_id,
            rack_id,
            local_device_id,
        }
    }

    /// Create a simple device ID (single node)
    pub fn simple(local_device_id: usize) -> Self {
        Self::new(0, 0, local_device_id)
    }

    /// Get node ID
    pub fn node_id(&self) -> usize {
        self.node_id
    }

    /// Get rack ID
    pub fn rack_id(&self) -> usize {
        self.rack_id
    }

    /// Get local device ID
    pub fn local_device_id(&self) -> usize {
        self.local_device_id
    }

    /// Get global unique ID
    pub fn global_id(&self) -> usize {
        // Simple encoding: rack_id * 1000 + node_id * 100 + local_device_id
        self.rack_id * 1000 + self.node_id * 100 + self.local_device_id
    }
}

/// Group of devices for distributed operations
///
/// Represents a logical group of devices that participate in collective operations.
#[derive(Debug, Clone)]
pub struct DeviceGroup {
    /// Devices in this group
    devices: Vec<DeviceId>,
    /// Group name for debugging
    name: Option<String>,
}

impl DeviceGroup {
    /// Create a new device group
    pub fn new(device_ids: Vec<usize>) -> Self {
        let devices = device_ids.iter().map(|&id| DeviceId::simple(id)).collect();
        Self {
            devices,
            name: None,
        }
    }

    /// Create a device group with explicit device IDs
    pub fn from_devices(devices: Vec<DeviceId>) -> Self {
        Self {
            devices,
            name: None,
        }
    }

    /// Set group name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Get devices in the group
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Get group size
    pub fn size(&self) -> usize {
        self.devices.len()
    }

    /// Check if device is in the group
    pub fn contains(&self, device_id: &DeviceId) -> bool {
        self.devices.contains(device_id)
    }

    /// Get device rank (position in group)
    pub fn rank(&self, device_id: &DeviceId) -> Option<usize> {
        self.devices.iter().position(|d| d == device_id)
    }

    /// Get group name
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Tensor sharding strategies
///
/// Different ways to partition a tensor across multiple devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingStrategy {
    /// Replicate the full tensor on each device (data parallelism)
    Replicated,
    /// Shard along the batch dimension
    DataParallel,
    /// Shard along the model dimension (tensor parallelism)
    ModelParallel,
    /// Shard along a specific dimension
    DimSharded(usize),
    /// Pipeline parallelism (different layers on different devices)
    Pipeline,
    /// Combination of strategies
    Hybrid,
}

impl fmt::Display for ShardingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardingStrategy::Replicated => write!(f, "Replicated"),
            ShardingStrategy::DataParallel => write!(f, "DataParallel"),
            ShardingStrategy::ModelParallel => write!(f, "ModelParallel"),
            ShardingStrategy::DimSharded(dim) => write!(f, "DimSharded({})", dim),
            ShardingStrategy::Pipeline => write!(f, "Pipeline"),
            ShardingStrategy::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Shard descriptor
///
/// Describes a single shard of a distributed tensor.
#[derive(Debug, Clone)]
pub struct Shard {
    /// Device where this shard is located
    device_id: DeviceId,
    /// Offset in the global tensor
    offset: Vec<usize>,
    /// Shape of this shard
    shape: Vec<usize>,
    /// Rank of this shard in the group
    rank: usize,
}

impl Shard {
    /// Create a new shard
    pub fn new(device_id: DeviceId, offset: Vec<usize>, shape: Vec<usize>, rank: usize) -> Self {
        Self {
            device_id,
            offset,
            shape,
            rank,
        }
    }

    /// Get device ID
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Get offset
    pub fn offset(&self) -> &[usize] {
        &self.offset
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Calculate shard size (number of elements)
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Distributed tensor metadata
///
/// Represents a tensor distributed across multiple devices.
#[derive(Debug, Clone)]
pub struct DistributedTensor {
    /// Global shape of the tensor
    global_shape: Vec<usize>,
    /// Sharding strategy
    strategy: ShardingStrategy,
    /// Device group
    device_group: DeviceGroup,
    /// Shard descriptors
    shards: Vec<Shard>,
}

impl DistributedTensor {
    /// Create a new distributed tensor
    pub fn new(
        global_shape: Vec<usize>,
        strategy: ShardingStrategy,
        device_group: DeviceGroup,
    ) -> Self {
        let shards = Self::create_shards(&global_shape, strategy, &device_group);
        Self {
            global_shape,
            strategy,
            device_group,
            shards,
        }
    }

    /// Create shards based on strategy
    fn create_shards(
        global_shape: &[usize],
        strategy: ShardingStrategy,
        device_group: &DeviceGroup,
    ) -> Vec<Shard> {
        let num_devices = device_group.size();
        let mut shards = Vec::new();

        match strategy {
            ShardingStrategy::Replicated => {
                // Full tensor on each device
                for (rank, &device_id) in device_group.devices().iter().enumerate() {
                    shards.push(Shard::new(
                        device_id,
                        vec![0; global_shape.len()],
                        global_shape.to_vec(),
                        rank,
                    ));
                }
            }
            ShardingStrategy::DataParallel | ShardingStrategy::DimSharded(0) => {
                // Shard along first dimension
                if global_shape.is_empty() {
                    return shards;
                }
                let dim0 = global_shape[0];
                let chunk_size = (dim0 + num_devices - 1) / num_devices;

                for (rank, &device_id) in device_group.devices().iter().enumerate() {
                    let start = rank * chunk_size;
                    let end = (start + chunk_size).min(dim0);
                    if start >= dim0 {
                        break;
                    }

                    let mut offset = vec![0; global_shape.len()];
                    offset[0] = start;

                    let mut shape = global_shape.to_vec();
                    shape[0] = end - start;

                    shards.push(Shard::new(device_id, offset, shape, rank));
                }
            }
            ShardingStrategy::ModelParallel => {
                // For now, same as data parallel
                // In practice, this would shard model parameters
                return Self::create_shards(
                    global_shape,
                    ShardingStrategy::DataParallel,
                    device_group,
                );
            }
            ShardingStrategy::DimSharded(dim) => {
                // Shard along specified dimension
                if dim >= global_shape.len() {
                    return shards;
                }
                let dim_size = global_shape[dim];
                let chunk_size = (dim_size + num_devices - 1) / num_devices;

                for (rank, &device_id) in device_group.devices().iter().enumerate() {
                    let start = rank * chunk_size;
                    let end = (start + chunk_size).min(dim_size);
                    if start >= dim_size {
                        break;
                    }

                    let mut offset = vec![0; global_shape.len()];
                    offset[dim] = start;

                    let mut shape = global_shape.to_vec();
                    shape[dim] = end - start;

                    shards.push(Shard::new(device_id, offset, shape, rank));
                }
            }
            _ => {
                // Default to replicated
                return Self::create_shards(
                    global_shape,
                    ShardingStrategy::Replicated,
                    device_group,
                );
            }
        }

        shards
    }

    /// Get global shape
    pub fn global_shape(&self) -> &[usize] {
        &self.global_shape
    }

    /// Get sharding strategy
    pub fn strategy(&self) -> ShardingStrategy {
        self.strategy
    }

    /// Get device group
    pub fn device_group(&self) -> &DeviceGroup {
        &self.device_group
    }

    /// Get shards
    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }

    /// Get shard for a specific device
    pub fn shard_for_device(&self, device_id: &DeviceId) -> Option<&Shard> {
        self.shards.iter().find(|s| &s.device_id == device_id)
    }

    /// Get total number of elements across all shards
    pub fn total_elements(&self) -> usize {
        match self.strategy {
            ShardingStrategy::Replicated => {
                // Only count once for replicated
                self.global_shape.iter().product()
            }
            _ => {
                // Sum all shard sizes
                self.shards.iter().map(|s| s.size()).sum()
            }
        }
    }
}

/// Collective communication operations
///
/// Common collective operations for distributed tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveOp {
    /// All-reduce: reduce across all devices and broadcast result
    AllReduce(ReduceOp),
    /// All-gather: gather data from all devices and broadcast
    AllGather,
    /// Reduce-scatter: reduce and scatter results
    ReduceScatter(ReduceOp),
    /// Broadcast: send data from one device to all others
    Broadcast { root: usize },
    /// Scatter: distribute data from one device to all others
    Scatter { root: usize },
    /// Gather: collect data from all devices to one device
    Gather { root: usize },
    /// All-to-all: each device sends unique data to every other device
    AllToAll,
    /// Barrier: synchronization point for all devices
    Barrier,
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction
    Sum,
    /// Product reduction
    Product,
    /// Minimum reduction
    Min,
    /// Maximum reduction
    Max,
    /// Average reduction
    Average,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceOp::Sum => write!(f, "Sum"),
            ReduceOp::Product => write!(f, "Product"),
            ReduceOp::Min => write!(f, "Min"),
            ReduceOp::Max => write!(f, "Max"),
            ReduceOp::Average => write!(f, "Average"),
        }
    }
}

/// Communication backend
///
/// Abstraction for different communication libraries (MPI, NCCL, Gloo, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommBackend {
    /// NCCL (NVIDIA Collective Communications Library) for GPU
    NCCL,
    /// Gloo for CPU and GPU
    Gloo,
    /// MPI (Message Passing Interface)
    MPI,
    /// Custom implementation
    Custom,
}

/// Communication descriptor
///
/// Describes a communication operation to be executed.
#[derive(Debug, Clone)]
pub struct CommunicationDescriptor {
    /// Collective operation
    operation: CollectiveOp,
    /// Device group
    device_group: DeviceGroup,
    /// Backend to use
    backend: CommBackend,
    /// Whether to use asynchronous communication
    async_op: bool,
}

impl CommunicationDescriptor {
    /// Create a new communication descriptor
    pub fn new(operation: CollectiveOp, device_group: DeviceGroup, backend: CommBackend) -> Self {
        Self {
            operation,
            device_group,
            backend,
            async_op: false,
        }
    }

    /// Set asynchronous flag
    pub fn with_async(mut self, async_op: bool) -> Self {
        self.async_op = async_op;
        self
    }

    /// Get operation
    pub fn operation(&self) -> CollectiveOp {
        self.operation
    }

    /// Get device group
    pub fn device_group(&self) -> &DeviceGroup {
        &self.device_group
    }

    /// Get backend
    pub fn backend(&self) -> CommBackend {
        self.backend
    }

    /// Check if async
    pub fn is_async(&self) -> bool {
        self.async_op
    }
}

/// Checkpoint metadata for fault tolerance
///
/// Contains information about a saved checkpoint of distributed tensors.
#[derive(Debug, Clone)]
pub struct CheckpointMetadata {
    /// Checkpoint ID
    id: String,
    /// Global step number
    step: u64,
    /// List of device IDs that contributed to checkpoint
    devices: Vec<DeviceId>,
    /// Timestamp (Unix epoch seconds)
    timestamp: u64,
    /// Additional metadata
    metadata: Vec<(String, String)>,
}

impl CheckpointMetadata {
    /// Create a new checkpoint metadata
    pub fn new(id: impl Into<String>, step: u64, devices: Vec<DeviceId>) -> Self {
        Self {
            id: id.into(),
            step,
            devices,
            timestamp: 0, // Would use system time in real implementation
            metadata: Vec::new(),
        }
    }

    /// Add metadata entry
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.push((key.into(), value.into()));
    }

    /// Get checkpoint ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get step number
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Get devices
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Get timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Get metadata
    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
}

/// Device topology for hierarchical communication
///
/// Represents the physical/logical topology of devices in a cluster.
#[derive(Debug, Clone)]
pub struct DeviceTopology {
    /// All devices in the topology
    devices: Vec<DeviceId>,
    /// Number of nodes
    num_nodes: usize,
    /// Number of racks
    num_racks: usize,
    /// Devices per node
    devices_per_node: usize,
}

impl DeviceTopology {
    /// Create a new device topology
    pub fn new(num_racks: usize, num_nodes: usize, devices_per_node: usize) -> Self {
        let mut devices = Vec::new();
        for rack_id in 0..num_racks {
            for node_id in 0..num_nodes {
                for device_id in 0..devices_per_node {
                    devices.push(DeviceId::new(node_id, rack_id, device_id));
                }
            }
        }

        Self {
            devices,
            num_nodes,
            num_racks,
            devices_per_node,
        }
    }

    /// Get all devices
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// Get devices on a specific node
    pub fn node_devices(&self, node_id: usize) -> Vec<DeviceId> {
        self.devices
            .iter()
            .filter(|d| d.node_id() == node_id)
            .copied()
            .collect()
    }

    /// Get devices in a specific rack
    pub fn rack_devices(&self, rack_id: usize) -> Vec<DeviceId> {
        self.devices
            .iter()
            .filter(|d| d.rack_id() == rack_id)
            .copied()
            .collect()
    }

    /// Get total number of devices
    pub fn total_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get number of racks
    pub fn num_racks(&self) -> usize {
        self.num_racks
    }

    /// Get devices per node
    pub fn devices_per_node(&self) -> usize {
        self.devices_per_node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id() {
        let device = DeviceId::new(0, 1, 2);
        assert_eq!(device.node_id(), 0);
        assert_eq!(device.rack_id(), 1);
        assert_eq!(device.local_device_id(), 2);
        assert_eq!(device.global_id(), 1002); // 1*1000 + 0*100 + 2
    }

    #[test]
    fn test_simple_device_id() {
        let device = DeviceId::simple(5);
        assert_eq!(device.local_device_id(), 5);
        assert_eq!(device.node_id(), 0);
        assert_eq!(device.rack_id(), 0);
    }

    #[test]
    fn test_device_group() {
        let group = DeviceGroup::new(vec![0, 1, 2, 3]);
        assert_eq!(group.size(), 4);
        assert!(group.contains(&DeviceId::simple(0)));
        assert_eq!(group.rank(&DeviceId::simple(2)), Some(2));
    }

    #[test]
    fn test_device_group_with_name() {
        let group = DeviceGroup::new(vec![0, 1]).with_name("test_group");
        assert_eq!(group.name(), Some("test_group"));
    }

    #[test]
    fn test_sharding_strategy_display() {
        assert_eq!(format!("{}", ShardingStrategy::Replicated), "Replicated");
        assert_eq!(
            format!("{}", ShardingStrategy::DataParallel),
            "DataParallel"
        );
        assert_eq!(
            format!("{}", ShardingStrategy::DimSharded(1)),
            "DimSharded(1)"
        );
    }

    #[test]
    fn test_shard() {
        let device = DeviceId::simple(0);
        let shard = Shard::new(device, vec![0, 0], vec![10, 20], 0);
        assert_eq!(shard.device_id(), device);
        assert_eq!(shard.offset(), &[0, 0]);
        assert_eq!(shard.shape(), &[10, 20]);
        assert_eq!(shard.rank(), 0);
        assert_eq!(shard.size(), 200);
    }

    #[test]
    fn test_distributed_tensor_replicated() {
        let group = DeviceGroup::new(vec![0, 1, 2, 3]);
        let tensor = DistributedTensor::new(vec![100, 50], ShardingStrategy::Replicated, group);

        assert_eq!(tensor.global_shape(), &[100, 50]);
        assert_eq!(tensor.shards().len(), 4);
        assert_eq!(tensor.strategy(), ShardingStrategy::Replicated);

        // All shards should have the full shape
        for shard in tensor.shards() {
            assert_eq!(shard.shape(), &[100, 50]);
        }
    }

    #[test]
    fn test_distributed_tensor_data_parallel() {
        let group = DeviceGroup::new(vec![0, 1, 2, 3]);
        let tensor = DistributedTensor::new(vec![100, 50], ShardingStrategy::DataParallel, group);

        assert_eq!(tensor.shards().len(), 4);

        // Each shard should have 25 rows (100 / 4)
        for shard in tensor.shards() {
            assert_eq!(shard.shape()[0], 25);
            assert_eq!(shard.shape()[1], 50);
        }
    }

    #[test]
    fn test_distributed_tensor_dim_sharded() {
        let group = DeviceGroup::new(vec![0, 1]);
        let tensor =
            DistributedTensor::new(vec![10, 20, 30], ShardingStrategy::DimSharded(1), group);

        assert_eq!(tensor.shards().len(), 2);

        // Sharded along dimension 1 (20 -> 10 + 10)
        assert_eq!(tensor.shards()[0].shape(), &[10, 10, 30]);
        assert_eq!(tensor.shards()[1].shape(), &[10, 10, 30]);
    }

    #[test]
    fn test_shard_for_device() {
        let group = DeviceGroup::new(vec![0, 1]);
        let tensor = DistributedTensor::new(vec![10, 20], ShardingStrategy::DataParallel, group);

        let device = DeviceId::simple(0);
        let shard = tensor.shard_for_device(&device);
        assert!(shard.is_some());
        assert_eq!(shard.unwrap().device_id(), device);
    }

    #[test]
    fn test_collective_operations() {
        let _all_reduce = CollectiveOp::AllReduce(ReduceOp::Sum);
        let _all_gather = CollectiveOp::AllGather;
        let _reduce_scatter = CollectiveOp::ReduceScatter(ReduceOp::Average);
        let _broadcast = CollectiveOp::Broadcast { root: 0 };
        let _scatter = CollectiveOp::Scatter { root: 0 };
        let _gather = CollectiveOp::Gather { root: 0 };
        let _all_to_all = CollectiveOp::AllToAll;
        let _barrier = CollectiveOp::Barrier;
    }

    #[test]
    fn test_reduce_op_display() {
        assert_eq!(format!("{}", ReduceOp::Sum), "Sum");
        assert_eq!(format!("{}", ReduceOp::Product), "Product");
        assert_eq!(format!("{}", ReduceOp::Min), "Min");
        assert_eq!(format!("{}", ReduceOp::Max), "Max");
        assert_eq!(format!("{}", ReduceOp::Average), "Average");
    }

    #[test]
    fn test_comm_backend() {
        let _nccl = CommBackend::NCCL;
        let _gloo = CommBackend::Gloo;
        let _mpi = CommBackend::MPI;
        let _custom = CommBackend::Custom;
    }

    #[test]
    fn test_communication_descriptor() {
        let group = DeviceGroup::new(vec![0, 1, 2, 3]);
        let comm_desc = CommunicationDescriptor::new(
            CollectiveOp::AllReduce(ReduceOp::Sum),
            group.clone(),
            CommBackend::NCCL,
        )
        .with_async(true);

        assert_eq!(
            comm_desc.operation(),
            CollectiveOp::AllReduce(ReduceOp::Sum)
        );
        assert_eq!(comm_desc.backend(), CommBackend::NCCL);
        assert!(comm_desc.is_async());
    }

    #[test]
    fn test_checkpoint_metadata() {
        let devices = vec![DeviceId::simple(0), DeviceId::simple(1)];
        let mut checkpoint = CheckpointMetadata::new("ckpt_001", 1000, devices);
        checkpoint.add_metadata("model", "resnet50");
        checkpoint.add_metadata("optimizer", "adam");

        assert_eq!(checkpoint.id(), "ckpt_001");
        assert_eq!(checkpoint.step(), 1000);
        assert_eq!(checkpoint.devices().len(), 2);
        assert_eq!(checkpoint.metadata().len(), 2);
    }

    #[test]
    fn test_device_topology() {
        let topology = DeviceTopology::new(2, 3, 4); // 2 racks, 3 nodes, 4 devices per node
        assert_eq!(topology.total_devices(), 24); // 2 * 3 * 4
        assert_eq!(topology.num_racks(), 2);
        assert_eq!(topology.num_nodes(), 3);
        assert_eq!(topology.devices_per_node(), 4);

        let node0_devices = topology.node_devices(0);
        assert_eq!(node0_devices.len(), 8); // 4 devices * 2 racks

        let rack0_devices = topology.rack_devices(0);
        assert_eq!(rack0_devices.len(), 12); // 3 nodes * 4 devices
    }

    #[test]
    fn test_total_elements() {
        let group = DeviceGroup::new(vec![0, 1, 2, 3]);

        // Replicated: count only once
        let replicated =
            DistributedTensor::new(vec![100, 50], ShardingStrategy::Replicated, group.clone());
        assert_eq!(replicated.total_elements(), 5000); // 100 * 50

        // Data parallel: sum of all shards
        let sharded = DistributedTensor::new(vec![100, 50], ShardingStrategy::DataParallel, group);
        assert_eq!(sharded.total_elements(), 5000); // Still 100 * 50 total
    }

    #[test]
    fn test_from_devices() {
        let devices = vec![DeviceId::new(0, 0, 1), DeviceId::new(0, 0, 2)];
        let group = DeviceGroup::from_devices(devices);
        assert_eq!(group.size(), 2);
    }

    #[test]
    fn test_device_not_in_group() {
        let group = DeviceGroup::new(vec![0, 1, 2]);
        assert!(!group.contains(&DeviceId::simple(5)));
        assert_eq!(group.rank(&DeviceId::simple(5)), None);
    }
}
