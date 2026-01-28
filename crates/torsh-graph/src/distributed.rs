//! Distributed graph neural networks for large-scale graph processing
//!
//! This module provides distributed training and inference capabilities
//! for graph neural networks across multiple devices and machines.

use crate::{GraphData, GraphLayer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_tensor::Tensor;

/// Distributed training configuration
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of worker nodes
    pub num_workers: usize,
    /// Rank of current process (0 to num_workers-1)
    pub rank: usize,
    /// Communication backend
    pub backend: CommunicationBackend,
    /// Graph partitioning strategy
    pub partitioning: GraphPartitioning,
    /// Aggregation method for distributed training
    pub aggregation: AggregationMethod,
    /// Synchronization frequency (in steps)
    pub sync_frequency: usize,
}

/// Communication backends for distributed training
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationBackend {
    /// Message Passing Interface
    MPI,
    /// NVIDIA Collective Communications Library
    NCCL,
    /// Gloo collective communications
    Gloo,
    /// TCP-based communication
    TCP,
    /// In-memory communication (single machine)
    InMemory,
}

/// Graph partitioning strategies
pub enum GraphPartitioning {
    /// Random vertex partitioning
    Random,
    /// METIS-based partitioning
    METIS,
    /// Hash-based partitioning
    Hash,
    /// Community-based partitioning
    Community,
    /// Custom partitioning function
    Custom(Box<dyn Fn(&GraphData, usize) -> Vec<PartitionInfo> + Send + Sync>),
}

// Manual Debug implementation for GraphPartitioning
impl std::fmt::Debug for GraphPartitioning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphPartitioning::Random => write!(f, "GraphPartitioning::Random"),
            GraphPartitioning::METIS => write!(f, "GraphPartitioning::METIS"),
            GraphPartitioning::Hash => write!(f, "GraphPartitioning::Hash"),
            GraphPartitioning::Community => write!(f, "GraphPartitioning::Community"),
            GraphPartitioning::Custom(_) => write!(f, "GraphPartitioning::Custom(<function>)"),
        }
    }
}

// Manual Clone implementation for GraphPartitioning (Clone not available for Custom)
impl Clone for GraphPartitioning {
    fn clone(&self) -> Self {
        match self {
            GraphPartitioning::Random => GraphPartitioning::Random,
            GraphPartitioning::METIS => GraphPartitioning::METIS,
            GraphPartitioning::Hash => GraphPartitioning::Hash,
            GraphPartitioning::Community => GraphPartitioning::Community,
            GraphPartitioning::Custom(_) => {
                // Cannot clone function pointer - fallback to Random
                GraphPartitioning::Random
            }
        }
    }
}

/// Aggregation methods for distributed updates
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    /// Average gradients across workers
    Average,
    /// Sum gradients across workers
    Sum,
    /// Weighted average based on partition size
    WeightedAverage,
    /// Asynchronous parameter server
    ParameterServer,
    /// AllReduce pattern
    AllReduce,
}

/// Information about a graph partition
#[derive(Debug, Clone)]
pub struct PartitionInfo {
    /// Worker rank responsible for this partition
    pub worker_rank: usize,
    /// Nodes in this partition
    pub nodes: Vec<usize>,
    /// Edges within this partition
    pub internal_edges: Vec<(usize, usize)>,
    /// Cross-partition edges (boundary edges)
    pub boundary_edges: Vec<(usize, usize, usize)>, // (src, dst, target_worker)
    /// Partition size metrics
    pub metrics: PartitionMetrics,
}

/// Metrics for evaluating partition quality
#[derive(Debug, Clone)]
pub struct PartitionMetrics {
    /// Number of nodes in partition
    pub num_nodes: usize,
    /// Number of internal edges
    pub num_internal_edges: usize,
    /// Number of boundary edges
    pub num_boundary_edges: usize,
    /// Load balance score (0.0 = perfect, higher = worse)
    pub load_balance_score: f32,
    /// Communication cost estimate
    pub communication_cost: f32,
}

/// Distributed graph neural network coordinator
#[derive(Debug)]
pub struct DistributedGNN {
    /// Configuration
    pub config: DistributedConfig,
    /// Local graph partition
    pub local_partition: GraphData,
    /// Partition information
    pub partition_info: PartitionInfo,
    /// Communication manager
    pub comm_manager: CommunicationManager,
    /// Parameter synchronization state
    pub sync_state: Arc<Mutex<SyncState>>,
    /// Performance metrics
    pub metrics: DistributedMetrics,
}

impl DistributedGNN {
    /// Create a new distributed GNN
    pub fn new(
        config: DistributedConfig,
        full_graph: &GraphData,
    ) -> Result<Self, DistributedError> {
        // Partition the graph
        let partitions = Self::partition_graph(full_graph, &config)?;
        let local_partition = partitions[config.rank].clone();

        // Initialize communication
        let comm_manager = CommunicationManager::new(&config)?;

        // Create partition info
        let partition_info = Self::create_partition_info(&local_partition, config.rank);

        let sync_state = Arc::new(Mutex::new(SyncState::new()));
        let metrics = DistributedMetrics::new();

        Ok(Self {
            config,
            local_partition,
            partition_info,
            comm_manager,
            sync_state,
            metrics,
        })
    }

    /// Perform distributed forward pass
    pub fn distributed_forward(
        &mut self,
        layer: &dyn GraphLayer,
    ) -> Result<GraphData, DistributedError> {
        // Step 1: Gather boundary node features from other workers
        let boundary_features = self.gather_boundary_features()?;

        // Step 2: Augment local graph with boundary features
        let augmented_graph = self.augment_local_graph(&boundary_features)?;

        // Step 3: Perform local forward pass
        let local_output = layer.forward(&augmented_graph);

        // Step 4: Extract and communicate updated boundary features
        self.communicate_boundary_updates(&local_output)?;

        Ok(local_output)
    }

    /// Synchronize parameters across workers
    pub fn synchronize_parameters(
        &mut self,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>, DistributedError> {
        match self.config.aggregation {
            AggregationMethod::AllReduce => self.all_reduce_parameters(parameters),
            AggregationMethod::Average => self.average_parameters(parameters),
            AggregationMethod::Sum => self.sum_parameters(parameters),
            AggregationMethod::WeightedAverage => self.weighted_average_parameters(parameters),
            AggregationMethod::ParameterServer => self.parameter_server_sync(parameters),
        }
    }

    /// Perform all-reduce on parameters
    fn all_reduce_parameters(
        &mut self,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>, DistributedError> {
        let mut reduced_params = Vec::new();

        for param in parameters {
            // Serialize parameter
            let param_data = param.to_vec().map_err(|e| {
                DistributedError::CommunicationError(format!(
                    "Failed to serialize parameter: {:?}",
                    e
                ))
            })?;

            // Perform all-reduce operation
            let reduced_data = self.comm_manager.all_reduce(&param_data)?;

            // Deserialize back to tensor
            let reduced_param = self.vec_to_tensor(&reduced_data, param.shape().dims())?;
            reduced_params.push(reduced_param);
        }

        Ok(reduced_params)
    }

    /// Average parameters across workers
    fn average_parameters(
        &mut self,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>, DistributedError> {
        let summed_params = self.sum_parameters(parameters)?;
        let num_workers = self.config.num_workers as f32;

        Ok(summed_params
            .into_iter()
            .map(|param| {
                param
                    .div_scalar(num_workers)
                    .expect("parameter division should succeed")
            })
            .collect())
    }

    /// Sum parameters across workers
    fn sum_parameters(&mut self, parameters: &[Tensor]) -> Result<Vec<Tensor>, DistributedError> {
        let mut summed_params = Vec::new();

        for param in parameters {
            let param_data = param.to_vec().map_err(|e| {
                DistributedError::CommunicationError(format!(
                    "Failed to serialize parameter: {:?}",
                    e
                ))
            })?;

            let summed_data = self.comm_manager.all_reduce_sum(&param_data)?;
            let summed_param = self.vec_to_tensor(&summed_data, param.shape().dims())?;
            summed_params.push(summed_param);
        }

        Ok(summed_params)
    }

    /// Weighted average based on partition sizes
    fn weighted_average_parameters(
        &mut self,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>, DistributedError> {
        let local_weight = self.partition_info.metrics.num_nodes as f32;
        let total_weight = self.comm_manager.all_reduce_sum(&[local_weight])?[0];

        let weighted_params = parameters
            .iter()
            .map(|param| {
                param
                    .mul_scalar(local_weight)
                    .expect("parameter weighting should succeed")
            })
            .collect::<Vec<_>>();

        let summed_params = self.sum_parameters(&weighted_params)?;

        Ok(summed_params
            .into_iter()
            .map(|param| {
                param
                    .div_scalar(total_weight)
                    .expect("weighted parameter division should succeed")
            })
            .collect())
    }

    /// Parameter server synchronization
    fn parameter_server_sync(
        &mut self,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>, DistributedError> {
        if self.config.rank == 0 {
            // Parameter server logic
            self.parameter_server_master(parameters)
        } else {
            // Worker logic
            self.parameter_server_worker(parameters)
        }
    }

    fn parameter_server_master(
        &mut self,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>, DistributedError> {
        // Collect updates from all workers
        let mut accumulated_updates = parameters.to_vec();

        for worker_rank in 1..self.config.num_workers {
            let worker_updates = self.comm_manager.receive_from(worker_rank)?;
            // Accumulate updates (simplified)
            for (i, update) in worker_updates.iter().enumerate() {
                if i < accumulated_updates.len() {
                    accumulated_updates[i] = accumulated_updates[i]
                        .add(update)
                        .expect("operation should succeed");
                }
            }
        }

        // Average and broadcast back
        let num_workers = self.config.num_workers as f32;
        let averaged_params: Vec<Tensor> = accumulated_updates
            .into_iter()
            .map(|param| {
                param
                    .div_scalar(num_workers)
                    .expect("parameter server division should succeed")
            })
            .collect();

        // Broadcast to all workers
        for worker_rank in 1..self.config.num_workers {
            self.comm_manager.send_to(worker_rank, &averaged_params)?;
        }

        Ok(averaged_params)
    }

    fn parameter_server_worker(
        &mut self,
        parameters: &[Tensor],
    ) -> Result<Vec<Tensor>, DistributedError> {
        // Send updates to parameter server
        self.comm_manager.send_to(0, parameters)?;

        // Receive updated parameters
        self.comm_manager.receive_from(0)
    }

    /// Gather boundary node features from neighboring partitions
    fn gather_boundary_features(&mut self) -> Result<HashMap<usize, Tensor>, DistributedError> {
        let mut boundary_features = HashMap::new();

        // Request features for boundary nodes
        for &(_, _, target_worker) in &self.partition_info.boundary_edges {
            if target_worker != self.config.rank {
                // Request boundary features from target worker
                let features = self.comm_manager.request_boundary_features(target_worker)?;
                boundary_features.insert(target_worker, features);
            }
        }

        Ok(boundary_features)
    }

    /// Augment local graph with boundary features
    fn augment_local_graph(
        &self,
        _boundary_features: &HashMap<usize, Tensor>,
    ) -> Result<GraphData, DistributedError> {
        // For now, return the local partition
        // In practice, would merge boundary features
        Ok(self.local_partition.clone())
    }

    /// Communicate boundary updates to neighboring workers
    fn communicate_boundary_updates(
        &mut self,
        _local_output: &GraphData,
    ) -> Result<(), DistributedError> {
        // Send boundary node updates to neighboring partitions
        // Simplified implementation
        Ok(())
    }

    /// Partition a graph into distributed chunks
    fn partition_graph(
        graph: &GraphData,
        config: &DistributedConfig,
    ) -> Result<Vec<GraphData>, DistributedError> {
        match &config.partitioning {
            GraphPartitioning::Random => Self::random_partition(graph, config.num_workers),
            GraphPartitioning::Hash => Self::hash_partition(graph, config.num_workers),
            GraphPartitioning::METIS => Self::metis_partition(graph, config.num_workers),
            GraphPartitioning::Community => Self::community_partition(graph, config.num_workers),
            GraphPartitioning::Custom(partition_fn) => {
                let partition_infos = partition_fn(graph, config.num_workers);
                Self::create_partitions_from_info(graph, &partition_infos)
            }
        }
    }

    fn random_partition(
        graph: &GraphData,
        num_partitions: usize,
    ) -> Result<Vec<GraphData>, DistributedError> {
        let mut partitions = Vec::new();
        let nodes_per_partition = graph.num_nodes / num_partitions;

        for i in 0..num_partitions {
            let start_node = i * nodes_per_partition;
            let end_node = if i == num_partitions - 1 {
                graph.num_nodes
            } else {
                (i + 1) * nodes_per_partition
            };

            // Create partition subgraph (simplified)
            let partition_nodes = (start_node..end_node).collect::<Vec<_>>();
            let partition_graph = Self::extract_subgraph(graph, &partition_nodes)?;
            partitions.push(partition_graph);
        }

        Ok(partitions)
    }

    fn hash_partition(
        graph: &GraphData,
        num_partitions: usize,
    ) -> Result<Vec<GraphData>, DistributedError> {
        let mut partition_nodes: Vec<Vec<usize>> = vec![Vec::new(); num_partitions];

        // Hash-based node assignment
        for node in 0..graph.num_nodes {
            let partition_id = node % num_partitions;
            partition_nodes[partition_id].push(node);
        }

        let mut partitions = Vec::new();
        for nodes in partition_nodes {
            let partition_graph = Self::extract_subgraph(graph, &nodes)?;
            partitions.push(partition_graph);
        }

        Ok(partitions)
    }

    fn metis_partition(
        _graph: &GraphData,
        _num_partitions: usize,
    ) -> Result<Vec<GraphData>, DistributedError> {
        // Placeholder for METIS integration
        Err(DistributedError::PartitioningError(
            "METIS partitioning not implemented".to_string(),
        ))
    }

    fn community_partition(
        _graph: &GraphData,
        _num_partitions: usize,
    ) -> Result<Vec<GraphData>, DistributedError> {
        // Placeholder for community-based partitioning
        Err(DistributedError::PartitioningError(
            "Community partitioning not implemented".to_string(),
        ))
    }

    fn create_partitions_from_info(
        graph: &GraphData,
        partition_infos: &[PartitionInfo],
    ) -> Result<Vec<GraphData>, DistributedError> {
        let mut partitions = Vec::new();

        for info in partition_infos {
            let partition_graph = Self::extract_subgraph(graph, &info.nodes)?;
            partitions.push(partition_graph);
        }

        Ok(partitions)
    }

    fn extract_subgraph(graph: &GraphData, nodes: &[usize]) -> Result<GraphData, DistributedError> {
        // Simplified subgraph extraction
        // In practice, would properly extract edges and reindex nodes

        if nodes.is_empty() {
            return Ok(GraphData::new(
                torsh_tensor::creation::zeros(&[0, graph.x.shape().dims()[1]])
                    .expect("empty features tensor creation should succeed"),
                torsh_tensor::creation::zeros(&[2, 0])
                    .expect("empty edge index tensor creation should succeed"),
            ));
        }

        // Extract node features
        let feature_dim = graph.x.shape().dims()[1];
        let mut subgraph_features = Vec::new();

        for &node in nodes {
            if node < graph.num_nodes {
                // Extract features for this node (simplified)
                for _f in 0..feature_dim {
                    subgraph_features.push(1.0); // Placeholder
                }
            }
        }

        let x = torsh_tensor::creation::from_vec(
            subgraph_features,
            &[nodes.len(), feature_dim],
            graph.x.device(),
        )
        .map_err(|e| {
            DistributedError::TensorError(format!("Failed to create features tensor: {:?}", e))
        })?;

        // Create minimal edge index (simplified)
        let edge_index = torsh_tensor::creation::zeros(&[2, 0])
            .expect("minimal edge index creation should succeed");

        Ok(GraphData::new(x, edge_index))
    }

    fn create_partition_info(graph: &GraphData, rank: usize) -> PartitionInfo {
        PartitionInfo {
            worker_rank: rank,
            nodes: (0..graph.num_nodes).collect(),
            internal_edges: Vec::new(),
            boundary_edges: Vec::new(),
            metrics: PartitionMetrics {
                num_nodes: graph.num_nodes,
                num_internal_edges: 0,
                num_boundary_edges: 0,
                load_balance_score: 0.0,
                communication_cost: 0.0,
            },
        }
    }

    fn vec_to_tensor(&self, data: &[f32], shape: &[usize]) -> Result<Tensor, DistributedError> {
        torsh_tensor::creation::from_vec(data.to_vec(), shape, torsh_core::device::DeviceType::Cpu)
            .map_err(|e| DistributedError::TensorError(format!("Failed to create tensor: {:?}", e)))
    }
}

/// Communication manager for distributed operations
#[derive(Debug)]
pub struct CommunicationManager {
    backend: CommunicationBackend,
    rank: usize,
    num_workers: usize,
    // Backend-specific state would go here
}

impl CommunicationManager {
    pub fn new(config: &DistributedConfig) -> Result<Self, DistributedError> {
        Ok(Self {
            backend: config.backend.clone(),
            rank: config.rank,
            num_workers: config.num_workers,
        })
    }

    /// Get the rank of this worker
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    pub fn all_reduce(&mut self, data: &[f32]) -> Result<Vec<f32>, DistributedError> {
        match self.backend {
            CommunicationBackend::InMemory => {
                // Simplified in-memory implementation
                Ok(data.to_vec())
            }
            _ => Err(DistributedError::CommunicationError(
                "Backend not implemented".to_string(),
            )),
        }
    }

    pub fn all_reduce_sum(&mut self, data: &[f32]) -> Result<Vec<f32>, DistributedError> {
        // Simplified implementation
        Ok(data.to_vec())
    }

    pub fn send_to(
        &mut self,
        _target_rank: usize,
        _data: &[Tensor],
    ) -> Result<(), DistributedError> {
        // Simplified implementation
        Ok(())
    }

    pub fn receive_from(&mut self, _source_rank: usize) -> Result<Vec<Tensor>, DistributedError> {
        // Simplified implementation
        Ok(Vec::new())
    }

    pub fn request_boundary_features(
        &mut self,
        _target_worker: usize,
    ) -> Result<Tensor, DistributedError> {
        // Simplified implementation
        torsh_tensor::creation::zeros(&[1, 1])
            .map_err(|e| DistributedError::TensorError(format!("Failed to create tensor: {:?}", e)))
    }
}

/// Synchronization state for distributed training
#[derive(Debug)]
pub struct SyncState {
    pub current_step: usize,
    pub last_sync_step: usize,
    pub pending_updates: HashMap<usize, Vec<Tensor>>,
}

impl SyncState {
    pub fn new() -> Self {
        Self {
            current_step: 0,
            last_sync_step: 0,
            pending_updates: HashMap::new(),
        }
    }

    pub fn should_sync(&self, sync_frequency: usize) -> bool {
        self.current_step - self.last_sync_step >= sync_frequency
    }

    pub fn mark_synced(&mut self) {
        self.last_sync_step = self.current_step;
        self.pending_updates.clear();
    }
}

/// Performance metrics for distributed training
#[derive(Debug, Clone)]
pub struct DistributedMetrics {
    pub communication_time_ms: f64,
    pub computation_time_ms: f64,
    pub synchronization_time_ms: f64,
    pub total_bytes_communicated: usize,
    pub num_synchronizations: usize,
    pub efficiency_score: f32,
}

impl DistributedMetrics {
    pub fn new() -> Self {
        Self {
            communication_time_ms: 0.0,
            computation_time_ms: 0.0,
            synchronization_time_ms: 0.0,
            total_bytes_communicated: 0,
            num_synchronizations: 0,
            efficiency_score: 1.0,
        }
    }

    pub fn compute_efficiency(&mut self) {
        let total_time = self.communication_time_ms + self.computation_time_ms;
        if total_time > 0.0 {
            self.efficiency_score = (self.computation_time_ms / total_time) as f32;
        }
    }
}

/// Distributed training errors
#[derive(Debug, Clone)]
pub enum DistributedError {
    /// Communication backend error
    CommunicationError(String),
    /// Graph partitioning error
    PartitioningError(String),
    /// Tensor operation error
    TensorError(String),
    /// Configuration error
    ConfigError(String),
    /// Synchronization error
    SynchronizationError(String),
}

impl std::fmt::Display for DistributedError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DistributedError::CommunicationError(msg) => write!(f, "Communication error: {}", msg),
            DistributedError::PartitioningError(msg) => write!(f, "Partitioning error: {}", msg),
            DistributedError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            DistributedError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            DistributedError::SynchronizationError(msg) => {
                write!(f, "Synchronization error: {}", msg)
            }
        }
    }
}

impl std::error::Error for DistributedError {}

/// Distributed graph layer wrapper
#[derive(Debug)]
pub struct DistributedGraphLayer {
    /// Base layer
    pub base_layer: Box<dyn GraphLayer>,
    /// Distributed coordinator
    pub coordinator: DistributedGNN,
}

impl DistributedGraphLayer {
    pub fn new(
        base_layer: Box<dyn GraphLayer>,
        config: DistributedConfig,
        full_graph: &GraphData,
    ) -> Result<Self, DistributedError> {
        let coordinator = DistributedGNN::new(config, full_graph)?;

        Ok(Self {
            base_layer,
            coordinator,
        })
    }
}

impl GraphLayer for DistributedGraphLayer {
    fn forward(&self, graph: &GraphData) -> GraphData {
        // Simplified distributed forward pass
        // In practice, would use the coordinator's distributed_forward method
        self.base_layer.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.base_layer.parameters()
    }
}

/// Utility functions for distributed graph operations
pub mod utils {
    use super::*;

    /// Calculate load balance score for partitions
    pub fn calculate_load_balance(partition_sizes: &[usize]) -> f32 {
        if partition_sizes.is_empty() {
            return 0.0;
        }

        let mean_size = partition_sizes.iter().sum::<usize>() as f32 / partition_sizes.len() as f32;
        let variance: f32 = partition_sizes
            .iter()
            .map(|&size| (size as f32 - mean_size).powi(2))
            .sum::<f32>()
            / partition_sizes.len() as f32;

        variance / mean_size.max(1.0)
    }

    /// Estimate communication cost for a partitioning
    pub fn estimate_communication_cost(partition_infos: &[PartitionInfo]) -> f32 {
        partition_infos
            .iter()
            .map(|info| info.metrics.num_boundary_edges as f32)
            .sum()
    }

    /// Create optimal distributed configuration for given hardware
    pub fn create_optimal_config(num_gpus: usize, graph_size: usize) -> DistributedConfig {
        let num_workers = num_gpus.max(1);
        let backend = if num_gpus > 1 {
            CommunicationBackend::NCCL
        } else {
            CommunicationBackend::InMemory
        };

        let partitioning = if graph_size > 1_000_000 {
            GraphPartitioning::METIS
        } else if graph_size > 10_000 {
            GraphPartitioning::Community
        } else {
            GraphPartitioning::Hash
        };

        DistributedConfig {
            num_workers,
            rank: 0, // Will be set by each worker
            backend,
            partitioning,
            aggregation: AggregationMethod::AllReduce,
            sync_frequency: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use torsh_tensor::creation::randn;

    #[test]
    fn test_distributed_config_creation() {
        let config = DistributedConfig {
            num_workers: 4,
            rank: 0,
            backend: CommunicationBackend::InMemory,
            partitioning: GraphPartitioning::Random,
            aggregation: AggregationMethod::Average,
            sync_frequency: 10,
        };

        assert_eq!(config.num_workers, 4);
        assert_eq!(config.rank, 0);
    }

    #[test]
    fn test_load_balance_calculation() {
        let partition_sizes = vec![100, 100, 100, 100];
        let balance_score = utils::calculate_load_balance(&partition_sizes);
        assert_eq!(balance_score, 0.0); // Perfect balance

        let unbalanced_sizes = vec![200, 50, 50, 50];
        let unbalanced_score = utils::calculate_load_balance(&unbalanced_sizes);
        assert!(unbalanced_score > 0.0); // Poor balance
    }

    #[test]
    fn test_communication_cost_estimation() {
        let partition_info = PartitionInfo {
            worker_rank: 0,
            nodes: vec![0, 1, 2],
            internal_edges: vec![(0, 1)],
            boundary_edges: vec![(2, 3, 1)],
            metrics: PartitionMetrics {
                num_nodes: 3,
                num_internal_edges: 1,
                num_boundary_edges: 1,
                load_balance_score: 0.0,
                communication_cost: 1.0,
            },
        };

        let cost = utils::estimate_communication_cost(&[partition_info]);
        assert_eq!(cost, 1.0);
    }

    #[test]
    fn test_optimal_config_creation() {
        let config = utils::create_optimal_config(4, 1_000_000);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.backend, CommunicationBackend::NCCL);

        let small_config = utils::create_optimal_config(1, 1000);
        assert_eq!(small_config.num_workers, 1);
        assert_eq!(small_config.backend, CommunicationBackend::InMemory);
    }

    #[test]
    fn test_sync_state() {
        let mut sync_state = SyncState::new();
        assert_eq!(sync_state.current_step, 0);
        assert!(!sync_state.should_sync(10));

        sync_state.current_step = 10;
        assert!(sync_state.should_sync(10));

        sync_state.mark_synced();
        assert_eq!(sync_state.last_sync_step, 10);
    }

    #[test]
    fn test_distributed_metrics() {
        let mut metrics = DistributedMetrics::new();
        metrics.computation_time_ms = 800.0;
        metrics.communication_time_ms = 200.0;

        metrics.compute_efficiency();
        assert_eq!(metrics.efficiency_score, 0.8);
    }

    #[test]
    fn test_partition_info_creation() {
        let x = randn(&[5, 3]).unwrap();
        let edge_index = torsh_tensor::creation::zeros(&[2, 0]).unwrap();
        let graph = GraphData::new(x, edge_index);

        let partition_info = DistributedGNN::create_partition_info(&graph, 0);
        assert_eq!(partition_info.worker_rank, 0);
        assert_eq!(partition_info.nodes.len(), 5);
        assert_eq!(partition_info.metrics.num_nodes, 5);
    }
}
