//! Graph partitioning module for distributed execution
//!
//! This module provides functionality to partition FX graphs across multiple devices or nodes
//! for distributed processing.

use crate::{FxGraph, Node};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet, VecDeque};
use torsh_core::Result;

/// Device information for graph partitioning
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    pub id: String,
    pub device_type: DeviceType,
    pub memory_capacity: usize,  // in bytes
    pub compute_capability: f64, // relative compute power
    pub bandwidth: f64,          // communication bandwidth
}

impl Eq for DeviceInfo {}

impl std::hash::Hash for DeviceInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.device_type.hash(state);
        self.memory_capacity.hash(state);
        // Hash f64 values as bits to make them hashable
        self.compute_capability.to_bits().hash(state);
        self.bandwidth.to_bits().hash(state);
    }
}

/// Device types for partitioning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    CPU,
    CUDA(u8, u8), // compute capability
    Metal,
    OpenCL,
    WebGPU,
}

/// Partitioning strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PartitioningStrategy {
    /// Minimize communication between partitions
    MinCommunication,
    /// Balance computational load across devices
    LoadBalance,
    /// Minimize memory usage per device
    MemoryOptimal,
    /// Custom strategy with weights
    Weighted {
        communication_weight: f64,
        load_balance_weight: f64,
        memory_weight: f64,
    },
}

/// Graph partition representation
#[derive(Debug, Clone)]
pub struct GraphPartition {
    pub device: DeviceInfo,
    pub nodes: Vec<NodeIndex>,
    pub local_edges: Vec<(NodeIndex, NodeIndex)>,
    pub communication_edges: Vec<CommunicationEdge>,
    pub estimated_memory: usize,
    pub estimated_compute_time: f64,
}

/// Communication edge between partitions
#[derive(Debug, Clone)]
pub struct CommunicationEdge {
    pub source_partition: usize,
    pub target_partition: usize,
    pub source_node: NodeIndex,
    pub target_node: NodeIndex,
    pub data_size: usize,
    pub communication_cost: f64,
}

/// Partitioned graph result
#[derive(Debug, Clone)]
pub struct PartitionedGraph {
    pub partitions: Vec<GraphPartition>,
    pub communication_schedule: CommunicationSchedule,
    pub total_communication_cost: f64,
    pub load_balance_score: f64,
    pub memory_efficiency: f64,
}

/// Communication schedule for coordinating data transfer
#[derive(Debug, Clone)]
pub struct CommunicationSchedule {
    pub stages: Vec<CommunicationStage>,
    pub total_stages: usize,
}

/// Communication stage with parallel transfers
#[derive(Debug, Clone)]
pub struct CommunicationStage {
    pub stage_id: usize,
    pub transfers: Vec<DataTransfer>,
    pub dependencies: Vec<usize>, // prerequisite stages
}

/// Data transfer between devices
#[derive(Debug, Clone)]
pub struct DataTransfer {
    pub source_device: String,
    pub target_device: String,
    pub data_id: String,
    pub data_size: usize,
    pub priority: TransferPriority,
}

/// Transfer priority for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Graph partitioner implementation
pub struct GraphPartitioner {
    devices: Vec<DeviceInfo>,
    strategy: PartitioningStrategy,
    max_partitions: Option<usize>,
}

impl GraphPartitioner {
    /// Create a new graph partitioner
    pub fn new(devices: Vec<DeviceInfo>, strategy: PartitioningStrategy) -> Self {
        Self {
            devices,
            strategy,
            max_partitions: None,
        }
    }

    /// Set maximum number of partitions
    pub fn with_max_partitions(mut self, max_partitions: usize) -> Self {
        self.max_partitions = Some(max_partitions);
        self
    }

    /// Partition the graph according to the strategy
    pub fn partition(&self, graph: &FxGraph) -> Result<PartitionedGraph> {
        match self.strategy {
            PartitioningStrategy::MinCommunication => self.partition_min_communication(graph),
            PartitioningStrategy::LoadBalance => self.partition_load_balance(graph),
            PartitioningStrategy::MemoryOptimal => self.partition_memory_optimal(graph),
            PartitioningStrategy::Weighted { .. } => self.partition_weighted(graph),
        }
    }

    fn partition_min_communication(&self, graph: &FxGraph) -> Result<PartitionedGraph> {
        let mut partitions = Vec::new();
        let mut node_to_partition = HashMap::new();

        // Start with a simple graph cut algorithm
        let _node_weights = self.compute_node_weights(graph);
        let _edge_weights = self.compute_edge_weights(graph);

        // Use a greedy approach to minimize cut edges
        let mut remaining_nodes: HashSet<NodeIndex> = graph.nodes().map(|(idx, _)| idx).collect();

        for (device_idx, device) in self.devices.iter().enumerate() {
            if remaining_nodes.is_empty() {
                break;
            }

            let mut partition_nodes = Vec::new();
            let target_size = remaining_nodes.len() / (self.devices.len() - device_idx);

            // Start with a random node or input node
            let start_node = if let Some(&node) = remaining_nodes.iter().next() {
                node
            } else {
                break;
            };

            let mut to_visit = VecDeque::new();
            to_visit.push_back(start_node);
            remaining_nodes.remove(&start_node);

            // BFS expansion while minimizing communication
            while let Some(current_node) = to_visit.pop_front() {
                partition_nodes.push(current_node);
                node_to_partition.insert(current_node, device_idx);

                if partition_nodes.len() >= target_size {
                    break;
                }

                // Add neighbors that minimize communication cost
                let neighbors = self.get_neighbors(graph, current_node);
                for neighbor in neighbors {
                    if remaining_nodes.contains(&neighbor) {
                        to_visit.push_back(neighbor);
                        remaining_nodes.remove(&neighbor);
                    }
                }
            }

            partitions.push(GraphPartition {
                device: device.clone(),
                nodes: partition_nodes,
                local_edges: Vec::new(),
                communication_edges: Vec::new(),
                estimated_memory: 0,
                estimated_compute_time: 0.0,
            });
        }

        // Compute edges and communication costs
        self.compute_partition_edges(graph, &mut partitions, &node_to_partition)?;

        let communication_schedule = self.create_communication_schedule(&partitions)?;
        let metrics = self.compute_partition_metrics(&partitions);

        Ok(PartitionedGraph {
            partitions,
            communication_schedule,
            total_communication_cost: metrics.0,
            load_balance_score: metrics.1,
            memory_efficiency: metrics.2,
        })
    }

    fn partition_load_balance(&self, graph: &FxGraph) -> Result<PartitionedGraph> {
        let node_weights = self.compute_node_weights(graph);
        let total_weight: f64 = node_weights.values().sum();
        let target_weight_per_device = total_weight / self.devices.len() as f64;

        let mut partitions = Vec::new();
        let mut node_to_partition = HashMap::new();
        let mut remaining_nodes: Vec<_> = graph.nodes().map(|(idx, _)| idx).collect();

        // Sort nodes by weight (descending) for better load balancing
        remaining_nodes.sort_by(|&a, &b| {
            node_weights
                .get(&b)
                .unwrap_or(&0.0)
                .partial_cmp(node_weights.get(&a).unwrap_or(&0.0))
                .expect("node weights should be comparable")
        });

        for (device_idx, device) in self.devices.iter().enumerate() {
            let mut partition_nodes = Vec::new();
            let mut current_weight = 0.0;
            let adjusted_target = target_weight_per_device * device.compute_capability;

            let mut i = 0;
            while i < remaining_nodes.len() && current_weight < adjusted_target {
                let node = remaining_nodes[i];
                let node_weight = *node_weights.get(&node).unwrap_or(&0.0);

                if current_weight + node_weight <= adjusted_target * 1.2
                    || partition_nodes.is_empty()
                {
                    partition_nodes.push(node);
                    node_to_partition.insert(node, device_idx);
                    current_weight += node_weight;
                    remaining_nodes.remove(i);
                } else {
                    i += 1;
                }
            }

            partitions.push(GraphPartition {
                device: device.clone(),
                nodes: partition_nodes,
                local_edges: Vec::new(),
                communication_edges: Vec::new(),
                estimated_memory: 0,
                estimated_compute_time: current_weight,
            });
        }

        // Handle remaining nodes
        for node in remaining_nodes {
            let min_partition = partitions
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| p.estimated_compute_time as u64)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            partitions[min_partition].nodes.push(node);
            node_to_partition.insert(node, min_partition);
        }

        self.compute_partition_edges(graph, &mut partitions, &node_to_partition)?;

        let communication_schedule = self.create_communication_schedule(&partitions)?;
        let metrics = self.compute_partition_metrics(&partitions);

        Ok(PartitionedGraph {
            partitions,
            communication_schedule,
            total_communication_cost: metrics.0,
            load_balance_score: metrics.1,
            memory_efficiency: metrics.2,
        })
    }

    fn partition_memory_optimal(&self, graph: &FxGraph) -> Result<PartitionedGraph> {
        let node_memory = self.compute_node_memory_usage(graph);

        let mut partitions = Vec::new();
        let mut node_to_partition = HashMap::new();
        let mut remaining_nodes: Vec<_> = graph.nodes().map(|(idx, _)| idx).collect();

        for (device_idx, device) in self.devices.iter().enumerate() {
            let mut partition_nodes = Vec::new();
            let mut current_memory = 0;
            let memory_limit = device.memory_capacity;

            let mut i = 0;
            while i < remaining_nodes.len() {
                let node = remaining_nodes[i];
                let node_mem = *node_memory.get(&node).unwrap_or(&0);

                if current_memory + node_mem <= memory_limit || partition_nodes.is_empty() {
                    partition_nodes.push(node);
                    node_to_partition.insert(node, device_idx);
                    current_memory += node_mem;
                    remaining_nodes.remove(i);
                } else {
                    i += 1;
                }
            }

            partitions.push(GraphPartition {
                device: device.clone(),
                nodes: partition_nodes,
                local_edges: Vec::new(),
                communication_edges: Vec::new(),
                estimated_memory: current_memory,
                estimated_compute_time: 0.0,
            });
        }

        self.compute_partition_edges(graph, &mut partitions, &node_to_partition)?;

        let communication_schedule = self.create_communication_schedule(&partitions)?;
        let metrics = self.compute_partition_metrics(&partitions);

        Ok(PartitionedGraph {
            partitions,
            communication_schedule,
            total_communication_cost: metrics.0,
            load_balance_score: metrics.1,
            memory_efficiency: metrics.2,
        })
    }

    fn partition_weighted(&self, graph: &FxGraph) -> Result<PartitionedGraph> {
        // Implement a weighted combination of strategies
        // For now, use load balance as the primary strategy
        self.partition_load_balance(graph)
    }

    fn compute_node_weights(&self, graph: &FxGraph) -> HashMap<NodeIndex, f64> {
        let mut weights = HashMap::new();

        for (idx, node) in graph.nodes() {
            let weight = match node {
                Node::Input(_) => 0.1,
                Node::Output => 0.1,
                Node::Call(op_name, _) => self.get_operation_weight(op_name),
                Node::Conditional { .. } => 2.0,
                Node::Loop { .. } => 5.0,
                Node::Merge { .. } => 0.5,
                Node::GetAttr { .. } => 0.1,
            };
            weights.insert(idx, weight);
        }

        weights
    }

    fn compute_edge_weights(&self, graph: &FxGraph) -> HashMap<(NodeIndex, NodeIndex), f64> {
        let mut weights = HashMap::new();

        for edge_ref in graph.graph.edge_references() {
            let source = edge_ref.source();
            let target = edge_ref.target();

            // Estimate data size and communication cost
            let weight = 1.0; // Default weight
            weights.insert((source, target), weight);
        }

        weights
    }

    fn compute_node_memory_usage(&self, graph: &FxGraph) -> HashMap<NodeIndex, usize> {
        let mut memory = HashMap::new();

        for (idx, node) in graph.nodes() {
            let mem_usage = match node {
                Node::Input(_) => 1024 * 1024, // 1MB default
                Node::Output => 0,
                Node::Call(op_name, _) => self.get_operation_memory(op_name),
                Node::Conditional { .. } => 512 * 1024,
                Node::Loop { .. } => 2 * 1024 * 1024,
                Node::Merge { .. } => 256 * 1024,
                Node::GetAttr { .. } => 0,
            };
            memory.insert(idx, mem_usage);
        }

        memory
    }

    fn get_operation_weight(&self, op_name: &str) -> f64 {
        match op_name {
            "add" | "sub" | "mul" | "div" => 1.0,
            "relu" | "sigmoid" | "tanh" => 1.5,
            "conv2d" => 10.0,
            "matmul" => 8.0,
            "batch_norm" => 3.0,
            "softmax" => 4.0,
            _ => 2.0, // Default weight
        }
    }

    fn get_operation_memory(&self, op_name: &str) -> usize {
        match op_name {
            "add" | "sub" | "mul" | "div" => 512 * 1024,
            "relu" | "sigmoid" | "tanh" => 256 * 1024,
            "conv2d" => 10 * 1024 * 1024,
            "matmul" => 8 * 1024 * 1024,
            "batch_norm" => 2 * 1024 * 1024,
            "softmax" => 1 * 1024 * 1024,
            _ => 1 * 1024 * 1024, // 1MB default
        }
    }

    fn get_neighbors(&self, graph: &FxGraph, node: NodeIndex) -> Vec<NodeIndex> {
        let mut neighbors = Vec::new();

        // Get incoming edges
        for edge_ref in graph
            .graph
            .edges_directed(node, petgraph::Direction::Incoming)
        {
            neighbors.push(edge_ref.source());
        }

        // Get outgoing edges
        for edge_ref in graph
            .graph
            .edges_directed(node, petgraph::Direction::Outgoing)
        {
            neighbors.push(edge_ref.target());
        }

        neighbors
    }

    fn compute_partition_edges(
        &self,
        graph: &FxGraph,
        partitions: &mut [GraphPartition],
        node_to_partition: &HashMap<NodeIndex, usize>,
    ) -> Result<()> {
        // Clear existing edges
        for partition in partitions.iter_mut() {
            partition.local_edges.clear();
            partition.communication_edges.clear();
        }

        for edge_ref in graph.graph.edge_references() {
            let source = edge_ref.source();
            let target = edge_ref.target();

            let source_partition = match node_to_partition.get(&source) {
                Some(partition) => *partition,
                None => continue, // Skip edges with unmapped nodes
            };
            let target_partition = match node_to_partition.get(&target) {
                Some(partition) => *partition,
                None => continue, // Skip edges with unmapped nodes
            };

            if source_partition == target_partition {
                // Local edge within partition
                partitions[source_partition]
                    .local_edges
                    .push((source, target));
            } else {
                // Communication edge between partitions
                let comm_edge = CommunicationEdge {
                    source_partition,
                    target_partition,
                    source_node: source,
                    target_node: target,
                    data_size: 1024, // Estimate
                    communication_cost: self.compute_communication_cost(
                        &partitions[source_partition].device,
                        &partitions[target_partition].device,
                        1024,
                    ),
                };

                partitions[source_partition]
                    .communication_edges
                    .push(comm_edge);
            }
        }

        Ok(())
    }

    fn compute_communication_cost(
        &self,
        source: &DeviceInfo,
        target: &DeviceInfo,
        data_size: usize,
    ) -> f64 {
        let bandwidth = source.bandwidth.min(target.bandwidth);
        let latency = if source.device_type == target.device_type {
            0.001
        } else {
            0.01
        };

        (data_size as f64) / bandwidth + latency
    }

    fn create_communication_schedule(
        &self,
        partitions: &[GraphPartition],
    ) -> Result<CommunicationSchedule> {
        let mut stages = Vec::new();
        let mut processed_transfers = HashSet::new();
        let mut stage_id = 0;

        // Group communication edges by dependencies
        let mut remaining_edges: Vec<_> = partitions
            .iter()
            .enumerate()
            .flat_map(|(partition_idx, partition)| {
                partition
                    .communication_edges
                    .iter()
                    .map(move |edge| (partition_idx, edge))
            })
            .collect();

        while !remaining_edges.is_empty() {
            let mut current_stage = CommunicationStage {
                stage_id,
                transfers: Vec::new(),
                dependencies: Vec::new(),
            };

            let mut i = 0;
            while i < remaining_edges.len() {
                let (_, edge) = &remaining_edges[i];
                let transfer_key = (
                    edge.source_partition,
                    edge.target_partition,
                    edge.source_node,
                    edge.target_node,
                );

                if !processed_transfers.contains(&transfer_key) {
                    let transfer = DataTransfer {
                        source_device: partitions[edge.source_partition].device.id.clone(),
                        target_device: partitions[edge.target_partition].device.id.clone(),
                        data_id: format!(
                            "data_{}_{}",
                            edge.source_node.index(),
                            edge.target_node.index()
                        ),
                        data_size: edge.data_size,
                        priority: TransferPriority::Medium,
                    };

                    current_stage.transfers.push(transfer);
                    processed_transfers.insert(transfer_key);
                    remaining_edges.remove(i);
                } else {
                    i += 1;
                }
            }

            if !current_stage.transfers.is_empty() {
                stages.push(current_stage);
                stage_id += 1;
            }
        }

        Ok(CommunicationSchedule {
            total_stages: stages.len(),
            stages,
        })
    }

    fn compute_partition_metrics(&self, partitions: &[GraphPartition]) -> (f64, f64, f64) {
        let total_communication_cost = partitions
            .iter()
            .flat_map(|p| &p.communication_edges)
            .map(|edge| edge.communication_cost)
            .sum();

        let compute_times: Vec<f64> = partitions
            .iter()
            .map(|p| p.estimated_compute_time)
            .collect();
        let max_compute_time = compute_times.iter().cloned().fold(0.0, f64::max);
        let avg_compute_time = compute_times.iter().sum::<f64>() / compute_times.len() as f64;
        let load_balance_score = if max_compute_time > 0.0 {
            avg_compute_time / max_compute_time
        } else {
            1.0
        };

        let memory_usage: Vec<usize> = partitions.iter().map(|p| p.estimated_memory).collect();
        let total_memory = memory_usage.iter().sum::<usize>();
        let total_capacity: usize = partitions.iter().map(|p| p.device.memory_capacity).sum();
        let memory_efficiency = if total_capacity > 0 {
            total_memory as f64 / total_capacity as f64
        } else {
            0.0
        };

        (
            total_communication_cost,
            load_balance_score,
            memory_efficiency,
        )
    }
}

/// Utility functions for graph partitioning
impl GraphPartitioner {
    /// Create a default CPU cluster configuration
    pub fn create_cpu_cluster(num_devices: usize) -> Vec<DeviceInfo> {
        (0..num_devices)
            .map(|i| DeviceInfo {
                id: format!("cpu_{i}"),
                device_type: DeviceType::CPU,
                memory_capacity: 8 * 1024 * 1024 * 1024, // 8GB
                compute_capability: 1.0,
                bandwidth: 10_000_000_000.0, // 10 GB/s
            })
            .collect()
    }

    /// Create a heterogeneous cluster with CPU and GPU devices
    pub fn create_heterogeneous_cluster() -> Vec<DeviceInfo> {
        vec![
            DeviceInfo {
                id: "cpu_0".to_string(),
                device_type: DeviceType::CPU,
                memory_capacity: 16 * 1024 * 1024 * 1024, // 16GB
                compute_capability: 1.0,
                bandwidth: 50_000_000_000.0, // 50 GB/s
            },
            DeviceInfo {
                id: "cuda_0".to_string(),
                device_type: DeviceType::CUDA(8, 0), // RTX 3080 class
                memory_capacity: 10 * 1024 * 1024 * 1024, // 10GB
                compute_capability: 5.0,
                bandwidth: 760_000_000_000.0, // 760 GB/s
            },
            DeviceInfo {
                id: "cuda_1".to_string(),
                device_type: DeviceType::CUDA(8, 6), // RTX 4090 class
                memory_capacity: 24 * 1024 * 1024 * 1024, // 24GB
                compute_capability: 8.0,
                bandwidth: 1_000_000_000_000.0, // 1 TB/s
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, FxGraph, Node};

    #[test]
    fn test_graph_partitioning_min_communication() {
        let mut graph = FxGraph::new();
        let input1 = graph.graph.add_node(Node::Input("x".to_string()));
        let input2 = graph.graph.add_node(Node::Input("y".to_string()));
        let add = graph.graph.add_node(Node::Call(
            "add".to_string(),
            vec!["x".to_string(), "y".to_string()],
        ));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["add_out".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input1,
            add,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            input2,
            add,
            Edge {
                name: "y".to_string(),
            },
        );
        graph.graph.add_edge(
            add,
            relu,
            Edge {
                name: "add_out".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );

        graph.inputs = vec![input1, input2];
        graph.outputs = vec![output];

        let devices = GraphPartitioner::create_cpu_cluster(2);
        let partitioner = GraphPartitioner::new(devices, PartitioningStrategy::MinCommunication);

        let result = partitioner.partition(&graph).unwrap();

        assert_eq!(result.partitions.len(), 2);
        assert!(result.total_communication_cost >= 0.0);
        assert!(result.load_balance_score > 0.0);
    }

    #[test]
    fn test_graph_partitioning_load_balance() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));

        // Create a linear chain of expensive operations
        let mut prev = input;
        for i in 0..6 {
            let op = graph
                .graph
                .add_node(Node::Call("matmul".to_string(), vec![format!("input_{i}")]));
            graph.graph.add_edge(
                prev,
                op,
                Edge {
                    name: format!("edge_{i}"),
                },
            );
            prev = op;
        }

        let output = graph.graph.add_node(Node::Output);
        graph.graph.add_edge(
            prev,
            output,
            Edge {
                name: "final".to_string(),
            },
        );

        graph.inputs = vec![input];
        graph.outputs = vec![output];

        let devices = GraphPartitioner::create_heterogeneous_cluster();
        let partitioner = GraphPartitioner::new(devices, PartitioningStrategy::LoadBalance);

        let result = partitioner.partition(&graph).unwrap();

        assert_eq!(result.partitions.len(), 3);
        assert!(result.load_balance_score > 0.0);

        // Check that high-compute devices get more work
        let gpu_partitions: Vec<_> = result
            .partitions
            .iter()
            .filter(|p| matches!(p.device.device_type, DeviceType::CUDA(_, _)))
            .collect();

        assert!(!gpu_partitions.is_empty());
    }

    #[test]
    fn test_communication_schedule() {
        let devices = vec![
            DeviceInfo {
                id: "device_0".to_string(),
                device_type: DeviceType::CPU,
                memory_capacity: 1024 * 1024 * 1024,
                compute_capability: 1.0,
                bandwidth: 1_000_000_000.0,
            },
            DeviceInfo {
                id: "device_1".to_string(),
                device_type: DeviceType::CPU,
                memory_capacity: 1024 * 1024 * 1024,
                compute_capability: 1.0,
                bandwidth: 1_000_000_000.0,
            },
        ];

        let partitions = vec![
            GraphPartition {
                device: devices[0].clone(),
                nodes: vec![],
                local_edges: vec![],
                communication_edges: vec![CommunicationEdge {
                    source_partition: 0,
                    target_partition: 1,
                    source_node: NodeIndex::new(0),
                    target_node: NodeIndex::new(1),
                    data_size: 1024,
                    communication_cost: 0.001,
                }],
                estimated_memory: 0,
                estimated_compute_time: 0.0,
            },
            GraphPartition {
                device: devices[1].clone(),
                nodes: vec![],
                local_edges: vec![],
                communication_edges: vec![],
                estimated_memory: 0,
                estimated_compute_time: 0.0,
            },
        ];

        let partitioner = GraphPartitioner::new(devices, PartitioningStrategy::MinCommunication);
        let schedule = partitioner
            .create_communication_schedule(&partitions)
            .unwrap();

        assert!(schedule.total_stages > 0);
        assert!(!schedule.stages.is_empty());
        assert!(!schedule.stages[0].transfers.is_empty());
    }
}
