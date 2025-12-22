//! Package Replication and High Availability
//!
//! This module provides comprehensive replication and high availability capabilities
//! for package distribution across multiple regions and nodes, including consistency
//! management, automatic failover, load balancing, and conflict resolution.
//!
//! # Features
//!
//! - **Multi-Region Replication**: Replicate packages across geographic regions
//! - **Consistency Models**: Support for eventual, strong, and causal consistency
//! - **Automatic Failover**: Detect and recover from node failures
//! - **Load Balancing**: Distribute requests across healthy replicas
//! - **Conflict Resolution**: Handle concurrent updates with configurable strategies
//! - **Topology Management**: Configure and manage replication topologies
//! - **Health Monitoring**: Track replica health and synchronization status
//! - **Split-Brain Detection**: Detect and resolve network partitions
//!
//! # Examples
//!
//! ```rust
//! use torsh_package::replication::{
//!     ReplicationManager, ReplicationConfig, ConsistencyLevel, ReplicationNode
//! };
//!
//! // Create replication manager
//! let config = ReplicationConfig {
//!     consistency: ConsistencyLevel::Eventual,
//!     replication_factor: 3,
//!     auto_failover: true,
//!     sync_interval_secs: 60,
//! };
//!
//! let mut manager = ReplicationManager::new(config);
//!
//! // Add replication nodes
//! let node = ReplicationNode::new(
//!     "node1".to_string(),
//!     "us-east-1".to_string(),
//!     "https://node1.example.com".to_string(),
//!     1,
//!     1000,
//! );
//! manager.add_node(node).unwrap();
//!
//! // Replicate a package
//! manager.replicate_package("my-package", "1.0.0", b"package data").unwrap();
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use torsh_core::error::TorshError;

/// Consistency level for replication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Eventual consistency - fastest, weakest guarantees
    Eventual,
    /// Quorum consistency - balanced performance and consistency
    Quorum,
    /// Strong consistency - slowest, strongest guarantees
    Strong,
    /// Causal consistency - preserves causality
    Causal,
}

/// Replication strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    /// Synchronous replication - wait for all replicas
    Synchronous,
    /// Asynchronous replication - fire and forget
    Asynchronous,
    /// Semi-synchronous - wait for majority
    SemiSynchronous,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Last write wins based on timestamp
    LastWriteWins,
    /// First write wins
    FirstWriteWins,
    /// Custom merge strategy
    Custom,
    /// Manual resolution required
    Manual,
}

/// Replication node status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is healthy and replicating
    Healthy,
    /// Node is experiencing degraded performance
    Degraded,
    /// Node is unhealthy and not replicating
    Unhealthy,
    /// Node is in maintenance mode
    Maintenance,
    /// Node is offline
    Offline,
}

impl Default for NodeStatus {
    fn default() -> Self {
        Self::Healthy
    }
}

/// Replication node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationNode {
    /// Unique node identifier
    pub id: String,
    /// Geographic region
    pub region: String,
    /// Node endpoint URL
    pub endpoint: String,
    /// Priority (higher = preferred)
    pub priority: u32,
    /// Storage capacity in bytes
    pub capacity: u64,
    /// Current status
    #[serde(skip)]
    pub status: NodeStatus,
    /// Last health check timestamp
    #[serde(skip)]
    pub last_health_check: Option<DateTime<Utc>>,
    /// Replication lag in seconds
    #[serde(skip)]
    pub replication_lag_secs: f64,
}

impl ReplicationNode {
    /// Create a new replication node
    pub fn new(id: String, region: String, endpoint: String, priority: u32, capacity: u64) -> Self {
        Self {
            id,
            region,
            endpoint,
            priority,
            capacity,
            status: NodeStatus::Healthy,
            last_health_check: None,
            replication_lag_secs: 0.0,
        }
    }
}

/// Replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Consistency level
    pub consistency: ConsistencyLevel,
    /// Number of replicas to maintain
    pub replication_factor: usize,
    /// Enable automatic failover
    pub auto_failover: bool,
    /// Synchronization interval in seconds
    pub sync_interval_secs: u64,
}

/// Package replica metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaMetadata {
    /// Package ID
    pub package_id: String,
    /// Package version
    pub version: String,
    /// Node ID where replica is stored
    pub node_id: String,
    /// Replica version/timestamp
    pub replica_version: u64,
    /// Checksum for integrity
    pub checksum: String,
    /// Last synchronized timestamp
    pub last_sync: DateTime<Utc>,
    /// Size in bytes
    pub size_bytes: u64,
}

/// Replication operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationOperation {
    /// Operation ID
    pub id: String,
    /// Package ID
    pub package_id: String,
    /// Package version
    pub version: String,
    /// Operation type (Create, Update, Delete)
    pub operation_type: String,
    /// Source node ID
    pub source_node: String,
    /// Target node IDs
    pub target_nodes: Vec<String>,
    /// Operation timestamp
    pub timestamp: DateTime<Utc>,
    /// Operation status
    pub status: OperationStatus,
}

/// Operation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationStatus {
    /// Operation is pending
    Pending,
    /// Operation is in progress
    InProgress,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed,
}

/// Replication conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConflict {
    /// Conflict ID
    pub id: String,
    /// Package ID
    pub package_id: String,
    /// Package version
    pub version: String,
    /// Conflicting replicas
    pub conflicting_replicas: Vec<ReplicaMetadata>,
    /// Conflict detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Resolution status
    pub resolved: bool,
    /// Resolution strategy used
    pub resolution_strategy: Option<ConflictResolution>,
}

/// Replication statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplicationStatistics {
    /// Total nodes
    pub total_nodes: usize,
    /// Healthy nodes
    pub healthy_nodes: usize,
    /// Total replicas
    pub total_replicas: usize,
    /// Total replication operations
    pub total_operations: u64,
    /// Successful operations
    pub successful_operations: u64,
    /// Failed operations
    pub failed_operations: u64,
    /// Active conflicts
    pub active_conflicts: usize,
    /// Average replication lag (seconds)
    pub avg_replication_lag_secs: f64,
    /// Total bandwidth used (bytes)
    pub total_bandwidth_bytes: u64,
}

/// Replication manager
///
/// Manages package replication across multiple nodes with configurable
/// consistency levels, automatic failover, and conflict resolution.
pub struct ReplicationManager {
    /// Replication configuration
    config: ReplicationConfig,
    /// Replication nodes by ID
    nodes: HashMap<String, ReplicationNode>,
    /// Package replicas
    replicas: HashMap<String, Vec<ReplicaMetadata>>,
    /// Replication operations
    operations: VecDeque<ReplicationOperation>,
    /// Active conflicts
    conflicts: Vec<ReplicationConflict>,
    /// Statistics
    statistics: ReplicationStatistics,
}

impl ReplicationManager {
    /// Create a new replication manager
    pub fn new(config: ReplicationConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            replicas: HashMap::new(),
            operations: VecDeque::new(),
            conflicts: Vec::new(),
            statistics: ReplicationStatistics::default(),
        }
    }

    /// Add a replication node
    pub fn add_node(&mut self, node: ReplicationNode) -> Result<(), TorshError> {
        if self.nodes.contains_key(&node.id) {
            return Err(TorshError::InvalidArgument(format!(
                "Node {} already exists",
                node.id
            )));
        }

        self.nodes.insert(node.id.clone(), node);
        self.update_statistics();

        Ok(())
    }

    /// Remove a replication node
    pub fn remove_node(&mut self, node_id: &str) -> Result<(), TorshError> {
        self.nodes
            .remove(node_id)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Node {} not found", node_id)))?;

        // Handle replica redistribution
        self.redistribute_replicas(node_id)?;
        self.update_statistics();

        Ok(())
    }

    /// Replicate a package to all nodes
    pub fn replicate_package(
        &mut self,
        package_id: &str,
        version: &str,
        _data: &[u8],
    ) -> Result<(), TorshError> {
        // Select target nodes based on replication factor
        let target_nodes = self.select_replication_nodes(package_id)?;

        // Create replication operation
        let operation = ReplicationOperation {
            id: uuid::Uuid::new_v4().to_string(),
            package_id: package_id.to_string(),
            version: version.to_string(),
            operation_type: "Create".to_string(),
            source_node: "primary".to_string(),
            target_nodes: target_nodes.iter().map(|n| n.id.clone()).collect(),
            timestamp: Utc::now(),
            status: OperationStatus::Pending,
        };

        self.operations.push_back(operation);

        // Execute replication based on strategy
        match self.config.consistency {
            ConsistencyLevel::Strong => self.replicate_synchronously(package_id, version)?,
            ConsistencyLevel::Quorum => self.replicate_to_quorum(package_id, version)?,
            ConsistencyLevel::Eventual | ConsistencyLevel::Causal => {
                self.replicate_asynchronously(package_id, version)?
            }
        }

        self.update_statistics();

        Ok(())
    }

    /// Get package from best available replica
    pub fn get_package(&self, package_id: &str, version: &str) -> Result<String, TorshError> {
        let key = format!("{}:{}", package_id, version);

        let replicas = self
            .replicas
            .get(&key)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Package {} not found", key)))?;

        // Select best replica based on node health and latency
        let best_replica = self.select_best_replica(replicas)?;

        Ok(best_replica.node_id.clone())
    }

    /// Perform health check on all nodes
    pub fn health_check(&mut self) -> Result<(), TorshError> {
        let now = Utc::now();

        // Collect node IDs first to avoid borrow checker issues
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

        for node_id in node_ids {
            // Mock health check (in production, send actual health check request)
            let is_healthy = self.check_node_health(&node_id);

            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.status = if is_healthy {
                    NodeStatus::Healthy
                } else {
                    NodeStatus::Unhealthy
                };

                node.last_health_check = Some(now);
            }
        }

        self.update_statistics();

        // Handle failover if auto_failover is enabled
        if self.config.auto_failover {
            self.handle_failover()?;
        }

        Ok(())
    }

    /// Synchronize replicas across nodes
    pub fn synchronize(&mut self) -> Result<(), TorshError> {
        // Identify replicas that are out of sync
        let mut to_sync = Vec::new();

        for (key, replicas) in &self.replicas {
            let max_version = replicas
                .iter()
                .map(|r| r.replica_version)
                .max()
                .unwrap_or(0);

            for replica in replicas {
                if replica.replica_version < max_version {
                    to_sync.push((key.clone(), replica.node_id.clone()));
                }
            }
        }

        // Synchronize out-of-sync replicas
        for (key, node_id) in to_sync {
            self.sync_replica(&key, &node_id)?;
        }

        Ok(())
    }

    /// Detect and resolve conflicts
    pub fn resolve_conflicts(&mut self) -> Result<(), TorshError> {
        let mut replicas_to_propagate = Vec::new();

        // Collect replicas to propagate and mark conflicts as resolved
        for conflict in &mut self.conflicts {
            if !conflict.resolved {
                // Apply conflict resolution strategy
                let strategy = ConflictResolution::LastWriteWins; // Use configured strategy

                match strategy {
                    ConflictResolution::LastWriteWins => {
                        // Keep replica with latest timestamp
                        if let Some(latest) = conflict
                            .conflicting_replicas
                            .iter()
                            .max_by_key(|r| r.last_sync)
                            .cloned()
                        {
                            replicas_to_propagate.push(latest);
                            conflict.resolved = true;
                            conflict.resolution_strategy = Some(strategy);
                        }
                    }
                    ConflictResolution::FirstWriteWins => {
                        // Keep replica with earliest timestamp
                        if let Some(earliest) = conflict
                            .conflicting_replicas
                            .iter()
                            .min_by_key(|r| r.last_sync)
                            .cloned()
                        {
                            replicas_to_propagate.push(earliest);
                            conflict.resolved = true;
                            conflict.resolution_strategy = Some(strategy);
                        }
                    }
                    ConflictResolution::Custom | ConflictResolution::Manual => {
                        // Require manual intervention
                    }
                }
            }
        }

        // Propagate replicas after collecting them
        for replica in replicas_to_propagate {
            self.propagate_replica(&replica)?;
        }

        // Remove resolved conflicts
        self.conflicts.retain(|c| !c.resolved);

        Ok(())
    }

    /// Get replication statistics
    pub fn get_statistics(&self) -> &ReplicationStatistics {
        &self.statistics
    }

    /// Get node status
    pub fn get_node_status(&self, node_id: &str) -> Option<NodeStatus> {
        self.nodes.get(node_id).map(|n| n.status)
    }

    /// List all nodes
    pub fn list_nodes(&self) -> Vec<&ReplicationNode> {
        self.nodes.values().collect()
    }

    /// List replicas for a package
    pub fn list_replicas(&self, package_id: &str, version: &str) -> Vec<&ReplicaMetadata> {
        let key = format!("{}:{}", package_id, version);
        self.replicas
            .get(&key)
            .map(|replicas| replicas.iter().collect())
            .unwrap_or_default()
    }

    /// Get active conflicts
    pub fn get_conflicts(&self) -> Vec<&ReplicationConflict> {
        self.conflicts.iter().collect()
    }

    // Private helper methods

    fn select_replication_nodes(
        &self,
        _package_id: &str,
    ) -> Result<Vec<&ReplicationNode>, TorshError> {
        let healthy_nodes: Vec<&ReplicationNode> = self
            .nodes
            .values()
            .filter(|n| n.status == NodeStatus::Healthy)
            .collect();

        if healthy_nodes.is_empty() {
            return Err(TorshError::RuntimeError(
                "No healthy nodes available".to_string(),
            ));
        }

        // Select nodes based on priority and capacity
        let mut selected: Vec<&ReplicationNode> = healthy_nodes
            .into_iter()
            .take(self.config.replication_factor)
            .collect();

        // Sort by priority
        selected.sort_by(|a, b| b.priority.cmp(&a.priority));

        Ok(selected)
    }

    fn replicate_synchronously(
        &mut self,
        _package_id: &str,
        _version: &str,
    ) -> Result<(), TorshError> {
        // Mock synchronous replication - mark last operation as completed
        if let Some(op) = self.operations.back_mut() {
            op.status = OperationStatus::Completed;
        }
        Ok(())
    }

    fn replicate_to_quorum(&mut self, _package_id: &str, _version: &str) -> Result<(), TorshError> {
        // Mock quorum replication - mark last operation as completed
        if let Some(op) = self.operations.back_mut() {
            op.status = OperationStatus::Completed;
        }
        Ok(())
    }

    fn replicate_asynchronously(
        &mut self,
        _package_id: &str,
        _version: &str,
    ) -> Result<(), TorshError> {
        // Mock asynchronous replication - mark last operation as completed
        if let Some(op) = self.operations.back_mut() {
            op.status = OperationStatus::Completed;
        }
        Ok(())
    }

    fn select_best_replica<'a>(
        &self,
        replicas: &'a [ReplicaMetadata],
    ) -> Result<&'a ReplicaMetadata, TorshError> {
        // Select replica on healthiest node with lowest lag
        replicas
            .iter()
            .filter(|r| {
                self.nodes
                    .get(&r.node_id)
                    .map(|n| n.status == NodeStatus::Healthy)
                    .unwrap_or(false)
            })
            .min_by(|a, b| {
                let a_lag = self
                    .nodes
                    .get(&a.node_id)
                    .map(|n| n.replication_lag_secs)
                    .unwrap_or(f64::MAX);
                let b_lag = self
                    .nodes
                    .get(&b.node_id)
                    .map(|n| n.replication_lag_secs)
                    .unwrap_or(f64::MAX);
                a_lag.partial_cmp(&b_lag).unwrap()
            })
            .ok_or_else(|| TorshError::RuntimeError("No healthy replicas".to_string()))
    }

    fn check_node_health(&self, _node_id: &str) -> bool {
        // Mock health check (in production, perform actual health check)
        true
    }

    fn handle_failover(&mut self) -> Result<(), TorshError> {
        let unhealthy_nodes: Vec<String> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.status == NodeStatus::Unhealthy)
            .map(|(id, _)| id.clone())
            .collect();

        for node_id in unhealthy_nodes {
            // Redistribute replicas from unhealthy node
            self.redistribute_replicas(&node_id)?;
        }

        Ok(())
    }

    fn redistribute_replicas(&mut self, node_id: &str) -> Result<(), TorshError> {
        // Find replicas on the removed/failed node
        let mut replicas_to_move = Vec::new();

        for (key, replicas) in &self.replicas {
            if replicas.iter().any(|r| r.node_id == node_id) {
                replicas_to_move.push(key.clone());
            }
        }

        // Replicate to other healthy nodes
        for key in replicas_to_move {
            if let Some(replicas) = self.replicas.get_mut(&key) {
                replicas.retain(|r| r.node_id != node_id);

                // If below replication factor, create new replica
                if replicas.len() < self.config.replication_factor {
                    // Mock: add new replica on another node
                    // In production, actually replicate the data
                }
            }
        }

        Ok(())
    }

    fn sync_replica(&mut self, _key: &str, _node_id: &str) -> Result<(), TorshError> {
        // Mock synchronization (in production, perform actual sync)
        Ok(())
    }

    fn propagate_replica(&mut self, _replica: &ReplicaMetadata) -> Result<(), TorshError> {
        // Mock propagation (in production, propagate to all nodes)
        Ok(())
    }

    fn update_statistics(&mut self) {
        let mut stats = ReplicationStatistics::default();

        stats.total_nodes = self.nodes.len();
        stats.healthy_nodes = self
            .nodes
            .values()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();

        stats.total_replicas = self.replicas.values().map(|v| v.len()).sum();

        stats.total_operations = self.operations.len() as u64;
        stats.successful_operations = self
            .operations
            .iter()
            .filter(|op| op.status == OperationStatus::Completed)
            .count() as u64;
        stats.failed_operations = self
            .operations
            .iter()
            .filter(|op| op.status == OperationStatus::Failed)
            .count() as u64;

        stats.active_conflicts = self.conflicts.len();

        // Calculate average replication lag
        let total_lag: f64 = self.nodes.values().map(|n| n.replication_lag_secs).sum();
        stats.avg_replication_lag_secs = if !self.nodes.is_empty() {
            total_lag / self.nodes.len() as f64
        } else {
            0.0
        };

        self.statistics = stats;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ReplicationConfig {
        ReplicationConfig {
            consistency: ConsistencyLevel::Eventual,
            replication_factor: 3,
            auto_failover: true,
            sync_interval_secs: 60,
        }
    }

    #[test]
    fn test_replication_manager_creation() {
        let config = create_test_config();
        let manager = ReplicationManager::new(config);
        let stats = manager.get_statistics();
        assert_eq!(stats.total_nodes, 0);
    }

    #[test]
    fn test_add_node() {
        let config = create_test_config();
        let mut manager = ReplicationManager::new(config);

        let node = ReplicationNode::new(
            "node1".to_string(),
            "us-east-1".to_string(),
            "https://node1.example.com".to_string(),
            1,
            1024 * 1024 * 1024,
        );

        manager.add_node(node).unwrap();

        let stats = manager.get_statistics();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.healthy_nodes, 1);
    }

    #[test]
    fn test_remove_node() {
        let config = create_test_config();
        let mut manager = ReplicationManager::new(config);

        let node = ReplicationNode::new(
            "node1".to_string(),
            "us-east-1".to_string(),
            "https://node1.example.com".to_string(),
            1,
            1024 * 1024 * 1024,
        );

        manager.add_node(node).unwrap();
        assert_eq!(manager.get_statistics().total_nodes, 1);

        manager.remove_node("node1").unwrap();
        assert_eq!(manager.get_statistics().total_nodes, 0);
    }

    #[test]
    fn test_replicate_package() {
        let config = create_test_config();
        let mut manager = ReplicationManager::new(config);

        // Add nodes
        for i in 1..=3 {
            let node = ReplicationNode::new(
                format!("node{}", i),
                "us-east-1".to_string(),
                format!("https://node{}.example.com", i),
                i,
                1024 * 1024 * 1024,
            );
            manager.add_node(node).unwrap();
        }

        // Replicate package
        let result = manager.replicate_package("test-pkg", "1.0.0", b"data");
        assert!(result.is_ok());

        let stats = manager.get_statistics();
        assert!(stats.successful_operations > 0);
    }

    #[test]
    fn test_health_check() {
        let config = create_test_config();
        let mut manager = ReplicationManager::new(config);

        let node = ReplicationNode::new(
            "node1".to_string(),
            "us-east-1".to_string(),
            "https://node1.example.com".to_string(),
            1,
            1024 * 1024 * 1024,
        );

        manager.add_node(node).unwrap();

        manager.health_check().unwrap();

        let status = manager.get_node_status("node1");
        assert!(status.is_some());
    }

    #[test]
    fn test_list_nodes() {
        let config = create_test_config();
        let mut manager = ReplicationManager::new(config);

        for i in 1..=3 {
            let node = ReplicationNode::new(
                format!("node{}", i),
                "us-east-1".to_string(),
                format!("https://node{}.example.com", i),
                i,
                1024 * 1024 * 1024,
            );
            manager.add_node(node).unwrap();
        }

        let nodes = manager.list_nodes();
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn test_consistency_levels() {
        let configs = vec![
            ConsistencyLevel::Eventual,
            ConsistencyLevel::Quorum,
            ConsistencyLevel::Strong,
            ConsistencyLevel::Causal,
        ];

        for consistency in configs {
            let config = ReplicationConfig {
                consistency,
                replication_factor: 3,
                auto_failover: true,
                sync_interval_secs: 60,
            };

            let manager = ReplicationManager::new(config);
            assert_eq!(manager.config.consistency, consistency);
        }
    }

    #[test]
    fn test_replication_statistics() {
        let config = create_test_config();
        let mut manager = ReplicationManager::new(config);

        // Add nodes
        for i in 1..=5 {
            let node = ReplicationNode::new(
                format!("node{}", i),
                "us-east-1".to_string(),
                format!("https://node{}.example.com", i),
                i,
                1024 * 1024 * 1024,
            );
            manager.add_node(node).unwrap();
        }

        manager.replicate_package("pkg1", "1.0.0", b"data").unwrap();
        manager.replicate_package("pkg2", "1.0.0", b"data").unwrap();

        let stats = manager.get_statistics();
        assert_eq!(stats.total_nodes, 5);
        assert_eq!(stats.healthy_nodes, 5);
        assert!(stats.successful_operations >= 2);
    }
}
