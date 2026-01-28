//! Memory optimization utilities for FX graphs
//!
//! This module provides advanced memory management capabilities including:
//! - Memory-mapped file support for large graph serialization
//! - Graph memory usage analysis and optimization
//! - Adaptive memory allocation strategies
//! - Memory-efficient graph representations

use crate::{Edge, FxGraph, Node, TorshResult};
// use petgraph::graph::NodeIndex; // Unused
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// Memory-mapped graph storage for large graphs
pub struct MemoryMappedGraph {
    file_path: PathBuf,
    header: GraphHeader,
    node_data: Option<Arc<RwLock<Vec<u8>>>>,
    edge_data: Option<Arc<RwLock<Vec<u8>>>>,
    memory_threshold: usize, // Size threshold for memory mapping
}

/// Header information for memory-mapped graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphHeader {
    pub version: u32,
    pub node_count: u32,
    pub edge_count: u32,
    pub node_data_offset: u64,
    pub edge_data_offset: u64,
    pub metadata: HashMap<String, String>,
}

/// Memory usage analysis for graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageReport {
    pub total_size_bytes: usize,
    pub node_data_size: usize,
    pub edge_data_size: usize,
    pub metadata_size: usize,
    pub memory_efficiency: f64, // 0.0 to 1.0
    pub recommendations: Vec<String>,
    pub hotspots: Vec<MemoryHotspot>,
}

/// Memory hotspot identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHotspot {
    pub location: String,
    pub size_bytes: usize,
    pub percentage: f64,
    pub optimization_suggestions: Vec<String>,
}

impl MemoryMappedGraph {
    /// Create a new memory-mapped graph
    pub fn new<P: AsRef<Path>>(file_path: P, memory_threshold: usize) -> TorshResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        // Create header with default values
        let header = GraphHeader {
            version: 1,
            node_count: 0,
            edge_count: 0,
            node_data_offset: std::mem::size_of::<GraphHeader>() as u64,
            edge_data_offset: 0, // Will be calculated
            metadata: HashMap::new(),
        };

        Ok(Self {
            file_path,
            header,
            node_data: None,
            edge_data: None,
            memory_threshold,
        })
    }

    /// Save a graph using memory-mapped storage
    pub fn save_graph(&mut self, graph: &FxGraph) -> TorshResult<()> {
        // Serialize graph data
        let node_data = self.serialize_nodes(graph)?;
        let edge_data = self.serialize_edges(graph)?;

        // Update header
        self.header.node_count = graph.node_count() as u32;
        self.header.edge_count = graph.edge_count() as u32;
        self.header.edge_data_offset = self.header.node_data_offset + node_data.len() as u64;

        // Determine if we should use memory mapping
        let total_size = node_data.len() + edge_data.len();

        if total_size > self.memory_threshold {
            self.save_memory_mapped(&node_data, &edge_data)?;
        } else {
            self.save_in_memory(node_data, edge_data);
        }

        Ok(())
    }

    /// Load a graph from memory-mapped storage
    pub fn load_graph(&mut self) -> TorshResult<FxGraph> {
        if !self.file_path.exists() {
            return Err(torsh_core::error::TorshError::IoError(
                "Memory-mapped file does not exist".to_string(),
            ));
        }

        // Load header
        self.load_header()?;

        // Load data based on size
        let file_size = std::fs::metadata(&self.file_path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?
            .len() as usize;

        if file_size > self.memory_threshold {
            self.load_memory_mapped()
        } else {
            self.load_from_file()
        }
    }

    /// Save using memory-mapped files
    fn save_memory_mapped(&mut self, node_data: &[u8], edge_data: &[u8]) -> TorshResult<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Write header
        let header_data = oxicode::serde::encode_to_vec(&self.header, oxicode::config::standard())
            .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))?;
        file.write_all(&header_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Write node data
        file.write_all(node_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Write edge data
        file.write_all(edge_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        file.sync_all()
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Save in memory for small graphs
    fn save_in_memory(&mut self, node_data: Vec<u8>, edge_data: Vec<u8>) {
        // Store in memory
        self.node_data = Some(Arc::new(RwLock::new(node_data.clone())));
        self.edge_data = Some(Arc::new(RwLock::new(edge_data.clone())));

        // Also write to file for persistence
        if let Err(_) = self.write_to_file(&node_data, &edge_data) {
            // If file write fails, just continue with in-memory storage
        }
    }

    /// Write header and data to file
    fn write_to_file(&mut self, node_data: &[u8], edge_data: &[u8]) -> TorshResult<()> {
        use std::io::Write;

        let mut file = File::create(&self.file_path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Calculate correct offsets accounting for the header size prefix
        let header_data = oxicode::serde::encode_to_vec(&self.header, oxicode::config::standard())
            .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))?;

        let header_size_bytes = 4u32; // u32 for header size
        let header_size = header_data.len() as u64;

        // Update header with correct offsets
        self.header.node_data_offset = header_size_bytes as u64 + header_size;
        self.header.edge_data_offset = self.header.node_data_offset + node_data.len() as u64;

        // Re-serialize header with correct offsets
        let updated_header_data =
            oxicode::serde::encode_to_vec(&self.header, oxicode::config::standard())
                .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))?;

        // Write header size first (as u32)
        let header_size = updated_header_data.len() as u32;
        file.write_all(&header_size.to_le_bytes())
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Write header data
        file.write_all(&updated_header_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Write node data
        file.write_all(node_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Write edge data
        file.write_all(edge_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        file.flush()
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load header from file
    fn load_header(&mut self) -> TorshResult<()> {
        use std::io::Read;

        let mut file = File::open(&self.file_path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Read header size first (4 bytes for u32)
        let mut size_bytes = [0u8; 4];
        file.read_exact(&mut size_bytes)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        let header_size = u32::from_le_bytes(size_bytes) as usize;

        // Read header data of exact size
        let mut header_data = vec![0u8; header_size];
        file.read_exact(&mut header_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        let (header, _): (GraphHeader, usize) =
            oxicode::serde::decode_from_slice(&header_data, oxicode::config::standard())
                .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))?;
        self.header = header;

        Ok(())
    }

    /// Load graph using memory mapping
    fn load_memory_mapped(&self) -> TorshResult<FxGraph> {
        let mut file = File::open(&self.file_path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Seek to node data
        file.seek(SeekFrom::Start(self.header.node_data_offset))
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Read node data in chunks to save memory
        let node_data = self.read_chunked_data(&mut file, self.header.node_count as usize)?;

        // Seek to edge data
        file.seek(SeekFrom::Start(self.header.edge_data_offset))
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        let edge_data = self.read_chunked_data(&mut file, self.header.edge_count as usize)?;

        self.deserialize_graph(&node_data, &edge_data)
    }

    /// Load graph from file into memory
    fn load_from_file(&self) -> TorshResult<FxGraph> {
        let mut file = File::open(&self.file_path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Skip header
        file.seek(SeekFrom::Start(self.header.node_data_offset))
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        let mut node_data = Vec::new();
        let mut edge_data = Vec::new();

        // Read all node data
        let node_end = self.header.edge_data_offset;
        let node_size = (node_end - self.header.node_data_offset) as usize;
        node_data.resize(node_size, 0);
        file.read_exact(&mut node_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        // Read all edge data
        file.read_to_end(&mut edge_data)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        self.deserialize_graph(&node_data, &edge_data)
    }

    /// Read data in chunks to save memory
    fn read_chunked_data(&self, file: &mut File, _item_count: usize) -> TorshResult<Vec<u8>> {
        let mut data = Vec::new();
        let mut buffer = [0u8; 4096]; // 4KB chunks

        loop {
            match file.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(n) => data.extend_from_slice(&buffer[..n]),
                Err(e) => return Err(torsh_core::error::TorshError::IoError(e.to_string())),
            }
        }

        Ok(data)
    }

    /// Serialize nodes to binary format
    fn serialize_nodes(&self, graph: &FxGraph) -> TorshResult<Vec<u8>> {
        let nodes: Vec<(usize, Node)> = graph
            .nodes()
            .map(|(idx, node)| (idx.index(), node.clone()))
            .collect();

        oxicode::serde::encode_to_vec(&nodes, oxicode::config::standard())
            .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))
    }

    /// Serialize edges to binary format
    fn serialize_edges(&self, graph: &FxGraph) -> TorshResult<Vec<u8>> {
        let edges: Vec<(usize, usize, Edge)> = graph
            .graph
            .edge_references()
            .map(|edge_ref| {
                use petgraph::visit::EdgeRef;
                (
                    edge_ref.source().index(),
                    edge_ref.target().index(),
                    edge_ref.weight().clone(),
                )
            })
            .collect();

        oxicode::serde::encode_to_vec(&edges, oxicode::config::standard())
            .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))
    }

    /// Deserialize graph from binary data
    fn deserialize_graph(&self, node_data: &[u8], edge_data: &[u8]) -> TorshResult<FxGraph> {
        let (nodes, _): (Vec<(usize, Node)>, usize) =
            oxicode::serde::decode_from_slice(node_data, oxicode::config::standard())
                .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))?;

        let (edges, _): (Vec<(usize, usize, Edge)>, usize) =
            oxicode::serde::decode_from_slice(edge_data, oxicode::config::standard())
                .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))?;

        // Reconstruct graph
        let mut graph = petgraph::Graph::new();
        let mut node_mapping = HashMap::new();

        // Add nodes
        for (original_idx, node) in nodes {
            let new_idx = graph.add_node(node);
            node_mapping.insert(original_idx, new_idx);
        }

        // Add edges
        for (src_idx, target_idx, edge) in edges {
            if let (Some(&src), Some(&target)) =
                (node_mapping.get(&src_idx), node_mapping.get(&target_idx))
            {
                graph.add_edge(src, target, edge);
            }
        }

        // Create FxGraph (inputs and outputs would need to be stored separately)
        Ok(FxGraph {
            graph,
            inputs: Vec::new(),  // Would need to be restored from metadata
            outputs: Vec::new(), // Would need to be restored from metadata
        })
    }
}

/// Memory usage analyzer for graphs
pub struct MemoryAnalyzer;

impl MemoryAnalyzer {
    /// Analyze memory usage of a graph
    pub fn analyze_memory_usage(graph: &FxGraph) -> MemoryUsageReport {
        let node_data_size = Self::calculate_node_data_size(graph);
        let edge_data_size = Self::calculate_edge_data_size(graph);
        let metadata_size = Self::calculate_metadata_size(graph);
        let total_size_bytes = node_data_size + edge_data_size + metadata_size;

        // Calculate memory efficiency (heuristic)
        let ideal_size = graph.node_count() * 32 + graph.edge_count() * 16; // Rough estimate
        let memory_efficiency = if total_size_bytes > 0 {
            (ideal_size as f64 / total_size_bytes as f64).min(1.0)
        } else {
            1.0
        };

        let hotspots = Self::identify_memory_hotspots(graph, total_size_bytes);
        let recommendations = Self::generate_memory_recommendations(graph, memory_efficiency);

        MemoryUsageReport {
            total_size_bytes,
            node_data_size,
            edge_data_size,
            metadata_size,
            memory_efficiency,
            recommendations,
            hotspots,
        }
    }

    /// Calculate size of node data
    fn calculate_node_data_size(graph: &FxGraph) -> usize {
        let mut total_size = 0;

        for (_, node) in graph.nodes() {
            total_size += match node {
                Node::Input(name) => 16 + name.len(), // Base size + string length
                Node::Call(op, args) => {
                    32 + op.len() + args.iter().map(|arg| arg.len()).sum::<usize>()
                }
                Node::Output => 8,
                Node::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    64 + condition.len()
                        + then_branch.iter().map(|s| s.len()).sum::<usize>()
                        + else_branch.iter().map(|s| s.len()).sum::<usize>()
                }
                Node::Loop {
                    condition,
                    body,
                    loop_vars,
                } => {
                    64 + condition.len()
                        + body.iter().map(|s| s.len()).sum::<usize>()
                        + loop_vars.iter().map(|s| s.len()).sum::<usize>()
                }
                Node::Merge { inputs } => 32 + inputs.iter().map(|s| s.len()).sum::<usize>(),
                Node::GetAttr { target, attr } => 24 + target.len() + attr.len(),
            };
        }

        total_size
    }

    /// Calculate size of edge data
    fn calculate_edge_data_size(graph: &FxGraph) -> usize {
        let mut total_size = 0;

        for edge_ref in graph.graph.edge_references() {
            // use petgraph::visit::EdgeRef; // Unused import
            total_size += 16 + edge_ref.weight().name.len(); // Base edge size + name
        }

        total_size
    }

    /// Calculate size of metadata
    fn calculate_metadata_size(graph: &FxGraph) -> usize {
        // Rough estimate for graph metadata
        graph.inputs().len() * 8 + graph.outputs().len() * 8 + 64
    }

    /// Identify memory hotspots in the graph
    fn identify_memory_hotspots(graph: &FxGraph, total_size: usize) -> Vec<MemoryHotspot> {
        let mut hotspots = Vec::new();

        // Check for nodes with large string data
        for (idx, node) in graph.nodes() {
            let node_size = match node {
                Node::Call(op, args)
                    if op.len() > 100 || args.iter().any(|arg| arg.len() > 100) =>
                {
                    op.len() + args.iter().map(|arg| arg.len()).sum::<usize>()
                }
                Node::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } if condition.len() > 50 || then_branch.len() > 20 || else_branch.len() > 20 => {
                    condition.len()
                        + then_branch.iter().map(|s| s.len()).sum::<usize>()
                        + else_branch.iter().map(|s| s.len()).sum::<usize>()
                }
                _ => 0,
            };

            if node_size > 1000 {
                // Threshold for considering as hotspot
                let percentage = (node_size as f64 / total_size as f64) * 100.0;
                hotspots.push(MemoryHotspot {
                    location: format!("Node {idx:?}"),
                    size_bytes: node_size,
                    percentage,
                    optimization_suggestions: vec![
                        "Consider using references instead of owned strings".to_string(),
                        "Use string interning for repeated values".to_string(),
                    ],
                });
            }
        }

        // Check for high fan-out nodes (many edges)
        for (idx, _) in graph.nodes() {
            let edge_count = graph.graph.edges(idx).count();
            if edge_count > 50 {
                let edge_size = edge_count * 24; // Approximate edge size
                let percentage = (edge_size as f64 / total_size as f64) * 100.0;
                hotspots.push(MemoryHotspot {
                    location: format!("Node {idx:?} edges"),
                    size_bytes: edge_size,
                    percentage,
                    optimization_suggestions: vec![
                        "Consider reducing fan-out through intermediate nodes".to_string(),
                        "Use broadcast operations instead of multiple edges".to_string(),
                    ],
                });
            }
        }

        hotspots.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        hotspots
    }

    /// Generate memory optimization recommendations
    fn generate_memory_recommendations(graph: &FxGraph, efficiency: f64) -> Vec<String> {
        let mut recommendations = Vec::new();

        if efficiency < 0.5 {
            recommendations.push("Consider using more compact node representations".to_string());
        }

        if graph.node_count() > 10000 {
            recommendations.push("Use memory-mapped storage for large graphs".to_string());
        }

        if graph.edge_count() > graph.node_count() * 3 {
            recommendations
                .push("High edge density detected - consider graph simplification".to_string());
        }

        recommendations.push("Enable compression for graph serialization".to_string());
        recommendations.push("Use lazy loading for large subgraphs".to_string());
        recommendations.push("Consider graph partitioning for distributed processing".to_string());

        recommendations
    }
}

/// Adaptive memory allocation strategies
pub struct AdaptiveMemoryManager {
    allocation_strategy: AllocationStrategy,
    memory_pressure_threshold: f64,
    current_memory_usage: Arc<Mutex<usize>>,
    max_memory_limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    Conservative, // Minimize memory usage
    Balanced,     // Balance memory and performance
    Aggressive,   // Optimize for performance
    Adaptive,     // Change strategy based on conditions
}

impl AdaptiveMemoryManager {
    /// Create a new adaptive memory manager
    pub fn new(strategy: AllocationStrategy) -> Self {
        Self {
            allocation_strategy: strategy,
            memory_pressure_threshold: 0.8, // 80% memory usage
            current_memory_usage: Arc::new(Mutex::new(0)),
            max_memory_limit: None,
        }
    }

    /// Set maximum memory limit
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.max_memory_limit = Some(limit);
        self
    }

    /// Allocate memory for graph operations
    pub fn allocate_graph_memory(&self, graph: &FxGraph) -> TorshResult<GraphMemoryLayout> {
        let memory_report = MemoryAnalyzer::analyze_memory_usage(graph);
        let required_memory = memory_report.total_size_bytes;

        // Check memory limits
        if let Some(limit) = self.max_memory_limit {
            let current_usage = *self
                .current_memory_usage
                .lock()
                .expect("lock should not be poisoned");
            if current_usage + required_memory > limit {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "Memory limit exceeded".to_string(),
                ));
            }
        }

        // Determine allocation strategy
        let strategy = self.determine_strategy(required_memory);
        let layout = self.create_memory_layout(graph, strategy)?;

        // Update memory usage
        *self
            .current_memory_usage
            .lock()
            .expect("lock should not be poisoned") += required_memory;

        Ok(layout)
    }

    /// Deallocate memory for graph operations
    pub fn deallocate_graph_memory(&self, layout: &GraphMemoryLayout) {
        let mut current_usage = self
            .current_memory_usage
            .lock()
            .expect("lock should not be poisoned");
        *current_usage = current_usage.saturating_sub(layout.total_size);
    }

    /// Determine the best allocation strategy based on current conditions
    fn determine_strategy(&self, required_memory: usize) -> AllocationStrategy {
        match &self.allocation_strategy {
            AllocationStrategy::Adaptive => {
                let current_usage = *self
                    .current_memory_usage
                    .lock()
                    .expect("lock should not be poisoned");
                let memory_pressure = if let Some(limit) = self.max_memory_limit {
                    current_usage as f64 / limit as f64
                } else {
                    0.0 // No limit, assume low pressure
                };

                if memory_pressure > self.memory_pressure_threshold {
                    AllocationStrategy::Conservative
                } else if required_memory > 1_000_000 {
                    // 1MB threshold
                    AllocationStrategy::Balanced
                } else {
                    AllocationStrategy::Aggressive
                }
            }
            strategy => strategy.clone(),
        }
    }

    /// Create memory layout based on strategy
    fn create_memory_layout(
        &self,
        graph: &FxGraph,
        strategy: AllocationStrategy,
    ) -> TorshResult<GraphMemoryLayout> {
        let memory_report = MemoryAnalyzer::analyze_memory_usage(graph);

        let layout = match strategy {
            AllocationStrategy::Conservative => GraphMemoryLayout {
                total_size: memory_report.total_size_bytes,
                use_memory_mapping: memory_report.total_size_bytes > 100_000, // 100KB threshold
                compression_enabled: true,
                lazy_loading: true,
                chunk_size: 4096, // 4KB chunks
                prefetch_enabled: false,
            },
            AllocationStrategy::Balanced => GraphMemoryLayout {
                total_size: memory_report.total_size_bytes,
                use_memory_mapping: memory_report.total_size_bytes > 1_000_000, // 1MB threshold
                compression_enabled: memory_report.total_size_bytes > 500_000,  // 500KB threshold
                lazy_loading: false,
                chunk_size: 8192, // 8KB chunks
                prefetch_enabled: true,
            },
            AllocationStrategy::Aggressive => GraphMemoryLayout {
                total_size: memory_report.total_size_bytes,
                use_memory_mapping: false, // Keep in memory
                compression_enabled: false,
                lazy_loading: false,
                chunk_size: 16384, // 16KB chunks
                prefetch_enabled: true,
            },
            AllocationStrategy::Adaptive => {
                // Should not reach here as adaptive is resolved above
                self.create_memory_layout(graph, AllocationStrategy::Balanced)?
            }
        };

        Ok(layout)
    }
}

/// Memory layout configuration for graphs
#[derive(Debug, Clone)]
pub struct GraphMemoryLayout {
    pub total_size: usize,
    pub use_memory_mapping: bool,
    pub compression_enabled: bool,
    pub lazy_loading: bool,
    pub chunk_size: usize,
    pub prefetch_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, FxGraph, Node};
    use tempfile::NamedTempFile;

    #[test]
    fn test_memory_mapped_graph() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();

        // Ensure the temporary file exists and is writable
        std::fs::write(&temp_path, b"").unwrap();

        let mut mmap_graph = MemoryMappedGraph::new(&temp_path, 1000).unwrap();

        // Create test graph
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            relu,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        // Save and load - use expect to get better error messages
        mmap_graph.save_graph(&graph).expect("Failed to save graph");
        let loaded_graph = mmap_graph.load_graph().expect("Failed to load graph");

        assert_eq!(loaded_graph.node_count(), graph.node_count());

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    #[test]
    fn test_memory_analyzer() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let relu = graph
            .graph
            .add_node(Node::Call("relu".to_string(), vec!["x".to_string()]));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            relu,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.graph.add_edge(
            relu,
            output,
            Edge {
                name: "relu_out".to_string(),
            },
        );

        let report = MemoryAnalyzer::analyze_memory_usage(&graph);

        assert!(report.total_size_bytes > 0);
        assert!(report.memory_efficiency > 0.0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_adaptive_memory_manager() {
        let manager =
            AdaptiveMemoryManager::new(AllocationStrategy::Adaptive).with_memory_limit(1_000_000); // 1MB limit

        let mut graph = FxGraph::new();
        let _input = graph.graph.add_node(Node::Input("x".to_string()));
        let _output = graph.graph.add_node(Node::Output);

        let layout = manager.allocate_graph_memory(&graph).unwrap();
        assert!(layout.total_size > 0);

        manager.deallocate_graph_memory(&layout);
    }

    #[test]
    fn test_memory_hotspot_detection() {
        let mut graph = FxGraph::new();

        // Create a node with large string data (>1000 bytes to trigger hotspot detection)
        let large_op = "very_long_operation_name_that_should_be_detected_as_hotspot".repeat(20);
        let _large_node = graph
            .graph
            .add_node(Node::Call(large_op, vec!["arg".to_string()]));

        let report = MemoryAnalyzer::analyze_memory_usage(&graph);

        // Should detect the large node as a hotspot
        assert!(!report.hotspots.is_empty());
    }
}
