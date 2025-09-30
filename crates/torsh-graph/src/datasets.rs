//! Graph dataset loaders with support for popular formats
//!
//! Implementation of comprehensive graph dataset loading capabilities
//! as specified in TODO.md, including GraphML, GML, and other formats.

use crate::{conv::HeteroGraphData, GraphData};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result as IoResult};
use std::path::Path;
use torsh_core::device::DeviceType;
use torsh_tensor::{creation::from_vec, Tensor};

/// Graph dataset loader trait
pub trait GraphDatasetLoader {
    /// Load graph from file
    fn load_from_file<P: AsRef<Path>>(&self, path: P) -> IoResult<GraphData>;

    /// Load multiple graphs from directory
    fn load_from_directory<P: AsRef<Path>>(&self, path: P) -> IoResult<Vec<GraphData>>;

    /// Get supported file extensions
    fn supported_extensions(&self) -> Vec<&'static str>;
}

/// Edge list format loader (simple format: src dst)
pub struct EdgeListLoader {
    /// Number of node features (will be filled with random values)
    pub num_features: usize,
    /// Whether edges are directed
    pub directed: bool,
    /// Delimiter for parsing
    pub delimiter: char,
}

impl EdgeListLoader {
    pub fn new(num_features: usize, directed: bool) -> Self {
        Self {
            num_features,
            directed,
            delimiter: ' ',
        }
    }

    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = delimiter;
        self
    }
}

impl GraphDatasetLoader for EdgeListLoader {
    fn load_from_file<P: AsRef<Path>>(&self, path: P) -> IoResult<GraphData> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut edges = Vec::new();
        let mut max_node_id = 0usize;

        // Read edges
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(self.delimiter).collect();
            if parts.len() >= 2 {
                if let (Ok(src), Ok(dst)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    edges.extend_from_slice(&[src, dst]);
                    max_node_id = max_node_id.max(src).max(dst);

                    // Add reverse edge for undirected graphs
                    if !self.directed && src != dst {
                        edges.extend_from_slice(&[dst, src]);
                    }
                }
            }
        }

        let num_nodes = max_node_id + 1;
        let num_edges = edges.len() / 2;

        // Create node features (random for now)
        let features = (0..num_nodes * self.num_features)
            .map(|i| (i as f32 * 0.1) % 1.0)
            .collect::<Vec<f32>>();
        let x =
            from_vec(features, &[num_nodes, self.num_features], DeviceType::Cpu).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Tensor creation failed: {:?}", e),
                )
            })?;

        // Create edge index
        let edge_index = from_vec(
            edges.into_iter().map(|e| e as i64).collect(),
            &[2, num_edges],
            DeviceType::Cpu,
        )
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Edge tensor creation failed: {:?}", e),
            )
        })?;

        Ok(GraphData::new(x, edge_index.to_f32_simd().unwrap()))
    }

    fn load_from_directory<P: AsRef<Path>>(&self, path: P) -> IoResult<Vec<GraphData>> {
        let mut graphs = Vec::new();
        let dir = std::fs::read_dir(path)?;

        for entry in dir {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if self
                    .supported_extensions()
                    .contains(&ext.to_str().unwrap_or(""))
                {
                    match self.load_from_file(&path) {
                        Ok(graph) => graphs.push(graph),
                        Err(e) => eprintln!("Warning: Failed to load {}: {}", path.display(), e),
                    }
                }
            }
        }

        Ok(graphs)
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["edges", "edgelist", "txt"]
    }
}

/// GML (Graph Modelling Language) format loader
pub struct GMLLoader;

impl GMLLoader {
    pub fn new() -> Self {
        Self
    }
}

impl GraphDatasetLoader for GMLLoader {
    fn load_from_file<P: AsRef<Path>>(&self, path: P) -> IoResult<GraphData> {
        let content = std::fs::read_to_string(path)?;
        self.parse_gml(&content)
    }

    fn load_from_directory<P: AsRef<Path>>(&self, path: P) -> IoResult<Vec<GraphData>> {
        let mut graphs = Vec::new();
        let dir = std::fs::read_dir(path)?;

        for entry in dir {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if self
                    .supported_extensions()
                    .contains(&ext.to_str().unwrap_or(""))
                {
                    match self.load_from_file(&path) {
                        Ok(graph) => graphs.push(graph),
                        Err(e) => {
                            eprintln!("Warning: Failed to load GML {}: {}", path.display(), e)
                        }
                    }
                }
            }
        }

        Ok(graphs)
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["gml"]
    }
}

impl GMLLoader {
    fn parse_gml(&self, content: &str) -> IoResult<GraphData> {
        let mut nodes = HashMap::new();
        let mut edges = Vec::new();
        let mut in_node = false;
        let mut in_edge = false;
        let mut current_node_id: Option<usize> = None;
        let mut current_edge_src: Option<usize> = None;
        let mut current_edge_dst: Option<usize> = None;

        for line in content.lines() {
            let line = line.trim();

            if line.starts_with("node") {
                in_node = true;
                in_edge = false;
            } else if line.starts_with("edge") {
                in_edge = true;
                in_node = false;
            } else if line == "]" {
                // End of node or edge block
                if in_node {
                    if let Some(id) = current_node_id {
                        nodes.insert(id, vec![1.0; 4]); // Default features
                    }
                    current_node_id = None;
                    in_node = false;
                } else if in_edge {
                    if let (Some(src), Some(dst)) = (current_edge_src, current_edge_dst) {
                        edges.extend_from_slice(&[src, dst]);
                    }
                    current_edge_src = None;
                    current_edge_dst = None;
                    in_edge = false;
                }
            } else if in_node && line.starts_with("id") {
                if let Some(id_str) = line.split_whitespace().nth(1) {
                    current_node_id = id_str.parse().ok();
                }
            } else if in_edge && line.starts_with("source") {
                if let Some(src_str) = line.split_whitespace().nth(1) {
                    current_edge_src = src_str.parse().ok();
                }
            } else if in_edge && line.starts_with("target") {
                if let Some(dst_str) = line.split_whitespace().nth(1) {
                    current_edge_dst = dst_str.parse().ok();
                }
            }
        }

        // Convert to tensors
        let num_nodes = nodes.len();
        let num_edges = edges.len() / 2;

        if num_nodes == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No nodes found in GML file",
            ));
        }

        // Create ordered node features
        let mut node_features = Vec::new();
        let mut node_mapping = HashMap::new();
        let mut new_id = 0;

        let mut sorted_nodes: Vec<_> = nodes.keys().collect();
        sorted_nodes.sort();

        for &original_id in sorted_nodes {
            node_mapping.insert(original_id, new_id);
            node_features.extend_from_slice(&nodes[&original_id]);
            new_id += 1;
        }

        // Remap edges
        let remapped_edges: Vec<i64> = edges
            .iter()
            .map(|&id| *node_mapping.get(&id).unwrap_or(&0) as i64)
            .collect();

        let x = from_vec(node_features, &[num_nodes, 4], DeviceType::Cpu).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Node features tensor creation failed: {:?}", e),
            )
        })?;

        let edge_index: Tensor<i64> = from_vec(remapped_edges, &[2, num_edges], DeviceType::Cpu)
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Edge index tensor creation failed: {:?}", e),
                )
            })?;

        Ok(GraphData::new(x, edge_index.to_f32_simd().unwrap()))
    }
}

/// JSON format loader for graph data
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonGraphData {
    pub nodes: Vec<JsonNode>,
    pub edges: Vec<JsonEdge>,
    pub directed: Option<bool>,
    pub multigraph: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonNode {
    pub id: usize,
    pub features: Option<Vec<f32>>,
    pub label: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JsonEdge {
    pub source: usize,
    pub target: usize,
    pub weight: Option<f32>,
    pub label: Option<String>,
}

pub struct JSONLoader {
    pub default_features: usize,
}

impl JSONLoader {
    pub fn new(default_features: usize) -> Self {
        Self { default_features }
    }
}

impl GraphDatasetLoader for JSONLoader {
    fn load_from_file<P: AsRef<Path>>(&self, path: P) -> IoResult<GraphData> {
        let content = std::fs::read_to_string(path)?;
        let data: JsonGraphData = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.convert_json_to_graph(data)
    }

    fn load_from_directory<P: AsRef<Path>>(&self, path: P) -> IoResult<Vec<GraphData>> {
        let mut graphs = Vec::new();
        let dir = std::fs::read_dir(path)?;

        for entry in dir {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if self
                    .supported_extensions()
                    .contains(&ext.to_str().unwrap_or(""))
                {
                    match self.load_from_file(&path) {
                        Ok(graph) => graphs.push(graph),
                        Err(e) => {
                            eprintln!("Warning: Failed to load JSON {}: {}", path.display(), e)
                        }
                    }
                }
            }
        }

        Ok(graphs)
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["json"]
    }
}

impl JSONLoader {
    fn convert_json_to_graph(&self, data: JsonGraphData) -> IoResult<GraphData> {
        let num_nodes = data.nodes.len();
        let num_edges = data.edges.len();

        if num_nodes == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No nodes in JSON graph data",
            ));
        }

        // Create node ID mapping
        let mut node_mapping = HashMap::new();
        for (i, node) in data.nodes.iter().enumerate() {
            node_mapping.insert(node.id, i);
        }

        // Extract node features
        let mut node_features = Vec::new();
        for node in &data.nodes {
            if let Some(ref features) = node.features {
                node_features.extend_from_slice(features);
            } else {
                // Use default features
                node_features.extend((0..self.default_features).map(|i| i as f32 * 0.1));
            }
        }

        let feature_dim = node_features.len() / num_nodes;

        // Extract edges
        let mut edges = Vec::new();
        for edge in &data.edges {
            if let (Some(&src), Some(&dst)) = (
                node_mapping.get(&edge.source),
                node_mapping.get(&edge.target),
            ) {
                edges.extend_from_slice(&[src as i64, dst as i64]);

                // Add reverse edge for undirected graphs
                if !data.directed.unwrap_or(false) && src != dst {
                    edges.extend_from_slice(&[dst as i64, src as i64]);
                }
            }
        }

        let final_num_edges = edges.len() / 2;

        let x =
            from_vec(node_features, &[num_nodes, feature_dim], DeviceType::Cpu).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Node features failed: {:?}", e),
                )
            })?;

        let edge_index = from_vec(edges, &[2, final_num_edges], DeviceType::Cpu).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Edge index failed: {:?}", e),
            )
        })?;

        Ok(GraphData::new(x, edge_index.to_f32_simd().unwrap()))
    }
}

/// Graph dataset collections for common benchmarks
pub struct GraphDatasetCollection;

impl GraphDatasetCollection {
    /// Create a synthetic dataset for testing
    pub fn create_synthetic_dataset(
        num_graphs: usize,
        nodes_per_graph: usize,
        edge_probability: f64,
        num_features: usize,
    ) -> Vec<GraphData> {
        use crate::scirs2_integration::generation;

        (0..num_graphs)
            .map(|_| {
                let mut graph = generation::erdos_renyi(nodes_per_graph, edge_probability);

                // Ensure correct feature dimension
                if graph.x.shape().dims()[1] != num_features {
                    let new_features = (0..nodes_per_graph * num_features)
                        .map(|i| (i as f32 * 0.01) % 1.0)
                        .collect();
                    let x = from_vec(
                        new_features,
                        &[nodes_per_graph, num_features],
                        DeviceType::Cpu,
                    )
                    .unwrap();
                    graph.x = x;
                }

                graph
            })
            .collect()
    }

    /// Load a collection of graphs with data augmentation
    pub fn load_with_augmentation<L: GraphDatasetLoader>(
        loader: L,
        path: impl AsRef<Path>,
        augmentation_factor: usize,
    ) -> IoResult<Vec<GraphData>> {
        let base_graphs = loader.load_from_directory(path)?;
        let mut augmented_graphs = base_graphs.clone();

        for _ in 1..augmentation_factor {
            for graph in &base_graphs {
                // Simple augmentation: add noise to features
                let augmented = Self::add_feature_noise(graph, 0.1);
                augmented_graphs.push(augmented);
            }
        }

        Ok(augmented_graphs)
    }

    /// Add noise to node features for data augmentation
    pub fn add_feature_noise(graph: &GraphData, noise_level: f32) -> GraphData {
        use scirs2_core::random::Rng;

        let mut rng = scirs2_core::random::thread_rng();
        let features = graph.x.to_vec().unwrap();
        let noisy_features: Vec<f32> = features
            .iter()
            .map(|&x| x + (rng.random::<f32>() - 0.5) * 2.0 * noise_level)
            .collect();

        let noisy_x = from_vec(noisy_features, graph.x.shape().dims(), DeviceType::Cpu).unwrap();

        GraphData {
            x: noisy_x,
            edge_index: graph.edge_index.clone(),
            edge_attr: graph.edge_attr.clone(),
            batch: graph.batch.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
        }
    }

    /// Split dataset into train/validation/test
    pub fn train_val_test_split(
        graphs: Vec<GraphData>,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (Vec<GraphData>, Vec<GraphData>, Vec<GraphData>) {
        let n = graphs.len();
        let train_size = (n as f64 * train_ratio) as usize;
        let val_size = (n as f64 * val_ratio) as usize;

        let mut train_graphs = Vec::new();
        let mut val_graphs = Vec::new();
        let mut test_graphs = Vec::new();

        for (i, graph) in graphs.into_iter().enumerate() {
            if i < train_size {
                train_graphs.push(graph);
            } else if i < train_size + val_size {
                val_graphs.push(graph);
            } else {
                test_graphs.push(graph);
            }
        }

        (train_graphs, val_graphs, test_graphs)
    }
}

/// Graph sampler for batch processing
pub struct GraphSampler {
    batch_size: usize,
    shuffle: bool,
}

impl GraphSampler {
    pub fn new(batch_size: usize, shuffle: bool) -> Self {
        Self {
            batch_size,
            shuffle,
        }
    }

    /// Sample batches from a dataset
    pub fn sample_batches<'a>(&self, graphs: &'a [GraphData]) -> Vec<Vec<&'a GraphData>> {
        let mut indices: Vec<usize> = (0..graphs.len()).collect();

        if self.shuffle {
            use scirs2_core::random::Rng;
            let mut rng = scirs2_core::random::thread_rng();
            for i in (1..indices.len()).rev() {
                let j = (rng.gen::<f64>() * (i + 1) as f64) as usize;
                indices.swap(i, j);
            }
        }

        indices
            .chunks(self.batch_size)
            .map(|chunk| chunk.iter().map(|&i| &graphs[i]).collect())
            .collect()
    }
}

/// Dynamic graph handling for temporal networks
pub struct TemporalGraphLoader {
    pub time_steps: usize,
    pub node_features: usize,
}

impl TemporalGraphLoader {
    pub fn new(time_steps: usize, node_features: usize) -> Self {
        Self {
            time_steps,
            node_features,
        }
    }

    /// Load temporal graph sequence
    pub fn load_temporal_sequence<P: AsRef<Path>>(&self, base_path: P) -> IoResult<Vec<GraphData>> {
        let mut graphs = Vec::new();
        let base_path = base_path.as_ref();

        for t in 0..self.time_steps {
            let file_path = base_path.join(format!("graph_t{}.edges", t));

            if file_path.exists() {
                let loader = EdgeListLoader::new(self.node_features, true);
                match loader.load_from_file(&file_path) {
                    Ok(graph) => graphs.push(graph),
                    Err(e) => {
                        eprintln!("Warning: Failed to load timestep {}: {}", t, e);
                        // Create empty graph as placeholder
                        let x = from_vec(
                            vec![0.0; self.node_features],
                            &[1, self.node_features],
                            DeviceType::Cpu,
                        )
                        .unwrap();
                        let edge_index = from_vec(vec![], &[2, 0], DeviceType::Cpu).unwrap();
                        graphs.push(GraphData::new(x, edge_index));
                    }
                }
            }
        }

        Ok(graphs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_edge_list_loader() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "0 1").unwrap();
        writeln!(temp_file, "1 2").unwrap();
        writeln!(temp_file, "2 0").unwrap();

        let loader = EdgeListLoader::new(3, false);
        let graph = loader.load_from_file(temp_file.path()).unwrap();

        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.num_edges, 6); // Undirected, so doubled
        assert_eq!(graph.x.shape().dims(), &[3, 3]);
    }

    #[test]
    fn test_json_loader() {
        let json_data = r#"{
            "nodes": [
                {"id": 0, "features": [1.0, 2.0]},
                {"id": 1, "features": [3.0, 4.0]}
            ],
            "edges": [
                {"source": 0, "target": 1}
            ],
            "directed": true
        }"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(json_data.as_bytes()).unwrap();

        let loader = JSONLoader::new(2);
        let graph = loader.load_from_file(temp_file.path()).unwrap();

        assert_eq!(graph.num_nodes, 2);
        assert_eq!(graph.num_edges, 1);
        assert_eq!(graph.x.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_graph_dataset_collection() {
        let graphs = GraphDatasetCollection::create_synthetic_dataset(5, 10, 0.2, 4);

        assert_eq!(graphs.len(), 5);
        for graph in graphs {
            assert_eq!(graph.num_nodes, 10);
            assert_eq!(graph.x.shape().dims(), &[10, 4]);
        }
    }

    #[test]
    fn test_train_val_test_split() {
        let graphs = GraphDatasetCollection::create_synthetic_dataset(100, 10, 0.1, 3);
        let (train, val, test) = GraphDatasetCollection::train_val_test_split(graphs, 0.7, 0.2);

        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 20);
        assert_eq!(test.len(), 10);
    }

    #[test]
    fn test_graph_sampler() {
        let graphs = GraphDatasetCollection::create_synthetic_dataset(10, 5, 0.3, 2);
        let sampler = GraphSampler::new(3, false);
        let batches = sampler.sample_batches(&graphs);

        assert_eq!(batches.len(), 4); // 10 graphs with batch_size 3 = 4 batches
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1); // Last batch has remainder
    }

    #[test]
    fn test_feature_noise_augmentation() {
        let base_graph = GraphDatasetCollection::create_synthetic_dataset(1, 5, 0.4, 3)[0].clone();
        let noisy_graph = GraphDatasetCollection::add_feature_noise(&base_graph, 0.1);

        assert_eq!(noisy_graph.num_nodes, base_graph.num_nodes);
        assert_eq!(noisy_graph.num_edges, base_graph.num_edges);
        assert_eq!(noisy_graph.x.shape().dims(), base_graph.x.shape().dims());

        // Features should be different due to noise
        let original_features = base_graph.x.to_vec().unwrap();
        let noisy_features = noisy_graph.x.to_vec().unwrap();

        let mut differences = 0;
        for (orig, noisy) in original_features.iter().zip(noisy_features.iter()) {
            if (orig - noisy).abs() > 1e-6 {
                differences += 1;
            }
        }

        // Should have some differences due to noise
        assert!(differences > 0);
    }
}
