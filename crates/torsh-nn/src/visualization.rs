//! Network architecture visualization tools
//!
//! This module provides tools for visualizing neural network architectures,
//! including graph representations, layer diagrams, and model flow charts.

use crate::Module;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use torsh_core::error::{Result, TorshError};

/// Represents a node in the network graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for the node
    pub id: String,
    /// Display name for the node
    pub name: String,
    /// Type of the layer/module
    pub layer_type: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of parameters
    pub parameter_count: usize,
    /// Position in the graph (x, y)
    pub position: Option<(f32, f32)>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(
        id: String,
        name: String,
        layer_type: String,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        parameter_count: usize,
    ) -> Self {
        Self {
            id,
            name,
            layer_type,
            input_shape,
            output_shape,
            parameter_count,
            position: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the position of the node
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position = Some((x, y));
        self
    }

    /// Add metadata to the node
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get a short description of the node
    pub fn short_description(&self) -> String {
        format!("{} ({})", self.name, self.layer_type)
    }

    /// Get a detailed description of the node
    pub fn detailed_description(&self) -> String {
        format!(
            "{}\nType: {}\nInput: {:?}\nOutput: {:?}\nParams: {}",
            self.name,
            self.layer_type,
            self.input_shape,
            self.output_shape,
            format_number(self.parameter_count)
        )
    }
}

/// Represents an edge connecting two nodes
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Data shape flowing through this edge
    pub shape: Vec<usize>,
    /// Edge weight or importance
    pub weight: f32,
    /// Edge style
    pub style: EdgeStyle,
}

/// Edge styling options
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeStyle {
    /// Regular connection
    Normal,
    /// Skip connection (residual)
    Skip,
    /// Attention connection
    Attention,
    /// Recurrent connection
    Recurrent,
}

impl GraphEdge {
    /// Create a new graph edge
    pub fn new(from: String, to: String, shape: Vec<usize>) -> Self {
        Self {
            from,
            to,
            shape,
            weight: 1.0,
            style: EdgeStyle::Normal,
        }
    }

    /// Set the edge style
    pub fn with_style(mut self, style: EdgeStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the edge weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

/// Complete network graph representation
#[derive(Debug, Clone)]
pub struct NetworkGraph {
    /// All nodes in the graph
    pub nodes: HashMap<String, GraphNode>,
    /// All edges in the graph
    pub edges: Vec<GraphEdge>,
    /// Input node IDs
    pub inputs: Vec<String>,
    /// Output node IDs
    pub outputs: Vec<String>,
    /// Graph metadata
    pub metadata: HashMap<String, String>,
}

impl NetworkGraph {
    /// Create a new empty network graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Set input nodes
    pub fn set_inputs(&mut self, inputs: Vec<String>) {
        self.inputs = inputs;
    }

    /// Set output nodes
    pub fn set_outputs(&mut self, outputs: Vec<String>) {
        self.outputs = outputs;
    }

    /// Get topological ordering of nodes
    pub fn topological_sort(&self) -> Result<Vec<String>> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize
        for node_id in self.nodes.keys() {
            in_degree.insert(node_id.clone(), 0);
            adj_list.insert(node_id.clone(), Vec::new());
        }

        // Build adjacency list and calculate in-degrees
        for edge in &self.edges {
            adj_list
                .get_mut(&edge.from)
                .expect("edge.from should exist in adj_list")
                .push(edge.to.clone());
            *in_degree
                .get_mut(&edge.to)
                .expect("edge.to should exist in in_degree") += 1;
        }

        // Kahn's algorithm
        let mut queue = Vec::new();
        let mut result = Vec::new();

        // Start with nodes that have no incoming edges
        for (node_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push(node_id.clone());
            }
        }

        while let Some(node_id) = queue.pop() {
            result.push(node_id.clone());

            if let Some(neighbors) = adj_list.get(&node_id) {
                for neighbor in neighbors {
                    let degree = in_degree
                        .get_mut(neighbor)
                        .expect("neighbor should exist in in_degree");
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push(neighbor.clone());
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(TorshError::InvalidArgument(
                "Graph contains cycles".to_string(),
            ));
        }

        Ok(result)
    }

    /// Calculate graph statistics
    pub fn calculate_statistics(&self) -> GraphStatistics {
        let total_params: usize = self.nodes.values().map(|n| n.parameter_count).sum();
        let total_nodes = self.nodes.len();
        let total_edges = self.edges.len();

        let layer_types: HashSet<String> =
            self.nodes.values().map(|n| n.layer_type.clone()).collect();

        let max_depth = self.calculate_depth();

        GraphStatistics {
            total_nodes,
            total_edges,
            total_parameters: total_params,
            unique_layer_types: layer_types.len(),
            max_depth,
            layer_type_counts: self.count_layer_types(),
        }
    }

    /// Calculate the maximum depth of the graph
    fn calculate_depth(&self) -> usize {
        // Simplified depth calculation
        if self.inputs.is_empty() {
            return 0;
        }

        let mut depths: HashMap<String, usize> = HashMap::new();

        // Set input depths to 0
        for input_id in &self.inputs {
            depths.insert(input_id.clone(), 0);
        }

        // Calculate depths using BFS
        if let Ok(sorted_nodes) = self.topological_sort() {
            for node_id in sorted_nodes {
                let mut max_input_depth = 0;

                // Find maximum depth of incoming edges
                for edge in &self.edges {
                    if edge.to == node_id {
                        if let Some(&depth) = depths.get(&edge.from) {
                            max_input_depth = max_input_depth.max(depth);
                        }
                    }
                }

                depths.insert(node_id, max_input_depth + 1);
            }
        }

        depths.values().max().copied().unwrap_or(0)
    }

    /// Count occurrences of each layer type
    fn count_layer_types(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for node in self.nodes.values() {
            *counts.entry(node.layer_type.clone()).or_insert(0) += 1;
        }
        counts
    }
}

impl Default for NetworkGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the network graph
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Total number of edges
    pub total_edges: usize,
    /// Total number of parameters
    pub total_parameters: usize,
    /// Number of unique layer types
    pub unique_layer_types: usize,
    /// Maximum depth of the graph
    pub max_depth: usize,
    /// Count of each layer type
    pub layer_type_counts: HashMap<String, usize>,
}

impl Display for GraphStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph Statistics:")?;
        writeln!(f, "  Total Nodes: {}", self.total_nodes)?;
        writeln!(f, "  Total Edges: {}", self.total_edges)?;
        writeln!(
            f,
            "  Total Parameters: {}",
            format_number(self.total_parameters)
        )?;
        writeln!(f, "  Unique Layer Types: {}", self.unique_layer_types)?;
        writeln!(f, "  Maximum Depth: {}", self.max_depth)?;
        writeln!(f, "  Layer Type Distribution:")?;

        for (layer_type, count) in &self.layer_type_counts {
            writeln!(f, "    {}: {}", layer_type, count)?;
        }

        Ok(())
    }
}

/// Configuration for graph visualization
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Width of the output
    pub width: usize,
    /// Height of the output
    pub height: usize,
    /// Whether to show parameter counts
    pub show_parameters: bool,
    /// Whether to show shapes
    pub show_shapes: bool,
    /// Whether to show layer types
    pub show_layer_types: bool,
    /// Layout algorithm to use
    pub layout: LayoutAlgorithm,
    /// Color scheme
    pub color_scheme: ColorScheme,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            show_parameters: true,
            show_shapes: true,
            show_layer_types: true,
            layout: LayoutAlgorithm::Hierarchical,
            color_scheme: ColorScheme::Default,
        }
    }
}

/// Layout algorithms for graph visualization
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutAlgorithm {
    /// Hierarchical top-down layout
    Hierarchical,
    /// Force-directed layout
    ForceDirected,
    /// Circular layout
    Circular,
    /// Grid layout
    Grid,
}

/// Color schemes for visualization
#[derive(Debug, Clone, PartialEq)]
pub enum ColorScheme {
    /// Default colors
    Default,
    /// Grayscale
    Grayscale,
    /// Colorful
    Colorful,
    /// High contrast
    HighContrast,
}

/// Generate a network graph from a model
pub fn create_graph_from_model<M: Module>(
    model: &M,
    input_shape: &[usize],
) -> Result<NetworkGraph> {
    let mut graph = NetworkGraph::new();

    // Create input node
    let input_node = GraphNode::new(
        "input".to_string(),
        "Input".to_string(),
        "Input".to_string(),
        vec![],
        input_shape.to_vec(),
        0,
    );
    graph.add_node(input_node);
    graph.set_inputs(vec!["input".to_string()]);

    // Create model node (simplified)
    let params = model.parameters();
    let param_count: usize = params
        .values()
        .map(|p| p.tensor().read().shape().dims().iter().product::<usize>())
        .sum();

    let model_node = GraphNode::new(
        "model".to_string(),
        "Model".to_string(),
        "Model".to_string(),
        input_shape.to_vec(),
        input_shape.to_vec(), // Simplified - would compute actual output shape
        param_count,
    );
    graph.add_node(model_node);

    // Create edge from input to model
    let edge = GraphEdge::new(
        "input".to_string(),
        "model".to_string(),
        input_shape.to_vec(),
    );
    graph.add_edge(edge);

    graph.set_outputs(vec!["model".to_string()]);

    Ok(graph)
}

/// Text-based graph renderer
pub struct TextRenderer {
    config: VisualizationConfig,
}

impl TextRenderer {
    /// Create a new text renderer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Render a graph as text
    pub fn render(&self, graph: &NetworkGraph) -> String {
        let mut output = String::new();

        output.push_str("Network Architecture Visualization\n");
        output.push_str("==================================\n\n");

        // Show statistics
        let stats = graph.calculate_statistics();
        output.push_str(&format!("{}\n", stats));

        // Show nodes
        output.push_str("Nodes:\n");
        output.push_str("------\n");

        for (id, node) in &graph.nodes {
            output.push_str(&format!("{}:\n", id));
            output.push_str(&format!("  Name: {}\n", node.name));
            output.push_str(&format!("  Type: {}\n", node.layer_type));

            if self.config.show_shapes {
                output.push_str(&format!("  Input Shape: {:?}\n", node.input_shape));
                output.push_str(&format!("  Output Shape: {:?}\n", node.output_shape));
            }

            if self.config.show_parameters && node.parameter_count > 0 {
                output.push_str(&format!(
                    "  Parameters: {}\n",
                    format_number(node.parameter_count)
                ));
            }

            output.push('\n');
        }

        // Show edges
        output.push_str("Connections:\n");
        output.push_str("------------\n");

        for edge in &graph.edges {
            let style_indicator = match edge.style {
                EdgeStyle::Normal => "->",
                EdgeStyle::Skip => "~~>",
                EdgeStyle::Attention => "==>",
                EdgeStyle::Recurrent => "<->",
            };

            output.push_str(&format!(
                "{} {} {} (shape: {:?})\n",
                edge.from, style_indicator, edge.to, edge.shape
            ));
        }

        output
    }
}

/// ASCII art graph renderer
pub struct AsciiRenderer {
    config: VisualizationConfig,
}

impl AsciiRenderer {
    /// Create a new ASCII renderer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Render a graph as ASCII art
    pub fn render(&self, graph: &NetworkGraph) -> String {
        let mut output = String::new();

        // Simple ASCII representation
        if let Ok(sorted_nodes) = graph.topological_sort() {
            for (i, node_id) in sorted_nodes.iter().enumerate() {
                if let Some(node) = graph.nodes.get(node_id) {
                    // Add indentation based on depth
                    let indent = "  ".repeat(i.min(10));

                    // Node representation
                    let node_repr = if self.config.show_parameters && node.parameter_count > 0 {
                        format!(
                            "[{}] {} ({})",
                            node.layer_type,
                            node.name,
                            format_number(node.parameter_count)
                        )
                    } else {
                        format!("[{}] {}", node.layer_type, node.name)
                    };

                    output.push_str(&format!("{}{}\n", indent, node_repr));

                    if self.config.show_shapes && !node.output_shape.is_empty() {
                        output
                            .push_str(&format!("{}  └─ Output: {:?}\n", indent, node.output_shape));
                    }

                    // Add connection lines
                    if i < sorted_nodes.len() - 1 {
                        output.push_str(&format!("{}  |\n", indent));
                    }
                }
            }
        }

        output
    }
}

/// DOT format renderer for Graphviz
pub struct DotRenderer {
    config: VisualizationConfig,
}

impl DotRenderer {
    /// Create a new DOT renderer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Render a graph in DOT format
    pub fn render(&self, graph: &NetworkGraph) -> String {
        let mut output = String::new();

        output.push_str("digraph NetworkGraph {\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  node [shape=box, style=filled];\n\n");

        // Add nodes
        for (id, node) in &graph.nodes {
            let label = if self.config.show_parameters && node.parameter_count > 0 {
                format!(
                    "{}\\n{}\\n{} params",
                    node.name,
                    node.layer_type,
                    format_number(node.parameter_count)
                )
            } else {
                format!("{}\\n{}", node.name, node.layer_type)
            };

            let color = self.get_node_color(&node.layer_type);

            output.push_str(&format!(
                "  \"{}\" [label=\"{}\", fillcolor=\"{}\"];\n",
                id, label, color
            ));
        }

        output.push('\n');

        // Add edges
        for edge in &graph.edges {
            let style = match edge.style {
                EdgeStyle::Normal => "solid",
                EdgeStyle::Skip => "dashed",
                EdgeStyle::Attention => "bold",
                EdgeStyle::Recurrent => "dotted",
            };

            output.push_str(&format!(
                "  \"{}\" -> \"{}\" [style={}];\n",
                edge.from, edge.to, style
            ));
        }

        output.push_str("}\n");
        output
    }

    /// Get color for a layer type
    fn get_node_color(&self, layer_type: &str) -> &'static str {
        match self.config.color_scheme {
            ColorScheme::Default => match layer_type {
                "Input" => "lightblue",
                "Linear" => "lightgreen",
                "Conv2d" => "orange",
                "ReLU" => "yellow",
                "BatchNorm2d" => "pink",
                _ => "lightgray",
            },
            ColorScheme::Grayscale => "lightgray",
            ColorScheme::Colorful => match layer_type {
                "Input" => "cyan",
                "Linear" => "green",
                "Conv2d" => "red",
                "ReLU" => "yellow",
                "BatchNorm2d" => "magenta",
                _ => "white",
            },
            ColorScheme::HighContrast => match layer_type {
                "Input" => "black",
                "Linear" => "white",
                "Conv2d" => "black",
                "ReLU" => "white",
                "BatchNorm2d" => "black",
                _ => "gray",
            },
        }
    }
}

/// Helper function to format numbers with appropriate units
fn format_number(num: usize) -> String {
    if num >= 1_000_000_000 {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}

/// Utility functions for common visualization tasks
pub mod utils {
    use super::*;

    /// Quick text visualization of a model
    pub fn quick_text_viz<M: Module>(model: &M, input_shape: &[usize]) -> Result<String> {
        let graph = create_graph_from_model(model, input_shape)?;
        let renderer = TextRenderer::new(VisualizationConfig::default());
        Ok(renderer.render(&graph))
    }

    /// Quick ASCII art visualization of a model
    pub fn quick_ascii_viz<M: Module>(model: &M, input_shape: &[usize]) -> Result<String> {
        let graph = create_graph_from_model(model, input_shape)?;
        let renderer = AsciiRenderer::new(VisualizationConfig::default());
        Ok(renderer.render(&graph))
    }

    /// Generate DOT format for Graphviz
    pub fn generate_dot<M: Module>(model: &M, input_shape: &[usize]) -> Result<String> {
        let graph = create_graph_from_model(model, input_shape)?;
        let renderer = DotRenderer::new(VisualizationConfig::default());
        Ok(renderer.render(&graph))
    }

    /// Print a quick visualization to stdout
    pub fn print_model_viz<M: Module>(model: &M, input_shape: &[usize]) -> Result<()> {
        let viz = quick_text_viz(model, input_shape)?;
        println!("{}", viz);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Linear;

    #[test]
    fn test_graph_node_creation() {
        let node = GraphNode::new(
            "linear1".to_string(),
            "Linear Layer 1".to_string(),
            "Linear".to_string(),
            vec![10, 20],
            vec![10, 30],
            630,
        );

        assert_eq!(node.id, "linear1");
        assert_eq!(node.layer_type, "Linear");
        assert_eq!(node.parameter_count, 630);
    }

    #[test]
    fn test_graph_edge_creation() {
        let edge = GraphEdge::new("input".to_string(), "linear1".to_string(), vec![10, 20])
            .with_style(EdgeStyle::Skip);

        assert_eq!(edge.from, "input");
        assert_eq!(edge.to, "linear1");
        assert_eq!(edge.style, EdgeStyle::Skip);
    }

    #[test]
    fn test_network_graph() {
        let mut graph = NetworkGraph::new();

        let node1 = GraphNode::new(
            "input".to_string(),
            "Input".to_string(),
            "Input".to_string(),
            vec![],
            vec![10, 20],
            0,
        );

        let node2 = GraphNode::new(
            "linear".to_string(),
            "Linear".to_string(),
            "Linear".to_string(),
            vec![10, 20],
            vec![10, 30],
            630,
        );

        graph.add_node(node1);
        graph.add_node(node2);

        let edge = GraphEdge::new("input".to_string(), "linear".to_string(), vec![10, 20]);
        graph.add_edge(edge);

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_topological_sort() -> Result<()> {
        let mut graph = NetworkGraph::new();

        // Create linear chain: A -> B -> C
        for (id, name) in [("A", "Node A"), ("B", "Node B"), ("C", "Node C")] {
            let node = GraphNode::new(
                id.to_string(),
                name.to_string(),
                "Test".to_string(),
                vec![10],
                vec![10],
                0,
            );
            graph.add_node(node);
        }

        graph.add_edge(GraphEdge::new("A".to_string(), "B".to_string(), vec![10]));
        graph.add_edge(GraphEdge::new("B".to_string(), "C".to_string(), vec![10]));

        let sorted = graph.topological_sort()?;
        assert_eq!(
            sorted,
            vec!["A".to_string(), "B".to_string(), "C".to_string()]
        );

        Ok(())
    }

    #[test]
    fn test_graph_statistics() -> Result<()> {
        let mut graph = NetworkGraph::new();

        let node1 = GraphNode::new(
            "linear1".to_string(),
            "Linear 1".to_string(),
            "Linear".to_string(),
            vec![10],
            vec![20],
            210,
        );

        let node2 = GraphNode::new(
            "linear2".to_string(),
            "Linear 2".to_string(),
            "Linear".to_string(),
            vec![20],
            vec![30],
            630,
        );

        graph.add_node(node1);
        graph.add_node(node2);

        let stats = graph.calculate_statistics();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_parameters, 840);
        assert_eq!(stats.unique_layer_types, 1);

        Ok(())
    }

    #[test]
    fn test_text_renderer() -> Result<()> {
        let model = Linear::new(64, 32, true);
        let graph = create_graph_from_model(&model, &[10, 64])?;

        let config = VisualizationConfig::default();
        let renderer = TextRenderer::new(config);
        let output = renderer.render(&graph);

        assert!(output.contains("Network Architecture Visualization"));
        assert!(output.contains("Nodes:"));
        assert!(output.contains("Connections:"));

        Ok(())
    }

    #[test]
    fn test_ascii_renderer() -> Result<()> {
        let model = Linear::new(32, 16, true);
        let graph = create_graph_from_model(&model, &[5, 32])?;

        let config = VisualizationConfig::default();
        let renderer = AsciiRenderer::new(config);
        let output = renderer.render(&graph);

        assert!(!output.is_empty());

        Ok(())
    }

    #[test]
    fn test_dot_renderer() -> Result<()> {
        let model = Linear::new(16, 8, true);
        let graph = create_graph_from_model(&model, &[3, 16])?;

        let config = VisualizationConfig::default();
        let renderer = DotRenderer::new(config);
        let output = renderer.render(&graph);

        assert!(output.contains("digraph NetworkGraph"));
        assert!(output.contains("->"));

        Ok(())
    }

    #[test]
    fn test_utils_functions() -> Result<()> {
        let model = Linear::new(128, 64, true);

        let text_viz = utils::quick_text_viz(&model, &[8, 128])?;
        assert!(!text_viz.is_empty());

        let ascii_viz = utils::quick_ascii_viz(&model, &[8, 128])?;
        assert!(!ascii_viz.is_empty());

        let dot_viz = utils::generate_dot(&model, &[8, 128])?;
        assert!(dot_viz.contains("digraph"));

        Ok(())
    }
}
