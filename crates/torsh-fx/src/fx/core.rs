//! Essential FxGraph implementation and core functionality

use crate::codegen::CodeGenerator;
use crate::fx::types::{Edge, Node};
use crate::onnx_export::{export_to_onnx, OnnxExporter, OnnxModel};
use crate::{FxGraph, TorshResult};
use petgraph::graph::{Graph, NodeIndex};

/// FX Graph representation
#[derive(Debug, Clone)]
pub struct FxGraphCore {
    pub graph: Graph<Node, Edge>,
    pub inputs: Vec<NodeIndex>,
    pub outputs: Vec<NodeIndex>,
}

impl FxGraph {
    /// Create new graph
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get input nodes
    pub fn inputs(&self) -> &[NodeIndex] {
        &self.inputs
    }

    /// Get output nodes
    pub fn outputs(&self) -> &[NodeIndex] {
        &self.outputs
    }

    /// Get a node by index
    pub fn get_node(&self, idx: NodeIndex) -> Option<&Node> {
        self.graph.node_weight(idx)
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        self.graph.add_node(node)
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, source: NodeIndex, target: NodeIndex, edge: Edge) {
        self.graph.add_edge(source, target, edge);
    }

    /// Add an input node
    pub fn add_input(&mut self, input: NodeIndex) {
        self.inputs.push(input);
    }

    /// Add an output node
    pub fn add_output(&mut self, output: NodeIndex) {
        self.outputs.push(output);
    }

    /// Iterate over all nodes
    pub fn nodes(&self) -> impl Iterator<Item = (NodeIndex, &Node)> {
        self.graph
            .node_indices()
            .map(move |idx| (idx, &self.graph[idx]))
    }

    /// Print the graph structure
    pub fn print(&self) {
        println!("FX Graph:");
        println!("  Nodes: {}", self.node_count());
        println!("  Edges: {}", self.edge_count());
        println!("  Inputs: {:?}", self.inputs);
        println!("  Outputs: {:?}", self.outputs);

        for (idx, node) in self.nodes() {
            println!("  Node {:?}: {:?}", idx, node);
        }
    }

    /// Generate code for the graph using the specified target language
    pub fn generate_code(&self, target: &str) -> TorshResult<String> {
        let generator = CodeGenerator::new();
        generator.generate_code(self, target)
    }

    /// Generate Python code for the graph
    pub fn to_python(&self) -> TorshResult<String> {
        self.generate_code("python")
    }

    /// Generate C++ code for the graph
    pub fn to_cpp(&self) -> TorshResult<String> {
        self.generate_code("cpp")
    }

    /// Export the graph to ONNX format
    pub fn to_onnx(&self) -> TorshResult<OnnxModel> {
        export_to_onnx(self, None)
    }

    /// Export the graph to ONNX format with a custom model name
    pub fn to_onnx_named(&self, model_name: String) -> TorshResult<OnnxModel> {
        export_to_onnx(self, Some(model_name))
    }

    /// Export the graph to ONNX JSON format
    pub fn to_onnx_json(&self) -> TorshResult<String> {
        let exporter = OnnxExporter::new();
        exporter.export_to_json(self)
    }
}

impl Default for FxGraph {
    fn default() -> Self {
        Self::new()
    }
}
