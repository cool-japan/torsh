//! TorchScript compatibility module
//!
//! This module provides functionality to import from and export to TorchScript format,
//! enabling interoperability with PyTorch models.

use crate::{Edge, FxGraph, Node};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use torsh_core::Result;

/// TorchScript model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorchScriptModel {
    pub name: String,
    pub version: String,
    pub producer_name: String,
    pub code: String,
    pub constants: HashMap<String, TorchScriptConstant>,
    pub parameters: Vec<TorchScriptParameter>,
    pub methods: Vec<TorchScriptMethod>,
    pub metadata: HashMap<String, String>,
}

/// TorchScript constant value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TorchScriptConstant {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Tensor(TensorConstant),
    List(Vec<TorchScriptConstant>),
    Dict(HashMap<String, TorchScriptConstant>),
}

/// Tensor constant representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorConstant {
    pub shape: Vec<i64>,
    pub dtype: String,
    pub data: Vec<u8>, // Serialized tensor data
}

/// TorchScript parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorchScriptParameter {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub requires_grad: bool,
    pub is_buffer: bool,
}

/// TorchScript method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorchScriptMethod {
    pub name: String,
    pub code: String,
    pub schema: MethodSchema,
    pub graph: Option<TorchScriptGraph>,
}

/// Method schema for type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodSchema {
    pub arguments: Vec<Argument>,
    pub returns: Vec<Return>,
}

/// Method argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    pub name: String,
    pub arg_type: String,
    pub default_value: Option<TorchScriptConstant>,
}

/// Method return type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Return {
    pub name: Option<String>,
    pub return_type: String,
}

/// TorchScript graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorchScriptGraph {
    pub nodes: Vec<TorchScriptNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// TorchScript node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorchScriptNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, TorchScriptConstant>,
    pub source_range: Option<SourceRange>,
}

/// Source code location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRange {
    pub filename: String,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

/// TorchScript importer
pub struct TorchScriptImporter {
    operator_mapping: HashMap<String, String>,
    #[allow(dead_code)]
    type_mapping: HashMap<String, String>,
}

impl Default for TorchScriptImporter {
    fn default() -> Self {
        let mut operator_mapping = HashMap::new();

        // Basic operators
        operator_mapping.insert("aten::add".to_string(), "add".to_string());
        operator_mapping.insert("aten::sub".to_string(), "sub".to_string());
        operator_mapping.insert("aten::mul".to_string(), "mul".to_string());
        operator_mapping.insert("aten::div".to_string(), "div".to_string());
        operator_mapping.insert("aten::relu".to_string(), "relu".to_string());
        operator_mapping.insert("aten::sigmoid".to_string(), "sigmoid".to_string());
        operator_mapping.insert("aten::tanh".to_string(), "tanh".to_string());
        operator_mapping.insert("aten::softmax".to_string(), "softmax".to_string());

        // Linear algebra
        operator_mapping.insert("aten::mm".to_string(), "matmul".to_string());
        operator_mapping.insert("aten::bmm".to_string(), "batch_matmul".to_string());
        operator_mapping.insert("aten::addmm".to_string(), "linear".to_string());

        // Convolution
        operator_mapping.insert("aten::conv2d".to_string(), "conv2d".to_string());
        operator_mapping.insert("aten::conv1d".to_string(), "conv1d".to_string());
        operator_mapping.insert("aten::conv3d".to_string(), "conv3d".to_string());

        // Pooling
        operator_mapping.insert("aten::max_pool2d".to_string(), "max_pool2d".to_string());
        operator_mapping.insert("aten::avg_pool2d".to_string(), "avg_pool2d".to_string());
        operator_mapping.insert(
            "aten::adaptive_avg_pool2d".to_string(),
            "adaptive_avg_pool2d".to_string(),
        );

        // Normalization
        operator_mapping.insert("aten::batch_norm".to_string(), "batch_norm".to_string());
        operator_mapping.insert("aten::layer_norm".to_string(), "layer_norm".to_string());
        operator_mapping.insert("aten::group_norm".to_string(), "group_norm".to_string());

        // Shape operations
        operator_mapping.insert("aten::view".to_string(), "reshape".to_string());
        operator_mapping.insert("aten::reshape".to_string(), "reshape".to_string());
        operator_mapping.insert("aten::transpose".to_string(), "transpose".to_string());
        operator_mapping.insert("aten::permute".to_string(), "permute".to_string());
        operator_mapping.insert("aten::squeeze".to_string(), "squeeze".to_string());
        operator_mapping.insert("aten::unsqueeze".to_string(), "unsqueeze".to_string());

        let mut type_mapping = HashMap::new();
        type_mapping.insert("Tensor".to_string(), "tensor".to_string());
        type_mapping.insert("int".to_string(), "i64".to_string());
        type_mapping.insert("float".to_string(), "f64".to_string());
        type_mapping.insert("bool".to_string(), "bool".to_string());
        type_mapping.insert("str".to_string(), "string".to_string());

        Self {
            operator_mapping,
            type_mapping,
        }
    }
}

impl TorchScriptImporter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Import a TorchScript model into an FX graph
    pub fn import_model(&self, model: &TorchScriptModel) -> Result<FxGraph> {
        if let Some(forward_method) = model.methods.iter().find(|m| m.name == "forward") {
            if let Some(graph) = &forward_method.graph {
                self.import_graph(graph)
            } else {
                // Parse from code if no graph is available
                self.parse_code_to_graph(&forward_method.code)
            }
        } else {
            Err(torsh_core::error::TorshError::InvalidArgument(
                "No forward method found in TorchScript model".to_string(),
            ))
        }
    }

    /// Import a TorchScript graph into an FX graph
    pub fn import_graph(&self, ts_graph: &TorchScriptGraph) -> Result<FxGraph> {
        let mut fx_graph = FxGraph::new();
        let mut node_mapping = HashMap::new();
        let mut value_to_node = HashMap::new();

        // Create input nodes
        for input_name in &ts_graph.inputs {
            let node = fx_graph.graph.add_node(Node::Input(input_name.clone()));
            fx_graph.inputs.push(node);
            value_to_node.insert(input_name.clone(), node);
        }

        // Process TorchScript nodes in topological order
        for ts_node in &ts_graph.nodes {
            let fx_node = self.convert_torchscript_node(ts_node)?;
            let node_idx = fx_graph.graph.add_node(fx_node);
            node_mapping.insert(ts_node.name.clone(), node_idx);

            // Map outputs to this node
            for output in &ts_node.outputs {
                value_to_node.insert(output.clone(), node_idx);
            }
        }

        // Create output nodes
        for output_name in &ts_graph.outputs {
            let output_node = fx_graph.graph.add_node(Node::Output);
            fx_graph.outputs.push(output_node);

            // Connect to the node that produces this output
            if let Some(&producer_node) = value_to_node.get(output_name) {
                fx_graph.graph.add_edge(
                    producer_node,
                    output_node,
                    Edge {
                        name: output_name.clone(),
                    },
                );
            }
        }

        // Create edges between nodes
        for ts_node in &ts_graph.nodes {
            if let Some(&target_node) = node_mapping.get(&ts_node.name) {
                for input_name in &ts_node.inputs {
                    if let Some(&source_node) = value_to_node.get(input_name) {
                        if source_node != target_node {
                            fx_graph.graph.add_edge(
                                source_node,
                                target_node,
                                Edge {
                                    name: input_name.clone(),
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(fx_graph)
    }

    fn convert_torchscript_node(&self, ts_node: &TorchScriptNode) -> Result<Node> {
        let op_name = self
            .operator_mapping
            .get(&ts_node.op_type)
            .unwrap_or(&ts_node.op_type)
            .clone();

        // Handle special cases
        match ts_node.op_type.as_str() {
            "prim::Constant" => {
                // Constants become inputs for now
                let node_name = &ts_node.name;
                Ok(Node::Input(format!("constant_{node_name}")))
            }
            "prim::If" => Ok(Node::Conditional {
                condition: ts_node
                    .inputs
                    .first()
                    .unwrap_or(&"condition".to_string())
                    .clone(),
                then_branch: vec!["true_branch".to_string()],
                else_branch: vec!["false_branch".to_string()],
            }),
            "prim::Loop" => Ok(Node::Loop {
                condition: ts_node
                    .inputs
                    .first()
                    .unwrap_or(&"condition".to_string())
                    .clone(),
                body: vec!["loop_body".to_string()],
                loop_vars: ts_node.inputs.iter().skip(1).cloned().collect(),
            }),
            "prim::GetAttr" => {
                let attr_name = ts_node
                    .attributes
                    .get("name")
                    .and_then(|v| {
                        if let TorchScriptConstant::String(s) = v {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| "attr".to_string());

                Ok(Node::GetAttr {
                    target: ts_node
                        .inputs
                        .first()
                        .unwrap_or(&"self".to_string())
                        .clone(),
                    attr: attr_name,
                })
            }
            _ => Ok(Node::Call(op_name, ts_node.inputs.clone())),
        }
    }

    fn parse_code_to_graph(&self, _code: &str) -> Result<FxGraph> {
        // This would require a full TorchScript parser
        // For now, return a simple placeholder graph
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("input".to_string()));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            output,
            Edge {
                name: "passthrough".to_string(),
            },
        );
        graph.inputs = vec![input];
        graph.outputs = vec![output];

        Ok(graph)
    }

    /// Add custom operator mapping
    pub fn add_operator_mapping(&mut self, torchscript_op: String, fx_op: String) {
        self.operator_mapping.insert(torchscript_op, fx_op);
    }
}

/// TorchScript exporter
pub struct TorchScriptExporter {
    operator_mapping: HashMap<String, String>,
    export_parameters: bool,
    optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

impl Default for TorchScriptExporter {
    fn default() -> Self {
        let mut operator_mapping = HashMap::new();

        // Reverse mapping from FX to TorchScript
        operator_mapping.insert("add".to_string(), "aten::add".to_string());
        operator_mapping.insert("sub".to_string(), "aten::sub".to_string());
        operator_mapping.insert("mul".to_string(), "aten::mul".to_string());
        operator_mapping.insert("div".to_string(), "aten::div".to_string());
        operator_mapping.insert("relu".to_string(), "aten::relu".to_string());
        operator_mapping.insert("sigmoid".to_string(), "aten::sigmoid".to_string());
        operator_mapping.insert("tanh".to_string(), "aten::tanh".to_string());
        operator_mapping.insert("softmax".to_string(), "aten::softmax".to_string());
        operator_mapping.insert("matmul".to_string(), "aten::mm".to_string());
        operator_mapping.insert("conv2d".to_string(), "aten::conv2d".to_string());
        operator_mapping.insert("max_pool2d".to_string(), "aten::max_pool2d".to_string());
        operator_mapping.insert("avg_pool2d".to_string(), "aten::avg_pool2d".to_string());
        operator_mapping.insert("batch_norm".to_string(), "aten::batch_norm".to_string());
        operator_mapping.insert("reshape".to_string(), "aten::view".to_string());
        operator_mapping.insert("transpose".to_string(), "aten::transpose".to_string());
        operator_mapping.insert("permute".to_string(), "aten::permute".to_string());

        Self {
            operator_mapping,
            export_parameters: true,
            optimization_level: OptimizationLevel::Basic,
        }
    }
}

impl TorchScriptExporter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parameters(mut self, export_parameters: bool) -> Self {
        self.export_parameters = export_parameters;
        self
    }

    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Export an FX graph to TorchScript model
    pub fn export_model(&self, graph: &FxGraph, model_name: &str) -> Result<TorchScriptModel> {
        let torchscript_graph = self.export_graph(graph)?;
        let forward_method = self.create_forward_method(&torchscript_graph)?;

        let model = TorchScriptModel {
            name: model_name.to_string(),
            version: "1.0".to_string(),
            producer_name: "torsh-fx".to_string(),
            code: self.generate_torchscript_code(&torchscript_graph)?,
            constants: HashMap::new(),
            parameters: if self.export_parameters {
                self.extract_parameters(graph)?
            } else {
                Vec::new()
            },
            methods: vec![forward_method],
            metadata: HashMap::new(),
        };

        Ok(model)
    }

    /// Export an FX graph to TorchScript graph
    pub fn export_graph(&self, fx_graph: &FxGraph) -> Result<TorchScriptGraph> {
        let mut nodes = Vec::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut node_name_counter = 0;
        let mut value_names = HashMap::new();

        // Process input nodes
        for &input_idx in &fx_graph.inputs {
            if let Some(node) = fx_graph.get_node(input_idx) {
                if let Node::Input(input_name) = node {
                    inputs.push(input_name.clone());
                    value_names.insert(input_idx, input_name.clone());
                }
            }
        }

        // Process all nodes in topological order
        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();

        // Start with input nodes
        for &input_idx in &fx_graph.inputs {
            queue.push_back(input_idx);
        }

        while let Some(current_idx) = queue.pop_front() {
            if visited.contains(&current_idx) {
                continue;
            }
            visited.insert(current_idx);

            if let Some(node) = fx_graph.get_node(current_idx) {
                if !matches!(node, Node::Input(_)) {
                    let ts_node = self.convert_fx_node(
                        node,
                        current_idx,
                        &mut node_name_counter,
                        &value_names,
                    )?;

                    // Update value names with outputs
                    for output in &ts_node.outputs {
                        value_names.insert(current_idx, output.clone());
                    }

                    nodes.push(ts_node);
                }

                // Add successors to queue
                for edge_ref in fx_graph
                    .graph
                    .edges_directed(current_idx, petgraph::Direction::Outgoing)
                {
                    queue.push_back(edge_ref.target());
                }
            }
        }

        // Process output nodes
        for &output_idx in &fx_graph.outputs {
            // Find the input to this output node
            for edge_ref in fx_graph
                .graph
                .edges_directed(output_idx, petgraph::Direction::Incoming)
            {
                let source_idx = edge_ref.source();
                if let Some(output_name) = value_names.get(&source_idx) {
                    outputs.push(output_name.clone());
                    break;
                }
            }
        }

        Ok(TorchScriptGraph {
            nodes,
            inputs,
            outputs,
        })
    }

    fn convert_fx_node(
        &self,
        fx_node: &Node,
        _node_idx: NodeIndex,
        name_counter: &mut usize,
        _value_names: &HashMap<NodeIndex, String>,
    ) -> Result<TorchScriptNode> {
        let counter = *name_counter;
        let node_name = format!("node_{counter}");
        *name_counter += 1;

        match fx_node {
            Node::Call(op_name, args) => {
                let ts_op_type = self
                    .operator_mapping
                    .get(op_name)
                    .unwrap_or(op_name)
                    .clone();

                Ok(TorchScriptNode {
                    name: node_name.clone(),
                    op_type: ts_op_type,
                    inputs: args.clone(),
                    outputs: vec![format!("{node_name}_output")],
                    attributes: HashMap::new(),
                    source_range: None,
                })
            }

            Node::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut attributes = HashMap::new();
                attributes.insert(
                    "then_block".to_string(),
                    TorchScriptConstant::List(
                        then_branch
                            .iter()
                            .map(|s| TorchScriptConstant::String(s.clone()))
                            .collect(),
                    ),
                );
                attributes.insert(
                    "else_block".to_string(),
                    TorchScriptConstant::List(
                        else_branch
                            .iter()
                            .map(|s| TorchScriptConstant::String(s.clone()))
                            .collect(),
                    ),
                );

                Ok(TorchScriptNode {
                    name: node_name.clone(),
                    op_type: "prim::If".to_string(),
                    inputs: vec![condition.clone()],
                    outputs: vec![format!("{node_name}_output")],
                    attributes,
                    source_range: None,
                })
            }

            Node::Loop {
                condition,
                body,
                loop_vars,
            } => {
                let mut attributes = HashMap::new();
                attributes.insert(
                    "body".to_string(),
                    TorchScriptConstant::List(
                        body.iter()
                            .map(|s| TorchScriptConstant::String(s.clone()))
                            .collect(),
                    ),
                );

                let mut inputs = vec![condition.clone()];
                inputs.extend(loop_vars.iter().cloned());

                Ok(TorchScriptNode {
                    name: node_name.clone(),
                    op_type: "prim::Loop".to_string(),
                    inputs,
                    outputs: vec![format!("{node_name}_output")],
                    attributes,
                    source_range: None,
                })
            }

            Node::GetAttr { target, attr } => {
                let mut attributes = HashMap::new();
                attributes.insert(
                    "name".to_string(),
                    TorchScriptConstant::String(attr.clone()),
                );

                Ok(TorchScriptNode {
                    name: node_name.clone(),
                    op_type: "prim::GetAttr".to_string(),
                    inputs: vec![target.clone()],
                    outputs: vec![format!("{node_name}_output")],
                    attributes,
                    source_range: None,
                })
            }

            Node::Merge { inputs } => Ok(TorchScriptNode {
                name: node_name.clone(),
                op_type: "prim::TupleConstruct".to_string(),
                inputs: inputs.clone(),
                outputs: vec![format!("{}_output", node_name)],
                attributes: HashMap::new(),
                source_range: None,
            }),

            _ => Ok(TorchScriptNode {
                name: node_name.clone(),
                op_type: "prim::Constant".to_string(),
                inputs: Vec::new(),
                outputs: vec![format!("{}_output", node_name)],
                attributes: HashMap::new(),
                source_range: None,
            }),
        }
    }

    fn create_forward_method(&self, graph: &TorchScriptGraph) -> Result<TorchScriptMethod> {
        let arguments = graph
            .inputs
            .iter()
            .map(|input| Argument {
                name: input.clone(),
                arg_type: "Tensor".to_string(),
                default_value: None,
            })
            .collect();

        let returns = graph
            .outputs
            .iter()
            .map(|output| Return {
                name: Some(output.clone()),
                return_type: "Tensor".to_string(),
            })
            .collect();

        let schema = MethodSchema { arguments, returns };

        Ok(TorchScriptMethod {
            name: "forward".to_string(),
            code: self.generate_torchscript_code(graph)?,
            schema,
            graph: Some(graph.clone()),
        })
    }

    fn generate_torchscript_code(&self, graph: &TorchScriptGraph) -> Result<String> {
        let mut code = String::new();

        // Function signature
        code.push_str("def forward(self");
        for input in &graph.inputs {
            code.push_str(&format!(", {}: Tensor", input));
        }
        code.push_str(") -> ");

        if graph.outputs.len() == 1 {
            code.push_str("Tensor");
        } else {
            code.push_str(&format!(
                "Tuple[{}]",
                vec!["Tensor"; graph.outputs.len()].join(", ")
            ));
        }
        code.push_str(":\n");

        // Function body
        for node in &graph.nodes {
            code.push_str(&self.generate_node_code(node)?);
            code.push('\n');
        }

        // Return statement
        if graph.outputs.len() == 1 {
            code.push_str(&format!("    return {}\n", graph.outputs[0]));
        } else {
            code.push_str(&format!("    return ({})\n", graph.outputs.join(", ")));
        }

        Ok(code)
    }

    fn generate_node_code(&self, node: &TorchScriptNode) -> Result<String> {
        let indent = "    ";

        match node.op_type.as_str() {
            "aten::add" => Ok(format!(
                "{}{} = {} + {}",
                indent,
                node.outputs[0],
                node.inputs.get(0).unwrap_or(&"input1".to_string()),
                node.inputs.get(1).unwrap_or(&"input2".to_string())
            )),

            "aten::relu" => Ok(format!(
                "{}{} = torch.relu({})",
                indent,
                node.outputs[0],
                node.inputs.get(0).unwrap_or(&"input".to_string())
            )),

            "aten::mm" => Ok(format!(
                "{}{} = torch.mm({}, {})",
                indent,
                node.outputs[0],
                node.inputs.get(0).unwrap_or(&"input1".to_string()),
                node.inputs.get(1).unwrap_or(&"input2".to_string())
            )),

            _ => Ok(format!(
                "{}{} = {}({})",
                indent,
                node.outputs[0],
                node.op_type,
                node.inputs.join(", ")
            )),
        }
    }

    fn extract_parameters(&self, _graph: &FxGraph) -> Result<Vec<TorchScriptParameter>> {
        // This would require analysis of the graph to identify learnable parameters
        // For now, return an empty list
        Ok(Vec::new())
    }

    /// Add custom operator mapping  
    pub fn add_operator_mapping(&mut self, fx_op: String, torchscript_op: String) {
        self.operator_mapping.insert(fx_op, torchscript_op);
    }
}

/// Utility functions for TorchScript compatibility
pub mod utils {
    use super::*;

    /// Load a TorchScript model from file
    pub fn load_torchscript_model(path: &str) -> Result<TorchScriptModel> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))?;

        serde_json::from_str(&content)
            .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))
    }

    /// Save a TorchScript model to file
    pub fn save_torchscript_model(model: &TorchScriptModel, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(model)
            .map_err(|e| torsh_core::error::TorshError::SerializationError(e.to_string()))?;

        std::fs::write(path, content)
            .map_err(|e| torsh_core::error::TorshError::IoError(e.to_string()))
    }

    /// Convert FX graph to TorchScript and back for validation
    pub fn validate_roundtrip(graph: &FxGraph) -> Result<bool> {
        let exporter = TorchScriptExporter::new();
        let model = exporter.export_model(graph, "test_model")?;

        let importer = TorchScriptImporter::new();
        let reconstructed = importer.import_model(&model)?;

        // Simple validation - check node counts
        Ok(graph.node_count() == reconstructed.node_count()
            && graph.edge_count() == reconstructed.edge_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, FxGraph, Node};

    #[test]
    fn test_torchscript_import_basic() {
        let ts_graph = TorchScriptGraph {
            nodes: vec![TorchScriptNode {
                name: "node_0".to_string(),
                op_type: "aten::relu".to_string(),
                inputs: vec!["input".to_string()],
                outputs: vec!["relu_out".to_string()],
                attributes: HashMap::new(),
                source_range: None,
            }],
            inputs: vec!["input".to_string()],
            outputs: vec!["relu_out".to_string()],
        };

        let importer = TorchScriptImporter::new();
        let fx_graph = importer.import_graph(&ts_graph).unwrap();

        assert_eq!(fx_graph.inputs.len(), 1);
        assert_eq!(fx_graph.outputs.len(), 1);
        assert!(fx_graph.node_count() >= 3); // input, relu, output
    }

    #[test]
    fn test_torchscript_export_basic() {
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
        graph.inputs = vec![input];
        graph.outputs = vec![output];

        let exporter = TorchScriptExporter::new();
        let ts_graph = exporter.export_graph(&graph).unwrap();

        assert!(!ts_graph.inputs.is_empty());
        assert!(!ts_graph.outputs.is_empty());
        assert!(!ts_graph.nodes.is_empty());

        // Check that relu was converted to aten::relu
        assert!(ts_graph
            .nodes
            .iter()
            .any(|node| node.op_type == "aten::relu"));
    }

    #[test]
    fn test_torchscript_roundtrip() {
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

        // Export to TorchScript
        let exporter = TorchScriptExporter::new();
        let model = exporter.export_model(&graph, "test_model").unwrap();

        // Import back to FX
        let importer = TorchScriptImporter::new();
        let reconstructed = importer.import_model(&model).unwrap();

        // Basic validation
        assert_eq!(graph.inputs.len(), reconstructed.inputs.len());
        assert_eq!(graph.outputs.len(), reconstructed.outputs.len());
    }

    #[test]
    fn test_torchscript_code_generation() {
        let ts_graph = TorchScriptGraph {
            nodes: vec![
                TorchScriptNode {
                    name: "add_node".to_string(),
                    op_type: "aten::add".to_string(),
                    inputs: vec!["x".to_string(), "y".to_string()],
                    outputs: vec!["add_out".to_string()],
                    attributes: HashMap::new(),
                    source_range: None,
                },
                TorchScriptNode {
                    name: "relu_node".to_string(),
                    op_type: "aten::relu".to_string(),
                    inputs: vec!["add_out".to_string()],
                    outputs: vec!["result".to_string()],
                    attributes: HashMap::new(),
                    source_range: None,
                },
            ],
            inputs: vec!["x".to_string(), "y".to_string()],
            outputs: vec!["result".to_string()],
        };

        let exporter = TorchScriptExporter::new();
        let code = exporter.generate_torchscript_code(&ts_graph).unwrap();

        assert!(code.contains("def forward(self, x: Tensor, y: Tensor) -> Tensor"));
        assert!(code.contains("add_out = x + y"));
        assert!(code.contains("result = torch.relu(add_out)"));
        assert!(code.contains("return result"));
    }
}
