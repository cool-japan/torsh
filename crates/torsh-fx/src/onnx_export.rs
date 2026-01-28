//! ONNX export functionality for FX graphs
//!
//! This module provides functionality to export FX graphs to ONNX format.
//! Currently implements a basic ONNX representation without external dependencies.

use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::{dtype::DType, error::TorshError};

/// ONNX attribute value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnnxAttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    Tensor(OnnxTensor),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

/// ONNX attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxAttribute {
    pub name: String,
    pub value: OnnxAttributeValue,
}

/// ONNX tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensor {
    pub dims: Vec<i64>,
    pub data_type: i32, // ONNX data type enum
    pub data: Vec<u8>,  // Raw tensor data
    pub name: Option<String>,
}

/// ONNX value info (for inputs/outputs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxValueInfo {
    pub name: String,
    pub type_info: OnnxTypeInfo,
}

/// ONNX type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTypeInfo {
    pub tensor_type: OnnxTensorType,
}

/// ONNX tensor type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensorType {
    pub elem_type: i32,
    pub shape: OnnxTensorShape,
}

/// ONNX tensor shape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensorShape {
    pub dim: Vec<OnnxDimension>,
}

/// ONNX dimension (can be static or dynamic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnnxDimension {
    DimValue(i64),
    DimParam(String),
}

/// ONNX node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxNode {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub name: Option<String>,
    pub op_type: String,
    pub domain: Option<String>,
    pub attribute: Vec<OnnxAttribute>,
}

/// ONNX graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxGraph {
    pub node: Vec<OnnxNode>,
    pub name: String,
    pub initializer: Vec<OnnxTensor>,
    pub input: Vec<OnnxValueInfo>,
    pub output: Vec<OnnxValueInfo>,
    pub value_info: Vec<OnnxValueInfo>,
}

/// ONNX model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub graph: OnnxGraph,
}

/// ONNX exporter
pub struct OnnxExporter {
    pub model_name: String,
    pub producer_name: String,
    pub producer_version: String,
    pub opset_version: i64,
}

impl Default for OnnxExporter {
    fn default() -> Self {
        Self {
            model_name: "exported_model".to_string(),
            producer_name: "torsh-fx".to_string(),
            producer_version: "0.1.0".to_string(),
            opset_version: 17, // ONNX opset version
        }
    }
}

impl OnnxExporter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model_name(mut self, name: String) -> Self {
        self.model_name = name;
        self
    }

    pub fn with_producer_info(mut self, name: String, version: String) -> Self {
        self.producer_name = name;
        self.producer_version = version;
        self
    }

    pub fn with_opset_version(mut self, version: i64) -> Self {
        self.opset_version = version;
        self
    }

    /// Export FX graph to ONNX model
    pub fn export(&self, graph: &FxGraph) -> TorshResult<OnnxModel> {
        let mut onnx_nodes = Vec::new();
        let value_infos = Vec::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut node_name_map = HashMap::new();

        // Generate unique names for each node
        for (idx, _) in graph.nodes() {
            let node_index = idx.index();
            let name = format!("node_{node_index}");
            node_name_map.insert(idx, name);
        }

        // Process each node in the graph
        for (idx, node) in graph.nodes() {
            match node {
                Node::Input(input_name) => {
                    let value_info = OnnxValueInfo {
                        name: input_name.clone(),
                        type_info: OnnxTypeInfo {
                            tensor_type: OnnxTensorType {
                                elem_type: self.dtype_to_onnx_type(DType::F32),
                                shape: OnnxTensorShape {
                                    dim: vec![OnnxDimension::DimParam("dynamic".to_string())],
                                },
                            },
                        },
                    };
                    inputs.push(value_info);
                }

                Node::Call(op_name, args) => {
                    let onnx_node =
                        self.convert_operation_to_onnx(idx, op_name, args, graph, &node_name_map)?;
                    onnx_nodes.push(onnx_node);
                }

                Node::Output => {
                    // Find the input to this output node
                    let predecessors: Vec<_> = graph
                        .graph
                        .neighbors_directed(idx, petgraph::Direction::Incoming)
                        .collect();

                    if let Some(pred_idx) = predecessors.first() {
                        let node_index = idx.index();
                        let default_name = format!("output_{node_index}");
                        let output_name = node_name_map.get(pred_idx).unwrap_or(&default_name);

                        let value_info = OnnxValueInfo {
                            name: output_name.clone(),
                            type_info: OnnxTypeInfo {
                                tensor_type: OnnxTensorType {
                                    elem_type: self.dtype_to_onnx_type(DType::F32),
                                    shape: OnnxTensorShape {
                                        dim: vec![OnnxDimension::DimParam("dynamic".to_string())],
                                    },
                                },
                            },
                        };
                        outputs.push(value_info);
                    }
                }

                Node::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    // ONNX doesn't have native conditionals, so we'd need to use If operator
                    // For now, create a placeholder node
                    let node_index = idx.index();
                    let onnx_node = OnnxNode {
                        input: vec![condition.clone()],
                        output: vec![node_name_map[&idx].clone()],
                        name: Some(format!("conditional_{node_index}")),
                        op_type: "If".to_string(),
                        domain: None,
                        attribute: vec![
                            OnnxAttribute {
                                name: "then_branch".to_string(),
                                value: OnnxAttributeValue::Strings(then_branch.clone()),
                            },
                            OnnxAttribute {
                                name: "else_branch".to_string(),
                                value: OnnxAttributeValue::Strings(else_branch.clone()),
                            },
                        ],
                    };
                    onnx_nodes.push(onnx_node);
                }

                Node::Loop {
                    condition,
                    body,
                    loop_vars,
                } => {
                    // ONNX has Loop operator for dynamic control flow
                    let node_index = idx.index();
                    let onnx_node = OnnxNode {
                        input: vec![condition.clone()],
                        output: vec![node_name_map[&idx].clone()],
                        name: Some(format!("loop_{node_index}")),
                        op_type: "Loop".to_string(),
                        domain: None,
                        attribute: vec![
                            OnnxAttribute {
                                name: "body".to_string(),
                                value: OnnxAttributeValue::Strings(body.clone()),
                            },
                            OnnxAttribute {
                                name: "loop_vars".to_string(),
                                value: OnnxAttributeValue::Strings(loop_vars.clone()),
                            },
                        ],
                    };
                    onnx_nodes.push(onnx_node);
                }

                Node::Merge {
                    inputs: merge_inputs,
                } => {
                    // Use Concat operator for merging
                    let node_index = idx.index();
                    let onnx_node = OnnxNode {
                        input: merge_inputs.clone(),
                        output: vec![node_name_map[&idx].clone()],
                        name: Some(format!("merge_{node_index}")),
                        op_type: "Concat".to_string(),
                        domain: None,
                        attribute: vec![OnnxAttribute {
                            name: "axis".to_string(),
                            value: OnnxAttributeValue::Int(0), // Default to axis 0
                        }],
                    };
                    onnx_nodes.push(onnx_node);
                }

                Node::GetAttr { target, attr } => {
                    // This would typically be a parameter/constant in ONNX
                    let onnx_node = OnnxNode {
                        input: vec![target.clone()],
                        output: vec![node_name_map[&idx].clone()],
                        name: Some(format!("getattr_{}_{}", target, attr)),
                        op_type: "Identity".to_string(), // Use Identity as placeholder
                        domain: None,
                        attribute: vec![OnnxAttribute {
                            name: "attribute_name".to_string(),
                            value: OnnxAttributeValue::String(attr.clone()),
                        }],
                    };
                    onnx_nodes.push(onnx_node);
                }
            }
        }

        let onnx_graph = OnnxGraph {
            node: onnx_nodes,
            name: self.model_name.clone(),
            initializer: Vec::new(), // Would contain model parameters
            input: inputs,
            output: outputs,
            value_info: value_infos,
        };

        Ok(OnnxModel {
            ir_version: 8, // ONNX IR version
            producer_name: self.producer_name.clone(),
            producer_version: self.producer_version.clone(),
            domain: "ai.torsh".to_string(),
            model_version: 1,
            graph: onnx_graph,
        })
    }

    /// Convert a torsh operation to ONNX node
    fn convert_operation_to_onnx(
        &self,
        node_idx: NodeIndex,
        op_name: &str,
        args: &[String],
        graph: &FxGraph,
        node_name_map: &HashMap<NodeIndex, String>,
    ) -> TorshResult<OnnxNode> {
        let node_name = &node_name_map[&node_idx];

        // Get input names from predecessor nodes
        let predecessors: Vec<_> = graph
            .graph
            .neighbors_directed(node_idx, petgraph::Direction::Incoming)
            .collect();

        let inputs: Vec<String> = predecessors
            .iter()
            .map(|&pred_idx| node_name_map[&pred_idx].clone())
            .collect();

        let (onnx_op_type, attributes) = self.map_operation_to_onnx(op_name, args)?;

        Ok(OnnxNode {
            input: inputs,
            output: vec![node_name.clone()],
            name: Some(format!("{}_{}", op_name, node_idx.index())),
            op_type: onnx_op_type,
            domain: None,
            attribute: attributes,
        })
    }

    /// Map torsh operations to ONNX operations
    fn map_operation_to_onnx(
        &self,
        op_name: &str,
        args: &[String],
    ) -> TorshResult<(String, Vec<OnnxAttribute>)> {
        match op_name {
            // Activation functions
            "relu" => Ok(("Relu".to_string(), vec![])),
            "sigmoid" => Ok(("Sigmoid".to_string(), vec![])),
            "tanh" => Ok(("Tanh".to_string(), vec![])),
            "gelu" => Ok(("Gelu".to_string(), vec![])),
            "softmax" => Ok((
                "Softmax".to_string(),
                vec![OnnxAttribute {
                    name: "axis".to_string(),
                    value: OnnxAttributeValue::Int(-1), // Default to last axis
                }],
            )),

            // Math operations
            "add" => Ok(("Add".to_string(), vec![])),
            "sub" => Ok(("Sub".to_string(), vec![])),
            "mul" => Ok(("Mul".to_string(), vec![])),
            "div" => Ok(("Div".to_string(), vec![])),
            "matmul" => Ok(("MatMul".to_string(), vec![])),

            // Tensor operations
            "reshape" => {
                // Extract shape from args
                let shape_args = if args.len() > 1 {
                    args[1..]
                        .iter()
                        .filter_map(|s| s.parse::<i64>().ok())
                        .collect()
                } else {
                    vec![-1] // Dynamic reshape
                };

                Ok((
                    "Reshape".to_string(),
                    vec![OnnxAttribute {
                        name: "shape".to_string(),
                        value: OnnxAttributeValue::Ints(shape_args),
                    }],
                ))
            }

            "transpose" => {
                let perm = if args.len() > 1 {
                    args[1..]
                        .iter()
                        .filter_map(|s| s.parse::<i64>().ok())
                        .collect()
                } else {
                    vec![] // Default transpose
                };

                Ok((
                    "Transpose".to_string(),
                    vec![OnnxAttribute {
                        name: "perm".to_string(),
                        value: OnnxAttributeValue::Ints(perm),
                    }],
                ))
            }

            "squeeze" => Ok(("Squeeze".to_string(), vec![])),
            "unsqueeze" => Ok(("Unsqueeze".to_string(), vec![])),

            // Neural network operations
            "conv2d" => Ok((
                "Conv".to_string(),
                vec![
                    OnnxAttribute {
                        name: "kernel_shape".to_string(),
                        value: OnnxAttributeValue::Ints(vec![3, 3]), // Default 3x3
                    },
                    OnnxAttribute {
                        name: "pads".to_string(),
                        value: OnnxAttributeValue::Ints(vec![1, 1, 1, 1]), // Default padding
                    },
                ],
            )),

            "max_pool2d" => Ok((
                "MaxPool".to_string(),
                vec![OnnxAttribute {
                    name: "kernel_shape".to_string(),
                    value: OnnxAttributeValue::Ints(vec![2, 2]), // Default 2x2
                }],
            )),

            "avg_pool2d" => Ok((
                "AveragePool".to_string(),
                vec![OnnxAttribute {
                    name: "kernel_shape".to_string(),
                    value: OnnxAttributeValue::Ints(vec![2, 2]), // Default 2x2
                }],
            )),

            "batch_norm" => Ok(("BatchNormalization".to_string(), vec![])),
            "dropout" => Ok(("Dropout".to_string(), vec![])),

            // Linear/Dense layer
            "linear" => Ok(("MatMul".to_string(), vec![])), // Linear is often just MatMul + Add

            // Default case
            _ => Ok(("Identity".to_string(), vec![])), // Use Identity as fallback
        }
    }

    /// Convert torsh DType to ONNX data type enum
    fn dtype_to_onnx_type(&self, dtype: DType) -> i32 {
        match dtype {
            DType::F32 => 1,  // FLOAT
            DType::F64 => 11, // DOUBLE
            DType::I32 => 6,  // INT32
            DType::I64 => 7,  // INT64
            DType::U8 => 2,   // UINT8
            DType::I8 => 3,   // INT8
            DType::I16 => 5,  // INT16
            DType::Bool => 9, // BOOL
            _ => 1,           // Default to FLOAT
        }
    }

    /// Export model to ONNX JSON format
    pub fn export_to_json(&self, graph: &FxGraph) -> TorshResult<String> {
        let model = self.export(graph)?;
        serde_json::to_string_pretty(&model).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize ONNX model to JSON: {}", e))
        })
    }

    /// Export model to ONNX binary format (protobuf would be used in real implementation)
    pub fn export_to_binary(&self, graph: &FxGraph) -> TorshResult<Vec<u8>> {
        let model = self.export(graph)?;
        oxicode::serde::encode_to_vec(&model, oxicode::config::standard()).map_err(|e| {
            TorshError::SerializationError(format!(
                "Failed to serialize ONNX model to binary: {}",
                e
            ))
        })
    }
}

/// Convenience function to export a graph to ONNX
pub fn export_to_onnx(graph: &FxGraph, model_name: Option<String>) -> TorshResult<OnnxModel> {
    let exporter = if let Some(name) = model_name {
        OnnxExporter::new().with_model_name(name)
    } else {
        OnnxExporter::new()
    };

    exporter.export(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, Node};

    #[test]
    fn test_basic_onnx_export() {
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

        let exporter = OnnxExporter::new();
        let onnx_model = exporter.export(&graph).unwrap();

        assert_eq!(onnx_model.producer_name, "torsh-fx");
        assert_eq!(onnx_model.graph.input.len(), 1);
        assert_eq!(onnx_model.graph.output.len(), 1);
        assert!(onnx_model.graph.node.iter().any(|n| n.op_type == "Relu"));
    }

    #[test]
    fn test_operation_mapping() {
        let exporter = OnnxExporter::new();

        let (op_type, _) = exporter.map_operation_to_onnx("relu", &[]).unwrap();
        assert_eq!(op_type, "Relu");

        let (op_type, _) = exporter.map_operation_to_onnx("add", &[]).unwrap();
        assert_eq!(op_type, "Add");

        let (op_type, attrs) = exporter.map_operation_to_onnx("softmax", &[]).unwrap();
        assert_eq!(op_type, "Softmax");
        assert!(!attrs.is_empty());
    }

    #[test]
    fn test_dtype_conversion() {
        let exporter = OnnxExporter::new();

        assert_eq!(exporter.dtype_to_onnx_type(DType::F32), 1);
        assert_eq!(exporter.dtype_to_onnx_type(DType::F64), 11);
        assert_eq!(exporter.dtype_to_onnx_type(DType::I32), 6);
    }

    #[test]
    fn test_json_export() {
        let mut graph = FxGraph::new();
        let input = graph.graph.add_node(Node::Input("x".to_string()));
        let output = graph.graph.add_node(Node::Output);

        graph.graph.add_edge(
            input,
            output,
            Edge {
                name: "x".to_string(),
            },
        );
        graph.inputs.push(input);
        graph.outputs.push(output);

        let exporter = OnnxExporter::new();
        let json = exporter.export_to_json(&graph).unwrap();

        assert!(json.contains("torsh-fx"));
        assert!(json.contains("exported_model"));
    }
}
