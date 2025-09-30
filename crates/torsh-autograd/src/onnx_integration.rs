use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::Path;

use crate::context::AutogradContext;

#[derive(Debug, Clone)]
pub struct OnnxGraph {
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
    pub initializers: Vec<OnnxTensor>,
    pub value_info: Vec<OnnxValueInfo>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttribute>,
}

#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    pub name: String,
    pub data_type: OnnxDataType,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub data_type: OnnxDataType,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(OnnxTensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OnnxDataType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
    String,
    Complex64,
    Complex128,
}

#[derive(Debug, Clone)]
pub struct AutogradGraphNode {
    pub id: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub gradient_fn: Option<String>,
    pub requires_grad: bool,
}

#[derive(Debug, Clone)]
pub struct AutogradGraphExporter {
    pub graph: OnnxGraph,
    pub node_mapping: HashMap<String, String>,
    pub gradient_nodes: HashMap<String, AutogradGraphNode>,
}

#[derive(Debug, Clone)]
pub struct AutogradGraphImporter {
    pub onnx_graph: OnnxGraph,
    pub autograd_nodes: HashMap<String, AutogradGraphNode>,
    pub tensor_mapping: HashMap<String, String>,
}

pub struct OnnxExportConfig {
    pub export_gradients: bool,
    pub include_training_info: bool,
    pub opset_version: i64,
    pub optimization_level: OptimizationLevel,
}

pub struct OnnxImportConfig {
    pub import_gradients: bool,
    pub validate_graph: bool,
    pub device: String,
    pub optimization_level: OptimizationLevel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Extended,
    All,
}

impl Default for OnnxExportConfig {
    fn default() -> Self {
        Self {
            export_gradients: true,
            include_training_info: true,
            opset_version: 17,
            optimization_level: OptimizationLevel::Basic,
        }
    }
}

impl Default for OnnxImportConfig {
    fn default() -> Self {
        Self {
            import_gradients: true,
            validate_graph: true,
            device: "cpu".to_string(),
            optimization_level: OptimizationLevel::Basic,
        }
    }
}

impl AutogradGraphExporter {
    pub fn new() -> Self {
        Self {
            graph: OnnxGraph {
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                initializers: Vec::new(),
                value_info: Vec::new(),
                name: "autograd_graph".to_string(),
            },
            node_mapping: HashMap::new(),
            gradient_nodes: HashMap::new(),
        }
    }

    pub fn export_to_onnx(
        &mut self,
        context: &AutogradContext,
        config: &OnnxExportConfig,
    ) -> Result<OnnxGraph, OnnxError> {
        self.convert_autograd_nodes_to_onnx(context, config)?;

        if config.export_gradients {
            self.add_gradient_computation_nodes(config)?;
        }

        if config.include_training_info {
            self.add_training_information(config)?;
        }

        self.optimize_graph(config)?;

        Ok(self.graph.clone())
    }

    fn convert_autograd_nodes_to_onnx(
        &mut self,
        context: &AutogradContext,
        config: &OnnxExportConfig,
    ) -> Result<(), OnnxError> {
        let graph_nodes = self.extract_graph_nodes(context)?;

        for node in graph_nodes {
            let onnx_node = self.convert_node_to_onnx(&node, config)?;
            self.graph.nodes.push(onnx_node);

            if config.export_gradients && node.requires_grad {
                self.gradient_nodes.insert(node.id.clone(), node);
            }
        }

        Ok(())
    }

    fn extract_graph_nodes(
        &self,
        _context: &AutogradContext,
    ) -> Result<Vec<AutogradGraphNode>, OnnxError> {
        let mut nodes = Vec::new();

        nodes.push(AutogradGraphNode {
            id: "input_0".to_string(),
            op_type: "Identity".to_string(),
            inputs: vec![],
            outputs: vec!["input_0".to_string()],
            gradient_fn: None,
            requires_grad: true,
        });

        nodes.push(AutogradGraphNode {
            id: "add_1".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["input_0".to_string(), "input_1".to_string()],
            outputs: vec!["add_1".to_string()],
            gradient_fn: Some("AddBackward".to_string()),
            requires_grad: true,
        });

        Ok(nodes)
    }

    fn convert_node_to_onnx(
        &mut self,
        node: &AutogradGraphNode,
        config: &OnnxExportConfig,
    ) -> Result<OnnxNode, OnnxError> {
        let onnx_node = OnnxNode {
            op_type: self.map_op_type(&node.op_type)?,
            name: node.id.clone(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            attributes: self.create_node_attributes(node, config)?,
        };

        self.node_mapping
            .insert(node.id.clone(), onnx_node.name.clone());

        Ok(onnx_node)
    }

    fn map_op_type(&self, op_type: &str) -> Result<String, OnnxError> {
        let mapped = match op_type {
            "Identity" => "Identity",
            "Add" => "Add",
            "Sub" => "Sub",
            "Mul" => "Mul",
            "Div" => "Div",
            "MatMul" => "MatMul",
            "Conv2D" => "Conv",
            "Relu" => "Relu",
            "Softmax" => "Softmax",
            "Sigmoid" => "Sigmoid",
            "Tanh" => "Tanh",
            "BatchNorm" => "BatchNormalization",
            "Dropout" => "Dropout",
            "MaxPool" => "MaxPool",
            "AveragePool" => "AveragePool",
            "GlobalMaxPool" => "GlobalMaxPool",
            "GlobalAveragePool" => "GlobalAveragePool",
            "Reshape" => "Reshape",
            "Transpose" => "Transpose",
            "Concat" => "Concat",
            "Split" => "Split",
            "Gather" => "Gather",
            "Scatter" => "Scatter",
            _ => {
                return Err(OnnxError::UnsupportedOperation(format!(
                    "Unsupported operation: {}",
                    op_type
                )))
            }
        };

        Ok(mapped.to_string())
    }

    fn create_node_attributes(
        &self,
        node: &AutogradGraphNode,
        config: &OnnxExportConfig,
    ) -> Result<HashMap<String, OnnxAttribute>, OnnxError> {
        let mut attributes = HashMap::new();

        if config.export_gradients && node.requires_grad {
            attributes.insert("requires_grad".to_string(), OnnxAttribute::Int(1));
        }

        if let Some(grad_fn) = &node.gradient_fn {
            attributes.insert(
                "gradient_fn".to_string(),
                OnnxAttribute::String(grad_fn.clone()),
            );
        }

        match node.op_type.as_str() {
            "Conv2D" => {
                attributes.insert("kernel_shape".to_string(), OnnxAttribute::Ints(vec![3, 3]));
                attributes.insert("pads".to_string(), OnnxAttribute::Ints(vec![1, 1, 1, 1]));
                attributes.insert("strides".to_string(), OnnxAttribute::Ints(vec![1, 1]));
            }
            "BatchNorm" => {
                attributes.insert("epsilon".to_string(), OnnxAttribute::Float(1e-5));
                attributes.insert("momentum".to_string(), OnnxAttribute::Float(0.9));
            }
            "Dropout" => {
                attributes.insert("ratio".to_string(), OnnxAttribute::Float(0.5));
            }
            "MaxPool" | "AveragePool" => {
                attributes.insert("kernel_shape".to_string(), OnnxAttribute::Ints(vec![2, 2]));
                attributes.insert("strides".to_string(), OnnxAttribute::Ints(vec![2, 2]));
            }
            _ => {}
        }

        Ok(attributes)
    }

    fn add_gradient_computation_nodes(
        &mut self,
        config: &OnnxExportConfig,
    ) -> Result<(), OnnxError> {
        let gradient_nodes: Vec<_> = self.gradient_nodes.values().cloned().collect();

        for node in gradient_nodes {
            let grad_node = self.create_gradient_node(&node, config)?;
            self.graph.nodes.push(grad_node);
        }

        Ok(())
    }

    fn create_gradient_node(
        &self,
        node: &AutogradGraphNode,
        _config: &OnnxExportConfig,
    ) -> Result<OnnxNode, OnnxError> {
        let grad_op_type = match node.op_type.as_str() {
            "Add" => "AddGrad",
            "Sub" => "SubGrad",
            "Mul" => "MulGrad",
            "Div" => "DivGrad",
            "MatMul" => "MatMulGrad",
            "Conv2D" => "ConvGrad",
            "Relu" => "ReluGrad",
            "Sigmoid" => "SigmoidGrad",
            "Tanh" => "TanhGrad",
            _ => "GenericGrad",
        };

        let grad_node = OnnxNode {
            op_type: grad_op_type.to_string(),
            name: format!("{}_grad", node.id),
            inputs: vec![format!("{}_grad_input", node.id)],
            outputs: vec![format!("{}_grad_output", node.id)],
            attributes: HashMap::new(),
        };

        Ok(grad_node)
    }

    fn add_training_information(&mut self, config: &OnnxExportConfig) -> Result<(), OnnxError> {
        let training_info = OnnxAttribute::String("training_mode".to_string());
        let opset_version = OnnxAttribute::Int(config.opset_version);

        for node in &mut self.graph.nodes {
            if !node.attributes.contains_key("training_info") {
                node.attributes
                    .insert("training_info".to_string(), training_info.clone());
            }
            if !node.attributes.contains_key("opset_version") {
                node.attributes
                    .insert("opset_version".to_string(), opset_version.clone());
            }
        }

        Ok(())
    }

    fn optimize_graph(&mut self, config: &OnnxExportConfig) -> Result<(), OnnxError> {
        match config.optimization_level {
            OptimizationLevel::None => {}
            OptimizationLevel::Basic => {
                self.remove_identity_nodes()?;
                self.merge_consecutive_operations()?;
            }
            OptimizationLevel::Extended => {
                self.remove_identity_nodes()?;
                self.merge_consecutive_operations()?;
                self.fold_constants()?;
                self.eliminate_dead_code()?;
            }
            OptimizationLevel::All => {
                self.remove_identity_nodes()?;
                self.merge_consecutive_operations()?;
                self.fold_constants()?;
                self.eliminate_dead_code()?;
                self.fuse_operations()?;
            }
        }

        Ok(())
    }

    fn remove_identity_nodes(&mut self) -> Result<(), OnnxError> {
        let mut nodes_to_remove = Vec::new();
        let mut input_output_mapping = HashMap::new();

        for (i, node) in self.graph.nodes.iter().enumerate() {
            if node.op_type == "Identity" && node.inputs.len() == 1 && node.outputs.len() == 1 {
                nodes_to_remove.push(i);
                input_output_mapping.insert(node.outputs[0].clone(), node.inputs[0].clone());
            }
        }

        for &index in nodes_to_remove.iter().rev() {
            self.graph.nodes.remove(index);
        }

        for node in &mut self.graph.nodes {
            for input in &mut node.inputs {
                if let Some(replacement) = input_output_mapping.get(input) {
                    *input = replacement.clone();
                }
            }
        }

        Ok(())
    }

    fn merge_consecutive_operations(&mut self) -> Result<(), OnnxError> {
        let mut merged_any = true;

        while merged_any {
            merged_any = false;

            for i in 0..self.graph.nodes.len() {
                for j in (i + 1)..self.graph.nodes.len() {
                    if self.can_merge_nodes(&self.graph.nodes[i], &self.graph.nodes[j]) {
                        let merged_node =
                            self.merge_nodes(&self.graph.nodes[i], &self.graph.nodes[j])?;
                        self.graph.nodes.remove(j);
                        self.graph.nodes.remove(i);
                        self.graph.nodes.push(merged_node);
                        merged_any = true;
                        break;
                    }
                }
                if merged_any {
                    break;
                }
            }
        }

        Ok(())
    }

    fn can_merge_nodes(&self, node1: &OnnxNode, node2: &OnnxNode) -> bool {
        if node1.op_type == "Add" && node2.op_type == "Add" {
            return node1
                .outputs
                .iter()
                .any(|output| node2.inputs.contains(output));
        }

        if node1.op_type == "Mul" && node2.op_type == "Mul" {
            return node1
                .outputs
                .iter()
                .any(|output| node2.inputs.contains(output));
        }

        false
    }

    fn merge_nodes(&self, node1: &OnnxNode, node2: &OnnxNode) -> Result<OnnxNode, OnnxError> {
        let merged_name = format!("{}_{}_merged", node1.name, node2.name);

        let mut merged_inputs = node1.inputs.clone();
        for input in &node2.inputs {
            if !node1.outputs.contains(input) && !merged_inputs.contains(input) {
                merged_inputs.push(input.clone());
            }
        }

        let merged_outputs = node2.outputs.clone();

        let merged_node = OnnxNode {
            op_type: format!("{}_{}", node1.op_type, node2.op_type),
            name: merged_name,
            inputs: merged_inputs,
            outputs: merged_outputs,
            attributes: node1.attributes.clone(),
        };

        Ok(merged_node)
    }

    fn fold_constants(&mut self) -> Result<(), OnnxError> {
        let mut constant_values = HashMap::new();

        for initializer in &self.graph.initializers {
            constant_values.insert(initializer.name.clone(), initializer.clone());
        }

        let mut nodes_to_remove = Vec::new();

        for (i, node) in self.graph.nodes.iter().enumerate() {
            if self.can_fold_node(node, &constant_values) {
                let folded_value = self.fold_node(node, &constant_values)?;
                self.graph.initializers.push(folded_value);
                nodes_to_remove.push(i);
            }
        }

        for &index in nodes_to_remove.iter().rev() {
            self.graph.nodes.remove(index);
        }

        Ok(())
    }

    fn can_fold_node(
        &self,
        node: &OnnxNode,
        constant_values: &HashMap<String, OnnxTensor>,
    ) -> bool {
        match node.op_type.as_str() {
            "Add" | "Sub" | "Mul" | "Div" => node
                .inputs
                .iter()
                .all(|input| constant_values.contains_key(input)),
            _ => false,
        }
    }

    fn fold_node(
        &self,
        node: &OnnxNode,
        _constant_values: &HashMap<String, OnnxTensor>,
    ) -> Result<OnnxTensor, OnnxError> {
        let folded_tensor = OnnxTensor {
            name: node.outputs[0].clone(),
            data_type: OnnxDataType::Float32,
            shape: vec![1],
            data: vec![0u8; 4],
        };

        Ok(folded_tensor)
    }

    fn eliminate_dead_code(&mut self) -> Result<(), OnnxError> {
        let mut used_outputs = HashSet::new();

        for output in &self.graph.outputs {
            used_outputs.insert(output.name.clone());
        }

        for node in &self.graph.nodes {
            for input in &node.inputs {
                used_outputs.insert(input.clone());
            }
        }

        self.graph.nodes.retain(|node| {
            node.outputs
                .iter()
                .any(|output| used_outputs.contains(output))
        });

        Ok(())
    }

    fn fuse_operations(&mut self) -> Result<(), OnnxError> {
        let mut fused_any = true;

        while fused_any {
            fused_any = false;

            for i in 0..self.graph.nodes.len() {
                for j in (i + 1)..self.graph.nodes.len() {
                    if self.can_fuse_nodes(&self.graph.nodes[i], &self.graph.nodes[j]) {
                        let fused_node =
                            self.fuse_nodes(&self.graph.nodes[i], &self.graph.nodes[j])?;
                        self.graph.nodes.remove(j);
                        self.graph.nodes.remove(i);
                        self.graph.nodes.push(fused_node);
                        fused_any = true;
                        break;
                    }
                }
                if fused_any {
                    break;
                }
            }
        }

        Ok(())
    }

    fn can_fuse_nodes(&self, node1: &OnnxNode, node2: &OnnxNode) -> bool {
        matches!(
            (node1.op_type.as_str(), node2.op_type.as_str()),
            ("Conv", "Relu") | ("MatMul", "Add") | ("Add", "Relu") | ("BatchNormalization", "Relu")
        )
    }

    fn fuse_nodes(&self, node1: &OnnxNode, node2: &OnnxNode) -> Result<OnnxNode, OnnxError> {
        let fused_name = format!("{}_{}_fused", node1.name, node2.name);
        let fused_op_type = format!("{}_{}", node1.op_type, node2.op_type);

        let mut fused_attributes = node1.attributes.clone();
        for (key, value) in &node2.attributes {
            fused_attributes.insert(key.clone(), value.clone());
        }

        let fused_node = OnnxNode {
            op_type: fused_op_type,
            name: fused_name,
            inputs: node1.inputs.clone(),
            outputs: node2.outputs.clone(),
            attributes: fused_attributes,
        };

        Ok(fused_node)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), OnnxError> {
        let serialized = self.serialize_graph()?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    fn serialize_graph(&self) -> Result<Vec<u8>, OnnxError> {
        let mut data = Vec::new();

        data.extend_from_slice(b"ONNX_GRAPH");
        data.extend_from_slice(&(self.graph.nodes.len() as u32).to_le_bytes());

        for node in &self.graph.nodes {
            let node_data = self.serialize_node(node)?;
            data.extend_from_slice(&(node_data.len() as u32).to_le_bytes());
            data.extend_from_slice(&node_data);
        }

        Ok(data)
    }

    fn serialize_node(&self, node: &OnnxNode) -> Result<Vec<u8>, OnnxError> {
        let mut data = Vec::new();

        data.extend_from_slice(node.op_type.as_bytes());
        data.push(0);
        data.extend_from_slice(node.name.as_bytes());
        data.push(0);

        data.extend_from_slice(&(node.inputs.len() as u32).to_le_bytes());
        for input in &node.inputs {
            data.extend_from_slice(input.as_bytes());
            data.push(0);
        }

        data.extend_from_slice(&(node.outputs.len() as u32).to_le_bytes());
        for output in &node.outputs {
            data.extend_from_slice(output.as_bytes());
            data.push(0);
        }

        Ok(data)
    }
}

impl AutogradGraphImporter {
    pub fn new() -> Self {
        Self {
            onnx_graph: OnnxGraph {
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                initializers: Vec::new(),
                value_info: Vec::new(),
                name: String::new(),
            },
            autograd_nodes: HashMap::new(),
            tensor_mapping: HashMap::new(),
        }
    }

    pub fn import_from_onnx(
        &mut self,
        onnx_graph: OnnxGraph,
        config: &OnnxImportConfig,
    ) -> Result<Vec<AutogradGraphNode>, OnnxError> {
        self.onnx_graph = onnx_graph;

        if config.validate_graph {
            self.validate_onnx_graph()?;
        }

        let autograd_nodes = self.convert_onnx_nodes_to_autograd(config)?;

        if config.import_gradients {
            self.setup_gradient_computation(&autograd_nodes)?;
        }

        Ok(autograd_nodes)
    }

    fn validate_onnx_graph(&self) -> Result<(), OnnxError> {
        if self.onnx_graph.nodes.is_empty() {
            return Err(OnnxError::InvalidGraph("Empty graph".to_string()));
        }

        let mut defined_values = HashSet::new();
        let mut used_values = HashSet::new();

        for input in &self.onnx_graph.inputs {
            defined_values.insert(input.name.clone());
        }

        for initializer in &self.onnx_graph.initializers {
            defined_values.insert(initializer.name.clone());
        }

        for node in &self.onnx_graph.nodes {
            for input in &node.inputs {
                used_values.insert(input.clone());
            }

            for output in &node.outputs {
                defined_values.insert(output.clone());
            }
        }

        for used_value in &used_values {
            if !defined_values.contains(used_value) {
                return Err(OnnxError::InvalidGraph(format!(
                    "Undefined value: {}",
                    used_value
                )));
            }
        }

        Ok(())
    }

    fn convert_onnx_nodes_to_autograd(
        &mut self,
        config: &OnnxImportConfig,
    ) -> Result<Vec<AutogradGraphNode>, OnnxError> {
        let mut autograd_nodes = Vec::new();

        // Clone the nodes to avoid borrow conflicts
        let nodes = self.onnx_graph.nodes.clone();
        for onnx_node in &nodes {
            let autograd_node = self.convert_onnx_node_to_autograd(onnx_node, config)?;
            autograd_nodes.push(autograd_node);
        }

        Ok(autograd_nodes)
    }

    fn convert_onnx_node_to_autograd(
        &mut self,
        onnx_node: &OnnxNode,
        config: &OnnxImportConfig,
    ) -> Result<AutogradGraphNode, OnnxError> {
        let requires_grad = if config.import_gradients {
            onnx_node
                .attributes
                .get("requires_grad")
                .map(|attr| match attr {
                    OnnxAttribute::Int(val) => *val != 0,
                    _ => false,
                })
                .unwrap_or(false)
        } else {
            false
        };

        let gradient_fn = if config.import_gradients {
            onnx_node
                .attributes
                .get("gradient_fn")
                .and_then(|attr| match attr {
                    OnnxAttribute::String(s) => Some(s.clone()),
                    _ => None,
                })
        } else {
            None
        };

        let autograd_node = AutogradGraphNode {
            id: onnx_node.name.clone(),
            op_type: self.map_onnx_op_type_to_autograd(&onnx_node.op_type)?,
            inputs: onnx_node.inputs.clone(),
            outputs: onnx_node.outputs.clone(),
            gradient_fn,
            requires_grad,
        };

        self.autograd_nodes
            .insert(onnx_node.name.clone(), autograd_node.clone());

        Ok(autograd_node)
    }

    fn map_onnx_op_type_to_autograd(&self, onnx_op_type: &str) -> Result<String, OnnxError> {
        let mapped = match onnx_op_type {
            "Identity" => "Identity",
            "Add" => "Add",
            "Sub" => "Sub",
            "Mul" => "Mul",
            "Div" => "Div",
            "MatMul" => "MatMul",
            "Conv" => "Conv2D",
            "Relu" => "Relu",
            "Sigmoid" => "Sigmoid",
            "Tanh" => "Tanh",
            "Softmax" => "Softmax",
            "BatchNormalization" => "BatchNorm",
            "Dropout" => "Dropout",
            "MaxPool" => "MaxPool",
            "AveragePool" => "AveragePool",
            "GlobalMaxPool" => "GlobalMaxPool",
            "GlobalAveragePool" => "GlobalAveragePool",
            "Reshape" => "Reshape",
            "Transpose" => "Transpose",
            "Concat" => "Concat",
            "Split" => "Split",
            "Gather" => "Gather",
            "Scatter" => "Scatter",
            _ => {
                return Err(OnnxError::UnsupportedOperation(format!(
                    "Unsupported ONNX operation: {}",
                    onnx_op_type
                )))
            }
        };

        Ok(mapped.to_string())
    }

    fn setup_gradient_computation(
        &mut self,
        autograd_nodes: &[AutogradGraphNode],
    ) -> Result<(), OnnxError> {
        for node in autograd_nodes {
            if node.requires_grad {
                self.create_backward_function_for_node(node)?;
            }
        }

        Ok(())
    }

    fn create_backward_function_for_node(
        &mut self,
        node: &AutogradGraphNode,
    ) -> Result<(), OnnxError> {
        let backward_function = match node.op_type.as_str() {
            "Add" => "AddBackward",
            "Sub" => "SubBackward",
            "Mul" => "MulBackward",
            "Div" => "DivBackward",
            "MatMul" => "MatMulBackward",
            "Conv2D" => "Conv2DBackward",
            "Relu" => "ReluBackward",
            "Sigmoid" => "SigmoidBackward",
            "Tanh" => "TanhBackward",
            _ => "GenericBackward",
        };

        self.tensor_mapping.insert(
            format!("{}_backward", node.id),
            backward_function.to_string(),
        );

        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<OnnxGraph, OnnxError> {
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let graph = self.deserialize_graph(&buffer)?;
        self.onnx_graph = graph.clone();

        Ok(graph)
    }

    fn deserialize_graph(&self, data: &[u8]) -> Result<OnnxGraph, OnnxError> {
        if data.len() < 14 {
            return Err(OnnxError::InvalidFormat("Insufficient data".to_string()));
        }

        let header = &data[0..10];
        if header != b"ONNX_GRAPH" {
            return Err(OnnxError::InvalidFormat("Invalid header".to_string()));
        }

        let node_count = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
        let mut nodes = Vec::new();
        let mut offset = 14;

        for _ in 0..node_count {
            if offset + 4 > data.len() {
                return Err(OnnxError::InvalidFormat(
                    "Insufficient data for node size".to_string(),
                ));
            }

            let node_size = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + node_size > data.len() {
                return Err(OnnxError::InvalidFormat(
                    "Insufficient data for node".to_string(),
                ));
            }

            let node_data = &data[offset..offset + node_size];
            let node = self.deserialize_node(node_data)?;
            nodes.push(node);

            offset += node_size;
        }

        Ok(OnnxGraph {
            nodes,
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
            value_info: Vec::new(),
            name: "imported_graph".to_string(),
        })
    }

    fn deserialize_node(&self, data: &[u8]) -> Result<OnnxNode, OnnxError> {
        let mut offset = 0;

        let op_type_end = data[offset..]
            .iter()
            .position(|&b| b == 0)
            .ok_or(OnnxError::InvalidFormat("Op type not found".to_string()))?;
        let op_type = String::from_utf8(data[offset..offset + op_type_end].to_vec())
            .map_err(|_| OnnxError::InvalidFormat("Invalid op type".to_string()))?;
        offset += op_type_end + 1;

        let name_end = data[offset..]
            .iter()
            .position(|&b| b == 0)
            .ok_or(OnnxError::InvalidFormat("Name not found".to_string()))?;
        let name = String::from_utf8(data[offset..offset + name_end].to_vec())
            .map_err(|_| OnnxError::InvalidFormat("Invalid name".to_string()))?;
        offset += name_end + 1;

        if offset + 4 > data.len() {
            return Err(OnnxError::InvalidFormat(
                "Insufficient data for input count".to_string(),
            ));
        }

        let input_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut inputs = Vec::new();
        for _ in 0..input_count {
            let input_end = data[offset..]
                .iter()
                .position(|&b| b == 0)
                .ok_or(OnnxError::InvalidFormat("Input not found".to_string()))?;
            let input = String::from_utf8(data[offset..offset + input_end].to_vec())
                .map_err(|_| OnnxError::InvalidFormat("Invalid input".to_string()))?;
            inputs.push(input);
            offset += input_end + 1;
        }

        if offset + 4 > data.len() {
            return Err(OnnxError::InvalidFormat(
                "Insufficient data for output count".to_string(),
            ));
        }

        let output_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut outputs = Vec::new();
        for _ in 0..output_count {
            let output_end = data[offset..]
                .iter()
                .position(|&b| b == 0)
                .ok_or(OnnxError::InvalidFormat("Output not found".to_string()))?;
            let output = String::from_utf8(data[offset..offset + output_end].to_vec())
                .map_err(|_| OnnxError::InvalidFormat("Invalid output".to_string()))?;
            outputs.push(output);
            offset += output_end + 1;
        }

        Ok(OnnxNode {
            op_type,
            name,
            inputs,
            outputs,
            attributes: HashMap::new(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum OnnxError {
    InvalidGraph(String),
    InvalidFormat(String),
    UnsupportedOperation(String),
    SerializationError(String),
    IoError(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::InvalidGraph(msg) => write!(f, "Invalid graph: {}", msg),
            OnnxError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            OnnxError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            OnnxError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            OnnxError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<std::io::Error> for OnnxError {
    fn from(error: std::io::Error) -> Self {
        OnnxError::IoError(error.to_string())
    }
}

pub fn export_autograd_graph_to_onnx(
    context: &AutogradContext,
    config: Option<OnnxExportConfig>,
) -> Result<OnnxGraph, OnnxError> {
    let mut exporter = AutogradGraphExporter::new();
    let config = config.unwrap_or_default();
    exporter.export_to_onnx(context, &config)
}

pub fn import_onnx_graph_to_autograd(
    onnx_graph: OnnxGraph,
    config: Option<OnnxImportConfig>,
) -> Result<Vec<AutogradGraphNode>, OnnxError> {
    let mut importer = AutogradGraphImporter::new();
    let config = config.unwrap_or_default();
    importer.import_from_onnx(onnx_graph, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_graph_creation() {
        let graph = OnnxGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            initializers: vec![],
            value_info: vec![],
            name: "test_graph".to_string(),
        };

        assert_eq!(graph.name, "test_graph");
        assert!(graph.nodes.is_empty());
    }

    #[test]
    fn test_autograd_graph_exporter() {
        let exporter = AutogradGraphExporter::new();
        assert_eq!(exporter.graph.name, "autograd_graph");
        assert!(exporter.node_mapping.is_empty());
    }

    #[test]
    fn test_autograd_graph_importer() {
        let importer = AutogradGraphImporter::new();
        assert!(importer.onnx_graph.nodes.is_empty());
        assert!(importer.autograd_nodes.is_empty());
    }

    #[test]
    fn test_op_type_mapping() {
        let exporter = AutogradGraphExporter::new();

        assert_eq!(exporter.map_op_type("Add").unwrap(), "Add");
        assert_eq!(exporter.map_op_type("Conv2D").unwrap(), "Conv");
        assert_eq!(exporter.map_op_type("Relu").unwrap(), "Relu");

        assert!(exporter.map_op_type("UnknownOp").is_err());
    }

    #[test]
    fn test_onnx_node_creation() {
        let node = OnnxNode {
            op_type: "Add".to_string(),
            name: "add_node".to_string(),
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
            attributes: HashMap::new(),
        };

        assert_eq!(node.op_type, "Add");
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);
    }

    #[test]
    fn test_autograd_graph_node_creation() {
        let node = AutogradGraphNode {
            id: "node_1".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["input1".to_string()],
            outputs: vec!["output1".to_string()],
            gradient_fn: Some("AddBackward".to_string()),
            requires_grad: true,
        };

        assert_eq!(node.id, "node_1");
        assert_eq!(node.op_type, "Add");
        assert!(node.requires_grad);
    }

    #[test]
    fn test_export_config() {
        let config = OnnxExportConfig::default();
        assert!(config.export_gradients);
        assert!(config.include_training_info);
        assert_eq!(config.opset_version, 17);
    }

    #[test]
    fn test_import_config() {
        let config = OnnxImportConfig::default();
        assert!(config.import_gradients);
        assert!(config.validate_graph);
        assert_eq!(config.device, "cpu");
    }

    #[test]
    fn test_optimization_levels() {
        assert_eq!(OptimizationLevel::None, OptimizationLevel::None);
        assert_ne!(OptimizationLevel::Basic, OptimizationLevel::Extended);
    }

    #[test]
    fn test_onnx_data_types() {
        assert_eq!(OnnxDataType::Float32, OnnxDataType::Float32);
        assert_ne!(OnnxDataType::Float32, OnnxDataType::Float64);
    }

    #[test]
    fn test_onnx_error_display() {
        let error = OnnxError::InvalidGraph("Test error".to_string());
        assert_eq!(format!("{}", error), "Invalid graph: Test error");
    }

    #[test]
    fn test_gradient_node_creation() {
        let exporter = AutogradGraphExporter::new();
        let node = AutogradGraphNode {
            id: "add_node".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
            gradient_fn: Some("AddBackward".to_string()),
            requires_grad: true,
        };

        let grad_node = exporter
            .create_gradient_node(&node, &OnnxExportConfig::default())
            .unwrap();
        assert_eq!(grad_node.op_type, "AddGrad");
        assert_eq!(grad_node.name, "add_node_grad");
    }
}
