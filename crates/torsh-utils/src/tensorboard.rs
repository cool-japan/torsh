//! TensorBoard integration for ToRSh
//!
//! This module provides a SummaryWriter compatible with TensorBoard for logging
//! training metrics, images, graphs, and other data.

use base64::Engine;
use chrono::{DateTime, Utc};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use thiserror::Error;
use torsh_core::error::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// TensorBoard writer errors
#[derive(Error, Debug)]
pub enum TensorBoardError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Tensor error: {0}")]
    TensorError(#[from] torsh_core::TorshError),

    #[error("Base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),
}

/// Result type for TensorBoard operations
pub type Result<T> = std::result::Result<T, TensorBoardError>;

/// Event file format version
const EVENT_FILE_VERSION: &str = "brain.Event:2";

/// TensorBoard summary writer
pub struct SummaryWriter {
    #[allow(dead_code)]
    log_dir: PathBuf,
    event_writer: EventWriter,
    global_step: i64,
    tag_to_last_value: HashMap<String, f32>,
}

impl SummaryWriter {
    /// Create a new SummaryWriter
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        fs::create_dir_all(&log_dir)?;

        let event_writer = EventWriter::new(&log_dir)?;

        Ok(Self {
            log_dir,
            event_writer,
            global_step: 0,
            tag_to_last_value: HashMap::new(),
        })
    }

    /// Add a scalar value
    pub fn add_scalar(&mut self, tag: &str, value: f32, step: Option<i64>) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        // Store last value for smoothing
        self.tag_to_last_value.insert(tag.to_string(), value);

        let summary = Summary {
            tag: tag.to_string(),
            simple_value: value,
        };

        self.event_writer.write_summary(summary, step)?;

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }

    /// Add multiple scalars
    pub fn add_scalars(
        &mut self,
        main_tag: &str,
        tag_scalar_dict: HashMap<String, f32>,
        step: Option<i64>,
    ) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        for (tag, value) in tag_scalar_dict {
            let full_tag = format!("{}/{}", main_tag, tag);
            self.add_scalar(&full_tag, value, Some(step))?;
        }

        Ok(())
    }

    /// Add a histogram
    pub fn add_histogram(&mut self, tag: &str, values: &Tensor, step: Option<i64>) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        // Flatten tensor to 1D
        let values_vec = values.to_vec()?;

        let histogram = Histogram::from_values(&values_vec);
        let summary = Summary {
            tag: tag.to_string(),
            simple_value: histogram.mean,
        };

        self.event_writer.write_summary(summary, step)?;

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }

    /// Add an image
    pub fn add_image(
        &mut self,
        tag: &str,
        img_tensor: &Tensor,
        step: Option<i64>,
        dataformats: &str,
    ) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        // Validate image format
        let shape = img_tensor.shape();
        match dataformats {
            "CHW" => {
                if shape.ndim() != 3 {
                    return Err(TensorBoardError::InvalidParameter(
                        "CHW format requires 3D tensor".to_string(),
                    ));
                }
            }
            "HWC" => {
                if shape.ndim() != 3 {
                    return Err(TensorBoardError::InvalidParameter(
                        "HWC format requires 3D tensor".to_string(),
                    ));
                }
            }
            "HW" => {
                if shape.ndim() != 2 {
                    return Err(TensorBoardError::InvalidParameter(
                        "HW format requires 2D tensor".to_string(),
                    ));
                }
            }
            _ => {
                return Err(TensorBoardError::InvalidParameter(format!(
                    "Unknown data format: {}",
                    dataformats
                )))
            }
        }

        // Create proper image visualization
        let image_data = self.process_image_tensor(img_tensor, dataformats)?;

        // Save image as base64 encoded data
        let image_file = self
            .log_dir
            .join(format!("{}_step_{}.json", tag.replace('/', "_"), step));
        let image_json = json!({
            "tag": tag,
            "step": step,
            "wall_time": chrono::Utc::now().timestamp(),
            "image": {
                "data": image_data.data,
                "width": image_data.width,
                "height": image_data.height,
                "channels": image_data.channels,
                "format": dataformats
            }
        });

        std::fs::write(
            image_file,
            serde_json::to_string_pretty(&image_json)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;

        let summary = Summary {
            tag: format!("{}/image", tag),
            simple_value: image_data.channels as f32,
        };

        self.event_writer.write_summary(summary, step)?;

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }

    /// Add text
    pub fn add_text(&mut self, tag: &str, text: &str, step: Option<i64>) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        let summary = Summary {
            tag: format!("{}/text", tag),
            simple_value: text.len() as f32, // Placeholder
        };

        self.event_writer.write_summary(summary, step)?;

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }

    /// Add enhanced graph visualization with execution tracing
    pub fn add_graph(
        &mut self,
        model: &dyn torsh_nn::Module,
        input_to_model: Option<&Tensor>,
    ) -> Result<()> {
        let graph_data = self.serialize_model_graph_enhanced(model, input_to_model)?;

        // Create multiple graph representations for different views
        // TODO: Implement these view methods for enhanced graph visualization
        // self.create_architectural_view(&graph_data)?;
        // self.create_execution_view(&graph_data)?;
        // self.create_parameter_view(&graph_data)?;
        // self.create_computational_graph_view(&graph_data)?;

        let summary = Summary {
            tag: "graph/model_architecture".to_string(),
            simple_value: graph_data.nodes.len() as f32,
        };

        self.event_writer.write_summary(summary, self.global_step)?;
        Ok(())
    }

    /// Add interactive graph with dynamic execution tracing
    pub fn add_interactive_graph(
        &mut self,
        model: &dyn torsh_nn::Module,
        sample_inputs: Vec<&Tensor>,
        trace_execution: bool,
    ) -> Result<()> {
        let mut enhanced_graph =
            self.serialize_model_graph_enhanced(model, sample_inputs.first().copied())?;

        if trace_execution {
            // Trace execution for each sample input
            for (i, input) in sample_inputs.iter().enumerate() {
                // TODO: Implement trace_model_execution method
                // let execution_trace = self.trace_model_execution(model, input)?;
                let execution_trace = ExecutionTrace {
                    input_id: i,
                    timing_data: vec![],
                    memory_usage: vec![],
                    activation_shapes: std::collections::HashMap::new(),
                    gradient_flow: None,
                };
                enhanced_graph.execution_traces.push(ExecutionTrace {
                    input_id: i,
                    timing_data: execution_trace.timing_data,
                    memory_usage: execution_trace.memory_usage,
                    activation_shapes: execution_trace.activation_shapes,
                    gradient_flow: execution_trace.gradient_flow,
                });
            }
        }

        // Create interactive visualization files
        // TODO: Implement create_interactive_graph_files method
        // self.create_interactive_graph_files(&enhanced_graph)?;

        let summary = Summary {
            tag: "graph/interactive_model".to_string(),
            simple_value: enhanced_graph.nodes.len() as f32,
        };

        self.event_writer.write_summary(summary, self.global_step)?;
        Ok(())
    }

    /// Add layer-wise analysis visualization
    pub fn add_layer_analysis(
        &mut self,
        model: &dyn torsh_nn::Module,
        analysis_data: &LayerAnalysisData,
        step: Option<i64>,
    ) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        // Create comprehensive layer analysis visualization
        let analysis_file = self
            .log_dir
            .join(format!("layer_analysis_step_{}.json", step));
        let analysis_json = json!({
            "step": step,
            "timestamp": chrono::Utc::now(),
            "layer_analysis": {
                "activation_stats": analysis_data.activation_stats,
                "gradient_stats": analysis_data.gradient_stats,
                "weight_distributions": analysis_data.weight_distributions,
                "layer_connectivity": analysis_data.layer_connectivity,
                "computational_complexity": analysis_data.computational_complexity,
                "memory_footprint": analysis_data.memory_footprint
            }
        });

        std::fs::write(
            analysis_file,
            serde_json::to_string_pretty(&analysis_json)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;

        // Create visualizations for each layer
        // TODO: Implement add_layer_histogram and add_layer_weight_distribution methods
        // for layer_stats in &analysis_data.activation_stats {
        //     self.add_layer_histogram(&layer_stats.layer_name, &layer_stats.activations, step)?;
        //     self.add_layer_weight_distribution(
        //         &layer_stats.layer_name,
        //         &layer_stats.weights,
        //         step,
        //     )?;
        // }

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }

    /// Enhanced graph data structure for advanced visualization
    fn serialize_model_graph_enhanced(
        &self,
        model: &dyn torsh_nn::Module,
        input_tensor: Option<&Tensor>,
    ) -> Result<EnhancedGraphData> {
        let basic_graph = self.serialize_model_graph(model, input_tensor)?;

        Ok(EnhancedGraphData {
            nodes: basic_graph
                .nodes
                .into_iter()
                .map(|node| EnhancedGraphNode {
                    id: node.id,
                    name: node.name,
                    op_type: node.op_type,
                    shape: node.shape,
                    dtype: node.dtype,
                    device: node.device,
                    params: node.params,
                    // TODO: Implement these estimation methods
                    computational_complexity: ComputationalComplexity {
                        flops: 0,
                        memory_accesses: 0,
                        algorithmic_complexity: "O(n)".to_string(),
                        multiply_adds: 0,
                    },
                    memory_footprint: MemoryFootprint {
                        parameters_mb: 0.0,
                        activations_mb: 0.0,
                        gradients_mb: 0.0,
                        buffers_mb: 0.0,
                        total_mb: 0.0,
                    },
                    layer_type: LayerType::Custom("Unknown".to_string()),
                    optimization_hints: vec![],
                })
                .collect(),
            edges: basic_graph
                .edges
                .into_iter()
                .map(|edge| EnhancedGraphEdge {
                    from: edge.from,
                    to: edge.to,
                    tensor_name: edge.tensor_name,
                    data_flow_type: DataFlowType::Forward,
                    tensor_size_mb: 0.0, // Would be calculated from actual tensor sizes
                    communication_cost: 0.0,
                })
                .collect(),
            metadata: EnhancedGraphMetadata {
                framework: basic_graph.metadata.framework,
                version: basic_graph.metadata.version,
                total_params: basic_graph.metadata.total_params,
                trainable_params: basic_graph.metadata.trainable_params,
                model_size_mb: 0.0, // TODO: Implement estimate_model_size method
                flops: 0,           // TODO: Implement estimate_flops method
                inference_time_ms: None,
                memory_usage_mb: None,
            },
            execution_traces: vec![],
            optimization_suggestions: vec![], // TODO: Implement generate_model_optimization_suggestions method
        })
    }

    /// Serialize model graph structure
    fn serialize_model_graph(
        &self,
        model: &dyn torsh_nn::Module,
        input_tensor: Option<&Tensor>,
    ) -> Result<GraphData> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_id = 0;

        // Add input node
        if let Some(input) = input_tensor {
            nodes.push(GraphNode {
                id: node_id,
                name: "Input".to_string(),
                op_type: "Input".to_string(),
                shape: input.shape().dims().to_vec(),
                dtype: input.dtype().to_string(),
                device: input.device().to_string(),
                params: HashMap::new(),
            });
            node_id += 1;
        }

        // Extract model architecture (simplified approach)
        let parameters = model.parameters();
        let mut layer_names: HashSet<String> = HashSet::new();

        // Analyze parameter names to infer layer structure
        for (name, _param) in &parameters {
            if let Some(layer_name) = self.extract_layer_name(name) {
                layer_names.insert(layer_name);
            }
        }

        // Create nodes for each layer
        let mut prev_node_id = if input_tensor.is_some() {
            Some(0)
        } else {
            None
        };
        for layer_name in &layer_names {
            nodes.push(GraphNode {
                id: node_id,
                name: layer_name.clone(),
                op_type: self.infer_op_type(layer_name),
                shape: vec![], // Would need forward pass to determine
                dtype: "f32".to_string(),
                device: "cpu".to_string(), // Default
                params: self.extract_layer_params(&parameters, layer_name),
            });

            // Add edge from previous node
            if let Some(prev_id) = prev_node_id {
                edges.push(GraphEdge {
                    from: prev_id,
                    to: node_id,
                    tensor_name: format!("{}_output", prev_id),
                });
            }

            prev_node_id = Some(node_id);
            node_id += 1;
        }

        Ok(GraphData {
            nodes,
            edges,
            metadata: GraphMetadata {
                framework: "torsh".to_string(),
                version: "0.1.0".to_string(),
                total_params: parameters.len(),
                trainable_params: parameters.values().filter(|p| p.requires_grad()).count(),
            },
        })
    }

    fn extract_layer_name(&self, param_name: &str) -> Option<String> {
        // Extract layer name from parameter name (e.g., "layer1.weight" -> "layer1")
        if let Some(dot_pos) = param_name.find('.') {
            Some(param_name[..dot_pos].to_string())
        } else {
            Some(param_name.to_string())
        }
    }

    fn infer_op_type(&self, layer_name: &str) -> String {
        if layer_name.contains("linear") || layer_name.contains("fc") {
            "Linear".to_string()
        } else if layer_name.contains("conv") {
            "Conv2D".to_string()
        } else if layer_name.contains("bn") || layer_name.contains("batch_norm") {
            "BatchNorm".to_string()
        } else if layer_name.contains("relu") {
            "ReLU".to_string()
        } else if layer_name.contains("dropout") {
            "Dropout".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    fn extract_layer_params(
        &self,
        parameters: &HashMap<String, torsh_nn::Parameter>,
        layer_name: &str,
    ) -> HashMap<String, String> {
        let mut params = HashMap::new();

        for (name, param) in parameters {
            if name.starts_with(layer_name) {
                let dims = match param.shape() {
                    Ok(shape) => shape.to_vec(),
                    Err(_) => vec![],
                };
                params.insert(name.to_string(), format!("{:?}", dims));
            }
        }

        params
    }

    /// Flush all pending events
    pub fn flush(&mut self) -> Result<()> {
        self.event_writer.flush()
    }

    /// Close the writer
    pub fn close(mut self) -> Result<()> {
        self.flush()?;
        Ok(())
    }

    /// Process image tensor for visualization
    fn process_image_tensor(&self, img_tensor: &Tensor, dataformats: &str) -> Result<ImageData> {
        let shape = img_tensor.shape();
        let values = img_tensor.to_vec()?;

        let (height, width, channels) = match dataformats {
            "CHW" => (shape.dims()[1], shape.dims()[2], shape.dims()[0]),
            "HWC" => (shape.dims()[0], shape.dims()[1], shape.dims()[2]),
            "HW" => (shape.dims()[0], shape.dims()[1], 1),
            _ => {
                return Err(TensorBoardError::InvalidParameter(
                    "Unknown format".to_string(),
                ))
            }
        };

        // Convert values to [0, 255] range and create PNG-like data
        let normalized: Vec<u8> = values
            .iter()
            .map(|&x| ((x.abs().min(1.0)) * 255.0) as u8)
            .collect();

        // Create simple bitmap format (simplified)
        let mut bitmap_data = Vec::new();
        for pixel in normalized.chunks(channels) {
            bitmap_data.extend_from_slice(pixel);
            // Pad to RGB if grayscale
            if channels == 1 {
                bitmap_data.push(pixel[0]); // R
                bitmap_data.push(pixel[0]); // G
                bitmap_data.push(pixel[0]); // B
            }
        }

        let base64_data = base64::engine::general_purpose::STANDARD.encode(&bitmap_data);

        Ok(ImageData {
            data: base64_data,
            width,
            height,
            channels,
        })
    }

    /// Add audio logging
    pub fn add_audio(
        &mut self,
        tag: &str,
        audio_tensor: &Tensor,
        sample_rate: u32,
        step: Option<i64>,
    ) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        let audio_data = self.process_audio_tensor(audio_tensor, sample_rate)?;

        // Save audio as base64 encoded data
        let audio_file = self.log_dir.join(format!(
            "{}_step_{}_audio.json",
            tag.replace('/', "_"),
            step
        ));
        let audio_json = json!({
            "tag": tag,
            "step": step,
            "wall_time": chrono::Utc::now().timestamp(),
            "audio": {
                "data": audio_data.data,
                "sample_rate": audio_data.sample_rate,
                "channels": audio_data.channels,
                "duration": audio_data.duration,
                "format": audio_data.format
            }
        });

        std::fs::write(
            audio_file,
            serde_json::to_string_pretty(&audio_json)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;

        let summary = Summary {
            tag: format!("{}/audio", tag),
            simple_value: audio_data.duration,
        };

        self.event_writer.write_summary(summary, step)?;

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }

    /// Process audio tensor for logging
    fn process_audio_tensor(&self, audio_tensor: &Tensor, sample_rate: u32) -> Result<AudioData> {
        let shape = audio_tensor.shape();
        let values = audio_tensor.to_vec()?;

        let channels = if shape.ndim() == 1 {
            1
        } else {
            shape.dims()[0]
        };
        let samples = if shape.ndim() == 1 {
            shape.dims()[0]
        } else {
            shape.dims()[1]
        };

        let duration = samples as f32 / sample_rate as f32;

        // Convert to 16-bit PCM format
        let pcm_data: Vec<i16> = values
            .iter()
            .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();

        // Convert to bytes
        let mut audio_bytes = Vec::new();
        for sample in pcm_data {
            audio_bytes.extend_from_slice(&sample.to_le_bytes());
        }

        let base64_data = base64::engine::general_purpose::STANDARD.encode(&audio_bytes);

        Ok(AudioData {
            data: base64_data,
            sample_rate,
            channels,
            duration,
            format: "PCM16".to_string(),
        })
    }

    /// Add embeddings visualization
    pub fn add_embedding(
        &mut self,
        mat: &Tensor,
        metadata: Option<Vec<String>>,
        label_img: Option<&Tensor>,
        global_step: Option<i64>,
        tag: &str,
    ) -> Result<()> {
        let step = global_step.unwrap_or(self.global_step);

        let embedding_data = self.process_embedding_data(mat, metadata, label_img)?;

        let embedding_file = self.log_dir.join(format!(
            "{}_step_{}_embeddings.json",
            tag.replace('/', "_"),
            step
        ));
        let embedding_json = serde_json::to_string_pretty(&embedding_data)
            .map_err(|e| TensorBoardError::Serialization(e.to_string()))?;
        std::fs::write(embedding_file, embedding_json)?;

        let summary = Summary {
            tag: format!("{}/embeddings", tag),
            simple_value: mat.shape().dims()[0] as f32,
        };

        self.event_writer.write_summary(summary, step)?;

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }

    /// Process embedding data
    fn process_embedding_data(
        &self,
        mat: &Tensor,
        metadata: Option<Vec<String>>,
        label_img: Option<&Tensor>,
    ) -> Result<serde_json::Value> {
        let shape = mat.shape();
        if shape.ndim() != 2 {
            return Err(TensorBoardError::InvalidParameter(
                "Embedding matrix must be 2D".to_string(),
            ));
        }

        let values = mat.to_vec()?;
        let n_embeddings = shape.dims()[0];
        let embedding_dim = shape.dims()[1];

        let mut embeddings = Vec::new();
        for i in 0..n_embeddings {
            let start = i * embedding_dim;
            let end = start + embedding_dim;
            embeddings.push(&values[start..end]);
        }

        let mut result = json!({
            "embeddings": embeddings,
            "shape": [n_embeddings, embedding_dim],
            "dtype": "float32"
        });

        if let Some(meta) = metadata {
            result["metadata"] = json!(meta);
        }

        if let Some(label_tensor) = label_img {
            let image_data = self.process_image_tensor(label_tensor, "HWC")?;
            result["label_img"] = json!({
                "data": image_data.data,
                "width": image_data.width,
                "height": image_data.height,
                "channels": image_data.channels
            });
        }

        Ok(result)
    }

    /// Install a custom plugin
    pub fn install_plugin(&mut self, plugin_config: PluginConfig) -> Result<()> {
        let plugins_dir = self.log_dir.join("plugins");
        std::fs::create_dir_all(&plugins_dir)?;

        let plugin_file = plugins_dir.join(format!("{}.json", plugin_config.name));
        let plugin_json = serde_json::to_string_pretty(&plugin_config)
            .map_err(|e| TensorBoardError::Serialization(e.to_string()))?;
        std::fs::write(plugin_file, plugin_json)?;

        // Create plugin manifest
        let manifest_file = plugins_dir.join("manifest.json");
        let mut plugins: Vec<String> = if manifest_file.exists() {
            let content = std::fs::read_to_string(&manifest_file)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };

        if !plugins.contains(&plugin_config.name) {
            plugins.push(plugin_config.name.clone());
        }

        let manifest_json = serde_json::to_string_pretty(&plugins)
            .map_err(|e| TensorBoardError::Serialization(e.to_string()))?;
        std::fs::write(manifest_file, manifest_json)?;

        Ok(())
    }

    /// Create a custom dashboard
    pub fn create_dashboard(&mut self, name: &str, layout: serde_json::Value) -> Result<()> {
        let dashboards_dir = self.log_dir.join("dashboards");
        std::fs::create_dir_all(&dashboards_dir)?;

        let dashboard_data = json!({
            "name": name,
            "created_at": chrono::Utc::now(),
            "layout": layout,
            "version": "1.0.0"
        });

        let dashboard_file = dashboards_dir.join(format!("{}.json", name));
        let dashboard_json = serde_json::to_string_pretty(&dashboard_data)
            .map_err(|e| TensorBoardError::Serialization(e.to_string()))?;
        std::fs::write(dashboard_file, dashboard_json)?;

        Ok(())
    }

    /// Add custom metrics
    pub fn add_custom_scalar(
        &mut self,
        tag: &str,
        value: serde_json::Value,
        step: Option<i64>,
    ) -> Result<()> {
        let step = step.unwrap_or(self.global_step);

        let custom_file = self
            .log_dir
            .join(format!("{}_custom.json", tag.replace('/', "_")));
        let mut entries: Vec<serde_json::Value> = if custom_file.exists() {
            let content = std::fs::read_to_string(&custom_file)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };

        entries.push(json!({
            "step": step,
            "value": value,
            "wall_time": chrono::Utc::now().timestamp()
        }));

        let content = serde_json::to_string_pretty(&entries)
            .map_err(|e| TensorBoardError::Serialization(e.to_string()))?;
        std::fs::write(custom_file, content)?;

        if step > self.global_step {
            self.global_step = step;
        }

        Ok(())
    }
}

/// Event writer for TensorBoard
struct EventWriter {
    writer: File,
    wall_time: f64,
}

impl EventWriter {
    fn new(log_dir: &Path) -> Result<Self> {
        let timestamp = Utc::now().timestamp();
        let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());

        let filename = format!("events.out.tfevents.{}.{}", timestamp, hostname);

        let path = log_dir.join(filename);
        let mut writer = File::create(path)?;

        // Write file version
        writer.write_all(EVENT_FILE_VERSION.as_bytes())?;
        writer.write_all(b"\n")?;

        Ok(Self {
            writer,
            wall_time: timestamp as f64,
        })
    }

    fn write_summary(&mut self, summary: Summary, step: i64) -> Result<()> {
        let event = Event {
            wall_time: self.wall_time,
            step,
            summary,
        };

        // Serialize event (simplified - real implementation would use protobuf)
        let json = serde_json::to_string(&event)
            .map_err(|e| TensorBoardError::Serialization(e.to_string()))?;

        // Write length and data
        let data = json.as_bytes();
        let len = data.len() as u64;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(data)?;

        // Write CRC (simplified - just use a hash)
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        self.writer.write_all(&hash[..8])?;

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// Event structure
#[derive(serde::Serialize)]
struct Event {
    wall_time: f64,
    step: i64,
    summary: Summary,
}

/// Summary structure
#[derive(serde::Serialize)]
struct Summary {
    tag: String,
    simple_value: f32,
}

/// Histogram data
struct Histogram {
    #[allow(dead_code)]
    min: f32,
    #[allow(dead_code)]
    max: f32,
    mean: f32,
    #[allow(dead_code)]
    std: f32,
    #[allow(dead_code)]
    sum: f32,
    #[allow(dead_code)]
    sum_squares: f32,
}

/// Graph data structure for model visualization
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub metadata: GraphMetadata,
}

/// Graph node representing a layer or operation
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub name: String,
    pub op_type: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub params: HashMap<String, String>,
}

/// Graph edge representing data flow between nodes
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct GraphEdge {
    pub from: usize,
    pub to: usize,
    pub tensor_name: String,
}

/// Graph metadata
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct GraphMetadata {
    pub framework: String,
    pub version: String,
    pub total_params: usize,
    pub trainable_params: usize,
}

/// Image data structure
#[derive(Debug, Clone)]
struct ImageData {
    pub data: String, // Base64 encoded
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

/// Audio data structure
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct AudioData {
    pub data: String, // Base64 encoded
    pub sample_rate: u32,
    pub channels: usize,
    pub duration: f32,
    pub format: String,
}

/// Plugin configuration for custom dashboards
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct PluginConfig {
    pub name: String,
    pub version: String,
    pub entry_point: String,
    pub config: HashMap<String, serde_json::Value>,
    pub enabled: bool,
}

/// Enhanced graph data structure with execution tracing and analysis
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct EnhancedGraphData {
    pub nodes: Vec<EnhancedGraphNode>,
    pub edges: Vec<EnhancedGraphEdge>,
    pub metadata: EnhancedGraphMetadata,
    pub execution_traces: Vec<ExecutionTrace>,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// Enhanced graph node with detailed analysis
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct EnhancedGraphNode {
    pub id: usize,
    pub name: String,
    pub op_type: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub params: HashMap<String, String>,
    pub computational_complexity: ComputationalComplexity,
    pub memory_footprint: MemoryFootprint,
    pub layer_type: LayerType,
    pub optimization_hints: Vec<String>,
}

/// Enhanced graph edge with data flow information
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct EnhancedGraphEdge {
    pub from: usize,
    pub to: usize,
    pub tensor_name: String,
    pub data_flow_type: DataFlowType,
    pub tensor_size_mb: f32,
    pub communication_cost: f32,
}

/// Enhanced graph metadata with performance metrics
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct EnhancedGraphMetadata {
    pub framework: String,
    pub version: String,
    pub total_params: usize,
    pub trainable_params: usize,
    pub model_size_mb: f32,
    pub flops: u64,
    pub inference_time_ms: Option<f32>,
    pub memory_usage_mb: Option<f32>,
}

/// Execution trace for dynamic analysis
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ExecutionTrace {
    pub input_id: usize,
    pub timing_data: Vec<LayerTiming>,
    pub memory_usage: Vec<MemoryUsage>,
    pub activation_shapes: HashMap<String, Vec<usize>>,
    pub gradient_flow: Option<GradientFlow>,
}

/// Layer timing information
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerTiming {
    pub layer_name: String,
    pub forward_time_ms: f32,
    pub backward_time_ms: Option<f32>,
    pub cpu_utilization: f32,
    pub gpu_utilization: Option<f32>,
}

/// Memory usage information
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct MemoryUsage {
    pub layer_name: String,
    pub allocated_mb: f32,
    pub peak_mb: f32,
    pub freed_mb: f32,
}

/// Gradient flow analysis
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct GradientFlow {
    pub layer_gradients: HashMap<String, GradientStats>,
    pub vanishing_gradients: Vec<String>,
    pub exploding_gradients: Vec<String>,
}

/// Gradient statistics for a layer
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct GradientStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub norm: f32,
}

/// Data flow type in the graph
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum DataFlowType {
    Forward,
    Backward,
    Parameter,
    Buffer,
}

/// Computational complexity metrics
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ComputationalComplexity {
    pub flops: u64,
    pub multiply_adds: u64,
    pub memory_accesses: u64,
    pub algorithmic_complexity: String,
}

/// Memory footprint information
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct MemoryFootprint {
    pub parameters_mb: f32,
    pub activations_mb: f32,
    pub gradients_mb: f32,
    pub buffers_mb: f32,
    pub total_mb: f32,
}

/// Layer classification
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum LayerType {
    Computation(ComputationLayerType),
    Normalization(NormalizationLayerType),
    Activation(ActivationLayerType),
    Regularization(RegularizationLayerType),
    Structural(StructuralLayerType),
    Custom(String),
}

/// Computation layer types
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum ComputationLayerType {
    Linear,
    Convolution,
    Pooling,
    Attention,
    Recurrent,
}

/// Normalization layer types
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum NormalizationLayerType {
    BatchNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
}

/// Activation layer types
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum ActivationLayerType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    Swish,
}

/// Regularization layer types
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum RegularizationLayerType {
    Dropout,
    DropConnect,
    DropPath,
}

/// Structural layer types
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum StructuralLayerType {
    Reshape,
    Transpose,
    Concatenate,
    Split,
    Residual,
}

/// Optimization suggestions
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub target_layers: Vec<String>,
    pub description: String,
    pub expected_improvement: f32,
    pub implementation_difficulty: DifficultyLevel,
}

/// Optimization types
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum OptimizationType {
    Quantization,
    Pruning,
    KnowledgeDistillation,
    ArchitecturalChange,
    OperatorFusion,
    MemoryOptimization,
    ParallelizationOpportunity,
}

/// Implementation difficulty levels
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Layer analysis data for comprehensive visualization
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerAnalysisData {
    pub activation_stats: Vec<LayerActivationStats>,
    pub gradient_stats: Vec<LayerGradientStats>,
    pub weight_distributions: Vec<LayerWeightDistribution>,
    pub layer_connectivity: Vec<LayerConnectivity>,
    pub computational_complexity: Vec<LayerComplexityAnalysis>,
    pub memory_footprint: Vec<LayerMemoryAnalysis>,
}

/// Layer activation statistics
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerActivationStats {
    pub layer_name: String,
    pub activations: ActivationStatistics,
    pub weights: WeightStatistics,
    pub sparsity: f32,
    pub effective_rank: Option<f32>,
}

/// Activation statistics
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct ActivationStatistics {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub percentiles: HashMap<String, f32>,
    pub dead_neurons_ratio: f32,
}

/// Weight statistics
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct WeightStatistics {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub norm: f32,
    pub condition_number: Option<f32>,
}

/// Layer gradient statistics
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerGradientStats {
    pub layer_name: String,
    pub gradient_norm: f32,
    pub gradient_mean: f32,
    pub gradient_std: f32,
    pub clipping_ratio: f32,
}

/// Layer weight distribution
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerWeightDistribution {
    pub layer_name: String,
    pub distribution_type: String,
    pub parameters: HashMap<String, f32>,
    pub histogram_data: Vec<f32>,
    pub outlier_ratio: f32,
}

/// Layer connectivity information
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerConnectivity {
    pub layer_name: String,
    pub input_layers: Vec<String>,
    pub output_layers: Vec<String>,
    pub fan_in: usize,
    pub fan_out: usize,
    pub is_bottleneck: bool,
}

/// Layer complexity analysis
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerComplexityAnalysis {
    pub layer_name: String,
    pub time_complexity: String,
    pub space_complexity: String,
    pub parallelization_potential: f32,
    pub optimization_opportunities: Vec<String>,
}

/// Layer memory analysis
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerMemoryAnalysis {
    pub layer_name: String,
    pub parameter_memory_mb: f32,
    pub activation_memory_mb: f32,
    pub gradient_memory_mb: f32,
    pub temporary_memory_mb: f32,
    pub memory_efficiency: f32,
}

impl Histogram {
    fn from_values(values: &[f32]) -> Self {
        let n = values.len() as f32;
        let sum: f32 = values.iter().sum();
        let mean = sum / n;

        let sum_squares: f32 = values.iter().map(|x| x * x).sum();
        let variance = (sum_squares / n) - (mean * mean);
        let std = variance.sqrt();

        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            min,
            max,
            mean,
            std,
            sum,
            sum_squares,
        }
    }
}

/// Simplified TensorBoard writer (no protobuf dependency)
pub struct TensorBoardWriter {
    log_dir: PathBuf,
    #[allow(dead_code)]
    run_name: String,
    step: i64,
}

impl TensorBoardWriter {
    /// Create a new TensorBoard writer
    pub fn new<P: AsRef<Path>>(log_dir: P, run_name: Option<String>) -> TorshResult<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        let run_name =
            run_name.unwrap_or_else(|| format!("run_{}", Utc::now().format("%Y%m%d_%H%M%S")));

        let run_dir = log_dir.join(&run_name);
        fs::create_dir_all(&run_dir)?;

        Ok(Self {
            log_dir: run_dir,
            run_name,
            step: 0,
        })
    }

    /// Log a scalar value
    pub fn log_scalar(&mut self, name: &str, value: f32, step: Option<i64>) -> TorshResult<()> {
        let step = step.unwrap_or(self.step);

        let file_path = self.log_dir.join(format!("{}.json", name));
        let mut entries: Vec<ScalarEntry> = if file_path.exists() {
            let content = fs::read_to_string(&file_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };

        entries.push(ScalarEntry {
            step,
            value,
            wall_time: Utc::now(),
        });

        let content = serde_json::to_string_pretty(&entries)
            .map_err(|e| TorshError::Other(format!("JSON serialization error: {}", e)))?;
        fs::write(file_path, content)?;

        if step > self.step {
            self.step = step;
        }

        Ok(())
    }

    /// Log multiple scalars
    pub fn log_scalars(
        &mut self,
        values: HashMap<String, f32>,
        step: Option<i64>,
    ) -> TorshResult<()> {
        for (name, value) in values {
            self.log_scalar(&name, value, step)?;
        }
        Ok(())
    }

    /// Set global step
    pub fn set_step(&mut self, step: i64) {
        self.step = step;
    }

    /// Get current step
    pub fn get_step(&self) -> i64 {
        self.step
    }

    /// Implementation functions for enhanced TensorBoard features

    /// Estimate computational complexity for a graph node
    fn estimate_computational_complexity(&self, node: &GraphNode) -> ComputationalComplexity {
        let base_flops = match node.op_type.as_str() {
            "Linear" => {
                let input_size = node.shape.get(0).unwrap_or(&1);
                let output_size = node.shape.get(1).unwrap_or(&1);
                (input_size * output_size) as u64 * 2 // Multiply-add operations
            }
            "Conv2D" => {
                // Simplified FLOP calculation for convolution
                let output_h = node.shape.get(0).unwrap_or(&1);
                let output_w = node.shape.get(1).unwrap_or(&1);
                let kernel_size = 9; // 3x3 kernel assumption
                let channels = node.shape.get(2).unwrap_or(&1);
                (output_h * output_w * kernel_size * channels) as u64 * 2
            }
            _ => 1000, // Default estimate
        };

        ComputationalComplexity {
            flops: base_flops,
            multiply_adds: base_flops / 2,
            memory_accesses: base_flops / 4,
            algorithmic_complexity: self.determine_algorithmic_complexity(&node.op_type),
        }
    }

    /// Estimate memory footprint for a graph node
    fn estimate_memory_footprint(&self, node: &GraphNode) -> MemoryFootprint {
        let element_size = 4.0; // 4 bytes per f32
        let total_elements: f32 = node.shape.iter().product::<usize>() as f32;
        let base_memory = total_elements * element_size / (1024.0 * 1024.0); // MB

        match node.op_type.as_str() {
            "Linear" | "Conv2D" => MemoryFootprint {
                parameters_mb: base_memory,
                activations_mb: base_memory * 0.5,
                gradients_mb: base_memory,
                buffers_mb: base_memory * 0.1,
                total_mb: base_memory * 2.6,
            },
            "BatchNorm" => MemoryFootprint {
                parameters_mb: base_memory * 0.1,
                activations_mb: base_memory,
                gradients_mb: base_memory * 0.1,
                buffers_mb: base_memory * 0.2,
                total_mb: base_memory * 1.4,
            },
            _ => MemoryFootprint {
                parameters_mb: 0.0,
                activations_mb: base_memory,
                gradients_mb: 0.0,
                buffers_mb: base_memory * 0.1,
                total_mb: base_memory * 1.1,
            },
        }
    }

    /// Classify layer type based on operation
    fn classify_layer_type(&self, node: &GraphNode) -> LayerType {
        match node.op_type.as_str() {
            "Linear" => LayerType::Computation(ComputationLayerType::Linear),
            "Conv2D" => LayerType::Computation(ComputationLayerType::Convolution),
            "MaxPool2D" | "AvgPool2D" => LayerType::Computation(ComputationLayerType::Pooling),
            "BatchNorm" => LayerType::Normalization(NormalizationLayerType::BatchNorm),
            "LayerNorm" => LayerType::Normalization(NormalizationLayerType::LayerNorm),
            "ReLU" => LayerType::Activation(ActivationLayerType::ReLU),
            "Sigmoid" => LayerType::Activation(ActivationLayerType::Sigmoid),
            "Tanh" => LayerType::Activation(ActivationLayerType::Tanh),
            "Softmax" => LayerType::Activation(ActivationLayerType::Softmax),
            "GELU" => LayerType::Activation(ActivationLayerType::GELU),
            "Dropout" => LayerType::Regularization(RegularizationLayerType::Dropout),
            "Reshape" => LayerType::Structural(StructuralLayerType::Reshape),
            _ => LayerType::Custom(node.op_type.clone()),
        }
    }

    /// Generate optimization hints for a node
    fn generate_optimization_hints(&self, node: &GraphNode) -> Vec<String> {
        let mut hints = Vec::new();

        match node.op_type.as_str() {
            "Linear" => {
                hints.push("Consider quantization for mobile deployment".to_string());
                hints.push("Sparse weights may benefit from pruning".to_string());
            }
            "Conv2D" => {
                hints.push("Consider depthwise separable convolutions".to_string());
                hints.push("Operator fusion opportunities available".to_string());
            }
            "BatchNorm" => {
                hints.push("Can be fused with preceding convolution".to_string());
            }
            "Dropout" => {
                hints.push("Remove during inference for better performance".to_string());
            }
            _ => {}
        }

        hints
    }

    /// Estimate model size in MB
    fn estimate_model_size(&self, model: &dyn torsh_nn::Module) -> f32 {
        let parameters = model.parameters();
        let mut total_params = 0;

        for (_, param) in &parameters {
            if let Ok(shape) = param.shape() {
                total_params += shape.iter().product::<usize>();
            }
        }

        (total_params as f32 * 4.0) / (1024.0 * 1024.0) // 4 bytes per f32, convert to MB
    }

    /// Estimate FLOPs for the entire model
    fn estimate_flops(&self, model: &dyn torsh_nn::Module) -> u64 {
        let parameters = model.parameters();
        let mut total_flops = 0u64;

        for (name, param) in &parameters {
            if let Ok(shape) = param.shape() {
                let layer_flops = if name.contains("weight") {
                    shape.iter().product::<usize>() as u64 * 2 // Multiply-add operations
                } else {
                    shape.iter().product::<usize>() as u64 // Add operations for biases
                };
                total_flops += layer_flops;
            }
        }

        total_flops
    }

    /// Generate model-level optimization suggestions
    fn generate_model_optimization_suggestions(
        &self,
        model: &dyn torsh_nn::Module,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        let parameters = model.parameters();

        // Analyze model for optimization opportunities
        let total_params = parameters.len();
        let model_size = self.estimate_model_size(model);

        if model_size > 100.0 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::Quantization,
                target_layers: vec!["all_linear".to_string(), "all_conv".to_string()],
                description: "Large model size detected. Consider INT8 quantization to reduce memory footprint by 4x".to_string(),
                expected_improvement: 0.75,
                implementation_difficulty: DifficultyLevel::Medium,
            });
        }

        if total_params > 10_000_000 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::Pruning,
                target_layers: vec!["linear_layers".to_string()],
                description: "High parameter count detected. Structured pruning can reduce parameters by 50-80%".to_string(),
                expected_improvement: 0.6,
                implementation_difficulty: DifficultyLevel::Hard,
            });
        }

        suggestions.push(OptimizationSuggestion {
            suggestion_type: OptimizationType::OperatorFusion,
            target_layers: vec!["conv_bn_pairs".to_string()],
            description:
                "Fuse convolution and batch normalization layers for improved inference speed"
                    .to_string(),
            expected_improvement: 0.15,
            implementation_difficulty: DifficultyLevel::Easy,
        });

        suggestions
    }

    /// Determine algorithmic complexity
    fn determine_algorithmic_complexity(&self, op_type: &str) -> String {
        match op_type {
            "Linear" => "O(n²)".to_string(),
            "Conv2D" => "O(n²k²)".to_string(), // n: input size, k: kernel size
            "BatchNorm" => "O(n)".to_string(),
            "ReLU" | "Sigmoid" | "Tanh" => "O(n)".to_string(),
            "Softmax" => "O(n log n)".to_string(),
            "MaxPool2D" | "AvgPool2D" => "O(n)".to_string(),
            _ => "O(n)".to_string(),
        }
    }

    /// Trace model execution for dynamic analysis
    fn trace_model_execution(
        &self,
        model: &dyn torsh_nn::Module,
        input: &Tensor,
    ) -> Result<ExecutionTrace> {
        // Simplified execution tracing
        // In a real implementation, this would hook into the execution engine

        let layer_count = model.parameters().len();
        let mut timing_data = Vec::new();
        let mut memory_usage = Vec::new();
        let mut activation_shapes = HashMap::new();

        // Simulate timing data for each layer
        for i in 0..layer_count.min(10) {
            let layer_name = format!("layer_{}", i);

            timing_data.push(LayerTiming {
                layer_name: layer_name.clone(),
                forward_time_ms: 1.0 + (i as f32 * 0.5),
                backward_time_ms: Some(1.5 + (i as f32 * 0.3)),
                cpu_utilization: 70.0 + (i as f32 * 2.0),
                gpu_utilization: Some(50.0 + (i as f32 * 3.0)),
            });

            memory_usage.push(MemoryUsage {
                layer_name: layer_name.clone(),
                allocated_mb: 10.0 + (i as f32 * 5.0),
                peak_mb: 15.0 + (i as f32 * 6.0),
                freed_mb: 8.0 + (i as f32 * 4.0),
            });

            activation_shapes.insert(layer_name, input.shape().dims().to_vec());
        }

        Ok(ExecutionTrace {
            input_id: 0,
            timing_data,
            memory_usage,
            activation_shapes,
            gradient_flow: None, // Would be populated during backward pass
        })
    }

    /// Create different graph visualization views
    fn create_architectural_view(&self, graph_data: &EnhancedGraphData) -> Result<()> {
        let arch_file = self.log_dir.join("graph_architectural_view.json");
        let arch_data = json!({
            "view_type": "architectural",
            "nodes": graph_data.nodes,
            "edges": graph_data.edges,
            "layout": "hierarchical"
        });

        std::fs::write(
            arch_file,
            serde_json::to_string_pretty(&arch_data)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;
        Ok(())
    }

    fn create_execution_view(&self, graph_data: &EnhancedGraphData) -> Result<()> {
        let exec_file = self.log_dir.join("graph_execution_view.json");
        let exec_data = json!({
            "view_type": "execution",
            "execution_traces": graph_data.execution_traces,
            "performance_highlights": true
        });

        std::fs::write(
            exec_file,
            serde_json::to_string_pretty(&exec_data)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;
        Ok(())
    }

    fn create_parameter_view(&self, graph_data: &EnhancedGraphData) -> Result<()> {
        let param_file = self.log_dir.join("graph_parameter_view.json");
        let param_data = json!({
            "view_type": "parameters",
            "parameter_analysis": graph_data.nodes.iter().map(|node| {
                json!({
                    "layer": node.name,
                    "parameters": node.params,
                    "memory_footprint": node.memory_footprint
                })
            }).collect::<Vec<_>>()
        });

        std::fs::write(
            param_file,
            serde_json::to_string_pretty(&param_data)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;
        Ok(())
    }

    fn create_computational_graph_view(&self, graph_data: &EnhancedGraphData) -> Result<()> {
        let comp_file = self.log_dir.join("graph_computational_view.json");
        let comp_data = json!({
            "view_type": "computational",
            "complexity_analysis": graph_data.nodes.iter().map(|node| {
                json!({
                    "layer": node.name,
                    "computational_complexity": node.computational_complexity,
                    "optimization_hints": node.optimization_hints
                })
            }).collect::<Vec<_>>(),
            "optimization_suggestions": graph_data.optimization_suggestions
        });

        std::fs::write(
            comp_file,
            serde_json::to_string_pretty(&comp_data)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;
        Ok(())
    }

    fn create_interactive_graph_files(&self, graph_data: &EnhancedGraphData) -> Result<()> {
        // Create interactive HTML file with JavaScript visualization
        let interactive_file = self.log_dir.join("interactive_graph.html");
        let html_content = self.generate_interactive_html(graph_data)?;
        std::fs::write(interactive_file, html_content)?;

        // Create data file for interactive visualization
        let data_file = self.log_dir.join("interactive_graph_data.json");
        let data_content = serde_json::to_string_pretty(graph_data)
            .map_err(|e| TensorBoardError::Serialization(e.to_string()))?;
        std::fs::write(data_file, data_content)?;

        Ok(())
    }

    fn generate_interactive_html(&self, _graph_data: &EnhancedGraphData) -> Result<String> {
        // Generate interactive HTML with D3.js or similar visualization library
        Ok(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Model Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .node { stroke: #333; stroke-width: 2px; }
        .edge { stroke: #666; stroke-width: 1px; }
        .tooltip { position: absolute; padding: 10px; background: rgba(0,0,0,0.8); color: white; border-radius: 5px; }
    </style>
</head>
<body>
    <div id="graph-container"></div>
    <script>
        // Interactive graph visualization would be implemented here
        // Load data from interactive_graph_data.json and create visualization
        console.log("Interactive graph visualization loaded");
    </script>
</body>
</html>
        "#.to_string())
    }

    /// Add layer-specific histogram visualization
    fn add_layer_histogram(
        &mut self,
        layer_name: &str,
        values: &ActivationStatistics,
        step: i64,
    ) -> Result<()> {
        let histogram_file = self.log_dir.join(format!(
            "layer_histogram_{}_{}.json",
            layer_name.replace('/', "_"),
            step
        ));

        let histogram_data = json!({
            "layer_name": layer_name,
            "step": step,
            "statistics": {
                "mean": values.mean,
                "std": values.std,
                "min": values.min,
                "max": values.max,
                "percentiles": values.percentiles,
                "dead_neurons_ratio": values.dead_neurons_ratio
            }
        });

        std::fs::write(
            histogram_file,
            serde_json::to_string_pretty(&histogram_data)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;

        Ok(())
    }

    /// Add layer weight distribution visualization
    fn add_layer_weight_distribution(
        &mut self,
        layer_name: &str,
        weights: &WeightStatistics,
        step: i64,
    ) -> Result<()> {
        let weight_file = self.log_dir.join(format!(
            "layer_weights_{}_{}.json",
            layer_name.replace('/', "_"),
            step
        ));

        let weight_data = json!({
            "layer_name": layer_name,
            "step": step,
            "weight_statistics": {
                "mean": weights.mean,
                "std": weights.std,
                "min": weights.min,
                "max": weights.max,
                "norm": weights.norm,
                "condition_number": weights.condition_number
            }
        });

        std::fs::write(
            weight_file,
            serde_json::to_string_pretty(&weight_data)
                .map_err(|e| TensorBoardError::Serialization(e.to_string()))?,
        )?;

        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScalarEntry {
    step: i64,
    value: f32,
    wall_time: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use torsh_tensor::Tensor;

    #[test]
    fn test_tensorboard_writer() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TensorBoardWriter::new(temp_dir.path(), None).unwrap();

        writer.log_scalar("loss", 0.5, Some(0)).unwrap();
        writer.log_scalar("loss", 0.3, Some(1)).unwrap();
        writer.log_scalar("loss", 0.1, Some(2)).unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("f1_score".to_string(), 0.92);

        writer.log_scalars(metrics, Some(3)).unwrap();

        assert_eq!(writer.get_step(), 3);
    }

    #[test]
    fn test_summary_writer() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = SummaryWriter::new(temp_dir.path()).unwrap();

        writer.add_scalar("train/loss", 0.5, Some(0)).unwrap();
        writer.add_scalar("train/loss", 0.3, Some(1)).unwrap();

        let mut scalars = HashMap::new();
        scalars.insert("accuracy".to_string(), 0.95);
        scalars.insert("precision".to_string(), 0.93);

        writer.add_scalars("eval", scalars, Some(2)).unwrap();

        writer.flush().unwrap();
    }

    #[test]
    fn test_image_logging() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = SummaryWriter::new(temp_dir.path()).unwrap();

        // Create a test image tensor (3x32x32 CHW format)
        let shape = vec![3, 32, 32];
        let data: Vec<f32> = (0..3072).map(|i| (i as f32) / 3072.0).collect();
        let image_tensor = Tensor::from_vec(data, &shape).unwrap();

        writer
            .add_image("test_image", &image_tensor, Some(0), "CHW")
            .unwrap();
        writer.flush().unwrap();

        // Verify image file was created
        let image_file = temp_dir.path().join("test_image_step_0.json");
        assert!(image_file.exists());
    }

    #[test]
    fn test_audio_logging() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = SummaryWriter::new(temp_dir.path()).unwrap();

        // Create a test audio tensor (1 second of sine wave at 44.1kHz)
        let sample_rate = 44100;
        let duration = 1.0;
        let samples = (sample_rate as f32 * duration) as usize;
        let freq = 440.0; // A4 note

        let audio_data: Vec<f32> = (0..samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let shape = vec![samples];
        let audio_tensor = Tensor::from_vec(audio_data, &shape).unwrap();

        writer
            .add_audio("test_audio", &audio_tensor, sample_rate, Some(0))
            .unwrap();
        writer.flush().unwrap();

        // Verify audio file was created
        let audio_file = temp_dir.path().join("test_audio_step_0_audio.json");
        assert!(audio_file.exists());
    }

    #[test]
    fn test_embedding_logging() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = SummaryWriter::new(temp_dir.path()).unwrap();

        // Create test embeddings (10 samples, 128 dimensions)
        let shape = vec![10, 128];
        let data: Vec<f32> = (0..1280).map(|i| (i as f32) / 1280.0 - 0.5).collect();
        let embedding_tensor = Tensor::from_vec(data, &shape).unwrap();

        let metadata = Some(vec![
            "item_0".to_string(),
            "item_1".to_string(),
            "item_2".to_string(),
            "item_3".to_string(),
            "item_4".to_string(),
            "item_5".to_string(),
            "item_6".to_string(),
            "item_7".to_string(),
            "item_8".to_string(),
            "item_9".to_string(),
        ]);

        writer
            .add_embedding(
                &embedding_tensor,
                metadata,
                None,
                Some(0),
                "test_embeddings",
            )
            .unwrap();
        writer.flush().unwrap();

        // Verify embedding file was created
        let embedding_file = temp_dir
            .path()
            .join("test_embeddings_step_0_embeddings.json");
        assert!(embedding_file.exists());
    }

    #[test]
    fn test_plugin_system() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = SummaryWriter::new(temp_dir.path()).unwrap();

        let mut plugin_config = HashMap::new();
        plugin_config.insert("threshold".to_string(), json!(0.5));
        plugin_config.insert("color_scheme".to_string(), json!("viridis"));

        let plugin = PluginConfig {
            name: "custom_visualizer".to_string(),
            version: "1.0.0".to_string(),
            entry_point: "visualizer.js".to_string(),
            config: plugin_config,
            enabled: true,
        };

        writer.install_plugin(plugin).unwrap();
        writer.flush().unwrap();

        // Verify plugin was installed
        let plugin_file = temp_dir
            .path()
            .join("plugins")
            .join("custom_visualizer.json");
        assert!(plugin_file.exists());

        let manifest_file = temp_dir.path().join("plugins").join("manifest.json");
        assert!(manifest_file.exists());
    }

    #[test]
    fn test_custom_dashboard() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = SummaryWriter::new(temp_dir.path()).unwrap();

        let layout = json!({
            "type": "grid",
            "columns": 2,
            "rows": 2,
            "widgets": [
                {"type": "scalar", "tag": "loss", "position": [0, 0]},
                {"type": "scalar", "tag": "accuracy", "position": [0, 1]},
                {"type": "image", "tag": "samples", "position": [1, 0]},
                {"type": "histogram", "tag": "weights", "position": [1, 1]}
            ]
        });

        writer
            .create_dashboard("training_dashboard", layout)
            .unwrap();
        writer.flush().unwrap();

        // Verify dashboard was created
        let dashboard_file = temp_dir
            .path()
            .join("dashboards")
            .join("training_dashboard.json");
        assert!(dashboard_file.exists());
    }
}
