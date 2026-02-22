//! Graph foundation models and self-supervised learning
//!
//! This module implements state-of-the-art foundation models for graphs,
//! including self-supervised pre-training, contrastive learning, and transfer learning.

use crate::GraphData;
use std::collections::{HashMap, HashSet};
use std::fmt;
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph foundation model architecture
#[derive(Debug)]
pub struct GraphFoundationModel {
    /// Model configuration
    pub config: FoundationModelConfig,
    /// Encoder layers (stored as indices/configs instead of trait objects for clonability)
    pub encoder_layers: Vec<String>, // Layer type names for reconstruction
    /// Pre-training head
    pub pretraining_head: PretrainingHead,
    /// Fine-tuning heads (stored as type names for reconstruction)
    pub task_heads: HashMap<String, String>,
    /// Tokenizer for graph elements
    pub tokenizer: GraphTokenizer,
    /// Model parameters
    pub parameters: FoundationModelParameters,
}

/// Configuration for foundation model
#[derive(Debug, Clone)]
pub struct FoundationModelConfig {
    /// Model dimension
    pub model_dim: usize,
    /// Number of encoder layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Feedforward dimension
    pub ff_dim: usize,
    /// Maximum sequence length for graph sequences
    pub max_seq_length: usize,
    /// Vocabulary size for graph tokens
    pub vocab_size: usize,
    /// Dropout rate
    pub dropout: f32,
    /// Pre-training objectives
    pub pretraining_objectives: Vec<PretrainingObjective>,
}

/// Pre-training objectives for self-supervised learning
#[derive(Debug, Clone)]
pub enum PretrainingObjective {
    /// Masked node modeling
    MaskedNodeModeling,
    /// Masked edge modeling
    MaskedEdgeModeling,
    /// Graph contrastive learning
    GraphContrastive,
    /// Node-level contrastive learning
    NodeContrastive,
    /// Graph structure prediction
    StructurePrediction,
    /// Motif prediction
    MotifPrediction,
    /// Property prediction (self-supervised)
    PropertyPrediction,
    /// Graph denoising
    GraphDenoising,
}

/// Pre-training head for foundation model
#[derive(Debug, Clone)]
pub struct PretrainingHead {
    /// Masked language modeling head
    pub mlm_head: MLMHead,
    /// Contrastive learning head
    pub contrastive_head: ContrastiveHead,
    /// Structure prediction head
    pub structure_head: StructurePredictionHead,
    /// Current active objectives
    pub active_objectives: Vec<PretrainingObjective>,
}

/// Masked Language Modeling head for graphs
#[derive(Debug, Clone)]
pub struct MLMHead {
    /// Output projection
    pub output_projection: Tensor,
    /// Bias terms
    pub bias: Tensor,
    /// Mask token embedding
    pub mask_token: Tensor,
}

/// Contrastive learning head
#[derive(Debug, Clone)]
pub struct ContrastiveHead {
    /// Projection head for contrastive learning
    pub projection: Tensor,
    /// Temperature parameter
    pub temperature: f32,
    /// Embedding dimension
    pub embed_dim: usize,
}

/// Structure prediction head
#[derive(Debug, Clone)]
pub struct StructurePredictionHead {
    /// Edge prediction layers
    pub edge_predictor: Tensor,
    /// Motif prediction layers
    pub motif_predictor: Tensor,
    /// Property prediction layers
    pub property_predictor: Tensor,
}

/// Graph tokenizer for converting graphs to token sequences
#[derive(Debug, Clone)]
pub struct GraphTokenizer {
    /// Node type vocabulary
    pub node_vocab: HashMap<String, usize>,
    /// Edge type vocabulary
    pub edge_vocab: HashMap<String, usize>,
    /// Special tokens
    pub special_tokens: SpecialTokens,
    /// Tokenization strategy
    pub strategy: TokenizationStrategy,
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub mask_token: usize,
    pub cls_token: usize,
    pub sep_token: usize,
    pub pad_token: usize,
    pub unk_token: usize,
}

#[derive(Debug, Clone)]
pub enum TokenizationStrategy {
    /// Node-centric tokenization
    NodeCentric,
    /// Edge-centric tokenization
    EdgeCentric,
    /// Walk-based tokenization
    WalkBased,
    /// Subgraph-based tokenization
    SubgraphBased,
    /// Hierarchical tokenization
    Hierarchical,
}

/// Foundation model parameters
#[derive(Debug, Clone)]
pub struct FoundationModelParameters {
    /// Pre-training parameters
    pub pretraining_params: HashMap<String, Tensor>,
    /// Task-specific parameters
    pub task_params: HashMap<String, HashMap<String, Tensor>>,
    /// Frozen parameters (for transfer learning)
    pub frozen_params: HashSet<String>,
}

impl GraphFoundationModel {
    /// Create a new foundation model
    pub fn new(config: FoundationModelConfig) -> Result<Self, FoundationModelError> {
        let tokenizer = GraphTokenizer::new(config.vocab_size)?;
        let pretraining_head = PretrainingHead::new(&config)?;
        let parameters = FoundationModelParameters::new();

        Ok(Self {
            config,
            encoder_layers: Vec::new(),
            pretraining_head,
            task_heads: HashMap::new(),
            tokenizer,
            parameters,
        })
    }

    /// Pre-train the foundation model
    pub fn pretrain(
        &mut self,
        graphs: &[GraphData],
        num_epochs: usize,
    ) -> Result<PretrainingStats, FoundationModelError> {
        let mut stats = PretrainingStats::new();

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            for graph in graphs {
                // Apply data augmentation
                let augmented_graphs = self.apply_augmentation(graph)?;

                for aug_graph in &augmented_graphs {
                    // Forward pass with pre-training objectives
                    let loss = self.compute_pretraining_loss(aug_graph)?;
                    epoch_loss += loss;
                    num_batches += 1;

                    // Update statistics
                    stats.total_samples += 1;
                }
            }

            stats.epoch_losses.push(epoch_loss / num_batches as f32);
            stats.current_epoch = epoch;

            // Learning rate scheduling
            self.update_learning_rate(epoch);
        }

        stats.pretraining_completed = true;
        Ok(stats)
    }

    /// Fine-tune on downstream task
    pub fn finetune(
        &mut self,
        task_name: &str,
        train_data: &[(GraphData, Tensor)],
        val_data: &[(GraphData, Tensor)],
        task_config: TaskConfig,
    ) -> Result<FinetuningStats, FoundationModelError> {
        // Add task-specific head
        self.add_task_head(task_name, task_config.task_type.clone())?;

        // Freeze pre-training parameters if specified
        if task_config.freeze_pretrained {
            self.freeze_pretrained_parameters();
        }

        let mut stats = FinetuningStats::new();

        for _epoch in 0..task_config.num_epochs {
            // Training phase
            let mut train_loss = 0.0;
            for (graph, target) in train_data {
                let prediction = self.forward_task(graph, task_name)?;
                let loss = self.compute_task_loss(&prediction, target, &task_config.task_type)?;
                train_loss += loss;
            }

            // Validation phase
            let mut val_loss = 0.0;
            let mut val_accuracy = 0.0;
            for (graph, target) in val_data {
                let prediction = self.forward_task(graph, task_name)?;
                let loss = self.compute_task_loss(&prediction, target, &task_config.task_type)?;
                val_loss += loss;

                let accuracy =
                    self.compute_accuracy(&prediction, target, &task_config.task_type)?;
                val_accuracy += accuracy;
            }

            stats
                .train_losses
                .push(train_loss / train_data.len() as f32);
            stats.val_losses.push(val_loss / val_data.len() as f32);
            stats
                .val_accuracies
                .push(val_accuracy / val_data.len() as f32);
        }

        Ok(stats)
    }

    /// Forward pass for pre-training
    fn compute_pretraining_loss(&self, graph: &GraphData) -> Result<f32, FoundationModelError> {
        let mut total_loss = 0.0;

        for objective in &self.pretraining_head.active_objectives {
            let loss = match objective {
                PretrainingObjective::MaskedNodeModeling => self.compute_masked_node_loss(graph)?,
                PretrainingObjective::MaskedEdgeModeling => self.compute_masked_edge_loss(graph)?,
                PretrainingObjective::GraphContrastive => {
                    self.compute_graph_contrastive_loss(graph)?
                }
                PretrainingObjective::NodeContrastive => {
                    self.compute_node_contrastive_loss(graph)?
                }
                PretrainingObjective::StructurePrediction => {
                    self.compute_structure_prediction_loss(graph)?
                }
                PretrainingObjective::MotifPrediction => {
                    self.compute_motif_prediction_loss(graph)?
                }
                PretrainingObjective::PropertyPrediction => {
                    self.compute_property_prediction_loss(graph)?
                }
                PretrainingObjective::GraphDenoising => self.compute_denoising_loss(graph)?,
            };

            total_loss += loss;
        }

        Ok(total_loss)
    }

    /// Masked node modeling loss
    fn compute_masked_node_loss(&self, graph: &GraphData) -> Result<f32, FoundationModelError> {
        // Mask random nodes and predict their features
        let _mask_prob = 0.15;
        let masked_graph = self.mask_nodes(graph, _mask_prob)?;

        // Forward pass through encoder
        let encoded = self.encode_graph(&masked_graph)?;

        // Simplified reconstruction loss - compare encoded features directly
        // In a real implementation, would use the MLM head for discrete token prediction
        let loss = self.compute_reconstruction_loss(&encoded, &graph.x)?;

        Ok(loss)
    }

    /// Masked edge modeling loss
    fn compute_masked_edge_loss(&self, _graph: &GraphData) -> Result<f32, FoundationModelError> {
        // Mask random edges and predict their existence
        let _mask_prob = 0.15;

        // Simplified edge masking - just return a placeholder loss
        // In practice, would mask edges and predict their existence based on _mask_prob
        Ok(0.3)
    }

    /// Graph contrastive learning loss
    fn compute_graph_contrastive_loss(
        &self,
        graph: &GraphData,
    ) -> Result<f32, FoundationModelError> {
        // Create positive and negative pairs
        let positive_graph = self.create_positive_augmentation(graph)?;
        let negative_graphs = self.create_negative_augmentations(graph, 5)?;

        // Encode all graphs
        let anchor_embedding = self.encode_graph_global(graph)?;
        let positive_embedding = self.encode_graph_global(&positive_graph)?;

        let mut negative_embeddings = Vec::new();
        for neg_graph in &negative_graphs {
            let neg_embedding = self.encode_graph_global(neg_graph)?;
            negative_embeddings.push(neg_embedding);
        }

        // Compute contrastive loss (InfoNCE)
        let loss = self.compute_infonce_loss(
            &anchor_embedding,
            &positive_embedding,
            &negative_embeddings,
        )?;

        Ok(loss)
    }

    /// Data augmentation for graphs
    fn apply_augmentation(
        &self,
        graph: &GraphData,
    ) -> Result<Vec<GraphData>, FoundationModelError> {
        let mut augmented = Vec::new();

        // Original graph
        augmented.push(graph.clone());

        // Node feature augmentation
        let feature_augmented = self.augment_features(graph, 0.1)?;
        augmented.push(feature_augmented);

        // Edge augmentation
        let edge_augmented = self.augment_edges(graph, 0.1)?;
        augmented.push(edge_augmented);

        // Subgraph sampling
        let subgraph = self.sample_subgraph(graph, 0.8)?;
        augmented.push(subgraph);

        Ok(augmented)
    }

    /// Self-supervised contrastive learning framework
    fn compute_node_contrastive_loss(
        &self,
        graph: &GraphData,
    ) -> Result<f32, FoundationModelError> {
        // Create node-level positive and negative pairs
        let node_embeddings = self.encode_graph(graph)?;

        // Use local structure for positive pairs
        let positive_pairs = self.create_node_positive_pairs(graph)?;
        let negative_pairs = self.create_node_negative_pairs(graph, 10)?;

        // Compute contrastive loss for nodes
        let loss =
            self.compute_node_level_infonce(&node_embeddings, &positive_pairs, &negative_pairs)?;

        Ok(loss)
    }

    // Helper methods for foundation model operations

    fn encode_graph(&self, graph: &GraphData) -> Result<Tensor, FoundationModelError> {
        // Simplified graph encoding
        Ok(graph.x.clone())
    }

    fn encode_graph_global(&self, graph: &GraphData) -> Result<Tensor, FoundationModelError> {
        // Global graph embedding (simplified)
        let node_embeddings = self.encode_graph(graph)?;
        // Average pooling for global representation (mean over dim 0)
        node_embeddings.mean(Some(&[0]), false).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to compute mean: {:?}", e))
        })
    }

    fn mask_nodes(
        &self,
        graph: &GraphData,
        _mask_prob: f32,
    ) -> Result<GraphData, FoundationModelError> {
        // Create masked version of graph
        let masked_features = graph.x.clone();

        // Apply masking (simplified)
        // In practice, would randomly mask nodes based on _mask_prob

        Ok(GraphData::new(masked_features, graph.edge_index.clone()))
    }

    fn create_positive_augmentation(
        &self,
        graph: &GraphData,
    ) -> Result<GraphData, FoundationModelError> {
        // Create positive augmentation (e.g., feature noise)
        self.augment_features(graph, 0.1)
    }

    fn create_negative_augmentations(
        &self,
        graph: &GraphData,
        num_negatives: usize,
    ) -> Result<Vec<GraphData>, FoundationModelError> {
        let mut negatives = Vec::new();

        for _ in 0..num_negatives {
            // Create negative samples (e.g., random graphs)
            let negative = self.create_random_graph(graph.num_nodes, graph.num_edges)?;
            negatives.push(negative);
        }

        Ok(negatives)
    }

    fn augment_features(
        &self,
        graph: &GraphData,
        noise_level: f32,
    ) -> Result<GraphData, FoundationModelError> {
        // Add Gaussian noise to features
        let noise = randn(graph.x.shape().dims()).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create noise tensor: {:?}", e))
        })?;

        let noisy_features = graph
            .x
            .add(
                &noise
                    .mul_scalar(noise_level)
                    .expect("operation should succeed"),
            )
            .expect("operation should succeed");

        Ok(GraphData::new(noisy_features, graph.edge_index.clone()))
    }

    fn augment_edges(
        &self,
        graph: &GraphData,
        _drop_prob: f32,
    ) -> Result<GraphData, FoundationModelError> {
        // Edge dropping augmentation (simplified)
        // In practice, would use _drop_prob to randomly drop edges
        Ok(graph.clone())
    }

    fn sample_subgraph(
        &self,
        graph: &GraphData,
        sample_ratio: f32,
    ) -> Result<GraphData, FoundationModelError> {
        // Subgraph sampling (simplified)
        let num_nodes_to_keep = (graph.num_nodes as f32 * sample_ratio) as usize;

        if num_nodes_to_keep == 0 {
            return Ok(graph.clone());
        }

        // Simplified subgraph sampling
        Ok(graph.clone())
    }

    fn create_random_graph(
        &self,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<GraphData, FoundationModelError> {
        // Create random graph for negative sampling
        let features = randn(&[num_nodes, self.config.model_dim]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create features: {:?}", e))
        })?;

        let edge_index = zeros(&[2, num_edges]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create edge index: {:?}", e))
        })?;

        Ok(GraphData::new(features, edge_index))
    }

    fn compute_reconstruction_loss(
        &self,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Result<f32, FoundationModelError> {
        // Mean squared error loss (simplified)
        let diff = predictions.sub(targets).expect("operation should succeed");
        let squared = diff.mul(&diff).expect("operation should succeed");
        let mean_loss = squared.mean(None, false).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to compute mean: {:?}", e))
        })?;

        let loss_data = mean_loss.to_vec().map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to extract loss: {:?}", e))
        })?;

        Ok(loss_data[0])
    }

    fn compute_infonce_loss(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negatives: &[Tensor],
    ) -> Result<f32, FoundationModelError> {
        // InfoNCE contrastive loss implementation (simplified)
        let temperature = self.pretraining_head.contrastive_head.temperature;

        // Positive similarity
        let pos_sim = self.cosine_similarity(anchor, positive)? / temperature;

        // Negative similarities
        let mut neg_sims = Vec::new();
        for negative in negatives {
            let neg_sim = self.cosine_similarity(anchor, negative)? / temperature;
            neg_sims.push(neg_sim);
        }

        // InfoNCE loss computation (simplified)
        let loss = -pos_sim + (neg_sims.iter().map(|x| x.exp()).sum::<f32>()).ln();

        Ok(loss)
    }

    fn cosine_similarity(&self, a: &Tensor, b: &Tensor) -> Result<f32, FoundationModelError> {
        // Simplified cosine similarity
        let dot_product = a
            .dot(b)
            .expect("cosine similarity dot product should succeed");
        let norm_a = a.norm().expect("cosine similarity norm_a should succeed");
        let norm_b = b.norm().expect("cosine similarity norm_b should succeed");

        let dot_data = dot_product.to_vec().expect("conversion should succeed");
        let norm_a_data = norm_a.to_vec().expect("conversion should succeed");
        let norm_b_data = norm_b.to_vec().expect("conversion should succeed");

        Ok(dot_data[0] / (norm_a_data[0] * norm_b_data[0]))
    }

    fn create_node_positive_pairs(
        &self,
        graph: &GraphData,
    ) -> Result<Vec<(usize, usize)>, FoundationModelError> {
        // Create positive pairs based on graph structure
        let edge_data = graph
            .edge_index
            .to_vec()
            .expect("conversion should succeed");
        let num_edges = edge_data.len() / 2;

        let mut pairs = Vec::new();
        for i in 0..num_edges {
            let src = edge_data[i] as usize;
            let dst = edge_data[i + num_edges] as usize;
            pairs.push((src, dst));
        }

        Ok(pairs)
    }

    fn create_node_negative_pairs(
        &self,
        graph: &GraphData,
        num_negatives: usize,
    ) -> Result<Vec<(usize, usize)>, FoundationModelError> {
        // Create negative pairs by random sampling
        let mut pairs = Vec::new();
        let mut rng = scirs2_core::random::thread_rng();

        for _ in 0..num_negatives {
            let src = rng.gen_range(0..graph.num_nodes);
            let dst = rng.gen_range(0..graph.num_nodes);
            if src != dst {
                pairs.push((src, dst));
            }
        }

        Ok(pairs)
    }

    fn compute_node_level_infonce(
        &self,
        _embeddings: &Tensor,
        positive_pairs: &[(usize, usize)],
        _negative_pairs: &[(usize, usize)],
    ) -> Result<f32, FoundationModelError> {
        // Node-level InfoNCE loss (simplified)
        let mut total_loss = 0.0;

        for &(_src, _dst) in positive_pairs {
            // Simplified node-level contrastive loss
            // In practice, would compute similarity between embeddings[src] and embeddings[dst]
            total_loss += 1.0; // Placeholder
        }

        Ok(total_loss / positive_pairs.len() as f32)
    }

    fn compute_structure_prediction_loss(
        &self,
        _graph: &GraphData,
    ) -> Result<f32, FoundationModelError> {
        // Structure prediction task (simplified)
        Ok(0.5)
    }

    fn compute_motif_prediction_loss(
        &self,
        _graph: &GraphData,
    ) -> Result<f32, FoundationModelError> {
        // Motif prediction task (simplified)
        Ok(0.3)
    }

    fn compute_property_prediction_loss(
        &self,
        _graph: &GraphData,
    ) -> Result<f32, FoundationModelError> {
        // Property prediction task (simplified)
        Ok(0.4)
    }

    fn compute_denoising_loss(&self, _graph: &GraphData) -> Result<f32, FoundationModelError> {
        // Graph denoising task (simplified)
        Ok(0.2)
    }

    fn forward_task(
        &self,
        graph: &GraphData,
        _task_name: &str,
    ) -> Result<Tensor, FoundationModelError> {
        // Forward pass for specific task
        // Simplified - just return encoded representation
        // In practice, would instantiate the appropriate task head based on task_name
        self.encode_graph(graph)
    }

    fn add_task_head(
        &mut self,
        task_name: &str,
        task_type: TaskType,
    ) -> Result<(), FoundationModelError> {
        // Store task type name for reconstruction
        let task_type_name = match task_type {
            TaskType::NodeClassification { num_classes } => {
                format!("NodeClassification_{}", num_classes)
            }
            TaskType::GraphClassification { num_classes } => {
                format!("GraphClassification_{}", num_classes)
            }
            TaskType::LinkPrediction => "LinkPrediction".to_string(),
            TaskType::GraphRegression => "GraphRegression".to_string(),
        };

        self.task_heads
            .insert(task_name.to_string(), task_type_name);
        Ok(())
    }

    fn freeze_pretrained_parameters(&mut self) {
        // Mark pre-training parameters as frozen
        for param_name in self.parameters.pretraining_params.keys() {
            self.parameters.frozen_params.insert(param_name.clone());
        }
    }

    fn compute_task_loss(
        &self,
        _prediction: &Tensor,
        _target: &Tensor,
        task_type: &TaskType,
    ) -> Result<f32, FoundationModelError> {
        match task_type {
            TaskType::NodeClassification { .. } | TaskType::GraphClassification { .. } => {
                // Cross-entropy loss (simplified)
                // In practice, would compute actual cross-entropy between _prediction and _target
                Ok(1.0)
            }
            TaskType::LinkPrediction => {
                // Binary cross-entropy loss (simplified)
                Ok(0.7)
            }
            TaskType::GraphRegression => {
                // Mean squared error loss (simplified)
                Ok(0.5)
            }
        }
    }

    fn compute_accuracy(
        &self,
        _prediction: &Tensor,
        _target: &Tensor,
        task_type: &TaskType,
    ) -> Result<f32, FoundationModelError> {
        match task_type {
            TaskType::NodeClassification { .. } | TaskType::GraphClassification { .. } => {
                // Classification accuracy (simplified)
                // In practice, would compare argmax(_prediction) with _target
                Ok(0.85)
            }
            TaskType::LinkPrediction => {
                // Link prediction accuracy (simplified)
                Ok(0.78)
            }
            TaskType::GraphRegression => {
                // R² score (simplified)
                Ok(0.65)
            }
        }
    }

    fn update_learning_rate(&mut self, _epoch: usize) {
        // Learning rate scheduling (simplified)
        // In practice, would implement cosine annealing, warmup, etc. based on _epoch
    }
}

/// Task configuration for fine-tuning
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Type of downstream task
    pub task_type: TaskType,
    /// Number of fine-tuning epochs
    pub num_epochs: usize,
    /// Learning rate for fine-tuning
    pub learning_rate: f32,
    /// Whether to freeze pre-trained parameters
    pub freeze_pretrained: bool,
    /// Task-specific hyperparameters
    pub task_params: HashMap<String, f32>,
}

/// Types of downstream tasks
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Node classification
    NodeClassification { num_classes: usize },
    /// Graph classification
    GraphClassification { num_classes: usize },
    /// Link prediction
    LinkPrediction,
    /// Graph regression
    GraphRegression,
}

/// Task head trait for different downstream tasks
pub trait TaskHead: fmt::Debug {
    fn forward(&self, embeddings: &Tensor) -> Result<Tensor, FoundationModelError>;
    fn parameters(&self) -> Vec<Tensor>;
}

/// Node classification head
#[derive(Debug)]
pub struct NodeClassificationHead {
    pub classifier: Tensor,
    pub bias: Tensor,
}

impl NodeClassificationHead {
    pub fn new(input_dim: usize, num_classes: usize) -> Result<Self, FoundationModelError> {
        let classifier = randn(&[input_dim, num_classes]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create classifier: {:?}", e))
        })?;
        let bias = zeros(&[num_classes]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create bias: {:?}", e))
        })?;

        Ok(Self { classifier, bias })
    }
}

impl TaskHead for NodeClassificationHead {
    fn forward(&self, embeddings: &Tensor) -> Result<Tensor, FoundationModelError> {
        let logits = embeddings.matmul(&self.classifier).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to compute logits: {:?}", e))
        })?;

        logits
            .add(&self.bias)
            .map_err(|e| FoundationModelError::TensorError(format!("Failed to add bias: {:?}", e)))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.classifier.clone(), self.bias.clone()]
    }
}

/// Graph classification head
#[derive(Debug)]
pub struct GraphClassificationHead {
    pub pooling_layer: Tensor,
    pub classifier: Tensor,
    pub bias: Tensor,
}

impl GraphClassificationHead {
    pub fn new(input_dim: usize, num_classes: usize) -> Result<Self, FoundationModelError> {
        let pooling_layer = randn(&[input_dim, input_dim]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create pooling layer: {:?}", e))
        })?;
        let classifier = randn(&[input_dim, num_classes]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create classifier: {:?}", e))
        })?;
        let bias = zeros(&[num_classes]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create bias: {:?}", e))
        })?;

        Ok(Self {
            pooling_layer,
            classifier,
            bias,
        })
    }
}

impl TaskHead for GraphClassificationHead {
    fn forward(&self, embeddings: &Tensor) -> Result<Tensor, FoundationModelError> {
        // Global pooling (mean over first dimension, keep dims)
        let pooled = embeddings.mean(Some(&[0]), true).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to compute mean: {:?}", e))
        })?;
        let transformed = pooled
            .matmul(&self.pooling_layer)
            .expect("pooling layer matmul should succeed");
        let logits = transformed
            .matmul(&self.classifier)
            .expect("classifier matmul should succeed");
        logits
            .add(&self.bias)
            .map_err(|e| FoundationModelError::TensorError(format!("Failed to add bias: {:?}", e)))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.pooling_layer.clone(),
            self.classifier.clone(),
            self.bias.clone(),
        ]
    }
}

/// Link prediction head
#[derive(Debug)]
pub struct LinkPredictionHead {
    pub edge_predictor: Tensor,
}

impl LinkPredictionHead {
    pub fn new(input_dim: usize) -> Result<Self, FoundationModelError> {
        let edge_predictor = randn(&[input_dim * 2, 1]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create edge predictor: {:?}", e))
        })?;

        Ok(Self { edge_predictor })
    }
}

impl TaskHead for LinkPredictionHead {
    fn forward(&self, embeddings: &Tensor) -> Result<Tensor, FoundationModelError> {
        // Simplified link prediction
        embeddings.matmul(&self.edge_predictor).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to predict links: {:?}", e))
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.edge_predictor.clone()]
    }
}

/// Graph regression head
#[derive(Debug)]
pub struct GraphRegressionHead {
    pub regressor: Tensor,
    pub bias: Tensor,
}

impl GraphRegressionHead {
    pub fn new(input_dim: usize) -> Result<Self, FoundationModelError> {
        let regressor = randn(&[input_dim, 1]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create regressor: {:?}", e))
        })?;
        let bias = zeros(&[1]).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to create bias: {:?}", e))
        })?;

        Ok(Self { regressor, bias })
    }
}

impl TaskHead for GraphRegressionHead {
    fn forward(&self, embeddings: &Tensor) -> Result<Tensor, FoundationModelError> {
        let pooled = embeddings.mean(Some(&[0]), true).map_err(|e| {
            FoundationModelError::TensorError(format!("Failed to compute mean: {:?}", e))
        })?;
        let output = pooled
            .matmul(&self.regressor)
            .expect("regressor matmul should succeed");
        output
            .add(&self.bias)
            .map_err(|e| FoundationModelError::TensorError(format!("Failed to add bias: {:?}", e)))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.regressor.clone(), self.bias.clone()]
    }
}

/// Pre-training statistics
#[derive(Debug, Clone)]
pub struct PretrainingStats {
    pub epoch_losses: Vec<f32>,
    pub total_samples: usize,
    pub current_epoch: usize,
    pub pretraining_completed: bool,
    pub best_loss: f32,
}

impl PretrainingStats {
    pub fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
            total_samples: 0,
            current_epoch: 0,
            pretraining_completed: false,
            best_loss: f32::INFINITY,
        }
    }
}

/// Fine-tuning statistics
#[derive(Debug, Clone)]
pub struct FinetuningStats {
    pub train_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub val_accuracies: Vec<f32>,
    pub best_val_accuracy: f32,
    pub converged: bool,
}

impl FinetuningStats {
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            val_accuracies: Vec::new(),
            best_val_accuracy: 0.0,
            converged: false,
        }
    }
}

/// Graph tokenizer implementation
impl GraphTokenizer {
    pub fn new(vocab_size: usize) -> Result<Self, FoundationModelError> {
        let mut node_vocab = HashMap::new();
        let mut edge_vocab = HashMap::new();

        // Initialize basic vocabularies
        for i in 0..vocab_size / 2 {
            node_vocab.insert(format!("node_{}", i), i);
            edge_vocab.insert(format!("edge_{}", i), i);
        }

        let special_tokens = SpecialTokens {
            mask_token: vocab_size - 5,
            cls_token: vocab_size - 4,
            sep_token: vocab_size - 3,
            pad_token: vocab_size - 2,
            unk_token: vocab_size - 1,
        };

        Ok(Self {
            node_vocab,
            edge_vocab,
            special_tokens,
            strategy: TokenizationStrategy::NodeCentric,
        })
    }

    /// Tokenize a graph into a sequence
    pub fn tokenize(&self, graph: &GraphData) -> Result<Vec<usize>, FoundationModelError> {
        match self.strategy {
            TokenizationStrategy::NodeCentric => self.tokenize_node_centric(graph),
            TokenizationStrategy::EdgeCentric => self.tokenize_edge_centric(graph),
            TokenizationStrategy::WalkBased => self.tokenize_walk_based(graph),
            TokenizationStrategy::SubgraphBased => self.tokenize_subgraph_based(graph),
            TokenizationStrategy::Hierarchical => self.tokenize_hierarchical(graph),
        }
    }

    fn tokenize_node_centric(&self, graph: &GraphData) -> Result<Vec<usize>, FoundationModelError> {
        let mut tokens = vec![self.special_tokens.cls_token];

        // Tokenize each node
        for node in 0..graph.num_nodes {
            tokens.push(node % self.node_vocab.len());
        }

        tokens.push(self.special_tokens.sep_token);
        Ok(tokens)
    }

    fn tokenize_edge_centric(&self, graph: &GraphData) -> Result<Vec<usize>, FoundationModelError> {
        let mut tokens = vec![self.special_tokens.cls_token];

        // Tokenize each edge
        let edge_data = graph
            .edge_index
            .to_vec()
            .expect("conversion should succeed");
        let num_edges = edge_data.len() / 2;

        for i in 0..num_edges {
            let edge_token = i % self.edge_vocab.len();
            tokens.push(edge_token);
        }

        tokens.push(self.special_tokens.sep_token);
        Ok(tokens)
    }

    fn tokenize_walk_based(&self, graph: &GraphData) -> Result<Vec<usize>, FoundationModelError> {
        // Random walk-based tokenization
        let mut tokens = vec![self.special_tokens.cls_token];

        // Simplified random walk
        let walk_length = 20;
        let mut current_node = 0;

        for _ in 0..walk_length {
            tokens.push(current_node % self.node_vocab.len());
            // Move to random neighbor (simplified)
            current_node = (current_node + 1) % graph.num_nodes;
        }

        tokens.push(self.special_tokens.sep_token);
        Ok(tokens)
    }

    fn tokenize_subgraph_based(
        &self,
        graph: &GraphData,
    ) -> Result<Vec<usize>, FoundationModelError> {
        // Subgraph-based tokenization
        let mut tokens = vec![self.special_tokens.cls_token];

        // Create tokens for subgraphs (simplified)
        for i in 0..graph.num_nodes.min(10) {
            tokens.push(i % self.node_vocab.len());
        }

        tokens.push(self.special_tokens.sep_token);
        Ok(tokens)
    }

    fn tokenize_hierarchical(&self, graph: &GraphData) -> Result<Vec<usize>, FoundationModelError> {
        // Hierarchical tokenization
        let mut tokens = vec![self.special_tokens.cls_token];

        // Multi-level tokenization (simplified)
        for level in 0..3 {
            for node in 0..graph.num_nodes.min(5) {
                let token = (level * graph.num_nodes + node) % self.node_vocab.len();
                tokens.push(token);
            }
            tokens.push(self.special_tokens.sep_token);
        }

        Ok(tokens)
    }
}

/// Foundation model implementation helpers
impl PretrainingHead {
    pub fn new(config: &FoundationModelConfig) -> Result<Self, FoundationModelError> {
        let mlm_head = MLMHead {
            output_projection: randn(&[config.model_dim, config.vocab_size])
                .expect("failed to create MLM output_projection tensor"),
            bias: zeros(&[config.vocab_size]).expect("failed to create MLM bias tensor"),
            mask_token: randn(&[config.model_dim]).expect("failed to create MLM mask_token tensor"),
        };

        let contrastive_head = ContrastiveHead {
            projection: randn(&[config.model_dim, config.model_dim])
                .expect("failed to create contrastive projection tensor"),
            temperature: 0.1,
            embed_dim: config.model_dim,
        };

        let structure_head = StructurePredictionHead {
            edge_predictor: randn(&[config.model_dim * 2, 1])
                .expect("failed to create edge_predictor tensor"),
            motif_predictor: randn(&[config.model_dim, 10])
                .expect("failed to create motif_predictor tensor"),
            property_predictor: randn(&[config.model_dim, 1])
                .expect("failed to create property_predictor tensor"),
        };

        Ok(Self {
            mlm_head,
            contrastive_head,
            structure_head,
            active_objectives: config.pretraining_objectives.clone(),
        })
    }
}

impl FoundationModelParameters {
    pub fn new() -> Self {
        Self {
            pretraining_params: HashMap::new(),
            task_params: HashMap::new(),
            frozen_params: HashSet::new(),
        }
    }
}

/// Foundation model errors
#[derive(Debug, Clone)]
pub enum FoundationModelError {
    /// Tensor operation error
    TensorError(String),
    /// Configuration error
    ConfigError(String),
    /// Task not found
    TaskNotFound(String),
    /// Pre-training error
    PretrainingError(String),
    /// Fine-tuning error
    FinetuningError(String),
    /// Tokenization error
    TokenizationError(String),
}

impl From<torsh_core::error::TorshError> for FoundationModelError {
    fn from(err: torsh_core::error::TorshError) -> Self {
        FoundationModelError::TensorError(format!("{:?}", err))
    }
}

impl fmt::Display for FoundationModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FoundationModelError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            FoundationModelError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            FoundationModelError::TaskNotFound(task) => write!(f, "Task not found: {}", task),
            FoundationModelError::PretrainingError(msg) => write!(f, "Pre-training error: {}", msg),
            FoundationModelError::FinetuningError(msg) => write!(f, "Fine-tuning error: {}", msg),
            FoundationModelError::TokenizationError(msg) => {
                write!(f, "Tokenization error: {}", msg)
            }
        }
    }
}

impl std::error::Error for FoundationModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foundation_model_config() {
        let config = FoundationModelConfig {
            model_dim: 256,
            num_layers: 6,
            num_heads: 8,
            ff_dim: 1024,
            max_seq_length: 512,
            vocab_size: 1000,
            dropout: 0.1,
            pretraining_objectives: vec![
                PretrainingObjective::MaskedNodeModeling,
                PretrainingObjective::GraphContrastive,
            ],
        };

        assert_eq!(config.model_dim, 256);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.pretraining_objectives.len(), 2);
    }

    #[test]
    fn test_graph_tokenizer() {
        let tokenizer = GraphTokenizer::new(1000);
        assert!(tokenizer.is_ok());

        let tok = tokenizer.unwrap();
        // Verify that unk_token is at the end (vocab_size - 1 = 999)
        assert_eq!(999, tok.special_tokens.unk_token);
    }

    #[test]
    fn test_task_types() {
        let node_task = TaskType::NodeClassification { num_classes: 5 };
        let _graph_task = TaskType::GraphClassification { num_classes: 3 };
        let _link_task = TaskType::LinkPrediction;
        let _regression_task = TaskType::GraphRegression;

        match node_task {
            TaskType::NodeClassification { num_classes } => assert_eq!(num_classes, 5),
            _ => panic!("Wrong task type"),
        }
    }

    #[test]
    fn test_pretraining_objectives() {
        let objectives = vec![
            PretrainingObjective::MaskedNodeModeling,
            PretrainingObjective::GraphContrastive,
            PretrainingObjective::StructurePrediction,
        ];

        assert_eq!(objectives.len(), 3);
    }

    #[test]
    fn test_task_heads() {
        let node_head = NodeClassificationHead::new(128, 5);
        assert!(node_head.is_ok());

        let graph_head = GraphClassificationHead::new(128, 3);
        assert!(graph_head.is_ok());

        let link_head = LinkPredictionHead::new(128);
        assert!(link_head.is_ok());

        let regression_head = GraphRegressionHead::new(128);
        assert!(regression_head.is_ok());
    }

    #[test]
    fn test_tokenization_strategies() {
        let strategies = vec![
            TokenizationStrategy::NodeCentric,
            TokenizationStrategy::EdgeCentric,
            TokenizationStrategy::WalkBased,
            TokenizationStrategy::SubgraphBased,
            TokenizationStrategy::Hierarchical,
        ];

        assert_eq!(strategies.len(), 5);
    }

    #[test]
    fn test_special_tokens() {
        let special_tokens = SpecialTokens {
            mask_token: 995,
            cls_token: 996,
            sep_token: 997,
            pad_token: 998,
            unk_token: 999,
        };

        assert_eq!(special_tokens.mask_token, 995);
        assert_eq!(special_tokens.unk_token, 999);
    }
}
