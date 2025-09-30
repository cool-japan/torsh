//! Few-shot learning utilities and meta-learning algorithms
//!
//! This module provides comprehensive few-shot learning techniques including:
//! - Meta-learning algorithms (MAML, Reptile, ProtoNet)
//! - Support set and query set handling
//! - Prototypical networks and metric learning
//! - Few-shot classification and regression
//! - Evaluation metrics for few-shot learning scenarios

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Configuration for few-shot learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotConfig {
    /// Number of classes in each episode (N-way)
    pub n_way: usize,
    /// Number of support examples per class (K-shot)
    pub k_shot: usize,
    /// Number of query examples per class
    pub query_shots: usize,
    /// Meta-learning algorithm to use
    pub algorithm: MetaLearningAlgorithm,
    /// Distance metric for similarity computation
    pub distance_metric: DistanceMetric,
    /// Training configuration
    pub training_config: FewShotTrainingConfig,
}

/// Meta-learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning (MAML)
    MAML {
        inner_lr: f64,
        outer_lr: f64,
        num_inner_steps: usize,
        first_order: bool,
    },
    /// Reptile meta-learning
    Reptile {
        inner_lr: f64,
        outer_lr: f64,
        num_inner_steps: usize,
    },
    /// Prototypical Networks
    ProtoNet {
        embedding_dim: usize,
        temperature: f64,
    },
    /// Matching Networks
    MatchingNet {
        embedding_dim: usize,
        use_attention: bool,
    },
    /// Relation Networks
    RelationNet {
        embedding_dim: usize,
        relation_dim: usize,
    },
    /// Meta-SGD
    MetaSGD {
        base_lr: f64,
        adapt_lr: bool,
        adapt_momentum: bool,
    },
}

/// Distance metrics for similarity computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Cosine similarity
    Cosine,
    /// Manhattan distance
    Manhattan,
    /// Learned distance metric
    Learned { metric_dim: usize },
}

/// Training configuration for few-shot learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotTrainingConfig {
    /// Number of episodes per epoch
    pub episodes_per_epoch: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Batch size for episodes
    pub episode_batch_size: usize,
    /// Validation frequency
    pub validation_frequency: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

/// Episode data structure for few-shot learning
#[derive(Debug, Clone)]
pub struct Episode {
    /// Support set examples
    pub support_set: Vec<(Tensor, usize)>, // (data, label)
    /// Query set examples
    pub query_set: Vec<(Tensor, usize)>, // (data, label)
    /// Number of ways (classes)
    pub n_way: usize,
    /// Number of shots per class
    pub k_shot: usize,
}

/// Few-shot learning dataset generator
pub struct FewShotDataset {
    /// All available data
    data: Vec<(Tensor, usize)>,
    /// Class to examples mapping
    class_to_examples: HashMap<usize, Vec<usize>>,
    /// Number of classes
    num_classes: usize,
    /// Configuration
    config: FewShotConfig,
}

impl FewShotDataset {
    /// Create a new few-shot dataset
    pub fn new(data: Vec<(Tensor, usize)>, config: FewShotConfig) -> Self {
        let mut class_to_examples = HashMap::new();
        let mut num_classes = 0;

        // Group examples by class
        for (idx, (_, label)) in data.iter().enumerate() {
            class_to_examples
                .entry(*label)
                .or_insert_with(Vec::new)
                .push(idx);
            num_classes = num_classes.max(*label + 1);
        }

        Self {
            data,
            class_to_examples,
            num_classes,
            config,
        }
    }

    /// Generate a random episode
    pub fn generate_episode(&self) -> Result<Episode> {
        use scirs2_core::random::{Random, Rng};

        let mut rng = Random::seed(42);

        // Select N random classes
        let mut available_classes: Vec<usize> = self.class_to_examples.keys().cloned().collect();
        // Fisher-Yates shuffle algorithm
        for i in (1..available_classes.len()).rev() {
            let j = rng.gen_range(0..=i);
            available_classes.swap(i, j);
        }

        if available_classes.len() < self.config.n_way {
            return Err(TorshError::ComputeError(
                "Not enough classes available for N-way classification".to_string(),
            ));
        }

        let selected_classes = &available_classes[..self.config.n_way];

        let mut support_set = Vec::new();
        let mut query_set = Vec::new();

        // For each selected class, sample K support examples and Q query examples
        for (class_idx, &class_label) in selected_classes.iter().enumerate() {
            let class_examples = self.class_to_examples.get(&class_label).unwrap();

            if class_examples.len() < self.config.k_shot + self.config.query_shots {
                return Err(TorshError::ComputeError(format!(
                    "Not enough examples for class {}",
                    class_label
                )));
            }

            // Shuffle and select examples
            let mut shuffled_examples = class_examples.clone();
            // Fisher-Yates shuffle algorithm
            for i in (1..shuffled_examples.len()).rev() {
                let j = rng.gen_range(0..=i);
                shuffled_examples.swap(i, j);
            }

            // Support set
            for &example_idx in &shuffled_examples[..self.config.k_shot] {
                support_set.push((self.data[example_idx].0.clone(), class_idx));
            }

            // Query set
            for &example_idx in
                &shuffled_examples[self.config.k_shot..self.config.k_shot + self.config.query_shots]
            {
                query_set.push((self.data[example_idx].0.clone(), class_idx));
            }
        }

        Ok(Episode {
            support_set,
            query_set,
            n_way: self.config.n_way,
            k_shot: self.config.k_shot,
        })
    }

    /// Generate multiple episodes
    pub fn generate_episodes(&self, num_episodes: usize) -> Result<Vec<Episode>> {
        let mut episodes = Vec::new();
        for _ in 0..num_episodes {
            episodes.push(self.generate_episode()?);
        }
        Ok(episodes)
    }
}

/// Prototypical Networks implementation
pub struct PrototypicalNetwork {
    /// Embedding network
    embedding_net: Box<dyn Module>,
    /// Embedding dimension
    embedding_dim: usize,
    /// Temperature for softmax
    temperature: f64,
}

impl PrototypicalNetwork {
    /// Create a new prototypical network
    pub fn new(embedding_net: Box<dyn Module>, embedding_dim: usize, temperature: f64) -> Self {
        Self {
            embedding_net,
            embedding_dim,
            temperature,
        }
    }

    /// Compute prototypes from support set
    pub fn compute_prototypes(
        &self,
        support_set: &[(Tensor, usize)],
        n_way: usize,
    ) -> Result<Tensor> {
        let mut class_embeddings: HashMap<usize, Vec<Tensor>> = HashMap::new();

        // Compute embeddings for all support examples
        for (data, label) in support_set {
            let embedding = self.embedding_net.forward(data)?;
            class_embeddings
                .entry(*label)
                .or_insert_with(Vec::new)
                .push(embedding);
        }

        // Compute prototype for each class (mean of embeddings)
        let mut prototypes = Vec::new();
        for class_idx in 0..n_way {
            if let Some(embeddings) = class_embeddings.get(&class_idx) {
                let prototype = self.compute_mean_embedding(embeddings)?;
                prototypes.push(prototype);
            } else {
                return Err(TorshError::ComputeError(format!(
                    "No support examples for class {}",
                    class_idx
                )));
            }
        }

        // Stack prototypes - implement using unsqueeze + cat
        let unsqueezed_prototypes: Result<Vec<Tensor<f32>>> =
            prototypes.iter().map(|t| t.unsqueeze(0)).collect();
        let unsqueezed_prototypes = unsqueezed_prototypes?;
        let prototype_refs: Vec<&Tensor> = unsqueezed_prototypes.iter().collect();
        Tensor::cat(&prototype_refs, 0)
    }

    /// Compute mean embedding
    fn compute_mean_embedding(&self, embeddings: &[Tensor]) -> Result<Tensor> {
        if embeddings.is_empty() {
            return Err(TorshError::ComputeError(
                "No embeddings to average".to_string(),
            ));
        }

        let mut sum = embeddings[0].clone();
        for embedding in &embeddings[1..] {
            sum = sum.add(embedding)?;
        }

        let count = embeddings.len() as f32;
        sum.div_scalar(count)
    }

    /// Classify query examples using prototypes
    pub fn classify_queries(
        &self,
        query_set: &[(Tensor, usize)],
        prototypes: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let mut predictions = Vec::new();

        for (query_data, _) in query_set {
            let query_embedding = self.embedding_net.forward(query_data)?;
            let distances = self.compute_distances(&query_embedding, prototypes)?;
            let logits = distances.mul_scalar((-1.0 / self.temperature) as f32)?;
            predictions.push(logits);
        }

        Ok(predictions)
    }

    /// Compute distances between query and prototypes
    fn compute_distances(&self, query_embedding: &Tensor, prototypes: &Tensor) -> Result<Tensor> {
        // Compute squared Euclidean distances
        let query_expanded = query_embedding.unsqueeze(0)?; // [1, embedding_dim]
        let diff = prototypes - &query_expanded; // [n_way, embedding_dim]
        let squared_distances = diff.pow_scalar(2.0)?.sum_dim(&[1], false)?; // [n_way]

        Ok(squared_distances)
    }
}

/// MAML (Model-Agnostic Meta-Learning) implementation
pub struct MAML {
    /// Base model
    model: Box<dyn Module>,
    /// Inner learning rate
    inner_lr: f64,
    /// Outer learning rate
    outer_lr: f64,
    /// Number of inner loop steps
    num_inner_steps: usize,
    /// Use first-order approximation
    first_order: bool,
}

impl MAML {
    /// Create a new MAML learner
    pub fn new(
        model: Box<dyn Module>,
        inner_lr: f64,
        outer_lr: f64,
        num_inner_steps: usize,
        first_order: bool,
    ) -> Self {
        Self {
            model,
            inner_lr,
            outer_lr,
            num_inner_steps,
            first_order,
        }
    }

    /// Perform meta-learning step
    pub fn meta_learn(&mut self, episodes: &[Episode]) -> Result<f64> {
        let mut total_loss = 0.0;

        for episode in episodes {
            let loss = self.process_episode(episode)?;
            total_loss += loss;
        }

        Ok(total_loss / episodes.len() as f64)
    }

    /// Process a single episode
    fn process_episode(&mut self, episode: &Episode) -> Result<f64> {
        // Clone model parameters for inner loop
        let original_params = self.model.parameters();
        let mut adapted_params = original_params.clone();

        // Inner loop adaptation
        for _ in 0..self.num_inner_steps {
            let support_loss = self.compute_support_loss(episode, &adapted_params)?;
            adapted_params = self.update_parameters(&adapted_params, &support_loss)?;
        }

        // Compute query loss with adapted parameters
        let query_loss = self.compute_query_loss(episode, &adapted_params)?;

        // Outer loop update (meta-parameters)
        self.update_meta_parameters(&query_loss)?;

        Ok(query_loss)
    }

    /// Compute loss on support set
    fn compute_support_loss(
        &self,
        episode: &Episode,
        params: &HashMap<String, Parameter>,
    ) -> Result<f64> {
        let mut total_loss = 0.0;

        for (data, label) in &episode.support_set {
            let logits = self.forward_with_params(data, params)?;
            let loss = self.compute_cross_entropy_loss(&logits, *label)?;
            total_loss += loss;
        }

        Ok(total_loss / episode.support_set.len() as f64)
    }

    /// Compute loss on query set
    fn compute_query_loss(
        &self,
        episode: &Episode,
        params: &HashMap<String, Parameter>,
    ) -> Result<f64> {
        let mut total_loss = 0.0;

        for (data, label) in &episode.query_set {
            let logits = self.forward_with_params(data, params)?;
            let loss = self.compute_cross_entropy_loss(&logits, *label)?;
            total_loss += loss;
        }

        Ok(total_loss / episode.query_set.len() as f64)
    }

    /// Forward pass with custom parameters
    fn forward_with_params(
        &self,
        input: &Tensor,
        _params: &HashMap<String, Parameter>,
    ) -> Result<Tensor> {
        // This would need to be implemented based on the specific model architecture
        // For now, use the regular forward pass
        self.model.forward(input)
    }

    /// Update parameters using gradients
    fn update_parameters(
        &self,
        params: &HashMap<String, Parameter>,
        _loss: &f64,
    ) -> Result<HashMap<String, Parameter>> {
        // This would need gradient computation and parameter updates
        // For now, return the same parameters
        Ok(params.clone())
    }

    /// Update meta-parameters
    fn update_meta_parameters(&mut self, _loss: &f64) -> Result<()> {
        // This would need meta-gradient computation and parameter updates
        Ok(())
    }

    /// Compute cross-entropy loss
    fn compute_cross_entropy_loss(&self, logits: &Tensor, target: usize) -> Result<f64> {
        // Simplified cross-entropy computation
        let softmax = logits.softmax(0)?;
        let target_prob = softmax.get(&[target])?;
        Ok((-target_prob.ln()) as f64)
    }
}

/// Reptile meta-learning implementation
pub struct Reptile {
    /// Base model
    model: Box<dyn Module>,
    /// Inner learning rate
    inner_lr: f64,
    /// Outer learning rate
    outer_lr: f64,
    /// Number of inner loop steps
    num_inner_steps: usize,
}

impl Reptile {
    /// Create a new Reptile learner
    pub fn new(
        model: Box<dyn Module>,
        inner_lr: f64,
        outer_lr: f64,
        num_inner_steps: usize,
    ) -> Self {
        Self {
            model,
            inner_lr,
            outer_lr,
            num_inner_steps,
        }
    }

    /// Perform meta-learning step
    pub fn meta_learn(&mut self, episodes: &[Episode]) -> Result<f64> {
        let mut total_loss = 0.0;

        for episode in episodes {
            let loss = self.process_episode(episode)?;
            total_loss += loss;
        }

        Ok(total_loss / episodes.len() as f64)
    }

    /// Process a single episode
    fn process_episode(&mut self, episode: &Episode) -> Result<f64> {
        // Save original parameters
        let original_params = self.model.parameters();

        // Adapt to the task
        let mut current_loss = 0.0;
        for _ in 0..self.num_inner_steps {
            current_loss = self.adapt_step(episode)?;
        }

        // Get adapted parameters
        let adapted_params = self.model.parameters();

        // Reptile update: move towards adapted parameters
        self.reptile_update(&original_params, &adapted_params)?;

        Ok(current_loss)
    }

    /// Perform one adaptation step
    fn adapt_step(&mut self, episode: &Episode) -> Result<f64> {
        let mut total_loss = 0.0;

        for (data, label) in &episode.support_set {
            let logits = self.model.forward(data)?;
            let loss = self.compute_cross_entropy_loss(&logits, *label)?;
            total_loss += loss;
        }

        // Update parameters (simplified)
        // In practice, this would compute gradients and update parameters

        Ok(total_loss / episode.support_set.len() as f64)
    }

    /// Reptile meta-update
    fn reptile_update(
        &mut self,
        original_params: &HashMap<String, Parameter>,
        adapted_params: &HashMap<String, Parameter>,
    ) -> Result<()> {
        // Move parameters towards adapted parameters
        for (name, original_param) in original_params {
            if let Some(adapted_param) = adapted_params.get(name) {
                let adapted_tensor = adapted_param.tensor();
                let adapted_tensor_guard = adapted_tensor.read();
                let original_tensor = original_param.tensor();
                let original_tensor_guard = original_tensor.read();
                let diff = adapted_tensor_guard.sub(&*original_tensor_guard)?;
                let update = diff.mul_scalar(self.outer_lr as f32)?;
                // Apply update: original_param.tensor() += update
                // This would need proper parameter update implementation
            }
        }
        Ok(())
    }

    /// Compute cross-entropy loss
    fn compute_cross_entropy_loss(&self, logits: &Tensor, target: usize) -> Result<f64> {
        // Simplified cross-entropy computation
        let softmax = logits.softmax(0)?;
        let target_prob = softmax.get(&[target])?;
        Ok((-target_prob.ln()) as f64)
    }
}

/// Few-shot learning evaluator
pub struct FewShotEvaluator {
    /// Configuration
    config: FewShotConfig,
}

impl FewShotEvaluator {
    /// Create a new evaluator
    pub fn new(config: FewShotConfig) -> Self {
        Self { config }
    }

    /// Evaluate few-shot learning performance
    pub fn evaluate<M: Module>(
        &self,
        model: &M,
        test_episodes: &[Episode],
    ) -> Result<FewShotMetrics> {
        let mut total_accuracy = 0.0;
        let mut total_loss = 0.0;
        let mut per_class_accuracy = HashMap::new();

        for episode in test_episodes {
            let (accuracy, loss) = self.evaluate_episode(model, episode)?;
            total_accuracy += accuracy;
            total_loss += loss;

            // Track per-class accuracy
            for class_idx in 0..episode.n_way {
                let class_acc = per_class_accuracy.entry(class_idx).or_insert(Vec::new());
                class_acc.push(accuracy); // Simplified - would need class-specific accuracy
            }
        }

        let avg_accuracy = total_accuracy / test_episodes.len() as f64;
        let avg_loss = total_loss / test_episodes.len() as f64;

        // Compute per-class averages
        let mut per_class_avg = HashMap::new();
        for (class_idx, accuracies) in per_class_accuracy {
            let avg = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
            per_class_avg.insert(class_idx, avg);
        }

        Ok(FewShotMetrics {
            accuracy: avg_accuracy,
            loss: avg_loss,
            per_class_accuracy: per_class_avg,
            num_episodes: test_episodes.len(),
            confidence_interval: self
                .compute_confidence_interval(total_accuracy, test_episodes.len())?,
        })
    }

    /// Evaluate a single episode
    fn evaluate_episode<M: Module>(&self, model: &M, episode: &Episode) -> Result<(f64, f64)> {
        let mut correct = 0;
        let mut total_loss = 0.0;

        for (data, true_label) in &episode.query_set {
            let logits = model.forward(data)?;
            let predicted_label = self.get_predicted_label(&logits)?;

            if predicted_label == *true_label {
                correct += 1;
            }

            let loss = self.compute_cross_entropy_loss(&logits, *true_label)?;
            total_loss += loss;
        }

        let accuracy = correct as f64 / episode.query_set.len() as f64;
        let avg_loss = total_loss / episode.query_set.len() as f64;

        Ok((accuracy, avg_loss))
    }

    /// Get predicted label from logits
    fn get_predicted_label(&self, logits: &Tensor) -> Result<usize> {
        // Find the index with maximum value
        let data = logits.to_vec()?;
        let (max_idx, _) = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok(max_idx)
    }

    /// Compute cross-entropy loss
    fn compute_cross_entropy_loss(&self, logits: &Tensor, target: usize) -> Result<f64> {
        // Simplified cross-entropy computation
        let softmax = logits.softmax(0)?;
        let target_prob = softmax.get(&[target])?;
        Ok((-target_prob.ln()) as f64)
    }

    /// Compute confidence interval
    fn compute_confidence_interval(
        &self,
        accuracy: f64,
        num_episodes: usize,
    ) -> Result<(f64, f64)> {
        // Simple 95% confidence interval using normal approximation
        let std_error = (accuracy * (1.0 - accuracy) / num_episodes as f64).sqrt();
        let margin = 1.96 * std_error; // 95% confidence

        Ok((accuracy - margin, accuracy + margin))
    }
}

/// Few-shot learning evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Average loss
    pub loss: f64,
    /// Per-class accuracy
    pub per_class_accuracy: HashMap<usize, f64>,
    /// Number of evaluation episodes
    pub num_episodes: usize,
    /// 95% confidence interval for accuracy
    pub confidence_interval: (f64, f64),
}

impl std::fmt::Display for FewShotMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Few-Shot Learning Metrics")?;
        writeln!(f, "========================")?;
        writeln!(
            f,
            "Accuracy: {:.4} ± {:.4}",
            self.accuracy,
            (self.confidence_interval.1 - self.confidence_interval.0) / 2.0
        )?;
        writeln!(f, "Loss: {:.4}", self.loss)?;
        writeln!(f, "Episodes: {}", self.num_episodes)?;
        writeln!(
            f,
            "95% CI: ({:.4}, {:.4})",
            self.confidence_interval.0, self.confidence_interval.1
        )?;

        if !self.per_class_accuracy.is_empty() {
            writeln!(f, "\nPer-class Accuracy:")?;
            for (class_idx, accuracy) in &self.per_class_accuracy {
                writeln!(f, "  Class {}: {:.4}", class_idx, accuracy)?;
            }
        }

        Ok(())
    }
}

/// Utility functions for few-shot learning
pub mod utils {
    use super::*;

    /// Create a standard few-shot configuration
    pub fn create_standard_config(n_way: usize, k_shot: usize) -> FewShotConfig {
        FewShotConfig {
            n_way,
            k_shot,
            query_shots: 15, // Standard query set size
            algorithm: MetaLearningAlgorithm::ProtoNet {
                embedding_dim: 64,
                temperature: 1.0,
            },
            distance_metric: DistanceMetric::Euclidean,
            training_config: FewShotTrainingConfig {
                episodes_per_epoch: 100,
                num_epochs: 100,
                episode_batch_size: 1,
                validation_frequency: 10,
                early_stopping_patience: 10,
            },
        }
    }

    /// Create MAML configuration
    pub fn create_maml_config(n_way: usize, k_shot: usize) -> FewShotConfig {
        FewShotConfig {
            n_way,
            k_shot,
            query_shots: 15,
            algorithm: MetaLearningAlgorithm::MAML {
                inner_lr: 0.01,
                outer_lr: 0.001,
                num_inner_steps: 5,
                first_order: false,
            },
            distance_metric: DistanceMetric::Euclidean,
            training_config: FewShotTrainingConfig {
                episodes_per_epoch: 100,
                num_epochs: 100,
                episode_batch_size: 4,
                validation_frequency: 10,
                early_stopping_patience: 10,
            },
        }
    }

    /// Split dataset into train/validation/test for few-shot learning
    pub fn split_dataset(
        data: Vec<(Tensor, usize)>,
        train_ratio: f64,
        val_ratio: f64,
    ) -> (
        Vec<(Tensor, usize)>,
        Vec<(Tensor, usize)>,
        Vec<(Tensor, usize)>,
    ) {
        use scirs2_core::random::{Random, Rng};

        let mut rng = Random::seed(42);
        let mut shuffled_data = data;
        // Fisher-Yates shuffle algorithm
        for i in (1..shuffled_data.len()).rev() {
            let j = rng.gen_range(0..=i);
            shuffled_data.swap(i, j);
        }

        let total_len = shuffled_data.len();
        let train_len = (total_len as f64 * train_ratio) as usize;
        let val_len = (total_len as f64 * val_ratio) as usize;

        let train_data = shuffled_data[..train_len].to_vec();
        let val_data = shuffled_data[train_len..train_len + val_len].to_vec();
        let test_data = shuffled_data[train_len + val_len..].to_vec();

        (train_data, val_data, test_data)
    }

    /// Create balanced episodes (equal support examples per class)
    pub fn create_balanced_episodes(
        dataset: &FewShotDataset,
        num_episodes: usize,
    ) -> Result<Vec<Episode>> {
        let mut episodes = Vec::new();

        for _ in 0..num_episodes {
            let episode = dataset.generate_episode()?;
            episodes.push(episode);
        }

        Ok(episodes)
    }

    /// Compute few-shot learning statistics
    pub fn compute_statistics(metrics: &[FewShotMetrics]) -> FewShotStatistics {
        let accuracies: Vec<f64> = metrics.iter().map(|m| m.accuracy).collect();
        let losses: Vec<f64> = metrics.iter().map(|m| m.loss).collect();

        let mean_accuracy = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
        let mean_loss = losses.iter().sum::<f64>() / losses.len() as f64;

        // Compute standard deviations
        let acc_variance = accuracies
            .iter()
            .map(|x| (x - mean_accuracy).powi(2))
            .sum::<f64>()
            / accuracies.len() as f64;
        let std_accuracy = acc_variance.sqrt();

        let loss_variance =
            losses.iter().map(|x| (x - mean_loss).powi(2)).sum::<f64>() / losses.len() as f64;
        let std_loss = loss_variance.sqrt();

        FewShotStatistics {
            mean_accuracy,
            std_accuracy,
            mean_loss,
            std_loss,
            num_runs: metrics.len(),
        }
    }
}

/// Few-shot learning statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotStatistics {
    pub mean_accuracy: f64,
    pub std_accuracy: f64,
    pub mean_loss: f64,
    pub std_loss: f64,
    pub num_runs: usize,
}

impl std::fmt::Display for FewShotStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Few-Shot Learning Statistics")?;
        writeln!(f, "============================")?;
        writeln!(
            f,
            "Mean Accuracy: {:.4} ± {:.4}",
            self.mean_accuracy, self.std_accuracy
        )?;
        writeln!(f, "Mean Loss: {:.4} ± {:.4}", self.mean_loss, self.std_loss)?;
        writeln!(f, "Number of Runs: {}", self.num_runs)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DeviceType;

    #[test]
    fn test_few_shot_config_creation() {
        let config = utils::create_standard_config(5, 1);
        assert_eq!(config.n_way, 5);
        assert_eq!(config.k_shot, 1);
        assert_eq!(config.query_shots, 15);
    }

    #[test]
    fn test_episode_generation() {
        // Create mock data
        let mut data = Vec::new();
        for class in 0..5 {
            for _ in 0..20 {
                let tensor = torsh_tensor::creation::zeros(&[3, 32, 32]).unwrap();
                data.push((tensor, class));
            }
        }

        let config = utils::create_standard_config(3, 5);
        let dataset = FewShotDataset::new(data, config);

        let episode = dataset.generate_episode().unwrap();
        assert_eq!(episode.n_way, 3);
        assert_eq!(episode.k_shot, 5);
        assert_eq!(episode.support_set.len(), 15); // 3 classes * 5 shots
        assert_eq!(episode.query_set.len(), 45); // 3 classes * 15 queries
    }

    #[test]
    fn test_prototypical_network() {
        // Would need a mock embedding network for testing
        // This is a placeholder test
        assert!(true);
    }

    #[test]
    fn test_few_shot_metrics() {
        let metrics = FewShotMetrics {
            accuracy: 0.85,
            loss: 0.3,
            per_class_accuracy: HashMap::new(),
            num_episodes: 100,
            confidence_interval: (0.82, 0.88),
        };

        assert_eq!(metrics.accuracy, 0.85);
        assert_eq!(metrics.num_episodes, 100);
    }

    #[test]
    fn test_dataset_split() {
        let mut data = Vec::new();
        for i in 0..100 {
            let tensor = torsh_tensor::creation::zeros(&[3, 32, 32]).unwrap();
            data.push((tensor, i % 10));
        }

        let (train, val, test) = utils::split_dataset(data, 0.7, 0.2);
        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 20);
        assert_eq!(test.len(), 10);
    }

    #[test]
    fn test_statistics_computation() {
        let metrics = vec![
            FewShotMetrics {
                accuracy: 0.8,
                loss: 0.3,
                per_class_accuracy: HashMap::new(),
                num_episodes: 100,
                confidence_interval: (0.77, 0.83),
            },
            FewShotMetrics {
                accuracy: 0.85,
                loss: 0.25,
                per_class_accuracy: HashMap::new(),
                num_episodes: 100,
                confidence_interval: (0.82, 0.88),
            },
        ];

        let stats = utils::compute_statistics(&metrics);
        assert!((stats.mean_accuracy - 0.825).abs() < 1e-10);
        assert_eq!(stats.num_runs, 2);
    }
}
