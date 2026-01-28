//! Research-level quantization features
//!
//! This module implements cutting-edge quantization research techniques including:
//! - Learned Step Size Quantization (LSQ)
//! - Hessian AWare Quantization (HAWQ)
//! - Automatic Quantization (AutoQ)
//! - Differentiable Quantization

use crate::{QuantConfig, TorshResult};
// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand

use std::collections::HashMap;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Learned Step Size (LSQ) Quantization
/// Enables learning of quantization step sizes during training
#[derive(Debug, Clone)]
pub struct LearnedStepSizeQuantizer {
    /// Current step size (scale parameter)
    pub step_size: f32,
    /// Learning rate for step size updates
    pub lr: f32,
    /// Gradient for step size
    step_size_grad: f32,
    /// Moving average for gradient momentum
    momentum: f32,
    /// Configuration
    config: QuantConfig,
}

impl LearnedStepSizeQuantizer {
    /// Create new LSQ quantizer
    pub fn new(initial_step_size: f32, lr: f32, config: QuantConfig) -> Self {
        Self {
            step_size: initial_step_size,
            lr,
            step_size_grad: 0.0,
            momentum: 0.0,
            config,
        }
    }

    /// Forward pass with learnable quantization
    pub fn forward(&mut self, input: &Tensor) -> TorshResult<(Tensor, Tensor)> {
        let (qmin, qmax) = self.config.get_qint_range();

        // Compute quantized values
        let input_data = input.data()?;
        let mut quantized_data = Vec::new();
        let mut straight_through_data = Vec::new();

        for &x in input_data.iter() {
            // Quantize: q = clamp(round(x / step_size), qmin, qmax)
            let q_float = (x / self.step_size).round();
            let q_clamped = q_float.max(qmin as f32).min(qmax as f32);

            // Dequantize: x_hat = q * step_size
            let x_hat = q_clamped * self.step_size;

            quantized_data.push(q_clamped);
            straight_through_data.push(x_hat);
        }

        let quantized = Tensor::from_data(
            quantized_data,
            input.shape().dims().to_vec(),
            input.device(),
        );

        let straight_through = Tensor::from_data(
            straight_through_data,
            input.shape().dims().to_vec(),
            input.device(),
        );

        Ok((quantized?, straight_through?))
    }

    /// Compute gradients for step size learning
    pub fn backward(&mut self, grad_output: &Tensor, input: &Tensor) -> TorshResult<()> {
        let (qmin, qmax) = self.config.get_qint_range();
        let grad_data = grad_output.data()?;
        let input_data = input.data()?;

        let mut step_size_grad = 0.0;
        let num_elements = input_data.len() as f32;

        for (&x, &grad) in input_data.iter().zip(grad_data.iter()) {
            let q_float = x / self.step_size;
            let q_clamped = q_float.round().max(qmin as f32).min(qmax as f32);

            // Gradient w.r.t. step_size using straight-through estimator
            if q_float >= qmin as f32 && q_float <= qmax as f32 {
                step_size_grad += grad * (q_clamped - q_float) / self.step_size;
            }
        }

        self.step_size_grad = step_size_grad / num_elements;
        Ok(())
    }

    /// Update step size parameters
    pub fn update_parameters(&mut self) {
        // Apply momentum-based gradient descent
        self.momentum = 0.9 * self.momentum + self.lr * self.step_size_grad;
        self.step_size -= self.momentum;

        // Ensure step size remains positive and reasonable
        self.step_size = self.step_size.clamp(1e-8, 100.0);

        // Reset gradients
        self.step_size_grad = 0.0;
    }

    /// Get current quantization parameters
    pub fn get_params(&self) -> (f32, i32) {
        (self.step_size, 0) // LSQ typically uses zero-point = 0
    }
}

/// HAWQ (Hessian AWare Quantization) implementation
/// Uses second-order information to determine optimal bit-widths per layer
#[derive(Debug, Clone)]
pub struct HawqQuantizer {
    /// Per-layer bit-width assignments
    pub layer_bits: HashMap<String, u8>,
    /// Per-layer sensitivity scores
    pub layer_sensitivity: HashMap<String, f32>,
    /// Total bit budget
    pub total_bits: u32,
    /// Minimum bits per layer
    pub min_bits: u8,
    /// Maximum bits per layer
    pub max_bits: u8,
}

impl HawqQuantizer {
    /// Create new HAWQ quantizer
    pub fn new(total_bits: u32, min_bits: u8, max_bits: u8) -> Self {
        Self {
            layer_bits: HashMap::new(),
            layer_sensitivity: HashMap::new(),
            total_bits,
            min_bits,
            max_bits,
        }
    }

    /// Compute layer sensitivity using approximate Hessian
    pub fn compute_sensitivity(&mut self, layers: &HashMap<String, Tensor>) -> TorshResult<()> {
        for (layer_name, tensor) in layers {
            let sensitivity = self.estimate_hessian_trace(tensor)?;
            self.layer_sensitivity
                .insert(layer_name.clone(), sensitivity);
        }
        Ok(())
    }

    /// Estimate Hessian trace using finite differences
    fn estimate_hessian_trace(&self, tensor: &Tensor) -> TorshResult<f32> {
        let data = tensor.data()?;
        let mut trace = 0.0;
        let eps = 1e-4;

        // Sample a subset of parameters for efficiency
        let sample_size = data.len().min(1000);
        let step = data.len().max(1) / sample_size.max(1);

        for i in (0..data.len()).step_by(step) {
            // Approximate second derivative using finite differences
            let x = data[i];

            // Forward differences approximation: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            let f_plus = self.mock_loss_function(x + eps);
            let f_center = self.mock_loss_function(x);
            let f_minus = self.mock_loss_function(x - eps);

            let second_derivative = (f_plus - 2.0 * f_center + f_minus) / (eps * eps);
            trace += second_derivative.abs();
        }

        Ok(trace / sample_size as f32)
    }

    /// Mock loss function for Hessian estimation (in practice, this would be the actual loss)
    fn mock_loss_function(&self, x: f32) -> f32 {
        // Simple quadratic loss for demonstration
        x * x
    }

    /// Allocate bit-widths based on sensitivity scores
    pub fn allocate_bits(&mut self) -> TorshResult<()> {
        let mut sorted_layers: Vec<_> = self.layer_sensitivity.iter().collect();
        sorted_layers.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let num_layers = sorted_layers.len();
        if num_layers == 0 {
            return Ok(());
        }

        // Initialize all layers with minimum bits
        let mut remaining_bits =
            self.total_bits as i32 - (num_layers * self.min_bits as usize) as i32;

        for (layer_name, _) in &sorted_layers {
            self.layer_bits
                .insert(layer_name.to_string(), self.min_bits);
        }

        // Allocate additional bits to most sensitive layers
        while remaining_bits > 0 {
            let mut allocated = false;

            for (layer_name, _sensitivity) in &sorted_layers {
                let current_bits = self.layer_bits[*layer_name];
                if current_bits < self.max_bits && remaining_bits > 0 {
                    self.layer_bits
                        .insert(layer_name.to_string(), current_bits + 1);
                    remaining_bits -= 1;
                    allocated = true;
                }
            }

            if !allocated {
                break; // All layers at max bits
            }
        }

        Ok(())
    }

    /// Get bit-width for a specific layer
    pub fn get_layer_bits(&self, layer_name: &str) -> u8 {
        self.layer_bits
            .get(layer_name)
            .copied()
            .unwrap_or(self.min_bits)
    }

    /// Create quantization config for a specific layer
    pub fn create_layer_config(&self, layer_name: &str) -> QuantConfig {
        let bits = self.get_layer_bits(layer_name);

        match bits {
            1 => QuantConfig::binary(),
            2 => QuantConfig::ternary(),
            4 => QuantConfig::int4(),
            8 => QuantConfig::int8(),
            _ => QuantConfig::int8(), // Default to INT8 for other bit-widths
        }
    }
}

/// AutoQ (Automatic Quantization) implementation
/// Automatically searches for optimal quantization configurations
#[derive(Debug, Clone)]
pub struct AutoQuantizer {
    /// Search space configurations
    pub search_configs: Vec<QuantConfig>,
    /// Best configuration found
    pub best_config: Option<QuantConfig>,
    /// Best accuracy achieved
    pub best_accuracy: f32,
    /// Evaluation function
    eval_fn: Option<fn(&Tensor, &QuantConfig) -> f32>,
}

impl AutoQuantizer {
    /// Create new AutoQ quantizer
    pub fn new() -> Self {
        // Add various quantization configurations to search space
        let mut search_configs = vec![
            QuantConfig::int8(),
            QuantConfig::uint8(),
            QuantConfig::int4(),
            QuantConfig::binary(),
            QuantConfig::ternary(),
            QuantConfig::mixed_precision(),
        ];

        // Add per-channel variants
        for axis in 0..2 {
            search_configs.push(QuantConfig::per_channel(axis));
        }

        // Add group-wise variants
        for &group_size in &[16, 32, 64, 128] {
            search_configs.push(QuantConfig::group_wise(0, group_size));
        }

        Self {
            search_configs,
            best_config: None,
            best_accuracy: f32::NEG_INFINITY,
            eval_fn: None,
        }
    }

    /// Set evaluation function for configuration scoring
    pub fn set_eval_function(&mut self, eval_fn: fn(&Tensor, &QuantConfig) -> f32) {
        self.eval_fn = Some(eval_fn);
    }

    /// Search for optimal quantization configuration
    pub fn search(&mut self, tensor: &Tensor) -> TorshResult<QuantConfig> {
        let eval_fn = self.eval_fn.ok_or_else(|| {
            TorshError::InvalidArgument("Evaluation function not set".to_string())
        })?;

        for config in &self.search_configs {
            if config.validate().is_ok() {
                let accuracy = eval_fn(tensor, config);

                if accuracy > self.best_accuracy {
                    self.best_accuracy = accuracy;
                    self.best_config = Some(config.clone());
                }
            }
        }

        self.best_config
            .clone()
            .ok_or_else(|| TorshError::InvalidArgument("No valid configuration found".to_string()))
    }

    /// Get top-k configurations by performance
    pub fn get_top_k_configs(
        &self,
        k: usize,
        tensor: &Tensor,
    ) -> TorshResult<Vec<(QuantConfig, f32)>> {
        let eval_fn = self.eval_fn.ok_or_else(|| {
            TorshError::InvalidArgument("Evaluation function not set".to_string())
        })?;

        let mut scored_configs = Vec::new();

        for config in &self.search_configs {
            if config.validate().is_ok() {
                let score = eval_fn(tensor, config);
                scored_configs.push((config.clone(), score));
            }
        }

        // Sort by score (descending)
        scored_configs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        scored_configs.truncate(k);
        Ok(scored_configs)
    }

    /// Add custom configuration to search space
    pub fn add_config(&mut self, config: QuantConfig) {
        self.search_configs.push(config);
    }

    /// Remove configurations that don't meet criteria
    pub fn filter_configs<F>(&mut self, predicate: F)
    where
        F: Fn(&QuantConfig) -> bool,
    {
        self.search_configs.retain(predicate);
    }
}

impl Default for AutoQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Differentiable Quantization implementation
/// Enables end-to-end gradient flow through quantization operations
#[derive(Debug, Clone)]
pub struct DifferentiableQuantizer {
    /// Temperature parameter for soft quantization
    pub temperature: f32,
    /// Quantization configuration
    pub config: QuantConfig,
    /// Whether to use straight-through estimator
    pub use_ste: bool,
}

impl DifferentiableQuantizer {
    /// Create new differentiable quantizer
    pub fn new(temperature: f32, config: QuantConfig) -> Self {
        Self {
            temperature,
            config,
            use_ste: true,
        }
    }

    /// Soft quantization using sigmoid/tanh functions
    pub fn soft_quantize(&self, input: &Tensor) -> TorshResult<Tensor> {
        let (qmin, qmax) = self.config.get_qint_range();
        let input_data = input.data()?;

        let soft_quantized: Vec<f32> = input_data
            .iter()
            .map(|&x| {
                // Normalize to [0, 1] range
                let normalized = (x - qmin as f32) / (qmax - qmin) as f32;

                // Apply soft quantization using sigmoid
                let levels = (qmax - qmin + 1) as f32;
                let soft_level = self.sigmoid_quantize(normalized * levels, levels);

                // Denormalize back to original range
                qmin as f32 + soft_level * (qmax - qmin) as f32 / levels
            })
            .collect();

        Tensor::from_data(
            soft_quantized,
            input.shape().dims().to_vec(),
            input.device(),
        )
    }

    /// Sigmoid-based soft quantization
    fn sigmoid_quantize(&self, x: f32, levels: f32) -> f32 {
        let mut sum = 0.0;

        for k in 0..(levels as i32) {
            let sigmoid_arg = (x - k as f32) / self.temperature;
            let sigmoid_val = 1.0 / (1.0 + (-sigmoid_arg).exp());
            sum += sigmoid_val;
        }

        sum - 0.5 // Adjust to center the output
    }

    /// Forward pass with differentiable quantization
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        if self.temperature > 0.1 {
            // Use soft quantization for high temperature
            self.soft_quantize(input)
        } else {
            // Use straight-through estimator for low temperature
            self.hard_quantize_with_ste(input)
        }
    }

    /// Hard quantization with straight-through estimator
    fn hard_quantize_with_ste(&self, input: &Tensor) -> TorshResult<Tensor> {
        let (qmin, qmax) = self.config.get_qint_range();
        let input_data = input.data()?;

        // In forward pass: hard quantization
        // In backward pass: gradient flows through (straight-through)
        let quantized: Vec<f32> = input_data
            .iter()
            .map(|&x| {
                let q = x.round();
                q.max(qmin as f32).min(qmax as f32)
            })
            .collect();

        Tensor::from_data(quantized, input.shape().dims().to_vec(), input.device())
    }

    /// Annealing schedule for temperature
    pub fn anneal_temperature(&mut self, current_epoch: u32, total_epochs: u32) {
        let progress = current_epoch as f32 / total_epochs as f32;
        // Exponential decay from initial temperature to near-zero
        self.temperature *= (0.01_f32).powf(progress);
    }

    /// Set straight-through estimator usage
    pub fn set_ste(&mut self, use_ste: bool) {
        self.use_ste = use_ste;
    }
}

/// Neural Architecture Search for Quantization (NAS-Q)
/// Searches for optimal network architectures with quantization constraints
#[derive(Debug, Clone)]
pub struct NasQuantizer {
    /// Architecture search space
    pub search_space: Vec<ArchitectureConfig>,
    /// Current best architecture
    pub best_architecture: Option<ArchitectureConfig>,
    /// Population size for evolutionary search
    pub population_size: usize,
    /// Number of generations
    pub generations: u32,
}

/// Architecture configuration for NAS-Q
#[derive(Debug, Clone)]
pub struct ArchitectureConfig {
    /// Layer configurations
    pub layers: Vec<LayerConfig>,
    /// Global quantization settings
    pub global_config: QuantConfig,
}

/// Individual layer configuration
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Layer name
    pub name: String,
    /// Layer-specific quantization config
    pub quant_config: QuantConfig,
    /// Layer size/dimensions
    pub dimensions: Vec<usize>,
}

impl NasQuantizer {
    /// Create new NAS quantizer
    pub fn new(population_size: usize, generations: u32) -> Self {
        Self {
            search_space: Vec::new(),
            best_architecture: None,
            population_size,
            generations,
        }
    }

    /// Generate random architecture
    pub fn generate_random_architecture(&self) -> ArchitectureConfig {
        let mut rng = scirs2_core::random::thread_rng();
        let num_layers = 5 + rng.random_range(0..10); // 5-15 layers
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let layer_config = LayerConfig {
                name: format!("layer_{i}"),
                quant_config: self.random_quant_config(),
                dimensions: vec![vec![64, 128, 256, 512][rng.random_range(0..4)]],
            };
            layers.push(layer_config);
        }

        ArchitectureConfig {
            layers,
            global_config: QuantConfig::int8(),
        }
    }

    /// Generate random quantization configuration
    fn random_quant_config(&self) -> QuantConfig {
        let mut rng = scirs2_core::random::thread_rng();
        let configs = [
            QuantConfig::int8(),
            QuantConfig::int4(),
            QuantConfig::binary(),
            QuantConfig::ternary(),
        ];

        configs[rng.random_range(0..configs.len())].clone()
    }

    /// Evolutionary search for optimal architecture
    pub fn search(&mut self) -> TorshResult<ArchitectureConfig> {
        // Initialize population
        let mut population = Vec::new();
        for _ in 0..self.population_size {
            population.push(self.generate_random_architecture());
        }

        // Evolution loop
        for _generation in 0..self.generations {
            // Evaluate population (mock evaluation for now)
            let mut fitness_scores = Vec::new();
            for arch in &population {
                let fitness = self.evaluate_architecture(arch);
                fitness_scores.push(fitness);
            }

            // Selection and reproduction
            population = self.evolve_population(population, fitness_scores.clone());

            // Track best architecture
            if let Some((best_idx, _)) = fitness_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                self.best_architecture = Some(population[best_idx].clone());
            }
        }

        self.best_architecture
            .clone()
            .ok_or_else(|| TorshError::InvalidArgument("No architecture found".to_string()))
    }

    /// Evaluate architecture fitness (mock implementation)
    fn evaluate_architecture(&self, _arch: &ArchitectureConfig) -> f32 {
        // In practice, this would train/evaluate the architecture
        // For now, return random fitness
        let mut rng = scirs2_core::random::thread_rng();
        rng.random::<f32>()
    }

    /// Evolve population using genetic operators
    fn evolve_population(
        &self,
        population: Vec<ArchitectureConfig>,
        fitness: Vec<f32>,
    ) -> Vec<ArchitectureConfig> {
        let mut new_population = Vec::new();

        // Keep top 50% (elitism)
        let mut indexed_fitness: Vec<_> = fitness.iter().enumerate().collect();
        indexed_fitness.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let elite_count = self.population_size / 2;
        for (idx, _) in indexed_fitness.iter().take(elite_count) {
            new_population.push(population[*idx].clone());
        }

        // Generate remaining population through crossover and mutation
        while new_population.len() < self.population_size {
            // Tournament selection
            let parent1_idx = self.tournament_selection(&fitness);
            let parent2_idx = self.tournament_selection(&fitness);

            // Crossover
            let mut child = self.crossover(&population[parent1_idx], &population[parent2_idx]);

            // Mutation
            self.mutate(&mut child);

            new_population.push(child);
        }

        new_population
    }

    /// Tournament selection
    fn tournament_selection(&self, fitness: &[f32]) -> usize {
        let mut rng = scirs2_core::random::thread_rng();
        let tournament_size = 3;
        let mut best_idx = rng.random_range(0..fitness.len());
        let mut best_fitness = fitness[best_idx];

        for _ in 1..tournament_size {
            let idx = rng.random_range(0..fitness.len());
            if fitness[idx] > best_fitness {
                best_idx = idx;
                best_fitness = fitness[idx];
            }
        }

        best_idx
    }

    /// Crossover two architectures
    fn crossover(
        &self,
        parent1: &ArchitectureConfig,
        parent2: &ArchitectureConfig,
    ) -> ArchitectureConfig {
        let mut rng = scirs2_core::random::thread_rng();
        let mut child_layers = Vec::new();
        let max_layers = parent1.layers.len().min(parent2.layers.len());

        for i in 0..max_layers {
            // Randomly choose from either parent
            if rng.random_bool(0.5) {
                child_layers.push(parent1.layers[i].clone());
            } else {
                child_layers.push(parent2.layers[i].clone());
            }
        }

        ArchitectureConfig {
            layers: child_layers,
            global_config: if rng.random_bool(0.5) {
                parent1.global_config.clone()
            } else {
                parent2.global_config.clone()
            },
        }
    }

    /// Mutate architecture
    fn mutate(&self, arch: &mut ArchitectureConfig) {
        let mut rng = scirs2_core::random::thread_rng();
        let mutation_rate = 0.1;

        for layer in &mut arch.layers {
            if rng.random::<f32>() < mutation_rate {
                layer.quant_config = self.random_quant_config();
            }
        }

        if rng.random::<f32>() < mutation_rate {
            arch.global_config = self.random_quant_config();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_learned_step_size_quantizer() {
        let config = QuantConfig::int8();
        let mut lsq = LearnedStepSizeQuantizer::new(0.13, 0.01, config); // Use non-round step size to create quantization error

        let input = tensor_1d(&[0.1, 1.1, 2.1, 3.1]).unwrap(); // Use non-round values
        let (quantized, straight_through) = lsq.forward(&input).unwrap();

        assert_eq!(quantized.shape().dims(), input.shape().dims());
        assert_eq!(straight_through.shape().dims(), input.shape().dims());
        assert!(lsq.step_size > 0.0);

        // Test parameter update
        let grad = tensor_1d(&[0.1, 0.1, 0.1, 0.1]).unwrap();
        lsq.backward(&grad, &input).unwrap();
        let old_step_size = lsq.step_size;
        lsq.update_parameters();

        // Step size should change after update
        assert_ne!(lsq.step_size, old_step_size);
    }

    #[test]
    fn test_hawq_quantizer() {
        let mut hawq = HawqQuantizer::new(32, 2, 8);

        let mut layers = HashMap::new();
        layers.insert("layer1".to_string(), tensor_1d(&[1.0, 2.0, 3.0]).unwrap());
        layers.insert("layer2".to_string(), tensor_1d(&[4.0, 5.0, 6.0]).unwrap());

        hawq.compute_sensitivity(&layers).unwrap();
        hawq.allocate_bits().unwrap();

        assert_eq!(hawq.layer_sensitivity.len(), 2);
        assert_eq!(hawq.layer_bits.len(), 2);

        // Check bit allocation is within bounds
        for &bits in hawq.layer_bits.values() {
            assert!(bits >= hawq.min_bits && bits <= hawq.max_bits);
        }

        let config = hawq.create_layer_config("layer1");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_auto_quantizer() {
        let mut autoq = AutoQuantizer::new();
        assert!(!autoq.search_configs.is_empty());

        // Set a mock evaluation function
        autoq.set_eval_function(|_tensor, config| {
            // Mock evaluation: prefer INT8 configs
            if matches!(config.dtype, torsh_core::DType::I8) {
                1.0
            } else {
                0.5
            }
        });

        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let best_config = autoq.search(&tensor).unwrap();
        assert!(best_config.validate().is_ok());

        let top_configs = autoq.get_top_k_configs(3, &tensor).unwrap();
        assert_eq!(top_configs.len(), 3);
        assert!(top_configs[0].1 >= top_configs[1].1); // Scores should be descending
    }

    #[test]
    fn test_differentiable_quantizer() {
        let config = QuantConfig::int8();
        let mut diff_quant = DifferentiableQuantizer::new(1.0, config);

        let input = tensor_1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();

        // Test soft quantization
        let soft_output = diff_quant.soft_quantize(&input).unwrap();
        assert_eq!(soft_output.shape().dims(), input.shape().dims());

        // Test forward pass
        let output = diff_quant.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), input.shape().dims());

        // Test temperature annealing
        let initial_temp = diff_quant.temperature;
        diff_quant.anneal_temperature(50, 100);
        assert!(diff_quant.temperature < initial_temp);

        // Test STE toggle
        diff_quant.set_ste(false);
        assert!(!diff_quant.use_ste);
    }

    #[test]
    fn test_nas_quantizer() {
        let mut nas = NasQuantizer::new(10, 5);

        // Test random architecture generation
        let arch = nas.generate_random_architecture();
        assert!(!arch.layers.is_empty());
        assert!(arch.global_config.validate().is_ok());

        // Test architecture search (with small population for speed)
        let best_arch = nas.search().unwrap();
        assert!(!best_arch.layers.is_empty());

        assert!(nas.best_architecture.is_some());
    }

    #[test]
    fn test_layer_config() {
        let layer_config = LayerConfig {
            name: "test_layer".to_string(),
            quant_config: QuantConfig::int8(),
            dimensions: vec![128, 256],
        };

        assert_eq!(layer_config.name, "test_layer");
        assert!(layer_config.quant_config.validate().is_ok());
        assert_eq!(layer_config.dimensions, vec![128, 256]);
    }

    #[test]
    fn test_architecture_config() {
        let mut layers = Vec::new();
        for i in 0..3 {
            layers.push(LayerConfig {
                name: format!("layer_{i}"),
                quant_config: QuantConfig::int8(),
                dimensions: vec![64, 128],
            });
        }

        let arch_config = ArchitectureConfig {
            layers,
            global_config: QuantConfig::int8(),
        };

        assert_eq!(arch_config.layers.len(), 3);
        assert!(arch_config.global_config.validate().is_ok());
    }
}
