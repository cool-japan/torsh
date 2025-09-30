//! Parameter management system for neural network modules
//!
//! This module provides comprehensive parameter management including:
//! - Parameter struct with thread-safe tensor access
//! - Comprehensive initialization methods
//! - Parameter analysis and diagnostics
//! - Parameter collections and batch operations

use crate::init::Initializer;
use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Parameter wrapper for trainable tensors
#[derive(Clone, Debug)]
pub struct Parameter {
    data: Arc<RwLock<Tensor>>,
    requires_grad: bool,
}

impl Parameter {
    /// Create a new parameter
    pub fn new(tensor: Tensor) -> Self {
        Self {
            data: Arc::new(RwLock::new(tensor)),
            requires_grad: true,
        }
    }

    /// Create a parameter that doesn't require gradients
    pub fn new_no_grad(tensor: Tensor) -> Self {
        Self {
            data: Arc::new(RwLock::new(tensor)),
            requires_grad: false,
        }
    }

    /// Get the underlying tensor
    pub fn tensor(&self) -> Arc<RwLock<Tensor>> {
        self.data.clone()
    }

    /// Create a parameter from an existing tensor Arc
    pub fn from_tensor(tensor: Arc<RwLock<Tensor>>) -> Self {
        Self {
            data: tensor,
            requires_grad: true,
        }
    }

    /// Set whether this parameter requires gradients
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        // Note: torsh_tensor doesn't support requires_grad yet
        // This will be implemented when autograd is available
        self
    }

    /// Check if parameter requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get parameter shape
    pub fn shape(&self) -> Result<Vec<usize>> {
        Ok(self.data.read().shape().dims().to_vec())
    }

    /// Get parameter size (number of elements)
    pub fn numel(&self) -> Result<usize> {
        Ok(self.data.read().shape().numel())
    }

    /// Move parameter to device
    pub fn to_device(&mut self, device: DeviceType) -> Result<()> {
        // This would move the tensor to the specified device
        // For now, just update the device field when tensor supports it
        let _ = device; // Suppress warning
        Ok(())
    }

    /// Zero the parameter gradients
    pub fn zero_grad(&mut self) {
        // This would zero gradients when autograd is available
        // For now, this is a placeholder
    }

    /// Clone the parameter data
    pub fn clone_data(&self) -> Tensor {
        self.data.read().clone()
    }
}

/// Enhanced parameter management utilities
impl Parameter {
    /// Create parameter with specific initialization function
    ///
    /// This is the most flexible parameter creation method, allowing custom
    /// initialization logic.
    pub fn with_init<F>(shape: Vec<usize>, _device: DeviceType, init_fn: F) -> Result<Self>
    where
        F: FnOnce(Vec<usize>) -> Result<Tensor>,
    {
        let tensor = init_fn(shape)?;
        Ok(Self::new(tensor))
    }

    /// Create parameter from existing data
    ///
    /// Convenient method to create a parameter from a vector of data.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let tensor = torsh_tensor::Tensor::from_vec(data, &shape)?;
        Ok(Self::new(tensor))
    }

    /// Create parameter with automatic shape inference
    ///
    /// Creates a parameter where the shape is inferred from the provided data.
    pub fn from_data_auto_shape(data: Vec<f32>) -> Result<Self> {
        let shape = vec![data.len()];
        Self::from_data(data, shape)
    }

    /// Create parameter with random initialization and automatic fan calculation
    ///
    /// This method automatically chooses the best initialization based on the layer type.
    pub fn auto_init(shape: Vec<usize>, device: DeviceType, layer_type: LayerType) -> Result<Self> {
        use crate::init::InitMethod;

        let init_method = match layer_type {
            LayerType::Linear | LayerType::Dense => InitMethod::KaimingUniform {
                mode: crate::init::FanMode::FanIn,
                nonlinearity: crate::init::Nonlinearity::Linear,
            },
            LayerType::Conv => InitMethod::KaimingUniform {
                mode: crate::init::FanMode::FanOut,
                nonlinearity: crate::init::Nonlinearity::ReLU,
            },
            LayerType::RNN | LayerType::LSTM | LayerType::GRU => {
                InitMethod::XavierUniform { gain: 1.0 }
            }
            LayerType::Attention => InitMethod::XavierNormal { gain: 1.0 },
            LayerType::Embedding => InitMethod::Normal {
                mean: 0.0,
                std: 1.0,
            },
            LayerType::Bias => InitMethod::Constant { value: 0.0 },
            LayerType::BatchNorm => InitMethod::Constant { value: 1.0 },
            LayerType::Custom => InitMethod::KaimingUniform {
                mode: crate::init::FanMode::FanIn,
                nonlinearity: crate::init::Nonlinearity::ReLU,
            },
        };

        Self::with_init_method(shape, device, init_method)
    }

    /// Create parameter filled with zeros
    pub fn zeros(shape: Vec<usize>, _device: DeviceType) -> Result<Self> {
        use torsh_tensor::creation::zeros;
        let tensor = zeros(&shape)?;
        Ok(Self::new(tensor))
    }

    /// Create parameter filled with ones
    pub fn ones(shape: Vec<usize>, _device: DeviceType) -> Result<Self> {
        use torsh_tensor::creation::ones;
        let tensor = ones(&shape)?;
        Ok(Self::new(tensor))
    }

    /// Create parameter using InitMethod enum
    pub fn with_init_method(
        shape: Vec<usize>,
        _device: DeviceType,
        method: crate::init::InitMethod,
    ) -> Result<Self> {
        let tensor = method.initialize(&shape)?;
        Ok(Self::new(tensor))
    }

    /// Create parameter with uniform random initialization
    pub fn uniform(shape: Vec<usize>, device: DeviceType, low: f32, high: f32) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::Uniform { low, high })
    }

    /// Create parameter with normal random initialization
    pub fn normal(shape: Vec<usize>, device: DeviceType, mean: f32, std: f32) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::Normal { mean, std })
    }

    /// Create parameter with Xavier/Glorot uniform initialization
    pub fn xavier_uniform(shape: Vec<usize>, device: DeviceType, gain: f32) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::XavierUniform { gain })
    }

    /// Create parameter with Xavier/Glorot normal initialization
    pub fn xavier_normal(shape: Vec<usize>, device: DeviceType, gain: f32) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::XavierNormal { gain })
    }

    /// Create parameter with Kaiming/He uniform initialization
    pub fn kaiming_uniform(
        shape: Vec<usize>,
        device: DeviceType,
        nonlinearity: &str,
    ) -> Result<Self> {
        use crate::init::{FanMode, InitMethod, Nonlinearity};
        let nl = match nonlinearity {
            "relu" => Nonlinearity::ReLU,
            "leaky_relu" => Nonlinearity::LeakyReLU {
                negative_slope: 0.01,
            },
            "tanh" => Nonlinearity::Tanh,
            "sigmoid" => Nonlinearity::Sigmoid,
            "selu" => Nonlinearity::SELU,
            "elu" => Nonlinearity::ELU,
            "swish" => Nonlinearity::Swish,
            "linear" => Nonlinearity::Linear,
            _ => Nonlinearity::Linear,
        };
        Self::with_init_method(
            shape,
            device,
            InitMethod::KaimingUniform {
                mode: FanMode::FanIn,
                nonlinearity: nl,
            },
        )
    }

    /// Create parameter with Kaiming/He normal initialization
    pub fn kaiming_normal(
        shape: Vec<usize>,
        device: DeviceType,
        nonlinearity: &str,
    ) -> Result<Self> {
        use crate::init::{FanMode, InitMethod, Nonlinearity};
        let nl = match nonlinearity {
            "relu" => Nonlinearity::ReLU,
            "leaky_relu" => Nonlinearity::LeakyReLU {
                negative_slope: 0.01,
            },
            "tanh" => Nonlinearity::Tanh,
            "sigmoid" => Nonlinearity::Sigmoid,
            "selu" => Nonlinearity::SELU,
            "elu" => Nonlinearity::ELU,
            "swish" => Nonlinearity::Swish,
            "linear" => Nonlinearity::Linear,
            _ => Nonlinearity::Linear,
        };
        Self::with_init_method(
            shape,
            device,
            InitMethod::KaimingNormal {
                mode: FanMode::FanIn,
                nonlinearity: nl,
            },
        )
    }

    /// Create parameter with constant value
    pub fn constant(shape: Vec<usize>, device: DeviceType, value: f32) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::Constant { value })
    }

    /// Create parameter with orthogonal initialization
    pub fn orthogonal(shape: Vec<usize>, device: DeviceType, gain: f32) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::Orthogonal { gain })
    }

    /// Create parameter with sparse initialization
    pub fn sparse(shape: Vec<usize>, device: DeviceType, sparsity: f32, std: f32) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::Sparse { sparsity, std })
    }

    /// Create parameter with Lecun uniform initialization
    pub fn lecun_uniform(shape: Vec<usize>, device: DeviceType) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::LecunUniform)
    }

    /// Create parameter with Lecun normal initialization
    pub fn lecun_normal(shape: Vec<usize>, device: DeviceType) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::LecunNormal)
    }

    /// Create parameter with truncated normal initialization
    pub fn truncated_normal(
        shape: Vec<usize>,
        device: DeviceType,
        mean: f32,
        std: f32,
        a: f32,
        b: f32,
    ) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(
            shape,
            device,
            InitMethod::TruncatedNormal { mean, std, a, b },
        )
    }

    /// Create parameter with eye/identity initialization
    pub fn eye(shape: Vec<usize>, device: DeviceType) -> Result<Self> {
        use crate::init::InitMethod;
        Self::with_init_method(shape, device, InitMethod::Eye)
    }

    /// Get parameter statistics
    pub fn stats(&self) -> Result<ParameterStats> {
        let tensor = self.data.read();
        let data = tensor.to_vec()?;

        if data.is_empty() {
            return Ok(ParameterStats {
                mean: 0.0,
                std: 0.0,
                variance: 0.0,
                min: 0.0,
                max: 0.0,
                numel: 0,
                median: 0.0,
                q25: 0.0,
                q75: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
            });
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Calculate additional statistics
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted_data.len() % 2 == 0 {
            (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) / 2.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };

        let q25_idx = sorted_data.len() / 4;
        let q75_idx = 3 * sorted_data.len() / 4;
        let q25 = sorted_data.get(q25_idx).copied().unwrap_or(min);
        let q75 = sorted_data.get(q75_idx).copied().unwrap_or(max);

        // Basic skewness and kurtosis calculations
        let n = data.len() as f32;
        let skewness = if std > 0.0 {
            data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f32>() / n
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 {
            data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f32>() / n - 3.0
        } else {
            0.0
        };

        Ok(ParameterStats {
            mean,
            std,
            variance,
            min,
            max,
            numel: data.len(),
            median,
            q25,
            q75,
            skewness,
            kurtosis,
        })
    }

    /// Check if parameter has finite values (no NaN or infinity)
    pub fn is_finite(&self) -> Result<bool> {
        let tensor = self.data.read();
        let data = tensor.to_vec()?;
        Ok(data.iter().all(|x| x.is_finite()))
    }

    /// Reinitialize parameter with a new method
    pub fn reinitialize(&mut self, method: crate::init::InitMethod) -> Result<()> {
        let current_shape = self.shape()?;
        let new_tensor = method.initialize(&current_shape)?;
        *self.data.write() = new_tensor;
        Ok(())
    }

    /// Get parameter norm (L2 norm)
    pub fn norm(&self) -> Result<f32> {
        let tensor = self.data.read();
        let data = tensor.to_vec()?;
        let norm = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(norm)
    }

    /// Get parameter L1 norm
    pub fn l1_norm(&self) -> Result<f32> {
        let tensor = self.data.read();
        let data = tensor.to_vec()?;
        let norm = data.iter().map(|x| x.abs()).sum::<f32>();
        Ok(norm)
    }

    /// Get parameter L-infinity norm (max absolute value)
    pub fn linf_norm(&self) -> Result<f32> {
        let tensor = self.data.read();
        let data = tensor.to_vec()?;
        let norm = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        Ok(norm)
    }

    /// Clamp parameter values to a range
    pub fn clamp(&mut self, min: f32, max: f32) -> Result<()> {
        let mut tensor = self.data.write();
        let data = tensor.to_vec()?;
        let clamped_data: Vec<f32> = data.iter().map(|&x| x.clamp(min, max)).collect();
        let shape = tensor.shape().dims().to_vec();
        *tensor = torsh_tensor::Tensor::from_vec(clamped_data, &shape)?;
        Ok(())
    }

    /// Apply a function to all parameter values
    pub fn apply_fn<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(f32) -> f32,
    {
        let mut tensor = self.data.write();
        let data = tensor.to_vec()?;
        let transformed_data: Vec<f32> = data.iter().map(|&x| f(x)).collect();
        let shape = tensor.shape().dims().to_vec();
        *tensor = torsh_tensor::Tensor::from_vec(transformed_data, &shape)?;
        Ok(())
    }

    /// Scale parameter by a factor
    pub fn scale(&mut self, factor: f32) -> Result<()> {
        self.apply_fn(|x| x * factor)
    }

    /// Add noise to parameter
    pub fn add_noise(&mut self, std: f32) -> Result<()> {
        use scirs2_core::random::{Random, Rng, thread_rng};
        let mut rng = thread_rng();
        let mut tensor = self.data.write();
        let data = tensor.to_vec()?;
        let noisy_data: Vec<f32> = data
            .iter()
            .map(|&x| x + rng.gen::<f32>() * std)
            .collect();
        let shape = tensor.shape().dims().to_vec();
        *tensor = torsh_tensor::Tensor::from_vec(noisy_data, &shape)?;
        Ok(())
    }

    /// Get parameter histogram for analysis
    pub fn histogram(&self, bins: usize) -> Result<Vec<(f32, usize)>> {
        let tensor = self.data.read();
        let data = tensor.to_vec()?;

        if data.is_empty() {
            return Ok(Vec::new());
        }

        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if min_val == max_val {
            return Ok(vec![(min_val, data.len())]);
        }

        let bin_width = (max_val - min_val) / bins as f32;
        let mut histogram = vec![0; bins];

        for &value in &data {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }

        let result: Vec<(f32, usize)> = histogram
            .into_iter()
            .enumerate()
            .map(|(i, count)| (min_val + (i as f32 + 0.5) * bin_width, count))
            .collect();

        Ok(result)
    }

    /// Check for common parameter issues
    pub fn diagnose(&self) -> Result<ParameterDiagnostics> {
        let stats = self.stats()?;
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Check for NaN or infinite values
        if !self.is_finite()? {
            issues.push("Parameter contains NaN or infinite values".to_string());
        }

        // Check for suspicious statistics
        if stats.std < 1e-6 {
            warnings
                .push("Very low standard deviation - parameters may be too uniform".to_string());
        }

        if stats.std > 10.0 {
            warnings.push("Very high standard deviation - parameters may be unstable".to_string());
        }

        if stats.mean.abs() > 5.0 {
            warnings
                .push("High mean absolute value - parameters may be poorly centered".to_string());
        }

        // Check gradient-related issues
        let norm = self.norm()?;
        if norm < 1e-8 {
            warnings
                .push("Very small parameter norm - may indicate vanishing gradients".to_string());
        } else if norm > 100.0 {
            warnings
                .push("Very large parameter norm - may indicate exploding gradients".to_string());
        }

        Ok(ParameterDiagnostics {
            stats,
            issues,
            warnings,
            norm,
            is_finite: self.is_finite()?,
        })
    }
}

/// Layer type enumeration for automatic parameter initialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Linear,
    Dense,
    Conv,
    RNN,
    LSTM,
    GRU,
    Attention,
    Embedding,
    Bias,
    BatchNorm,
    Custom,
}

/// Parameter statistics for analysis and debugging
#[derive(Debug, Clone)]
pub struct ParameterStats {
    pub mean: f32,
    pub std: f32,
    pub variance: f32,
    pub min: f32,
    pub max: f32,
    pub numel: usize,
    pub median: f32,
    pub q25: f32,
    pub q75: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

impl ParameterStats {
    /// Create parameter statistics from data
    pub fn from_data(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::empty();
        }

        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_data[0];
        let max = sorted_data[sorted_data.len() - 1];
        let median = Self::percentile(&sorted_data, 0.5);
        let q25 = Self::percentile(&sorted_data, 0.25);
        let q75 = Self::percentile(&sorted_data, 0.75);

        // Calculate skewness and kurtosis
        let std_cubed = std.powi(3);
        let std_fourth = std.powi(4);

        let skewness = if std_cubed > 0.0 {
            data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f32>() / n
        } else {
            0.0
        };

        let kurtosis = if std_fourth > 0.0 {
            data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f32>() / n - 3.0
        } else {
            0.0
        };

        Self {
            mean,
            std,
            variance,
            min,
            max,
            numel: data.len(),
            median,
            q25,
            q75,
            skewness,
            kurtosis,
        }
    }

    /// Create empty statistics
    pub fn empty() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            variance: 0.0,
            min: 0.0,
            max: 0.0,
            numel: 0,
            median: 0.0,
            q25: 0.0,
            q75: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }

    /// Calculate percentile from sorted data
    fn percentile(sorted_data: &[f32], p: f32) -> f32 {
        if sorted_data.is_empty() {
            return 0.0;
        }

        let index = p * (sorted_data.len() - 1) as f32;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f32;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }

    /// Get interquartile range
    pub fn iqr(&self) -> f32 {
        self.q75 - self.q25
    }

    /// Check if distribution appears normal
    pub fn is_approximately_normal(&self) -> bool {
        // Simple heuristic: check if skewness and kurtosis are reasonable
        self.skewness.abs() < 1.0 && self.kurtosis.abs() < 3.0
    }
}

impl core::fmt::Display for ParameterStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ParameterStats(mean={:.4}, std={:.4}, min={:.4}, max={:.4}, numel={})",
            self.mean, self.std, self.min, self.max, self.numel
        )
    }
}

/// Parameter diagnostics for debugging and analysis
#[derive(Debug, Clone)]
pub struct ParameterDiagnostics {
    pub stats: ParameterStats,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub norm: f32,
    pub is_finite: bool,
}

impl core::fmt::Display for ParameterDiagnostics {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Parameter Diagnostics:")?;
        writeln!(f, "  {}", self.stats)?;
        writeln!(f, "  Norm: {:.6}", self.norm)?;
        writeln!(f, "  Finite: {}", self.is_finite)?;

        if !self.issues.is_empty() {
            writeln!(f, "  Issues:")?;
            for issue in &self.issues {
                writeln!(f, "    - {issue}")?;
            }
        }

        if !self.warnings.is_empty() {
            writeln!(f, "  Warnings:")?;
            for warning in &self.warnings {
                writeln!(f, "    - {warning}")?;
            }
        }

        Ok(())
    }
}

/// Parameter collection utility for managing multiple parameters
///
/// This provides convenient methods for working with collections of parameters,
/// such as applying operations to all parameters in a module.
#[derive(Debug, Clone)]
pub struct ParameterCollection {
    parameters: HashMap<String, Parameter>,
}

impl ParameterCollection {
    /// Create a new parameter collection
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    /// Create from a parameter map
    pub fn from_map(parameters: HashMap<String, Parameter>) -> Self {
        Self { parameters }
    }

    /// Add a parameter to the collection
    pub fn add(&mut self, name: String, parameter: Parameter) {
        self.parameters.insert(name, parameter);
    }

    /// Get a parameter by name
    pub fn get(&self, name: &str) -> Option<&Parameter> {
        self.parameters.get(name)
    }

    /// Get a mutable parameter by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Parameter> {
        self.parameters.get_mut(name)
    }

    /// Get all parameter names
    pub fn names(&self) -> Vec<&String> {
        self.parameters.keys().collect()
    }

    /// Get the number of parameters
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Apply a function to all parameters
    pub fn apply_to_all<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(&mut Parameter) -> Result<()>,
    {
        for param in self.parameters.values_mut() {
            f(param)?;
        }
        Ok(())
    }

    /// Get statistics for all parameters
    pub fn stats(&self) -> Result<HashMap<String, ParameterStats>> {
        let mut stats = HashMap::new();
        for (name, param) in &self.parameters {
            stats.insert(name.clone(), param.stats()?);
        }
        Ok(stats)
    }

    /// Get diagnostics for all parameters
    pub fn diagnose(&self) -> Result<HashMap<String, ParameterDiagnostics>> {
        let mut diagnostics = HashMap::new();
        for (name, param) in &self.parameters {
            diagnostics.insert(name.clone(), param.diagnose()?);
        }
        Ok(diagnostics)
    }

    /// Get total parameter count
    pub fn total_parameters(&self) -> usize {
        self.parameters
            .values()
            .map(|p| p.numel().unwrap_or(0))
            .sum()
    }

    /// Get total memory usage
    pub fn total_memory_usage(&self) -> usize {
        self.parameters
            .values()
            .map(|p| p.numel().unwrap_or(0) * 4) // Assume f32
            .sum()
    }

    /// Freeze all parameters
    pub fn freeze_all(&mut self) {
        for param in self.parameters.values_mut() {
            param.requires_grad = false;
        }
    }

    /// Unfreeze all parameters
    pub fn unfreeze_all(&mut self) {
        for param in self.parameters.values_mut() {
            param.requires_grad = true;
        }
    }

    /// Scale all parameters by a factor
    pub fn scale_all(&mut self, factor: f32) -> Result<()> {
        self.apply_to_all(|param| param.scale(factor))
    }

    /// Clamp all parameters to a range
    pub fn clamp_all(&mut self, min: f32, max: f32) -> Result<()> {
        self.apply_to_all(|param| param.clamp(min, max))
    }

    /// Add noise to all parameters
    pub fn add_noise_all(&mut self, std: f32) -> Result<()> {
        self.apply_to_all(|param| param.add_noise(std))
    }

    /// Filter parameters by name pattern
    pub fn filter_by_name(&self, pattern: &str) -> ParameterCollection {
        let filtered: HashMap<String, Parameter> = self
            .parameters
            .iter()
            .filter(|(name, _)| name.contains(pattern))
            .map(|(name, param)| (name.clone(), param.clone()))
            .collect();
        ParameterCollection::from_map(filtered)
    }

    /// Filter parameters by predicate
    pub fn filter_by<F>(&self, predicate: F) -> ParameterCollection
    where
        F: Fn(&String, &Parameter) -> bool,
    {
        let filtered: HashMap<String, Parameter> = self
            .parameters
            .iter()
            .filter(|(name, param)| predicate(name, param))
            .map(|(name, param)| (name.clone(), param.clone()))
            .collect();
        ParameterCollection::from_map(filtered)
    }

    /// Create a summary report of all parameters
    pub fn summary_report(&self) -> Result<String> {
        let mut report = String::new();
        report.push_str("Parameter Collection Summary\n");
        report.push_str(&format!("Total parameters: {}\n", self.len()));
        report.push_str(&format!("Total elements: {}\n", self.total_parameters()));
        report.push_str(&format!(
            "Memory usage: {:.2} MB\n",
            self.total_memory_usage() as f64 / (1024.0 * 1024.0)
        ));
        report.push_str("\nParameter Details:\n");

        for (name, param) in &self.parameters {
            let stats = param.stats()?;
            report.push_str(&format!(
                "  {}: {} elements, mean={:.4}, std={:.4}\n",
                name, stats.numel, stats.mean, stats.std
            ));
        }

        Ok(report)
    }
}

impl Default for ParameterCollection {
    fn default() -> Self {
        Self::new()
    }
}

impl From<HashMap<String, Parameter>> for ParameterCollection {
    fn from(parameters: HashMap<String, Parameter>) -> Self {
        Self::from_map(parameters)
    }
}

impl From<ParameterCollection> for HashMap<String, Parameter> {
    fn from(val: ParameterCollection) -> Self {
        val.parameters
    }
}
