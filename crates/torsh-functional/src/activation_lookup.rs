//! Lookup table optimizations for activation functions
//!
//! This module provides optimized implementations of activation functions using
//! precomputed lookup tables for improved performance on large tensors.

use std::sync::OnceLock;
use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Lookup table for activation functions
#[derive(Clone)]
pub struct ActivationLookupTable {
    /// Input range minimum
    pub min_val: f32,
    /// Input range maximum
    pub max_val: f32,
    /// Number of table entries
    pub table_size: usize,
    /// Step size between table entries
    pub step: f32,
    /// Precomputed values
    pub values: Vec<f32>,
}

impl ActivationLookupTable {
    /// Create a new lookup table
    pub fn new<F>(min_val: f32, max_val: f32, table_size: usize, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let step = (max_val - min_val) / (table_size - 1) as f32;
        let values: Vec<f32> = (0..table_size)
            .map(|i| {
                let x = min_val + i as f32 * step;
                f(x)
            })
            .collect();

        Self {
            min_val,
            max_val,
            table_size,
            step,
            values,
        }
    }

    /// Lookup value with linear interpolation
    #[allow(dead_code)]
    pub fn lookup(&self, x: f32) -> f32 {
        if x <= self.min_val {
            return self.values[0];
        }
        if x >= self.max_val {
            return self.values[self.table_size - 1];
        }

        let index_f = (x - self.min_val) / self.step;
        let index = index_f as usize;

        if index >= self.table_size - 1 {
            return self.values[self.table_size - 1];
        }

        // Linear interpolation
        let frac = index_f - index as f32;
        let v0 = self.values[index];
        let v1 = self.values[index + 1];
        v0 + frac * (v1 - v0)
    }

    /// Fast lookup without interpolation (nearest neighbor)
    #[allow(dead_code)]
    pub fn lookup_fast(&self, x: f32) -> f32 {
        if x <= self.min_val {
            return self.values[0];
        }
        if x >= self.max_val {
            return self.values[self.table_size - 1];
        }

        let index = ((x - self.min_val) / self.step + 0.5) as usize;
        let index = index.min(self.table_size - 1);
        self.values[index]
    }
}

/// Global lookup tables for common activation functions
static SIGMOID_TABLE: OnceLock<ActivationLookupTable> = OnceLock::new();
static TANH_TABLE: OnceLock<ActivationLookupTable> = OnceLock::new();
static SOFTPLUS_TABLE: OnceLock<ActivationLookupTable> = OnceLock::new();
#[allow(dead_code)]
static EXP_TABLE: OnceLock<ActivationLookupTable> = OnceLock::new();

/// Configuration for lookup table optimizations
pub struct LookupConfig {
    /// Minimum input value to table
    pub min_val: f32,
    /// Maximum input value to table  
    pub max_val: f32,
    /// Size of lookup table
    pub table_size: usize,
    /// Whether to use linear interpolation (slower but more accurate)
    pub use_interpolation: bool,
    /// Threshold size above which to use lookup tables
    pub size_threshold: usize,
}

impl Default for LookupConfig {
    fn default() -> Self {
        Self {
            min_val: -10.0,
            max_val: 10.0,
            table_size: 10000,
            use_interpolation: true,
            size_threshold: 1000,
        }
    }
}

/// Initialize lookup tables
fn init_sigmoid_table() -> &'static ActivationLookupTable {
    SIGMOID_TABLE.get_or_init(|| {
        let config = LookupConfig::default();
        ActivationLookupTable::new(config.min_val, config.max_val, config.table_size, |x| {
            1.0 / (1.0 + (-x).exp())
        })
    })
}

fn init_tanh_table() -> &'static ActivationLookupTable {
    TANH_TABLE.get_or_init(|| {
        let config = LookupConfig::default();
        ActivationLookupTable::new(config.min_val, config.max_val, config.table_size, |x| {
            x.tanh()
        })
    })
}

fn init_softplus_table() -> &'static ActivationLookupTable {
    SOFTPLUS_TABLE.get_or_init(|| {
        let config = LookupConfig::default();
        ActivationLookupTable::new(config.min_val, config.max_val, config.table_size, |x| {
            (1.0 + x.exp()).ln()
        })
    })
}

#[allow(dead_code)]
fn init_exp_table() -> &'static ActivationLookupTable {
    EXP_TABLE.get_or_init(|| {
        let config = LookupConfig::default();
        ActivationLookupTable::new(config.min_val, config.max_val, config.table_size, |x| {
            x.exp()
        })
    })
}

/// Optimized sigmoid using lookup table
#[allow(dead_code)]
pub fn sigmoid_lookup(input: &Tensor, config: Option<LookupConfig>) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();
    let data = input.data()?;

    // Use lookup table for large tensors
    if data.len() >= config.size_threshold {
        let table = init_sigmoid_table();
        let result_data: Vec<f32> = if config.use_interpolation {
            data.iter().map(|&x| table.lookup(x)).collect()
        } else {
            data.iter().map(|&x| table.lookup_fast(x)).collect()
        };

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    } else {
        // Fall back to standard implementation for small tensors
        input.sigmoid()
    }
}

/// Optimized tanh using lookup table
#[allow(dead_code)]
pub fn tanh_lookup(input: &Tensor, config: Option<LookupConfig>) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();
    let data = input.data()?;

    if data.len() >= config.size_threshold {
        let table = init_tanh_table();
        let result_data: Vec<f32> = if config.use_interpolation {
            data.iter().map(|&x| table.lookup(x)).collect()
        } else {
            data.iter().map(|&x| table.lookup_fast(x)).collect()
        };

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    } else {
        input.tanh()
    }
}

/// Optimized softplus using lookup table
#[allow(dead_code)]
pub fn softplus_lookup(
    input: &Tensor,
    beta: f64,
    threshold: f64,
    config: Option<LookupConfig>,
) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();
    let data = input.data()?;

    if data.len() >= config.size_threshold && beta == 1.0 {
        // Only use lookup table for beta=1.0 case
        let table = init_softplus_table();
        let result_data: Vec<f32> = data
            .iter()
            .map(|&x| {
                if x > threshold as f32 {
                    x // Linear approximation for large values
                } else if config.use_interpolation {
                    table.lookup(x)
                } else {
                    table.lookup_fast(x)
                }
            })
            .collect();

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    } else {
        // Fall back to standard implementation
        crate::activations::softplus(input, beta, threshold)
    }
}

/// Optimized GELU using lookup table for the erf component
#[allow(dead_code)]
pub fn gelu_lookup(input: &Tensor, config: Option<LookupConfig>) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();
    let data = input.data()?;

    if data.len() >= config.size_threshold {
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        // We can optimize the erf computation using a lookup table
        let sqrt_2_inv = 1.0 / 2.0_f32.sqrt();

        let result_data: Vec<f32> = data
            .iter()
            .map(|&x| {
                let x_scaled = x * sqrt_2_inv;
                // Approximate erf using tanh: erf(x) ≈ tanh(1.2*x + 0.44*x^3) for small x
                let erf_approx = if x_scaled.abs() < 2.0 {
                    let x3 = x_scaled * x_scaled * x_scaled;
                    (1.2 * x_scaled + 0.44 * x3).tanh()
                } else {
                    if x_scaled > 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                };
                0.5 * x * (1.0 + erf_approx)
            })
            .collect();

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    } else {
        input.gelu()
    }
}

/// Optimized Swish/SiLU using lookup table (x * sigmoid(x))
#[allow(dead_code)]
pub fn swish_lookup(input: &Tensor, config: Option<LookupConfig>) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();
    let data = input.data()?;

    if data.len() >= config.size_threshold {
        let table = init_sigmoid_table();
        let result_data: Vec<f32> = data
            .iter()
            .map(|&x| {
                let sigmoid_val = if config.use_interpolation {
                    table.lookup(x)
                } else {
                    table.lookup_fast(x)
                };
                x * sigmoid_val
            })
            .collect();

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    } else {
        // Fall back to standard implementation
        let sigmoid_result = input.sigmoid()?;
        input.mul_op(&sigmoid_result)
    }
}

/// Benchmark and choose between lookup table and standard implementation
#[allow(dead_code)]
pub fn adaptive_sigmoid(input: &Tensor) -> TorshResult<Tensor> {
    let data_size = input.numel();

    // Use heuristics to choose implementation
    if data_size > 10000 {
        sigmoid_lookup(input, None)
    } else {
        input.sigmoid()
    }
}

/// Multi-threaded activation function application
#[allow(dead_code)]
pub fn parallel_activation<F>(
    input: &Tensor,
    activation_fn: F,
    chunk_size: Option<usize>,
) -> TorshResult<Tensor>
where
    F: Fn(f32) -> f32 + Send + Sync,
{
    use rayon::prelude::*;

    let data = input.data()?;
    let chunk_size = chunk_size.unwrap_or(1000);

    // Only use parallel processing for large tensors
    if data.len() >= chunk_size * 4 {
        let result_data: Vec<f32> = data
            .par_chunks(chunk_size)
            .flat_map(|chunk| chunk.iter().map(|&x| activation_fn(x)).collect::<Vec<_>>())
            .collect();

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    } else {
        // Sequential for small tensors
        let result_data: Vec<f32> = data.iter().map(|&x| activation_fn(x)).collect();

        Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::rand;

    #[test]
    fn test_sigmoid_lookup_table() {
        let table = ActivationLookupTable::new(-10.0, 10.0, 1000, |x| 1.0 / (1.0 + (-x).exp()));

        // Test specific values
        let x = 0.0;
        let expected = 0.5;
        let actual = table.lookup(x);
        assert!((actual - expected).abs() < 0.01);

        let x = 2.0;
        let expected = 1.0 / (1.0 + (-2.0_f32).exp());
        let actual = table.lookup(x);
        assert!((actual - expected).abs() < 0.01);
    }

    #[test]
    fn test_parallel_activation() {
        let input = rand(&[1000]).unwrap();
        let result = parallel_activation(&input, |x| x.max(0.0), Some(100)).unwrap();

        // Check that result has same shape
        assert_eq!(input.shape(), result.shape());
    }
}
