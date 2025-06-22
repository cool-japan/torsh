//! Parameter initialization functions

use torsh_tensor::{Tensor, creation::*};
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;
use scirs2::neural::utils::initializers as sci_init;

/// Calculate fan-in and fan-out for a tensor shape
pub fn calculate_fan_in_fan_out(shape: &[usize]) -> (usize, usize) {
    let dimensions = shape.len();
    
    if dimensions < 2 {
        panic!("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");
    }
    
    let num_input_fmaps = shape[1];
    let num_output_fmaps = shape[0];
    
    let mut receptive_field_size = 1;
    if dimensions > 2 {
        for i in 2..dimensions {
            receptive_field_size *= shape[i];
        }
    }
    
    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;
    
    (fan_in, fan_out)
}

/// Xavier/Glorot uniform initialization
pub fn xavier_uniform(shape: &[usize]) -> Tensor<f32> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(shape);
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let bound = std * 3.0_f32.sqrt();
    
    uniform(shape, -bound, bound)
}

/// Xavier/Glorot normal initialization
pub fn xavier_normal(shape: &[usize]) -> Tensor<f32> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(shape);
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    
    normal(shape, 0.0, std)
}

/// Kaiming/He uniform initialization
pub fn kaiming_uniform(shape: &[usize], mode: &str) -> Tensor<f32> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(shape);
    let fan = match mode {
        "fan_in" => fan_in,
        "fan_out" => fan_out,
        _ => panic!("Mode {} not supported, please use one of 'fan_in' or 'fan_out'.", mode),
    };
    
    let gain = 2.0_f32.sqrt(); // For ReLU
    let std = gain / (fan as f32).sqrt();
    let bound = std * 3.0_f32.sqrt();
    
    uniform(shape, -bound, bound)
}

/// Kaiming/He normal initialization
pub fn kaiming_normal(shape: &[usize], mode: &str) -> Tensor<f32> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(shape);
    let fan = match mode {
        "fan_in" => fan_in,
        "fan_out" => fan_out,
        _ => panic!("Mode {} not supported, please use one of 'fan_in' or 'fan_out'.", mode),
    };
    
    let gain = 2.0_f32.sqrt(); // For ReLU
    let std = gain / (fan as f32).sqrt();
    
    normal(shape, 0.0, std)
}

/// Uniform initialization
pub fn uniform(shape: &[usize], low: f32, high: f32) -> Tensor<f32> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let uniform_dist = Uniform::new(low, high);
    
    let values: Vec<f32> = (0..size)
        .map(|_| uniform_dist.sample(&mut rng))
        .collect();
    
    Tensor::from_vec(values, shape)
}

/// Normal initialization
pub fn normal(shape: &[usize], mean: f32, std: f32) -> Tensor<f32> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let normal_dist = Normal::new(mean, std).unwrap();
    
    let values: Vec<f32> = (0..size)
        .map(|_| normal_dist.sample(&mut rng))
        .collect();
    
    Tensor::from_vec(values, shape)
}

/// Constant initialization
pub fn constant(shape: &[usize], value: f32) -> Tensor<f32> {
    full(shape, value)
}

/// Eye/Identity initialization
pub fn eye_init(n: usize) -> Tensor<f32> {
    eye(n)
}

/// Orthogonal initialization
pub fn orthogonal(shape: &[usize], gain: f32) -> Tensor<f32> {
    // Implement orthogonal initialization using SVD
    // For now, fall back to xavier normal until we implement SVD
    let _ = gain; // Suppress unused warning
    xavier_normal(shape)
}

/// Sparse initialization
pub fn sparse(shape: &[usize], sparsity: f32, std: f32) -> Tensor<f32> {
    if shape.len() != 2 {
        panic!("Only tensors with 2 dimensions are supported");
    }
    
    let rows = shape[0];
    let cols = shape[1];
    let num_zeros = (rows as f32 * sparsity) as usize;
    
    let mut tensor = normal(shape, 0.0, std);
    
    // Set random elements to zero
    let mut rng = rand::thread_rng();
    for i in 0..rows {
        let mut indices: Vec<usize> = (0..cols).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);
        
        for j in 0..num_zeros.min(cols) {
            // TODO: Set element at (i, indices[j]) to 0
        }
    }
    
    tensor
}

/// Initialize a tensor with a specific initialization method
pub fn init_tensor(
    tensor: &mut Tensor<f32>,
    method: &str,
    gain: Option<f32>,
    mode: Option<&str>,
) {
    let binding = tensor.shape();
    let shape = binding.dims();
    let gain = gain.unwrap_or(1.0);
    let mode = mode.unwrap_or("fan_in");
    
    let initialized = match method {
        "xavier_uniform" | "glorot_uniform" => xavier_uniform(shape),
        "xavier_normal" | "glorot_normal" => xavier_normal(shape),
        "kaiming_uniform" | "he_uniform" => kaiming_uniform(shape, mode),
        "kaiming_normal" | "he_normal" => kaiming_normal(shape, mode),
        "orthogonal" => orthogonal(shape, gain),
        "zeros" => zeros(shape),
        "ones" => ones(shape),
        _ => panic!("Unknown initialization method: {}", method),
    };
    
    *tensor = initialized;
}

/// Reset parameters of a module using default initialization
pub trait Initializable {
    fn reset_parameters(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fan_calculation() {
        let (fan_in, fan_out) = calculate_fan_in_fan_out(&[64, 32, 3, 3]);
        assert_eq!(fan_in, 32 * 3 * 3);
        assert_eq!(fan_out, 64 * 3 * 3);
    }
    
    #[test]
    fn test_xavier_uniform() {
        let tensor = xavier_uniform(&[10, 5]);
        assert_eq!(tensor.shape().dims(), &[10, 5]);
    }
}