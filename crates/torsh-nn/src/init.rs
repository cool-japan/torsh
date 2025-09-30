//! Parameter initialization functions

// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{thread_rng, Random, Rng, quick::random_f32};
use scirs2_core::slice_random::shuffle;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::{creation::*, Tensor};

/// Unified initialization interface
pub trait Initializer {
    /// Initialize a tensor with the given shape
    fn initialize(&self, shape: &[usize]) -> Result<Tensor>;
}

/// Enumeration of initialization methods
#[derive(Debug, Clone)]
pub enum InitMethod {
    /// Xavier/Glorot uniform initialization
    XavierUniform { gain: f32 },
    /// Xavier/Glorot normal initialization
    XavierNormal { gain: f32 },
    /// Kaiming/He uniform initialization
    KaimingUniform {
        mode: FanMode,
        nonlinearity: Nonlinearity,
    },
    /// Kaiming/He normal initialization  
    KaimingNormal {
        mode: FanMode,
        nonlinearity: Nonlinearity,
    },
    /// Uniform random initialization
    Uniform { low: f32, high: f32 },
    /// Normal random initialization
    Normal { mean: f32, std: f32 },
    /// Zero initialization
    Zeros,
    /// Ones initialization
    Ones,
    /// Constant initialization
    Constant { value: f32 },
    /// Orthogonal initialization
    Orthogonal { gain: f32 },
    /// Sparse initialization
    Sparse { sparsity: f32, std: f32 },
    /// Identity/Eye initialization
    Eye,
    /// Lecun uniform initialization
    LecunUniform,
    /// Lecun normal initialization
    LecunNormal,
    /// Truncated normal initialization
    TruncatedNormal { mean: f32, std: f32, a: f32, b: f32 },
}

/// Fan mode for Kaiming initialization
#[derive(Debug, Clone, Copy)]
pub enum FanMode {
    FanIn,
    FanOut,
    FanAvg,
}

/// Nonlinearity types for calculating gains
#[derive(Debug, Clone, Copy)]
pub enum Nonlinearity {
    ReLU,
    LeakyReLU { negative_slope: f32 },
    Tanh,
    Sigmoid,
    SELU,
    ELU,
    Swish,
    Linear,
}

impl Nonlinearity {
    /// Calculate the gain for this nonlinearity
    pub fn gain(&self) -> f32 {
        match self {
            Nonlinearity::ReLU => (2.0_f32).sqrt(),
            Nonlinearity::LeakyReLU { negative_slope } => {
                (2.0 / (1.0 + negative_slope.powi(2))).sqrt()
            }
            Nonlinearity::Tanh => (5.0_f32 / 3.0_f32).sqrt(),
            Nonlinearity::Sigmoid => 1.0,
            Nonlinearity::SELU => (3.0_f32 / 4.0_f32).sqrt(),
            Nonlinearity::ELU => (5.0_f32 / 3.0_f32).sqrt(),
            Nonlinearity::Swish => (2.0_f32).sqrt(),
            Nonlinearity::Linear => 1.0,
        }
    }
}

impl Initializer for InitMethod {
    fn initialize(&self, shape: &[usize]) -> Result<Tensor> {
        match self {
            InitMethod::XavierUniform { gain } => xavier_uniform_with_gain(shape, *gain),
            InitMethod::XavierNormal { gain } => xavier_normal_with_gain(shape, *gain),
            InitMethod::KaimingUniform { mode, nonlinearity } => {
                kaiming_uniform_with_nonlinearity(shape, *mode, *nonlinearity)
            }
            InitMethod::KaimingNormal { mode, nonlinearity } => {
                kaiming_normal_with_nonlinearity(shape, *mode, *nonlinearity)
            }
            InitMethod::Uniform { low, high } => uniform(shape, *low, *high),
            InitMethod::Normal { mean, std } => normal(shape, *mean, *std),
            InitMethod::Zeros => zeros(shape),
            InitMethod::Ones => ones(shape),
            InitMethod::Constant { value } => constant(shape, *value),
            InitMethod::Orthogonal { gain } => orthogonal_init(shape, *gain),
            InitMethod::Sparse { sparsity, std } => sparse_init(shape, *sparsity, *std),
            InitMethod::Eye => eye_init_tensor(shape),
            InitMethod::LecunUniform => lecun_uniform(shape),
            InitMethod::LecunNormal => lecun_normal(shape),
            InitMethod::TruncatedNormal { mean, std, a, b } => {
                truncated_normal(shape, *mean, *std, *a, *b)
            }
        }
    }
}

/// Create a constant tensor filled with a specific value
pub fn constant(shape: &[usize], value: f32) -> Result<Tensor> {
    let size = shape.iter().product();
    let values = vec![value; size];
    Tensor::from_vec(values, shape)
        .map_err(|e| TorshError::RuntimeError(format!("Failed to create constant tensor: {}", e)))
}

/// Helper function to create an initializer
pub fn init(method: InitMethod) -> impl Initializer {
    method
}

/// Calculate fan-in and fan-out for a tensor shape
pub fn calculate_fan_in_fan_out(shape: &[usize]) -> Result<(usize, usize)> {
    let dimensions = shape.len();

    if dimensions < 2 {
        return Err(TorshError::InvalidArgument(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
                .to_string(),
        ));
    }

    let num_input_fmaps = shape[1];
    let num_output_fmaps = shape[0];

    let mut receptive_field_size = 1;
    if dimensions > 2 {
        for &size in shape.iter().skip(2).take(dimensions - 2) {
            receptive_field_size *= size;
        }
    }

    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;

    Ok((fan_in, fan_out))
}

/// Calculate the appropriate fan value based on mode
pub fn calculate_fan(shape: &[usize], mode: FanMode) -> Result<usize> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(shape)?;

    match mode {
        FanMode::FanIn => Ok(fan_in),
        FanMode::FanOut => Ok(fan_out),
        FanMode::FanAvg => Ok((fan_in + fan_out) / 2),
    }
}

/// Xavier/Glorot uniform initialization
pub fn xavier_uniform(shape: &[usize]) -> Result<Tensor> {
    xavier_uniform_with_gain(shape, 1.0)
}

/// Xavier/Glorot uniform initialization with custom gain
pub fn xavier_uniform_with_gain(shape: &[usize], gain: f32) -> Result<Tensor> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(shape)?;
    let std = gain * (2.0 / (fan_in + fan_out) as f32).sqrt();
    let bound = std * 3.0_f32.sqrt();

    uniform(shape, -bound, bound)
}

/// Xavier/Glorot normal initialization
pub fn xavier_normal(shape: &[usize]) -> Result<Tensor> {
    xavier_normal_with_gain(shape, 1.0)
}

/// Xavier/Glorot normal initialization with custom gain
pub fn xavier_normal_with_gain(shape: &[usize], gain: f32) -> Result<Tensor> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(shape)?;
    let std = gain * (2.0 / (fan_in + fan_out) as f32).sqrt();

    normal(shape, 0.0, std)
}

/// Kaiming/He uniform initialization
pub fn kaiming_uniform(shape: &[usize], mode: &str) -> Result<Tensor> {
    let fan_mode = match mode {
        "fan_in" => FanMode::FanIn,
        "fan_out" => FanMode::FanOut,
        "fan_avg" => FanMode::FanAvg,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Mode {} not supported, please use one of 'fan_in', 'fan_out', or 'fan_avg'.",
                mode
            )))
        }
    };

    kaiming_uniform_with_nonlinearity(shape, fan_mode, Nonlinearity::ReLU)
}

/// Kaiming/He uniform initialization with nonlinearity specification
pub fn kaiming_uniform_with_nonlinearity(
    shape: &[usize],
    mode: FanMode,
    nonlinearity: Nonlinearity,
) -> Result<Tensor> {
    let fan = calculate_fan(shape, mode)?;
    let gain = nonlinearity.gain();
    let std = gain / (fan as f32).sqrt();
    let bound = std * 3.0_f32.sqrt();

    uniform(shape, -bound, bound)
}

/// Kaiming/He normal initialization
pub fn kaiming_normal(shape: &[usize], mode: &str) -> Result<Tensor> {
    let fan_mode = match mode {
        "fan_in" => FanMode::FanIn,
        "fan_out" => FanMode::FanOut,
        "fan_avg" => FanMode::FanAvg,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Mode {} not supported, please use one of 'fan_in', 'fan_out', or 'fan_avg'.",
                mode
            )))
        }
    };

    kaiming_normal_with_nonlinearity(shape, fan_mode, Nonlinearity::ReLU)
}

/// Kaiming/He normal initialization with nonlinearity specification
pub fn kaiming_normal_with_nonlinearity(
    shape: &[usize],
    mode: FanMode,
    nonlinearity: Nonlinearity,
) -> Result<Tensor> {
    let fan = calculate_fan(shape, mode)?;
    let gain = nonlinearity.gain();
    let std = gain / (fan as f32).sqrt();

    normal(shape, 0.0, std)
}

/// Uniform initialization
pub fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Tensor> {
    if low >= high {
        return Err(TorshError::InvalidArgument(
            "Low bound must be less than high bound for uniform initialization".to_string(),
        ));
    }

    let size = shape.iter().product();
    let range = high - low;
    let values: Vec<f32> = (0..size).map(|_| low + random_f32() * range).collect();

    Tensor::from_vec(values, shape)
        .map_err(|e| TorshError::RuntimeError(format!("Failed to create uniform tensor: {}", e)))
}

/// Normal initialization
pub fn normal(shape: &[usize], mean: f32, std: f32) -> Result<Tensor> {
    if std <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Standard deviation must be positive for normal initialization".to_string(),
        ));
    }

    let size = shape.iter().product();
    let values: Vec<f32> = (0..size).map(|_| {
        // Box-Muller transform for normal distribution
        let u1 = random_f32();
        let u2 = random_f32();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + z0 * std
    }).collect();

    Tensor::from_vec(values, shape)
        .map_err(|e| TorshError::RuntimeError(format!("Failed to create normal tensor: {}", e)))
}

/// Lecun uniform initialization
pub fn lecun_uniform(shape: &[usize]) -> Result<Tensor> {
    let fan_in = calculate_fan(shape, FanMode::FanIn)?;
    let limit = (3.0 / fan_in as f32).sqrt();
    uniform(shape, -limit, limit)
}

/// Lecun normal initialization
pub fn lecun_normal(shape: &[usize]) -> Result<Tensor> {
    let fan_in = calculate_fan(shape, FanMode::FanIn)?;
    let std = (1.0 / fan_in as f32).sqrt();
    normal(shape, 0.0, std)
}

/// Truncated normal initialization
pub fn truncated_normal(shape: &[usize], mean: f32, std: f32, a: f32, b: f32) -> Result<Tensor> {
    if std <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Standard deviation must be positive for truncated normal initialization".to_string(),
        ));
    }

    if a >= b {
        return Err(TorshError::InvalidArgument(
            "Lower bound must be less than upper bound for truncated normal initialization"
                .to_string(),
        ));
    }

    let size = shape.iter().product();
    let mut values = Vec::with_capacity(size);

    for _ in 0..size {
        loop {
            // Box-Muller transform for normal distribution
            let u1 = random_f32();
            let u2 = random_f32();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let sample = mean + z0 * std;
            if sample >= a && sample <= b {
                values.push(sample);
                break;
            }
        }
    }

    Tensor::from_vec(values, shape).map_err(|e| {
        TorshError::RuntimeError(format!("Failed to create truncated normal tensor: {}", e))
    })
}

/// Eye/Identity initialization for square matrices
pub fn eye_init(n: usize) -> Result<Tensor> {
    eye(n).map_err(|e| TorshError::RuntimeError(format!("Failed to create eye tensor: {}", e)))
}

/// Eye/Identity initialization for arbitrary tensor shapes
pub fn eye_init_tensor(shape: &[usize]) -> Result<Tensor> {
    if shape.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Eye initialization requires at least 2D tensor".to_string(),
        ));
    }

    let rows = shape[0];
    let cols = shape[1];

    if rows != cols {
        return Err(TorshError::InvalidArgument(
            "Eye initialization requires square matrices (rows == cols)".to_string(),
        ));
    }

    eye_init(rows)
}

/// Orthogonal initialization
pub fn orthogonal_init(shape: &[usize], gain: f32) -> Result<Tensor> {
    if shape.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Orthogonal initialization requires at least 2D tensor".to_string(),
        ));
    }

    // For now, use a simplified orthogonal approximation
    // TODO: Implement proper QR decomposition when available
    let num_rows = shape[0];
    let num_cols = shape[1];

    // For square matrices, use Xavier normal as approximation
    let tensor = if num_rows == num_cols {
        xavier_normal_with_gain(shape, gain)?
    } else {
        // For non-square, use truncated normal
        normal(shape, 0.0, (1.0 / num_cols.max(num_rows) as f32).sqrt())?
    };

    // Apply gain
    if gain != 1.0 {
        // TODO: Scale tensor by gain when tensor operations are available
    }

    Ok(tensor)
}

/// Sparse initialization
pub fn sparse_init(shape: &[usize], sparsity: f32, std: f32) -> Result<Tensor> {
    if shape.len() != 2 {
        return Err(TorshError::InvalidArgument(
            "Only tensors with 2 dimensions are supported for sparse initialization".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&sparsity) {
        return Err(TorshError::InvalidArgument(
            "Sparsity must be between 0.0 and 1.0".to_string(),
        ));
    }

    let rows = shape[0];
    let cols = shape[1];
    let total_elements = rows * cols;
    let num_zeros = (total_elements as f32 * sparsity) as usize;

    // Start with normal initialization
    let mut values = Vec::with_capacity(total_elements);
    let mut rng = thread_rng();

    for _ in 0..total_elements {
        // Box-Muller transform for normal distribution
        let u1 = random_f32();
        let u2 = random_f32();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        values.push(z0 * std); // mean = 0.0
    }

    // Randomly zero out elements
    // ✅ SciRS2 Policy Compliant - Using scirs2_core random shuffling
    let mut indices: Vec<usize> = (0..total_elements).collect();
    // Use scirs2_core's shuffle functionality
    shuffle(&mut indices);

    for &idx in indices.iter().take(num_zeros) {
        values[idx] = 0.0;
    }

    Tensor::from_vec(values, shape)
        .map_err(|e| TorshError::RuntimeError(format!("Failed to create sparse tensor: {}", e)))
}

/// Initialize a tensor with a specific initialization method
pub fn init_tensor(
    tensor: &mut Tensor,
    method: &str,
    gain: Option<f32>,
    mode: Option<&str>,
) -> Result<()> {
    let binding = tensor.shape();
    let shape = binding.dims();
    let gain = gain.unwrap_or(1.0);
    let mode = mode.unwrap_or("fan_in");

    let initialized = match method {
        "xavier_uniform" | "glorot_uniform" => xavier_uniform_with_gain(shape, gain),
        "xavier_normal" | "glorot_normal" => xavier_normal_with_gain(shape, gain),
        "kaiming_uniform" | "he_uniform" => kaiming_uniform(shape, mode),
        "kaiming_normal" | "he_normal" => kaiming_normal(shape, mode),
        "orthogonal" => orthogonal_init(shape, gain),
        "lecun_uniform" => lecun_uniform(shape),
        "lecun_normal" => lecun_normal(shape),
        "zeros" => zeros(shape),
        "ones" => ones(shape),
        "eye" => eye_init_tensor(shape),
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Unknown initialization method: {}",
                method
            )))
        }
    }?;

    *tensor = initialized;
    Ok(())
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
        let (fan_in, fan_out) = calculate_fan_in_fan_out(&[64, 32, 3, 3]).unwrap();
        assert_eq!(fan_in, 32 * 3 * 3);
        assert_eq!(fan_out, 64 * 3 * 3);
    }

    #[test]
    fn test_xavier_uniform() {
        let tensor = xavier_uniform(&[10, 5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);
    }

    #[test]
    fn test_init_method_enum() {
        let method = InitMethod::XavierUniform { gain: 1.0 };
        let tensor = method.initialize(&[5, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[5, 3]);
    }

    #[test]
    fn test_nonlinearity_gains() {
        assert!((Nonlinearity::ReLU.gain() - (2.0_f32).sqrt()).abs() < 1e-6);
        assert!((Nonlinearity::Linear.gain() - 1.0).abs() < 1e-6);
        assert!(
            (Nonlinearity::LeakyReLU {
                negative_slope: 0.01
            }
            .gain()
                - (2.0 / (1.0 + 0.01_f32.powi(2))).sqrt())
            .abs()
                < 1e-6
        );
    }

    #[test]
    fn test_sparse_initialization() {
        let tensor = sparse_init(&[10, 10], 0.5, 1.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        // Test with invalid sparsity
        assert!(sparse_init(&[10, 10], 1.5, 1.0).is_err());
        assert!(sparse_init(&[10, 10], -0.1, 1.0).is_err());
    }
}
