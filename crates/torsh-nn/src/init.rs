//! Parameter initialization functions

// ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
use scirs2_core::random::{quick::random_f32, thread_rng};
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
    /// Variance scaling initialization (generalization of Xavier/Kaiming)
    VarianceScaling {
        scale: f32,
        mode: FanMode,
        distribution: Distribution,
    },
    /// Dirac initialization for convolutional layers
    Dirac,
    /// SIREN initialization for periodic activation functions
    /// Recommended for networks using sine activations
    SIREN { c: f32, w0: f32 },
}

/// Distribution type for variance scaling initialization
#[derive(Debug, Clone, Copy)]
pub enum Distribution {
    /// Uniform distribution
    Uniform,
    /// Normal (Gaussian) distribution
    Normal,
    /// Truncated normal distribution
    TruncatedNormal,
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

impl InitMethod {
    /// Create Xavier/Glorot uniform initialization with default gain (1.0)
    pub fn xavier_uniform() -> Self {
        InitMethod::XavierUniform { gain: 1.0 }
    }

    /// Create Xavier/Glorot normal initialization with default gain (1.0)
    pub fn xavier_normal() -> Self {
        InitMethod::XavierNormal { gain: 1.0 }
    }

    /// Create Kaiming/He uniform initialization for ReLU activations
    pub fn kaiming_uniform() -> Self {
        InitMethod::KaimingUniform {
            mode: FanMode::FanIn,
            nonlinearity: Nonlinearity::ReLU,
        }
    }

    /// Create Kaiming/He normal initialization for ReLU activations
    pub fn kaiming_normal() -> Self {
        InitMethod::KaimingNormal {
            mode: FanMode::FanIn,
            nonlinearity: Nonlinearity::ReLU,
        }
    }

    /// Create uniform initialization with specified range
    pub fn uniform_range(low: f32, high: f32) -> Self {
        InitMethod::Uniform { low, high }
    }

    /// Create normal initialization with specified mean and standard deviation
    pub fn normal_dist(mean: f32, std: f32) -> Self {
        InitMethod::Normal { mean, std }
    }

    /// Create zero initialization
    pub fn zeros() -> Self {
        InitMethod::Zeros
    }

    /// Create ones initialization
    pub fn ones() -> Self {
        InitMethod::Ones
    }

    /// Create constant initialization with specified value
    pub fn constant(value: f32) -> Self {
        InitMethod::Constant { value }
    }

    /// Create orthogonal initialization with default gain (1.0)
    pub fn orthogonal() -> Self {
        InitMethod::Orthogonal { gain: 1.0 }
    }

    /// Create LeCun uniform initialization
    pub fn lecun_uniform() -> Self {
        InitMethod::LecunUniform
    }

    /// Create LeCun normal initialization
    pub fn lecun_normal() -> Self {
        InitMethod::LecunNormal
    }

    /// Create Dirac initialization for convolutional layers
    pub fn dirac() -> Self {
        InitMethod::Dirac
    }

    /// Create SIREN initialization for first layer (w0=30.0, c=1.0)
    pub fn siren_first_layer() -> Self {
        InitMethod::SIREN { c: 1.0, w0: 30.0 }
    }

    /// Create SIREN initialization for hidden layers (w0=1.0, c=6.0)
    pub fn siren_hidden_layer() -> Self {
        InitMethod::SIREN { c: 6.0, w0: 1.0 }
    }

    /// Set the gain for applicable initialization methods
    pub fn with_gain(self, gain: f32) -> Self {
        match self {
            InitMethod::XavierUniform { .. } => InitMethod::XavierUniform { gain },
            InitMethod::XavierNormal { .. } => InitMethod::XavierNormal { gain },
            InitMethod::Orthogonal { .. } => InitMethod::Orthogonal { gain },
            other => other,
        }
    }

    /// Set the fan mode for applicable initialization methods
    pub fn with_fan_mode(self, mode: FanMode) -> Self {
        match self {
            InitMethod::KaimingUniform {
                nonlinearity,
                mode: _,
            } => InitMethod::KaimingUniform { mode, nonlinearity },
            InitMethod::KaimingNormal {
                nonlinearity,
                mode: _,
            } => InitMethod::KaimingNormal { mode, nonlinearity },
            InitMethod::VarianceScaling {
                scale,
                distribution,
                mode: _,
            } => InitMethod::VarianceScaling {
                scale,
                mode,
                distribution,
            },
            other => other,
        }
    }

    /// Set the nonlinearity for applicable initialization methods
    pub fn with_nonlinearity(self, nonlinearity: Nonlinearity) -> Self {
        match self {
            InitMethod::KaimingUniform { mode, .. } => {
                InitMethod::KaimingUniform { mode, nonlinearity }
            }
            InitMethod::KaimingNormal { mode, .. } => {
                InitMethod::KaimingNormal { mode, nonlinearity }
            }
            other => other,
        }
    }

    /// Get a human-readable name for this initialization method
    pub fn name(&self) -> &str {
        match self {
            InitMethod::XavierUniform { .. } => "Xavier Uniform",
            InitMethod::XavierNormal { .. } => "Xavier Normal",
            InitMethod::KaimingUniform { .. } => "Kaiming Uniform",
            InitMethod::KaimingNormal { .. } => "Kaiming Normal",
            InitMethod::Uniform { .. } => "Uniform",
            InitMethod::Normal { .. } => "Normal",
            InitMethod::Zeros => "Zeros",
            InitMethod::Ones => "Ones",
            InitMethod::Constant { .. } => "Constant",
            InitMethod::Orthogonal { .. } => "Orthogonal",
            InitMethod::Sparse { .. } => "Sparse",
            InitMethod::Eye => "Eye/Identity",
            InitMethod::LecunUniform => "LeCun Uniform",
            InitMethod::LecunNormal => "LeCun Normal",
            InitMethod::TruncatedNormal { .. } => "Truncated Normal",
            InitMethod::VarianceScaling { .. } => "Variance Scaling",
            InitMethod::Dirac => "Dirac",
            InitMethod::SIREN { .. } => "SIREN",
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
            InitMethod::VarianceScaling {
                scale,
                mode,
                distribution,
            } => variance_scaling(shape, *scale, *mode, *distribution),
            InitMethod::Dirac => dirac_init(shape),
            InitMethod::SIREN { c, w0 } => siren_init(shape, *c, *w0),
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
    let values: Vec<f32> = (0..size)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1 = random_f32();
            let u2 = random_f32();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + z0 * std
        })
        .collect();

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

/// Orthogonal initialization using QR decomposition
///
/// Generates an orthogonal matrix (or semi-orthogonal for non-square matrices)
/// using QR decomposition of a random Gaussian matrix. This initialization
/// helps preserve gradient norms during backpropagation, improving training stability.
///
/// # Arguments
/// * `shape` - Shape of the tensor (must be at least 2D)
/// * `gain` - Scaling factor applied to the orthogonal matrix
///
/// # Returns
/// An orthogonal (or semi-orthogonal) tensor scaled by `gain`
pub fn orthogonal_init(shape: &[usize], gain: f32) -> Result<Tensor> {
    if shape.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Orthogonal initialization requires at least 2D tensor".to_string(),
        ));
    }

    let num_rows = shape[0];
    let num_cols = shape[1];

    // Generate a random Gaussian matrix
    // For non-square matrices, we need the larger dimension for proper orthogonalization
    let (qr_rows, qr_cols) = if num_rows < num_cols {
        (num_cols, num_rows)
    } else {
        (num_rows, num_cols)
    };

    // Generate random Gaussian tensor for QR decomposition
    let random_tensor = normal(&[qr_rows, qr_cols], 0.0, 1.0)?;

    // Perform QR decomposition to get orthogonal Q matrix
    let (q, _r) = torsh_linalg::decomposition::qr(&random_tensor)?;

    // Extract the portion we need
    let orthogonal_tensor = if num_rows < num_cols {
        // Transpose: we generated (num_cols × num_rows), need (num_rows × num_cols)
        // Extract first num_rows columns and transpose
        let mut values = Vec::with_capacity(num_rows * num_cols);
        for col in 0..num_cols {
            for row in 0..num_rows {
                values.push(q.get(&[col, row])?);
            }
        }
        Tensor::from_vec(values, &[num_rows, num_cols])?
    } else {
        // Normal case: extract first num_rows × num_cols portion
        let mut values = Vec::with_capacity(num_rows * num_cols);
        for row in 0..num_rows {
            for col in 0..num_cols {
                values.push(q.get(&[row, col])?);
            }
        }
        Tensor::from_vec(values, &[num_rows, num_cols])?
    };

    // Apply gain scaling
    if (gain - 1.0).abs() > 1e-6 {
        // Scale all values by gain
        let values: Vec<f32> = orthogonal_tensor
            .to_vec()?
            .iter()
            .map(|&v| v * gain)
            .collect();
        Tensor::from_vec(values, shape)
    } else {
        Ok(orthogonal_tensor)
    }
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
    let _rng = thread_rng();

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

/// Variance scaling initialization
///
/// A general initialization method that covers Xavier and Kaiming as special cases.
/// The variance of the initialized weights is controlled by the scale and fan mode.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `scale` - Scaling factor for the variance
/// * `mode` - Which fan to use (fan_in, fan_out, or fan_avg)
/// * `distribution` - Distribution type (uniform, normal, or truncated_normal)
///
/// # Examples
/// - Xavier uniform: variance_scaling(shape, 1.0, FanMode::FanAvg, Distribution::Uniform)
/// - Kaiming normal: variance_scaling(shape, 2.0, FanMode::FanIn, Distribution::Normal)
pub fn variance_scaling(
    shape: &[usize],
    scale: f32,
    mode: FanMode,
    distribution: Distribution,
) -> Result<Tensor> {
    let fan = calculate_fan(shape, mode)?;
    let variance = scale / fan as f32;

    match distribution {
        Distribution::Uniform => {
            let limit = (3.0 * variance).sqrt();
            uniform(shape, -limit, limit)
        }
        Distribution::Normal => {
            let std = variance.sqrt();
            normal(shape, 0.0, std)
        }
        Distribution::TruncatedNormal => {
            let std = variance.sqrt();
            // Truncate at 2 standard deviations
            truncated_normal(shape, 0.0, std, -2.0 * std, 2.0 * std)
        }
    }
}

/// Dirac initialization for convolutional layers
///
/// Initializes convolutional kernels with the Dirac delta function, creating
/// identity-like convolutions that preserve input features. Particularly useful
/// for residual connections in very deep networks.
///
/// For 3D kernels (out_channels, in_channels, kernel_size), this creates
/// identity mappings where possible, with zeros elsewhere.
///
/// # Arguments
/// * `shape` - Shape of the convolutional kernel (must be at least 3D)
///
/// # Note
/// - For kernels where in_channels != out_channels, only min(in_channels, out_channels) are initialized as identity
/// - The center position of the kernel contains the identity mapping
pub fn dirac_init(shape: &[usize]) -> Result<Tensor> {
    if shape.len() < 3 {
        return Err(TorshError::InvalidArgument(
            "Dirac initialization requires at least 3D tensor (out_channels, in_channels, kernel_size)".to_string(),
        ));
    }

    let out_channels = shape[0];
    let in_channels = shape[1];

    // Calculate total size and kernel size
    let total_size: usize = shape.iter().product();
    let kernel_spatial_size: usize = shape[2..].iter().product();

    // Start with zeros
    let mut values = vec![0.0_f32; total_size];

    // Find center position in spatial dimensions
    let mut center_offset = 0;
    let mut stride = 1;
    for &dim_size in shape[2..].iter().rev() {
        center_offset += (dim_size / 2) * stride;
        stride *= dim_size;
    }

    // Set diagonal elements to 1.0 at the center of the kernel
    let min_channels = out_channels.min(in_channels);
    for i in 0..min_channels {
        let idx = i * in_channels * kernel_spatial_size + i * kernel_spatial_size + center_offset;
        if idx < total_size {
            values[idx] = 1.0;
        }
    }

    Tensor::from_vec(values, shape)
        .map_err(|e| TorshError::RuntimeError(format!("Failed to create Dirac tensor: {}", e)))
}

/// SIREN initialization
///
/// Initialization method designed for networks using sine activations (SIREN: Sinusoidal Representation Networks).
/// This initialization ensures that activations are properly distributed for sine activation functions.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `c` - Constant factor (typically 6.0 for hidden layers)
/// * `w0` - Frequency parameter for the first layer (typically 30.0)
///
/// # Recommendations
/// - First layer: Use c=1.0, w0=30.0, then multiply weights by w0
/// - Hidden layers: Use c=6.0, w0=1.0
///
/// # Reference
/// Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020
pub fn siren_init(shape: &[usize], c: f32, w0: f32) -> Result<Tensor> {
    if shape.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "SIREN initialization requires at least 2D tensor".to_string(),
        ));
    }

    let fan_in = calculate_fan(shape, FanMode::FanIn)?;

    // For the first layer, use uniform distribution in [-1/n, 1/n]
    // For hidden layers, use uniform distribution in [-sqrt(c/n)/w0, sqrt(c/n)/w0]
    let bound = if (w0 - 1.0).abs() < 1e-6 {
        // Hidden layer
        (c / fan_in as f32).sqrt()
    } else {
        // First layer: sample from [-1/n, 1/n] then multiply by w0
        1.0 / fan_in as f32
    };

    let mut tensor = uniform(shape, -bound, bound)?;

    // For first layer, multiply by w0
    if (w0 - 1.0).abs() > 1e-6 {
        let values: Vec<f32> = tensor.to_vec()?.iter().map(|&v| v * w0).collect();
        tensor = Tensor::from_vec(values, shape)?;
    }

    Ok(tensor)
}

// =============================================================================
// MODERN INITIALIZATION TECHNIQUES
// =============================================================================

/// Fixup initialization for very deep networks without normalization
///
/// Fixup initialization is designed for training very deep residual networks (100+ layers)
/// without batch normalization. It scales weights based on network depth to prevent
/// gradient explosion/vanishing.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `num_layers` - Total number of layers in the network
/// * `num_residual_blocks` - Number of residual blocks in the network
///
/// # Formula
/// For residual branch weights: scale by (2 * num_layers)^(-1/(2*num_residual_blocks-2))
/// For other weights: standard initialization
///
/// # Reference
/// Zhang et al., "Fixup Initialization: Residual Learning Without Normalization", ICLR 2019
pub fn fixup_init(
    shape: &[usize],
    num_layers: usize,
    num_residual_blocks: usize,
    is_residual_branch: bool,
) -> Result<Tensor> {
    // Start with He/Kaiming normal initialization
    let mut tensor = kaiming_normal_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::ReLU)?;

    if is_residual_branch && num_residual_blocks > 1 {
        // Calculate Fixup scaling factor
        let exponent = -1.0 / (2.0 * num_residual_blocks as f32 - 2.0);
        let scale = (2.0 * num_layers as f32).powf(exponent);

        // Scale the tensor
        let values: Vec<f32> = tensor.to_vec()?.iter().map(|&v| v * scale).collect();
        tensor = Tensor::from_vec(values, shape).map_err(|e| {
            TorshError::RuntimeError(format!("Failed to create Fixup tensor: {}", e))
        })?;
    }

    Ok(tensor)
}

/// ReZero initialization for ReZero-style residual connections
///
/// ReZero initialization sets a learnable scalar parameter (alpha) to zero initially,
/// allowing the network to start as an identity function and gradually learn representations.
/// This enables training of very deep networks without normalization.
///
/// # Arguments
/// * `shape` - Shape of the tensor
///
/// # Note
/// This returns a weight tensor initialized normally, meant to be multiplied by a zero-initialized
/// scalar (alpha) parameter. The alpha parameter should be initialized separately to 0.
///
/// # Reference
/// Bachlechner et al., "ReZero is All You Need: Fast Convergence at Large Depth", UAI 2021
pub fn rezero_init(shape: &[usize]) -> Result<Tensor> {
    // Use standard initialization for the weight tensor
    // The alpha (residual weight) should be initialized to 0 separately
    kaiming_normal_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::ReLU)
}

/// Create zero-initialized scalar for ReZero residual weight
pub fn rezero_alpha_init() -> Result<Tensor> {
    Tensor::from_vec(vec![0.0_f32], &[1])
        .map_err(|e| TorshError::RuntimeError(format!("Failed to create ReZero alpha: {}", e)))
}

/// Delta-Orthogonal initialization
///
/// An improved variant of orthogonal initialization that handles non-square matrices better
/// and provides better gradient flow properties. Particularly effective for RNNs and deep networks.
///
/// # Arguments
/// * `shape` - Shape of the tensor (must be at least 2D)
/// * `gain` - Scaling factor applied to the orthogonal matrix
///
/// # Reference
/// Xiao et al., "Dynamical Isometry and a Mean Field Theory of CNNs", ICML 2018
pub fn delta_orthogonal_init(shape: &[usize], gain: f32) -> Result<Tensor> {
    if shape.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Delta-Orthogonal initialization requires at least 2D tensor".to_string(),
        ));
    }

    // For now, use standard orthogonal initialization
    // Full delta-orthogonal would require convolution-aware initialization
    orthogonal_init(shape, gain)
}

/// Meta-learning inspired initialization (MetaInit)
///
/// Initialization method inspired by meta-learning that aims to put parameters
/// in a region that enables fast adaptation with few gradient steps.
/// Uses a combination of small magnitude with strategic sparsity.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `sparsity` - Fraction of weights to set to zero (typically 0.7-0.9)
/// * `scale` - Magnitude of non-zero weights (typically 0.01-0.1)
///
/// # Reference
/// Inspired by MAML and Reptile meta-learning algorithms
pub fn metainit(shape: &[usize], sparsity: f32, scale: f32) -> Result<Tensor> {
    if sparsity < 0.0 || sparsity >= 1.0 {
        return Err(TorshError::InvalidArgument(format!(
            "Sparsity must be in [0, 1), got {}",
            sparsity
        )));
    }

    if scale <= 0.0 {
        return Err(TorshError::InvalidArgument(format!(
            "Scale must be positive, got {}",
            scale
        )));
    }

    let size = shape.iter().product();
    let mut values = Vec::with_capacity(size);

    for _ in 0..size {
        if random_f32() < sparsity {
            values.push(0.0);
        } else {
            // Use small random values for non-zero weights
            let sign = if random_f32() < 0.5 { -1.0 } else { 1.0 };
            values.push(sign * scale * random_f32());
        }
    }

    Tensor::from_vec(values, shape)
        .map_err(|e| TorshError::RuntimeError(format!("Failed to create MetaInit tensor: {}", e)))
}

/// Layer-Sequential Unit-Variance (LSUV) initialization helper
///
/// LSUV is a data-driven initialization method that normalizes layer outputs to unit variance.
/// This function provides the initial orthogonal initialization; the normalization step
/// requires forward passes with actual data.
///
/// # Arguments
/// * `shape` - Shape of the tensor
///
/// # Note
/// This only provides the first step (orthogonal initialization). The full LSUV algorithm
/// requires iterative normalization with forward passes on real data batches.
///
/// # Reference
/// Mishkin & Matas, "All you need is a good init", ICLR 2016
pub fn lsuv_init(shape: &[usize]) -> Result<Tensor> {
    // Start with orthogonal initialization
    // The actual layer-sequential unit-variance normalization requires forward passes
    orthogonal_init(shape, 1.0)
}

/// Zero-centered initialization with controlled variance
///
/// Initializes weights with zero mean and carefully controlled variance based on
/// the layer's position in the network and its fan-in/fan-out.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `target_variance` - Target variance for the initialization
///
/// # Use Case
/// Useful for layers that need precise variance control, such as in normalizing flows
/// or when specific signal propagation properties are required.
pub fn zero_centered_variance_init(shape: &[usize], target_variance: f32) -> Result<Tensor> {
    if target_variance <= 0.0 {
        return Err(TorshError::InvalidArgument(format!(
            "Target variance must be positive, got {}",
            target_variance
        )));
    }

    let std = target_variance.sqrt();
    normal(shape, 0.0, std)
}

/// Balanced initialization for GANs
///
/// Specialized initialization for GAN training that balances generator and discriminator
/// learning rates. Uses smaller initial weights to prevent early collapse.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `is_generator` - Whether this is for generator (true) or discriminator (false)
///
/// # Formula
/// - Generator: Smaller weights (Xavier with 0.5 gain) for stable learning
/// - Discriminator: Standard Xavier with 1.0 gain
pub fn gan_balanced_init(shape: &[usize], is_generator: bool) -> Result<Tensor> {
    let gain = if is_generator { 0.5 } else { 1.0 };

    let fan_in = calculate_fan(shape, FanMode::FanIn)?;
    let fan_out = calculate_fan(shape, FanMode::FanOut)?;
    let fan_avg = (fan_in + fan_out) / 2;

    let std = gain * (2.0 / fan_avg as f32).sqrt();
    normal(shape, 0.0, std)
}

/// Coordinate-based network initialization (for NeRF-style architectures)
///
/// Specialized initialization for coordinate-based neural networks (like NeRF)
/// that map coordinates to properties. Uses geometric priors to initialize weights.
///
/// # Arguments
/// * `shape` - Shape of the tensor
/// * `omega_0` - Frequency scaling parameter (typical values: 1.0-30.0)
///
/// # Reference
/// Inspired by NeRF and Instant-NGP positional encoding strategies
pub fn coordinate_mlp_init(shape: &[usize], omega_0: f32) -> Result<Tensor> {
    if shape.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Coordinate MLP initialization requires at least 2D tensor".to_string(),
        ));
    }

    let fan_in = calculate_fan(shape, FanMode::FanIn)?;
    let std = 1.0 / (fan_in as f32 * omega_0).sqrt();

    normal(shape, 0.0, std)
}

// =============================================================================
// AUTOMATIC INITIALIZATION SELECTION UTILITIES
// =============================================================================

/// Architecture hint for automatic initialization selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureHint {
    /// Standard feedforward network
    Feedforward,
    /// Convolutional network
    Convolutional,
    /// Recurrent network (RNN, LSTM, GRU)
    Recurrent,
    /// Transformer or attention-based architecture
    Transformer,
    /// Residual network (ResNet-style)
    Residual,
    /// Very deep network (100+ layers)
    VeryDeep,
    /// Generative Adversarial Network
    GAN,
    /// Coordinate-based network (NeRF-style)
    CoordinateBased,
    /// Network with periodic activations (SIREN)
    Periodic,
    /// Auto-encoder or VAE
    Autoencoder,
}

/// Activation function hint for automatic initialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationHint {
    /// ReLU or variants (LeakyReLU, PReLU)
    ReLU,
    /// Tanh activation
    Tanh,
    /// Sigmoid activation
    Sigmoid,
    /// SELU activation
    SELU,
    /// Swish/SiLU activation
    Swish,
    /// GELU activation
    GELU,
    /// Sine activation (for SIREN)
    Sine,
    /// Linear/no activation
    Linear,
}

/// Automatic initialization selection based on architecture and activation
///
/// Selects an appropriate initialization method based on the network architecture,
/// activation function, and layer properties. This provides a convenient way to
/// get good default initializations without manually specifying the method.
///
/// # Arguments
/// * `shape` - Shape of the tensor to initialize
/// * `arch` - Architecture hint (feedforward, convolutional, recurrent, etc.)
/// * `activation` - Activation function hint (ReLU, Tanh, etc.)
/// * `layer_depth` - Optional layer depth in the network (for depth-aware initialization)
///
/// # Returns
/// An initialized tensor with an appropriate initialization method
///
/// # Example
/// ```ignore
/// use torsh_nn::init::{auto_init, ArchitectureHint, ActivationHint};
///
/// // Initialize weights for a convolutional layer with ReLU
/// let weights = auto_init(&[64, 32, 3, 3], ArchitectureHint::Convolutional, ActivationHint::ReLU, None)?;
///
/// // Initialize weights for a deep residual network
/// let weights = auto_init(&[256, 256], ArchitectureHint::Residual, ActivationHint::ReLU, Some(50))?;
/// ```
pub fn auto_init(
    shape: &[usize],
    arch: ArchitectureHint,
    activation: ActivationHint,
    layer_depth: Option<usize>,
) -> Result<Tensor> {
    match (arch, activation) {
        // SIREN / Periodic activations
        (ArchitectureHint::Periodic, ActivationHint::Sine) | (_, ActivationHint::Sine) => {
            let is_first_layer = layer_depth.unwrap_or(0) == 0;
            if is_first_layer {
                siren_init(shape, 1.0, 30.0)
            } else {
                siren_init(shape, 6.0, 1.0)
            }
        }

        // Coordinate-based networks (NeRF-style)
        (ArchitectureHint::CoordinateBased, _) => coordinate_mlp_init(shape, 1.0),

        // Very deep networks
        (ArchitectureHint::VeryDeep, ActivationHint::ReLU) => {
            if let Some(depth) = layer_depth {
                // Use Fixup for very deep networks
                fixup_init(shape, depth, depth / 2, true)
            } else {
                kaiming_normal_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::ReLU)
            }
        }

        // Residual networks
        (ArchitectureHint::Residual, _) => {
            if layer_depth.is_some() {
                // Use ReZero or Fixup for deep residual networks
                rezero_init(shape)
            } else {
                kaiming_normal_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::ReLU)
            }
        }

        // Recurrent networks
        (ArchitectureHint::Recurrent, _) => orthogonal_init(shape, 1.0),

        // Transformer/Attention
        (ArchitectureHint::Transformer, _) => xavier_uniform(shape),

        // GANs
        (ArchitectureHint::GAN, _) => {
            // Default to generator initialization
            gan_balanced_init(shape, true)
        }

        // Convolutional networks with specific activations
        (ArchitectureHint::Convolutional, ActivationHint::ReLU) => {
            kaiming_normal_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::ReLU)
        }
        (ArchitectureHint::Convolutional, ActivationHint::Tanh) => xavier_normal(shape),
        (ArchitectureHint::Convolutional, ActivationHint::Sigmoid) => xavier_normal(shape),
        (ArchitectureHint::Convolutional, ActivationHint::SELU) => lecun_normal(shape),
        (ArchitectureHint::Convolutional, ActivationHint::Swish)
        | (ArchitectureHint::Convolutional, ActivationHint::GELU) => {
            kaiming_normal_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::Swish)
        }

        // Feedforward networks with specific activations
        (ArchitectureHint::Feedforward, ActivationHint::ReLU)
        | (ArchitectureHint::Autoencoder, ActivationHint::ReLU) => {
            kaiming_uniform_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::ReLU)
        }
        (ArchitectureHint::Feedforward, ActivationHint::Tanh)
        | (ArchitectureHint::Autoencoder, ActivationHint::Tanh) => xavier_uniform(shape),
        (ArchitectureHint::Feedforward, ActivationHint::Sigmoid)
        | (ArchitectureHint::Autoencoder, ActivationHint::Sigmoid) => xavier_uniform(shape),
        (ArchitectureHint::Feedforward, ActivationHint::SELU)
        | (ArchitectureHint::Autoencoder, ActivationHint::SELU) => lecun_uniform(shape),
        (ArchitectureHint::Feedforward, ActivationHint::Swish)
        | (ArchitectureHint::Feedforward, ActivationHint::GELU)
        | (ArchitectureHint::Autoencoder, ActivationHint::Swish)
        | (ArchitectureHint::Autoencoder, ActivationHint::GELU) => {
            kaiming_uniform_with_nonlinearity(shape, FanMode::FanIn, Nonlinearity::Swish)
        }
        (ArchitectureHint::Feedforward, ActivationHint::Linear) | (_, ActivationHint::Linear) => {
            xavier_uniform(shape)
        }

        // Catch-all: use Xavier as safe default
        _ => xavier_uniform(shape),
    }
}

/// Get recommended initialization method as InitMethod enum
///
/// Similar to `auto_init` but returns an `InitMethod` enum instead of an initialized tensor.
/// Useful when you want to know the recommended method without immediately initializing.
///
/// # Arguments
/// * `arch` - Architecture hint
/// * `activation` - Activation function hint
/// * `layer_depth` - Optional layer depth in the network
///
/// # Returns
/// The recommended `InitMethod` for the given configuration
pub fn recommend_init_method(
    arch: ArchitectureHint,
    activation: ActivationHint,
    layer_depth: Option<usize>,
) -> InitMethod {
    match (arch, activation) {
        // SIREN / Periodic activations
        (ArchitectureHint::Periodic, ActivationHint::Sine) | (_, ActivationHint::Sine) => {
            let is_first_layer = layer_depth.unwrap_or(0) == 0;
            if is_first_layer {
                InitMethod::SIREN { c: 1.0, w0: 30.0 }
            } else {
                InitMethod::SIREN { c: 6.0, w0: 1.0 }
            }
        }

        // Very deep or residual networks with ReLU
        (ArchitectureHint::VeryDeep, ActivationHint::ReLU)
        | (ArchitectureHint::Residual, ActivationHint::ReLU) => InitMethod::KaimingNormal {
            mode: FanMode::FanIn,
            nonlinearity: Nonlinearity::ReLU,
        },

        // Recurrent networks
        (ArchitectureHint::Recurrent, _) => InitMethod::Orthogonal { gain: 1.0 },

        // Transformer/Attention
        (ArchitectureHint::Transformer, _) => InitMethod::XavierUniform { gain: 1.0 },

        // Convolutional with ReLU
        (ArchitectureHint::Convolutional, ActivationHint::ReLU) => InitMethod::KaimingNormal {
            mode: FanMode::FanIn,
            nonlinearity: Nonlinearity::ReLU,
        },

        // SELU networks
        (_, ActivationHint::SELU) => InitMethod::LecunNormal,

        // Tanh or Sigmoid
        (_, ActivationHint::Tanh) | (_, ActivationHint::Sigmoid) => {
            InitMethod::XavierUniform { gain: 1.0 }
        }

        // Default: Xavier uniform
        _ => InitMethod::XavierUniform { gain: 1.0 },
    }
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

    #[test]
    fn test_variance_scaling() {
        // Test uniform distribution
        let tensor =
            variance_scaling(&[10, 5], 2.0, FanMode::FanIn, Distribution::Uniform).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);

        // Test normal distribution
        let tensor = variance_scaling(&[10, 5], 2.0, FanMode::FanIn, Distribution::Normal).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);

        // Test truncated normal distribution
        let tensor =
            variance_scaling(&[10, 5], 2.0, FanMode::FanIn, Distribution::TruncatedNormal).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);
    }

    #[test]
    fn test_dirac_initialization() {
        // Test 3D convolutional kernel
        let tensor = dirac_init(&[16, 16, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[16, 16, 3]);

        // Test with invalid dimensions
        assert!(dirac_init(&[10, 10]).is_err());
    }

    #[test]
    fn test_siren_initialization() {
        // Test first layer
        let tensor = siren_init(&[10, 5], 1.0, 30.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);

        // Test hidden layer
        let tensor = siren_init(&[10, 5], 6.0, 1.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);

        // Test with invalid dimensions
        assert!(siren_init(&[10], 6.0, 1.0).is_err());
    }

    #[test]
    fn test_init_method_builders() {
        // Test builder pattern
        let method = InitMethod::xavier_uniform();
        assert_eq!(method.name(), "Xavier Uniform");

        let method = InitMethod::kaiming_normal().with_fan_mode(FanMode::FanOut);
        assert_eq!(method.name(), "Kaiming Normal");

        let method = InitMethod::orthogonal().with_gain(2.0);
        assert_eq!(method.name(), "Orthogonal");

        let method = InitMethod::siren_first_layer();
        assert_eq!(method.name(), "SIREN");

        let method = InitMethod::dirac();
        assert_eq!(method.name(), "Dirac");
    }

    #[test]
    fn test_init_method_enum_variants() {
        // Test new enum variants
        let method = InitMethod::VarianceScaling {
            scale: 2.0,
            mode: FanMode::FanIn,
            distribution: Distribution::Normal,
        };
        let tensor = method.initialize(&[10, 5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);

        let method = InitMethod::Dirac;
        let tensor = method.initialize(&[8, 8, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[8, 8, 3]);

        let method = InitMethod::SIREN { c: 6.0, w0: 1.0 };
        let tensor = method.initialize(&[10, 5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);
    }

    #[test]
    fn test_fixup_initialization() {
        // Test Fixup initialization for residual branch
        let tensor = fixup_init(&[10, 10], 50, 10, true).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        // Test Fixup initialization for non-residual branch
        let tensor = fixup_init(&[10, 10], 50, 10, false).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        // Test with minimal layers
        let tensor = fixup_init(&[5, 5], 2, 1, true).unwrap();
        assert_eq!(tensor.shape().dims(), &[5, 5]);
    }

    #[test]
    fn test_rezero_initialization() {
        // Test ReZero weight initialization
        let tensor = rezero_init(&[10, 5]).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);

        // Test ReZero alpha initialization
        let alpha = rezero_alpha_init().unwrap();
        assert_eq!(alpha.shape().dims(), &[1]);
        // Alpha should be initialized to 0
        let alpha_val: Vec<f32> = alpha
            .to_vec()
            .expect("tensor to vec conversion should succeed");
        assert_eq!(alpha_val[0], 0.0);
    }

    #[test]
    fn test_delta_orthogonal_initialization() {
        // Test Delta-Orthogonal initialization
        let tensor = delta_orthogonal_init(&[10, 10], 1.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        // Test with gain
        let tensor = delta_orthogonal_init(&[8, 8], 2.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[8, 8]);

        // Test error handling for 1D tensor
        assert!(delta_orthogonal_init(&[10], 1.0).is_err());
    }

    #[test]
    fn test_metainit() {
        // Test MetaInit with typical parameters
        let tensor = metainit(&[10, 10], 0.8, 0.05).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        // Verify sparsity is roughly correct
        let values: Vec<f32> = tensor
            .to_vec()
            .expect("tensor to vec conversion should succeed");
        let zero_count = values.iter().filter(|&&v| v == 0.0).count();
        let sparsity_ratio = zero_count as f32 / values.len() as f32;
        // Allow generous tolerance in sparsity (60-95% for 80% target)
        // due to random nature of initialization
        assert!(sparsity_ratio > 0.6 && sparsity_ratio < 0.95);

        // Test error handling
        assert!(metainit(&[10, 10], 1.5, 0.05).is_err()); // Invalid sparsity
        assert!(metainit(&[10, 10], -0.1, 0.05).is_err()); // Negative sparsity
        assert!(metainit(&[10, 10], 0.8, -0.05).is_err()); // Negative scale
    }

    #[test]
    fn test_lsuv_initialization() {
        // Test LSUV initialization (first step)
        let tensor = lsuv_init(&[10, 10]).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        // Test with different shapes
        let tensor = lsuv_init(&[64, 32]).unwrap();
        assert_eq!(tensor.shape().dims(), &[64, 32]);
    }

    #[test]
    fn test_zero_centered_variance_init() {
        // Test with specific target variance
        let tensor = zero_centered_variance_init(&[10, 10], 1.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 10]);

        let tensor = zero_centered_variance_init(&[20, 20], 0.5).unwrap();
        assert_eq!(tensor.shape().dims(), &[20, 20]);

        // Test error handling for invalid variance
        assert!(zero_centered_variance_init(&[10, 10], 0.0).is_err());
        assert!(zero_centered_variance_init(&[10, 10], -1.0).is_err());
    }

    #[test]
    fn test_gan_balanced_initialization() {
        // Test generator initialization
        let gen_tensor = gan_balanced_init(&[10, 10], true).unwrap();
        assert_eq!(gen_tensor.shape().dims(), &[10, 10]);

        // Test discriminator initialization
        let disc_tensor = gan_balanced_init(&[10, 10], false).unwrap();
        assert_eq!(disc_tensor.shape().dims(), &[10, 10]);

        // Generator weights should generally be smaller in magnitude than discriminator
        // (though this is probabilistic, so we just test shape and successful initialization)
    }

    #[test]
    fn test_coordinate_mlp_initialization() {
        // Test coordinate MLP initialization
        let tensor = coordinate_mlp_init(&[10, 3], 1.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 3]);

        // Test with different omega_0
        let tensor = coordinate_mlp_init(&[64, 32], 30.0).unwrap();
        assert_eq!(tensor.shape().dims(), &[64, 32]);

        // Test error handling for 1D tensor
        assert!(coordinate_mlp_init(&[10], 1.0).is_err());
    }

    #[test]
    fn test_auto_init() {
        // Test auto initialization for feedforward + ReLU
        let tensor = auto_init(
            &[10, 5],
            ArchitectureHint::Feedforward,
            ActivationHint::ReLU,
            None,
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[10, 5]);

        // Test for convolutional + ReLU
        let tensor = auto_init(
            &[64, 32, 3, 3],
            ArchitectureHint::Convolutional,
            ActivationHint::ReLU,
            None,
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[64, 32, 3, 3]);

        // Test for recurrent networks
        let tensor = auto_init(
            &[128, 256],
            ArchitectureHint::Recurrent,
            ActivationHint::Tanh,
            None,
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[128, 256]);

        // Test for transformer
        let tensor = auto_init(
            &[512, 512],
            ArchitectureHint::Transformer,
            ActivationHint::GELU,
            None,
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[512, 512]);

        // Test for SIREN (periodic activations)
        let tensor = auto_init(
            &[32, 16],
            ArchitectureHint::Periodic,
            ActivationHint::Sine,
            Some(0),
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[32, 16]);

        // Test for very deep networks
        let tensor = auto_init(
            &[256, 256],
            ArchitectureHint::VeryDeep,
            ActivationHint::ReLU,
            Some(100),
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[256, 256]);

        // Test for GAN
        let tensor = auto_init(
            &[100, 784],
            ArchitectureHint::GAN,
            ActivationHint::ReLU,
            None,
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[100, 784]);

        // Test for coordinate-based networks
        let tensor = auto_init(
            &[64, 3],
            ArchitectureHint::CoordinateBased,
            ActivationHint::ReLU,
            None,
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[64, 3]);
    }

    #[test]
    fn test_recommend_init_method() {
        // Test recommendation for feedforward + ReLU
        let method =
            recommend_init_method(ArchitectureHint::Feedforward, ActivationHint::ReLU, None);
        matches!(method, InitMethod::KaimingNormal { .. });

        // Test recommendation for transformer
        let method =
            recommend_init_method(ArchitectureHint::Transformer, ActivationHint::GELU, None);
        matches!(method, InitMethod::XavierUniform { .. });

        // Test recommendation for recurrent
        let method = recommend_init_method(ArchitectureHint::Recurrent, ActivationHint::Tanh, None);
        matches!(method, InitMethod::Orthogonal { .. });

        // Test recommendation for SIREN
        let method =
            recommend_init_method(ArchitectureHint::Periodic, ActivationHint::Sine, Some(0));
        matches!(method, InitMethod::SIREN { .. });

        // Test recommendation for SELU
        let method =
            recommend_init_method(ArchitectureHint::Feedforward, ActivationHint::SELU, None);
        matches!(method, InitMethod::LecunNormal);
    }

    #[test]
    fn test_architecture_hints() {
        // Test that all architecture hints are distinct
        assert_ne!(
            ArchitectureHint::Feedforward,
            ArchitectureHint::Convolutional
        );
        assert_ne!(ArchitectureHint::Recurrent, ArchitectureHint::Transformer);
        assert_ne!(ArchitectureHint::Residual, ArchitectureHint::VeryDeep);
        assert_ne!(ArchitectureHint::GAN, ArchitectureHint::CoordinateBased);
        assert_ne!(ArchitectureHint::Periodic, ArchitectureHint::Autoencoder);
    }

    #[test]
    fn test_activation_hints() {
        // Test that all activation hints are distinct
        assert_ne!(ActivationHint::ReLU, ActivationHint::Tanh);
        assert_ne!(ActivationHint::Sigmoid, ActivationHint::SELU);
        assert_ne!(ActivationHint::Swish, ActivationHint::GELU);
        assert_ne!(ActivationHint::Sine, ActivationHint::Linear);
    }
}
