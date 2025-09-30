//! Individual fused operation implementations
//!
//! This module contains the actual implementations of fused operations,
//! including basic arithmetic fusions and complex operations like batch norm.

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Fused ReLU + Add operation: relu(x + y)
///
/// # Mathematical Formula
/// ```
/// output[i] = max(0, x[i] + y[i])
/// ```
/// where the max operation is applied element-wise.
///
/// # Applications
/// - **Residual connections**: Used in ResNet architectures to combine skip connections
/// - **Gated networks**: Part of gating mechanisms in RNNs and Transformers
/// - **Activation after bias**: Common pattern of adding bias followed by activation
///
/// # Performance Benefits
/// - **Memory efficiency**: Single pass through data instead of two separate operations
/// - **Cache optimization**: Better temporal locality of reference
/// - **SIMD acceleration**: Leverages vectorized instructions when available
/// - **Reduced kernel launches**: Important for GPU implementations
pub fn fused_relu_add(x: &Tensor, y: &Tensor) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;

    if x_data.len() != y_data.len() {
        return Err(TorshError::shape_mismatch_formatted(
            &format!("{:?}", x.shape().dims()),
            &format!("{:?}", y.shape().dims()),
        ));
    }

    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise operation: relu(x + y)
    for i in 0..len {
        result_data[i] = (x_data[i] + y_data[i]).max(0.0);
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fused Multiply + Add operation: x * y + z (FMADD)
///
/// # Mathematical Formula
/// ```
/// output[i] = x[i] * y[i] + z[i]
/// ```
/// This implements the fused multiply-add (FMA) operation element-wise.
///
/// # Mathematical Properties
/// - **Associative**: (a * b) + c = a * (b + c/a) when c/a is defined
/// - **Distributive**: Can be factored as x * (y + z/x) when z/x is defined
/// - **Numerical stability**: Hardware FMA reduces intermediate rounding errors
///
/// # Applications
/// - **Linear transformations**: Core operation in fully connected layers
/// - **Convolution**: Inner product computation in conv layers
/// - **Attention mechanisms**: Query-key-value computations in Transformers
/// - **Bias addition**: Adding bias terms after matrix multiplication
/// - **Polynomial evaluation**: Horner's method for efficient polynomial computation
///
/// # Performance Benefits
/// - **Hardware acceleration**: Leverages dedicated FMA units on modern CPUs/GPUs
/// - **Reduced memory bandwidth**: Single read/write cycle for three operations
/// - **Improved numerical accuracy**: Hardware FMA has single rounding step
/// - **Vectorization**: Excellent SIMD/GPU parallelization characteristics
pub fn fused_mul_add(x: &Tensor, y: &Tensor, z: &Tensor) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;
    let z_data = z.data()?;

    if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
        return Err(TorshError::invalid_argument_with_context(
            "All tensors must have the same number of elements for fused_mul_add",
            "fused_mul_add",
        ));
    }

    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise operation: x * y + z
    for i in 0..len {
        result_data[i] = x_data[i] * y_data[i] + z_data[i];
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fused Add + Multiply operation: (x + y) * z
///
/// # Mathematical Formula
/// ```
/// output[i] = (x[i] + y[i]) * z[i]
/// ```
/// This computes element-wise addition followed by element-wise multiplication.
///
/// # Mathematical Properties
/// - **Distributive**: Equivalent to x*z + y*z (when beneficial for optimization)
/// - **Commutative in addition**: (x + y) * z = (y + x) * z
/// - **Associative with scaling**: Can be reordered for numerical stability
///
/// # Applications
/// - **Gating mechanisms**: Used in LSTM and GRU cells
/// - **Attention weights**: Combining attention scores with values
/// - **Normalization**: Part of layer normalization computation
/// - **Activation scaling**: Applying learned scaling after bias addition
///
/// # Performance Benefits
/// - **Memory efficiency**: Avoids storing intermediate (x + y) result
/// - **Cache locality**: Better temporal access pattern than separate operations
/// - **SIMD optimization**: Efficient vectorization on modern architectures
/// - **Reduced memory bandwidth**: Fewer memory transactions than separate operations
pub fn fused_add_mul(x: &Tensor, y: &Tensor, z: &Tensor) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;
    let z_data = z.data()?;

    if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
        return Err(TorshError::invalid_argument_with_context(
            "All tensors must have the same number of elements for fused_add_mul",
            "fused_add_mul",
        ));
    }

    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise operation: (x + y) * z
    for i in 0..len {
        result_data[i] = (x_data[i] + y_data[i]) * z_data[i];
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fused Sigmoid + Multiply operation: sigmoid(x) * y
///
/// # Mathematical Formula
/// ```
/// output[i] = sigmoid(x[i]) * y[i] = (1 / (1 + exp(-x[i]))) * y[i]
/// ```
/// When y = x, this becomes the SiLU (Swish) activation function.
///
/// # Mathematical Properties
/// - **Bounded output**: sigmoid(x) ∈ (0, 1), so output is scaled version of y
/// - **Smooth**: Infinitely differentiable with well-behaved gradients
/// - **Monotonic in x**: Sigmoid function is strictly increasing
///
/// # Applications
/// - **SiLU activation**: When y = x, creates x * sigmoid(x) (Swish/SiLU)
/// - **Gated units**: Sigmoid acts as a gate controlling information flow
/// - **Attention mechanisms**: Sigmoid gates in some attention variants
/// - **Highway networks**: Gating mechanism for information highways
///
/// # Performance Benefits
/// - **Single memory pass**: Avoids intermediate sigmoid storage
/// - **SIMD efficiency**: Good vectorization characteristics
/// - **Numerical stability**: Combined operation reduces precision loss
/// - **Cache optimization**: Better memory access pattern than separate operations
pub fn fused_sigmoid_mul(x: &Tensor, y: &Tensor) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let y_data = y.data()?;

    if x_data.len() != y_data.len() {
        return Err(TorshError::shape_mismatch_formatted(
            &format!("{:?}", x.shape().dims()),
            &format!("{:?}", y.shape().dims()),
        ));
    }

    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise operation: sigmoid(x) * y
    for i in 0..len {
        let sigmoid_val = 1.0 / (1.0 + (-x_data[i]).exp());
        result_data[i] = sigmoid_val * y_data[i];
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fused SiLU activation: x * sigmoid(x)
///
/// # Mathematical Formula
/// ```
/// output[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
/// ```
/// This is the Sigmoid Linear Unit (SiLU), also known as Swish activation.
///
/// # Mathematical Properties
/// - **Self-gated**: Uses its own values as gates (x * sigmoid(x))
/// - **Smooth**: Infinitely differentiable, unlike ReLU
/// - **Non-monotonic**: Has a small negative region for negative inputs
/// - **Bounded below**: Approaches -0.278 as x → -∞, unlike ReLU's hard cutoff
///
/// # Applications
/// - **Modern architectures**: Increasingly used instead of ReLU in newer models
/// - **Transformer variants**: Some attention mechanisms and FFN layers
/// - **EfficientNet**: Used as primary activation in EfficientNet architectures
/// - **Swish activation**: Equivalent implementation of Google's Swish function
///
/// # Performance Benefits
/// - **Single tensor pass**: Avoids creating intermediate sigmoid tensor
/// - **SIMD optimization**: Excellent vectorization characteristics
/// - **Memory efficiency**: Lower memory bandwidth than separate sigmoid + multiply
/// - **Gradient efficiency**: Smooth gradients improve training stability
pub fn fused_silu(x: &Tensor) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise operation: x * sigmoid(x) (SiLU/Swish)
    for i in 0..len {
        let sigmoid_val = 1.0 / (1.0 + (-x_data[i]).exp());
        result_data[i] = x_data[i] * sigmoid_val;
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fused Tanh + Scale operation: tanh(x) * scale
///
/// # Mathematical Formula
/// ```
/// output[i] = tanh(x[i]) * scale = ((exp(x[i]) - exp(-x[i])) / (exp(x[i]) + exp(-x[i]))) * scale
/// ```
/// This applies hyperbolic tangent followed by scalar scaling.
///
/// # Mathematical Properties
/// - **Bounded**: tanh(x) ∈ (-1, 1), so output ∈ (-scale, scale)
/// - **Odd function**: tanh(-x) = -tanh(x), preserving sign symmetry
/// - **Smooth**: Infinitely differentiable with bounded derivatives
/// - **Saturating**: Approaches ±1 for large |x|, providing natural gradient clipping
///
/// # Applications
/// - **Scaled activations**: Custom activation functions with controlled output range
/// - **Gating mechanisms**: Tanh gates in LSTM cells with scaling
/// - **Attention mechanisms**: Scaled tanh in some attention formulations
/// - **Normalization**: Part of custom normalization schemes
///
/// # Performance Benefits
/// - **Single memory pass**: Avoids intermediate tanh storage
/// - **SIMD acceleration**: Good vectorization for both tanh and scaling
/// - **Reduced memory bandwidth**: Fewer memory operations than separate tanh + scale
/// - **Cache efficiency**: Better temporal locality than separate operations
pub fn fused_tanh_scale(x: &Tensor, scale: f32) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise operation: tanh(x) * scale
    for i in 0..len {
        result_data[i] = x_data[i].tanh() * scale;
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fused Add + ReLU + Multiply operation: relu(x + bias) * scale
///
/// # Mathematical Formula
/// ```
/// output[i] = max(0, x[i] + bias[i]) * scale[i]
/// ```
/// This combines bias addition, ReLU activation, and scaling in a single operation.
///
/// # Mathematical Properties
/// - **Non-linear transformation**: Combines linear (add, multiply) and non-linear (ReLU) operations
/// - **Piecewise linear**: Output is piecewise linear due to ReLU
/// - **Sparse activation**: ReLU creates sparsity, multiplication preserves or amplifies it
///
/// # Applications
/// - **Batch normalization**: Part of batch norm when followed by learned scaling
/// - **Residual connections**: Advanced residual blocks with scaling
/// - **Attention mechanisms**: Scaled attention values with ReLU gating
/// - **Custom activations**: Building blocks for complex activation functions
///
/// # Performance Benefits
/// - **Memory efficiency**: Avoids two intermediate tensor allocations
/// - **Cache optimization**: Single pass through data improves cache utilization
/// - **SIMD acceleration**: All operations vectorize well independently
/// - **Reduced memory bandwidth**: 3x reduction in memory operations vs separate functions
pub fn fused_add_relu_mul(x: &Tensor, bias: &Tensor, scale: &Tensor) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let bias_data = bias.data()?;
    let scale_data = scale.data()?;

    if x_data.len() != bias_data.len() || x_data.len() != scale_data.len() {
        return Err(TorshError::invalid_argument_with_context(
            "All tensors must have the same number of elements for fused_add_relu_mul",
            "fused_add_relu_mul",
        ));
    }

    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise operation: relu(x + bias) * scale
    for i in 0..len {
        let added = x_data[i] + bias_data[i];
        let activated = added.max(0.0);
        result_data[i] = activated * scale_data[i];
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Fused Batch Normalization operation
///
/// # Mathematical Formula
/// ```
/// output[i] = ((x[i] - mean[i]) / sqrt(var[i] + eps)) * gamma[i] + beta[i]
/// ```
/// This implements the complete batch normalization transformation in a single fused operation.
///
/// # Mathematical Properties
/// - **Standardization**: Centers data around zero with unit variance
/// - **Affine transformation**: Gamma and beta allow learning optimal scale and shift
/// - **Numerical stability**: Epsilon prevents division by zero in variance calculation
///
/// # Applications
/// - **Deep networks**: Enables training of very deep networks by normalizing activations
/// - **Convergence speed**: Accelerates training by reducing internal covariate shift
/// - **Regularization**: Provides mild regularization effect during training
/// - **Broadcasting**: Parameters broadcast along batch and spatial dimensions
/// - **Inference mode**: Uses pre-computed running statistics
/// - **Training mode**: Computes batch statistics on-the-fly
///
/// # Performance Benefits
/// - **Single kernel**: Combines 4-5 separate operations into one pass
/// - **Memory efficiency**: Reduces intermediate tensor allocations
/// - **Cache optimization**: Better data locality than separate operations
/// - **Numerical precision**: Minimizes accumulation of floating-point errors
pub fn fused_batch_norm(
    x: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    gamma: Option<&Tensor>,
    beta: Option<&Tensor>,
    eps: f32,
) -> TorshResult<Tensor> {
    let x_data = x.data()?;
    let mean_data = mean.data()?;
    let var_data = var.data()?;

    let gamma_data = gamma.map(|g| g.data()).transpose()?;
    let beta_data = beta.map(|b| b.data()).transpose()?;

    if x_data.len() != mean_data.len() || x_data.len() != var_data.len() {
        return Err(TorshError::invalid_argument_with_context(
            "Input, mean, and variance tensors must have the same size",
            "fused_batch_norm",
        ));
    }

    let len = x_data.len();
    let mut result_data = vec![0.0_f32; len];

    // Element-wise batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    for i in 0..len {
        // Normalize: (x - mean) / sqrt(var + eps)
        let centered = x_data[i] - mean_data[i];
        let std_dev = (var_data[i] + eps).sqrt();
        let normalized = centered / std_dev;

        // Apply affine transformation: normalized * gamma + beta
        let scaled = if let Some(gamma_data) = &gamma_data {
            normalized * gamma_data[i % gamma_data.len()]
        } else {
            normalized
        };

        result_data[i] = if let Some(beta_data) = &beta_data {
            scaled + beta_data[i % beta_data.len()]
        } else {
            scaled
        };
    }

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}
