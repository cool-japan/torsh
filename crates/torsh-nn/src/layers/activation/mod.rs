//! # Neural Network Activation Functions
//!
//! This module provides a comprehensive collection of activation functions for neural networks,
//! organized into specialized submodules for better maintainability and discoverability.
//! All activation functions implement the `Module` trait and can be used interchangeably
//! in neural network architectures.
//!
//! ## Module Organization
//!
//! The activation functions are organized into the following categories:
//!
//! ### Basic Activations ([`basic`])
//! Traditional, fundamental activation functions:
//! - **ReLU** - Rectified Linear Unit
//! - **Sigmoid** - Sigmoid function
//! - **Tanh** - Hyperbolic tangent
//! - **LeakyReLU** - Leaky Rectified Linear Unit
//! - **ReLU6** - ReLU clamped to 6
//! - **PReLU** - Parametric ReLU with learnable parameters
//!
//! ### Modern Activations ([`modern`])
//! Advanced activation functions used in state-of-the-art architectures:
//! - **GELU** - Gaussian Error Linear Unit
//! - **SiLU/Swish** - Sigmoid Linear Unit
//! - **Mish** - Self-regularized activation
//! - **Hardswish** - Efficient approximation to Swish
//! - **ELU** - Exponential Linear Unit
//! - **SELU** - Scaled Exponential Linear Unit
//!
//! ### Softmax Family ([`softmax`])
//! Probability distribution and classification functions:
//! - **Softmax** - Standard softmax for multi-class classification
//! - **LogSoftmax** - Log-softmax for numerical stability
//! - **LogSigmoid** - Log-sigmoid for binary classification
//!
//! ### Threshold Functions ([`threshold`])
//! Threshold-based and shrinking activation functions:
//! - **Hardshrink** - Hard shrinkage function
//! - **Softshrink** - Soft shrinkage function
//! - **Hardtanh** - Hard hyperbolic tangent
//! - **Threshold** - Basic threshold function
//! - **Tanhshrink** - Tanh shrinkage function
//!
//! ### Smooth Functions ([`smooth`])
//! Continuously differentiable activation functions:
//! - **Softplus** - Smooth approximation to ReLU
//! - **Softsign** - Smooth alternative to tanh
//! - **Hardsigmoid** - Piecewise linear sigmoid approximation
//!
//! ### Gated Functions ([`gated`])
//! Gated activation functions for advanced architectures:
//! - **GLU** - Gated Linear Unit
//! - **GEGLU** - Gaussian Error Gated Linear Unit
//! - **ReGLU** - ReLU Gated Linear Unit
//! - **SwiGLU** - Swish Gated Linear Unit
//!
//! ## Usage Examples
//!
//! ### Basic Usage
//! ```rust
//! use torsh_nn::layers::activation::{ReLU, Sigmoid, GELU};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Create activation functions
//! let relu = ReLU::new();
//! let sigmoid = Sigmoid::new();
//! let gelu = GELU::new();
//!
//! // Apply to tensors
//! let input = Tensor::randn(&[2, 3]);
//! let relu_output = relu.forward(&input)?;
//! let sigmoid_output = sigmoid.forward(&input)?;
//! let gelu_output = gelu.forward(&input)?;
//! ```
//!
//! ### Classification Example
//! ```rust
//! use torsh_nn::layers::activation::{Softmax, LogSoftmax};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Multi-class classification
//! let softmax = Softmax::new(Some(1)); // Apply along class dimension
//! let log_softmax = LogSoftmax::new(Some(1));
//!
//! let logits = Tensor::randn(&[32, 10]); // 32 samples, 10 classes
//! let probabilities = softmax.forward(&logits)?;
//! let log_probs = log_softmax.forward(&logits)?; // For NLL loss
//! ```
//!
//! ### Advanced Architectures
//! ```rust
//! use torsh_nn::layers::activation::{SwiGLU, GEGLU, Mish};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Modern transformer-style activations
//! let swiglu = SwiGLU::new(-1); // Gated activation
//! let geglu = GEGLU::new(-1);   // GELU-gated activation
//! let mish = Mish::new();       // Self-regularized activation
//!
//! let input = Tensor::randn(&[2, 8]); // Even last dimension for gated functions
//! let swiglu_output = swiglu.forward(&input)?; // Shape: [2, 4]
//! let geglu_output = geglu.forward(&input)?;   // Shape: [2, 4]
//! let mish_output = mish.forward(&input)?;     // Shape: [2, 8]
//! ```
//!
//! ### Parameterized Activations
//! ```rust
//! use torsh_nn::layers::activation::{LeakyReLU, PReLU, ELU, Softplus};
//! use torsh_nn::Module;
//! use torsh_tensor::Tensor;
//!
//! // Configurable activation functions
//! let leaky_relu = LeakyReLU::new(0.01);        // Negative slope
//! let prelu = PReLU::new(64)?;                  // Learnable parameters
//! let elu = ELU::new(1.0);                      // Alpha parameter
//! let softplus = Softplus::new(1.0, 20.0);     // Beta and threshold
//!
//! let input = Tensor::randn(&[2, 64]);
//! let outputs = vec![
//!     leaky_relu.forward(&input)?,
//!     prelu.forward(&input)?,
//!     elu.forward(&input)?,
//!     softplus.forward(&input)?,
//! ];
//! ```
//!
//! ## Performance Considerations
//!
//! - **Basic functions** (ReLU, Sigmoid, Tanh) are highly optimized and fastest
//! - **Modern functions** (GELU, SiLU, Mish) provide better performance but are more computationally expensive
//! - **Gated functions** (GLU variants) require even-sized inputs and are most expensive
//! - **Threshold functions** are efficient for creating sparse representations
//! - **Softmax functions** are optimized for numerical stability in classification tasks
//!
//! ## Integration with Neural Network Modules
//!
//! All activation functions implement the `Module` trait and can be used in:
//! - Sequential models
//! - Custom layer implementations
//! - Functional-style neural network construction
//! - As components in larger modules

// Declare all submodules
pub mod basic;
pub mod gated;
pub mod modern;
pub mod smooth;
pub mod softmax;
pub mod threshold;

// Re-export all basic activation functions
pub use basic::{LeakyReLU, PReLU, ReLU, ReLU6, Sigmoid, Tanh};

// Re-export all modern activation functions
pub use modern::{
    Hardswish,
    Mish,
    SiLU,
    Swish, // Type alias for SiLU
    ELU,
    GELU,
    SELU,
};

// Re-export all softmax family functions
pub use softmax::{LogSigmoid, LogSoftmax, Softmax};

// Re-export all threshold and shrinking functions
pub use threshold::{Hardshrink, Hardtanh, Softshrink, Tanhshrink, Threshold};

// Re-export all smooth activation functions
pub use smooth::{Hardsigmoid, Softplus, Softsign};

// Re-export all gated activation functions
pub use gated::{ReGLU, SwiGLU, GEGLU, GLU};

/// Prelude module for convenient imports of common activation functions
pub mod prelude {
    //! Commonly used activation functions for quick imports
    //!
    //! This module re-exports the most frequently used activation functions
    //! for convenient importing with `use torsh_nn::layers::activation::prelude::*;`

    pub use super::basic::{LeakyReLU, ReLU, Sigmoid, Tanh};
    pub use super::gated::{SwiGLU, GLU};
    pub use super::modern::{Mish, SiLU, GELU};
    pub use super::smooth::Softplus;
    pub use super::softmax::{LogSoftmax, Softmax};
    pub use super::threshold::Hardtanh;
}

/// Collections of activation functions grouped by use case
pub mod collections {
    //! Curated collections of activation functions for specific use cases

    /// Classical activation functions suitable for basic neural networks
    pub mod classical {
        pub use crate::layers::activation::basic::{ReLU, Sigmoid, Tanh};
        pub use crate::layers::activation::smooth::{Softplus, Softsign};
        pub use crate::layers::activation::threshold::Hardtanh;
    }

    /// Modern activation functions for state-of-the-art architectures
    pub mod modern {
        pub use crate::layers::activation::gated::{SwiGLU, GEGLU, GLU};
        pub use crate::layers::activation::modern::{Mish, SiLU, ELU, GELU, SELU};
    }

    /// Activation functions optimized for mobile and edge deployment
    pub mod mobile {
        pub use crate::layers::activation::basic::{ReLU, ReLU6};
        pub use crate::layers::activation::modern::Hardswish;
        pub use crate::layers::activation::smooth::Hardsigmoid;
        pub use crate::layers::activation::threshold::Hardtanh;
    }

    /// Activation functions for transformer architectures
    pub mod transformer {
        pub use crate::layers::activation::gated::{ReGLU, SwiGLU, GEGLU, GLU};
        pub use crate::layers::activation::modern::{SiLU, GELU};
    }

    /// Activation functions for classification tasks
    pub mod classification {
        pub use crate::layers::activation::basic::Sigmoid;
        pub use crate::layers::activation::softmax::{LogSigmoid, LogSoftmax, Softmax};
    }

    /// Sparse activation functions for creating sparse representations
    pub mod sparse {
        pub use crate::layers::activation::basic::ReLU;
        pub use crate::layers::activation::gated::ReGLU;
        pub use crate::layers::activation::threshold::{Hardshrink, Softshrink, Threshold};
    }
}

/// Factory functions for creating activation functions with common configurations
pub mod factory {
    //! Factory functions for creating pre-configured activation functions

    use super::*;
    use torsh_core::error::Result;

    /// Create a ReLU activation (most common activation)
    pub fn relu() -> ReLU {
        ReLU::new()
    }

    /// Create a GELU activation (common in transformers)
    pub fn gelu() -> GELU {
        GELU::new()
    }

    /// Create a SiLU/Swish activation (modern alternative to ReLU)
    pub fn silu() -> SiLU {
        SiLU::new()
    }

    /// Create a softmax for classification (along last dimension)
    pub fn softmax() -> Softmax {
        Softmax::new(Some(1))
    }

    /// Create a log-softmax for classification with NLL loss
    pub fn log_softmax() -> LogSoftmax {
        LogSoftmax::new(Some(1))
    }

    /// Create a SwiGLU for transformer feed-forward layers
    pub fn swiglu() -> SwiGLU {
        SwiGLU::new(-1)
    }

    /// Create a GEGLU for transformer feed-forward layers
    pub fn geglu() -> GEGLU {
        GEGLU::new(-1)
    }

    /// Create a PReLU with the specified number of parameters
    pub fn prelu(num_parameters: usize) -> Result<PReLU> {
        PReLU::new(num_parameters)
    }

    /// Create a Leaky ReLU with default slope (0.01)
    pub fn leaky_relu() -> LeakyReLU {
        LeakyReLU::default()
    }

    /// Create a Leaky ReLU with custom slope
    pub fn leaky_relu_with_slope(slope: f64) -> LeakyReLU {
        LeakyReLU::new(slope)
    }

    /// Create a standard SELU (self-normalizing)
    pub fn selu() -> SELU {
        SELU::new()
    }

    /// Create a Mish activation (self-regularized)
    pub fn mish() -> Mish {
        Mish::new()
    }

    /// Create a Hardswish for mobile deployment
    pub fn hardswish() -> Hardswish {
        Hardswish::new()
    }

    /// Create a ReLU6 for mobile deployment
    pub fn relu6() -> ReLU6 {
        ReLU6::new()
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::Module;
    use torsh_tensor::creation::*;
    use torsh_tensor::Tensor;

    #[test]
    fn test_all_activations_implement_module() -> torsh_core::error::Result<()> {
        let input = randn(&[2, 4])?;

        // Test basic activations
        let relu = ReLU::new();
        let sigmoid = Sigmoid::new();
        let tanh = Tanh::new();
        let leaky_relu = LeakyReLU::new(0.01);
        let relu6 = ReLU6::new();

        let _relu_out = relu.forward(&input)?;
        let _sigmoid_out = sigmoid.forward(&input)?;
        let _tanh_out = tanh.forward(&input)?;
        let _leaky_relu_out = leaky_relu.forward(&input)?;
        let _relu6_out = relu6.forward(&input)?;

        // Test modern activations
        let gelu = GELU::new();
        let silu = SiLU::new();
        let mish = Mish::new();
        let hardswish = Hardswish::new();
        let elu = ELU::new(1.0);
        let selu = SELU::new();

        let _gelu_out = gelu.forward(&input)?;
        let _silu_out = silu.forward(&input)?;
        let _mish_out = mish.forward(&input)?;
        let _hardswish_out = hardswish.forward(&input)?;
        let _elu_out = elu.forward(&input)?;
        let _selu_out = selu.forward(&input)?;

        // Test softmax family
        let softmax = Softmax::new(Some(1));
        let log_softmax = LogSoftmax::new(Some(1));
        let log_sigmoid = LogSigmoid::new();

        let _softmax_out = softmax.forward(&input)?;
        let _log_softmax_out = log_softmax.forward(&input)?;
        let _log_sigmoid_out = log_sigmoid.forward(&input)?;

        // Test threshold functions
        let hardshrink = Hardshrink::new(0.5);
        let softshrink = Softshrink::new(0.5);
        let hardtanh = Hardtanh::new(-1.0, 1.0);
        let threshold = Threshold::new(0.0, 0.0);
        let tanhshrink = Tanhshrink::new();

        let _hardshrink_out = hardshrink.forward(&input)?;
        let _softshrink_out = softshrink.forward(&input)?;
        let _hardtanh_out = hardtanh.forward(&input)?;
        let _threshold_out = threshold.forward(&input)?;
        let _tanhshrink_out = tanhshrink.forward(&input)?;

        // Test smooth functions
        let softplus = Softplus::new(1.0, 20.0);
        let softsign = Softsign::new();
        let hardsigmoid = Hardsigmoid::new();

        let _softplus_out = softplus.forward(&input)?;
        let _softsign_out = softsign.forward(&input)?;
        let _hardsigmoid_out = hardsigmoid.forward(&input)?;

        // Test gated functions (need even last dimension)
        let gated_input = randn(&[2, 8])?; // Even last dimension
        let glu = GLU::new(-1);
        let geglu = GEGLU::new(-1);
        let reglu = ReGLU::new(-1);
        let swiglu = SwiGLU::new(-1);

        let _glu_out = glu.forward(&gated_input)?;
        let _geglu_out = geglu.forward(&gated_input)?;
        let _reglu_out = reglu.forward(&gated_input)?;
        let _swiglu_out = swiglu.forward(&gated_input)?;

        Ok(())
    }

    #[test]
    fn test_factory_functions() -> torsh_core::error::Result<()> {
        let input = randn(&[2, 4])?;

        // Test factory functions
        let relu = factory::relu();
        let gelu = factory::gelu();
        let silu = factory::silu();
        let softmax = factory::softmax();
        let log_softmax = factory::log_softmax();
        let leaky_relu = factory::leaky_relu();
        let selu = factory::selu();
        let mish = factory::mish();
        let hardswish = factory::hardswish();
        let relu6 = factory::relu6();

        // Test forward passes
        let _relu_out = relu.forward(&input)?;
        let _gelu_out = gelu.forward(&input)?;
        let _silu_out = silu.forward(&input)?;
        let _softmax_out = softmax.forward(&input)?;
        let _log_softmax_out = log_softmax.forward(&input)?;
        let _leaky_relu_out = leaky_relu.forward(&input)?;
        let _selu_out = selu.forward(&input)?;
        let _mish_out = mish.forward(&input)?;
        let _hardswish_out = hardswish.forward(&input)?;
        let _relu6_out = relu6.forward(&input)?;

        // Test gated factory functions
        let gated_input = randn(&[2, 8])?;
        let swiglu = factory::swiglu();
        let geglu = factory::geglu();

        let _swiglu_out = swiglu.forward(&gated_input)?;
        let _geglu_out = geglu.forward(&gated_input)?;

        Ok(())
    }

    #[test]
    fn test_prelude_imports() -> torsh_core::error::Result<()> {
        use super::prelude::*;

        let input = randn(&[2, 4])?;

        // Test that all prelude types are accessible
        let relu = ReLU::new();
        let sigmoid = Sigmoid::new();
        let tanh = Tanh::new();
        let gelu = GELU::new();
        let silu = SiLU::new();
        let mish = Mish::new();
        let softmax = Softmax::new(Some(1));
        let log_softmax = LogSoftmax::new(Some(1));

        // Test forward passes
        let _outputs = vec![
            relu.forward(&input)?,
            sigmoid.forward(&input)?,
            tanh.forward(&input)?,
            gelu.forward(&input)?,
            silu.forward(&input)?,
            mish.forward(&input)?,
            softmax.forward(&input)?,
            log_softmax.forward(&input)?,
        ];

        Ok(())
    }

    #[test]
    fn test_collections() -> torsh_core::error::Result<()> {
        let input = randn(&[2, 4])?;

        // Test classical collection
        let classical_relu = collections::classical::ReLU::new();
        let classical_sigmoid = collections::classical::Sigmoid::new();
        let _relu_out = classical_relu.forward(&input)?;
        let _sigmoid_out = classical_sigmoid.forward(&input)?;

        // Test modern collection
        let modern_gelu = collections::modern::GELU::new();
        let modern_silu = collections::modern::SiLU::new();
        let _gelu_out = modern_gelu.forward(&input)?;
        let _silu_out = modern_silu.forward(&input)?;

        // Test mobile collection
        let mobile_relu = collections::mobile::ReLU::new();
        let mobile_relu6 = collections::mobile::ReLU6::new();
        let _mobile_relu_out = mobile_relu.forward(&input)?;
        let _mobile_relu6_out = mobile_relu6.forward(&input)?;

        // Test classification collection
        let class_softmax = collections::classification::Softmax::new(Some(1));
        let class_log_softmax = collections::classification::LogSoftmax::new(Some(1));
        let _softmax_out = class_softmax.forward(&input)?;
        let _log_softmax_out = class_log_softmax.forward(&input)?;

        Ok(())
    }

    #[test]
    fn test_backward_compatibility() -> torsh_core::error::Result<()> {
        // Test that all original activation function names are still accessible
        // This ensures complete backward compatibility with the original monolithic module

        let input = randn(&[2, 4])?;
        let gated_input = randn(&[2, 8])?;

        // All these should compile and work exactly as in the original module
        let activations: Vec<Box<dyn Fn(&Tensor) -> torsh_core::error::Result<Tensor>>> = vec![
            Box::new(|x| ReLU::new().forward(x)),
            Box::new(|x| Sigmoid::new().forward(x)),
            Box::new(|x| Tanh::new().forward(x)),
            Box::new(|x| GELU::new().forward(x)),
            Box::new(|x| SiLU::new().forward(x)),
            Box::new(|x| Mish::new().forward(x)),
            Box::new(|x| LeakyReLU::new(0.01).forward(x)),
            Box::new(|x| ReLU6::new().forward(x)),
            Box::new(|x| ELU::new(1.0).forward(x)),
            Box::new(|x| SELU::new().forward(x)),
            Box::new(|x| Hardswish::new().forward(x)),
            Box::new(|x| Softmax::new(Some(1)).forward(x)),
            Box::new(|x| LogSoftmax::new(Some(1)).forward(x)),
            Box::new(|x| LogSigmoid::new().forward(x)),
            Box::new(|x| Hardshrink::new(0.5).forward(x)),
            Box::new(|x| Softshrink::new(0.5).forward(x)),
            Box::new(|x| Hardtanh::new(-1.0, 1.0).forward(x)),
            Box::new(|x| Threshold::new(0.0, 0.0).forward(x)),
            Box::new(|x| Tanhshrink::new().forward(x)),
            Box::new(|x| Softplus::new(1.0, 20.0).forward(x)),
            Box::new(|x| Softsign::new().forward(x)),
            Box::new(|x| Hardsigmoid::new().forward(x)),
        ];

        for activation in activations {
            let _output = activation(&input)?;
        }

        // Test gated activations
        let gated_activations: Vec<Box<dyn Fn(&Tensor) -> torsh_core::error::Result<Tensor>>> = vec![
            Box::new(|x| GLU::new(-1).forward(x)),
            Box::new(|x| GEGLU::new(-1).forward(x)),
            Box::new(|x| ReGLU::new(-1).forward(x)),
            Box::new(|x| SwiGLU::new(-1).forward(x)),
        ];

        for activation in gated_activations {
            let _output = activation(&gated_input)?;
        }

        Ok(())
    }
}
