//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Module, ModuleBase};
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// 2D average pooling layer
pub struct AvgPool2d {
    pub base: ModuleBase,
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub padding: (usize, usize),
    pub ceil_mode: bool,
    #[allow(dead_code)]
    count_include_pad: bool,
}
impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        }
    }
    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new((kernel_size, kernel_size), None, (0, 0), false, true)
    }
}
/// Fractional max pooling 1D layer
pub struct FractionalMaxPool1d {
    pub base: ModuleBase,
    pub kernel_size: usize,
    pub output_size: Option<usize>,
    pub output_ratio: Option<f32>,
    pub return_indices: bool,
}
impl FractionalMaxPool1d {
    pub fn new(
        kernel_size: usize,
        output_size: Option<usize>,
        output_ratio: Option<f32>,
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            output_size,
            output_ratio,
            return_indices,
        }
    }
    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new(kernel_size, None, Some(0.5), false)
    }
    pub fn with_output_ratio(kernel_size: usize, output_ratio: f32) -> Self {
        Self::new(kernel_size, None, Some(output_ratio), false)
    }
    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let output_length = if let Some(ol) = self.output_size {
            ol
        } else if let Some(r) = self.output_ratio {
            (input_shape[2] as f32 * r) as usize
        } else {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Either output_size or output_ratio must be specified".to_string(),
            ));
        };
        let output_shape = [input_shape[0], input_shape[1], output_length];
        let output = zeros(&output_shape)?;
        let indices = if self.return_indices {
            Some(zeros(&output_shape)?)
        } else {
            None
        };
        Ok((output, indices))
    }
}
/// Fractional max pooling 3D layer
pub struct FractionalMaxPool3d {
    pub base: ModuleBase,
    pub kernel_size: (usize, usize, usize),
    pub output_size: Option<(usize, usize, usize)>,
    pub output_ratio: Option<(f32, f32, f32)>,
    pub return_indices: bool,
}
impl FractionalMaxPool3d {
    pub fn new(
        kernel_size: (usize, usize, usize),
        output_size: Option<(usize, usize, usize)>,
        output_ratio: Option<(f32, f32, f32)>,
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            output_size,
            output_ratio,
            return_indices,
        }
    }
    pub fn with_kernel_size(kernel_size: (usize, usize, usize)) -> Self {
        Self::new(kernel_size, None, Some((0.5, 0.5, 0.5)), false)
    }
    pub fn with_output_ratio(
        kernel_size: (usize, usize, usize),
        output_ratio: (f32, f32, f32),
    ) -> Self {
        Self::new(kernel_size, None, Some(output_ratio), false)
    }
    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let (output_depth, output_height, output_width) =
            if let Some((od, oh, ow)) = self.output_size {
                (od, oh, ow)
            } else if let Some((rd, rh, rw)) = self.output_ratio {
                (
                    (input_shape[2] as f32 * rd) as usize,
                    (input_shape[3] as f32 * rh) as usize,
                    (input_shape[4] as f32 * rw) as usize,
                )
            } else {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "Either output_size or output_ratio must be specified".to_string(),
                ));
            };
        let output_shape = [
            input_shape[0],
            input_shape[1],
            output_depth,
            output_height,
            output_width,
        ];
        let output = zeros(&output_shape)?;
        let indices = if self.return_indices {
            Some(zeros(&output_shape)?)
        } else {
            None
        };
        Ok((output, indices))
    }
}
/// Adaptive 1D average pooling layer
pub struct AdaptiveAvgPool1d {
    pub base: ModuleBase,
    pub output_size: usize,
}
impl AdaptiveAvgPool1d {
    pub fn new(output_size: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
        }
    }
}
/// Adaptive 3D average pooling layer
pub struct AdaptiveAvgPool3d {
    pub base: ModuleBase,
    pub output_size: (Option<usize>, Option<usize>, Option<usize>),
}
impl AdaptiveAvgPool3d {
    pub fn new(output_size: (Option<usize>, Option<usize>, Option<usize>)) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
        }
    }
    pub fn with_output_size(output_size: usize) -> Self {
        Self::new((Some(output_size), Some(output_size), Some(output_size)))
    }
}
/// Adaptive 2D average pooling layer
pub struct AdaptiveAvgPool2d {
    pub base: ModuleBase,
    pub output_size: (Option<usize>, Option<usize>),
}
impl AdaptiveAvgPool2d {
    pub fn new(output_size: (Option<usize>, Option<usize>)) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
        }
    }
    pub fn with_output_size(output_size: usize) -> Self {
        Self::new((Some(output_size), Some(output_size)))
    }
}
/// Adaptive 2D max pooling layer
///
/// This layer automatically computes the kernel size and stride to produce
/// the specified output size, regardless of the input size. This is useful
/// in modern CNNs where different input sizes need to produce the same
/// feature map size (e.g., ResNet, VGG).
///
/// # PyTorch Equivalence
/// ```python
/// import torch.nn as nn
/// pool = nn.AdaptiveMaxPool2d((7, 7))
/// ```
///
/// # Examples
/// ```rust,ignore
/// use torsh_nn::layers::pooling::AdaptiveMaxPool2d;
/// use torsh_nn::Module;
/// use torsh_tensor::creation::zeros;
///
/// let pool = AdaptiveMaxPool2d::with_output_size(7);
/// let input1 = zeros(&[1, 64, 224, 224])?; // ImageNet size
/// let input2 = zeros(&[1, 64, 384, 384])?; // Larger input
///
/// let output1 = pool.forward(&input1)?; // [1, 64, 7, 7]
/// let output2 = pool.forward(&input2)?; // [1, 64, 7, 7] - same output size!
/// ```
pub struct AdaptiveMaxPool2d {
    pub base: ModuleBase,
    pub output_size: (Option<usize>, Option<usize>),
    pub return_indices: bool,
}
impl AdaptiveMaxPool2d {
    /// Creates a new AdaptiveMaxPool2d layer
    ///
    /// # Arguments
    /// * `output_size` - Target output size (H, W). Use None to keep dimension unchanged.
    /// * `return_indices` - Whether to return max indices along with output
    pub fn new(output_size: (Option<usize>, Option<usize>), return_indices: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
            return_indices,
        }
    }
    /// Creates an AdaptiveMaxPool2d with square output size
    pub fn with_output_size(output_size: usize) -> Self {
        Self::new((Some(output_size), Some(output_size)), false)
    }
    /// Computes adaptive pooling parameters for a single dimension
    ///
    /// Formula:
    /// - stride = floor(input_size / output_size)
    /// - kernel_size = input_size - (output_size - 1) * stride
    ///
    /// # Arguments
    /// * `input_size` - Input dimension size
    /// * `output_size` - Desired output dimension size
    ///
    /// # Returns
    /// (kernel_size, stride, padding)
    fn compute_pooling_params(input_size: usize, output_size: usize) -> (usize, usize, usize) {
        if output_size == 1 {
            return (input_size, input_size, 0);
        }
        let stride = input_size / output_size;
        let kernel_size = input_size - (output_size - 1) * stride;
        let padding = 0;
        (kernel_size, stride, padding)
    }
    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        if input_shape.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "AdaptiveMaxPool2d expects 4D input [N, C, H, W], got {}D: {:?}",
                input_shape.len(),
                input_shape
            )));
        }
        let output_height = self.output_size.0.unwrap_or(input_shape[2]);
        let output_width = self.output_size.1.unwrap_or(input_shape[3]);
        let _h_params = Self::compute_pooling_params(input_shape[2], output_height);
        let _w_params = Self::compute_pooling_params(input_shape[3], output_width);
        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];
        let output = zeros(&output_shape)?;
        let indices = if self.return_indices {
            Some(zeros(&output_shape)?)
        } else {
            None
        };
        Ok((output, indices))
    }
}
/// Adaptive 3D max pooling layer
pub struct AdaptiveMaxPool3d {
    pub base: ModuleBase,
    pub output_size: (Option<usize>, Option<usize>, Option<usize>),
    pub return_indices: bool,
}
impl AdaptiveMaxPool3d {
    pub fn new(
        output_size: (Option<usize>, Option<usize>, Option<usize>),
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
            return_indices,
        }
    }
    pub fn with_output_size(output_size: usize) -> Self {
        Self::new(
            (Some(output_size), Some(output_size), Some(output_size)),
            false,
        )
    }
    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let output_depth = self.output_size.0.unwrap_or(input_shape[2]);
        let output_height = self.output_size.1.unwrap_or(input_shape[3]);
        let output_width = self.output_size.2.unwrap_or(input_shape[4]);
        let output_shape = [
            input_shape[0],
            input_shape[1],
            output_depth,
            output_height,
            output_width,
        ];
        let output = zeros(&output_shape)?;
        let indices = if self.return_indices {
            Some(zeros(&output_shape)?)
        } else {
            None
        };
        Ok((output, indices))
    }
}
/// Fractional max pooling 2D layer
///
/// Fractional max pooling provides regularization through stochastic pooling
/// with fractional reduction ratios. Unlike standard max pooling which uses
/// fixed integer stride/kernel sizes, fractional pooling uses randomized
/// pooling regions that provide better generalization.
///
/// # Key Features
/// - Stochastic pooling regions (training mode)
/// - Deterministic pooling (evaluation mode)
/// - Fractional reduction ratios (e.g., 0.5, 0.7, 0.9)
/// - Regularization through randomness
///
/// # PyTorch Equivalence
/// ```python
/// import torch.nn as nn
/// pool = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=0.5)
/// ```
///
/// # Examples
/// ```rust,ignore
/// use torsh_nn::layers::pooling::FractionalMaxPool2d;
/// use torsh_nn::Module;
/// use torsh_tensor::creation::zeros;
///
/// let mut pool = FractionalMaxPool2d::with_output_ratio((2, 2), (0.5, 0.5));
/// let input = zeros(&[1, 64, 32, 32])?;
///
/// // Training: stochastic pooling (randomized)
/// pool.train();
/// let output1 = pool.forward(&input)?; // [1, 64, 16, 16]
/// let output2 = pool.forward(&input)?; // Different result due to randomness
///
/// // Evaluation: deterministic pooling
/// pool.eval();
/// let output3 = pool.forward(&input)?; // [1, 64, 16, 16]
/// let output4 = pool.forward(&input)?; // Same result (deterministic)
/// ```
pub struct FractionalMaxPool2d {
    pub base: ModuleBase,
    pub kernel_size: (usize, usize),
    pub output_size: Option<(usize, usize)>,
    pub output_ratio: Option<(f32, f32)>,
    pub return_indices: bool,
}
impl FractionalMaxPool2d {
    /// Creates a new FractionalMaxPool2d layer
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling kernel (H, W)
    /// * `output_size` - Explicit output size (overrides output_ratio if set)
    /// * `output_ratio` - Fractional reduction ratio (typically 0.5-0.9)
    /// * `return_indices` - Whether to return max indices along with output
    ///
    /// # Note
    /// Either `output_size` or `output_ratio` must be specified (not both).
    pub fn new(
        kernel_size: (usize, usize),
        output_size: Option<(usize, usize)>,
        output_ratio: Option<(f32, f32)>,
        return_indices: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            output_size,
            output_ratio,
            return_indices,
        }
    }
    /// Creates a FractionalMaxPool2d with default 0.5 output ratio
    pub fn with_kernel_size(kernel_size: (usize, usize)) -> Self {
        Self::new(kernel_size, None, Some((0.5, 0.5)), false)
    }
    /// Creates a FractionalMaxPool2d with specified output ratio
    ///
    /// # Arguments
    /// * `kernel_size` - Size of the pooling kernel
    /// * `output_ratio` - Fractional reduction ratio (e.g., 0.5 = 50% size reduction)
    pub fn with_output_ratio(kernel_size: (usize, usize), output_ratio: (f32, f32)) -> Self {
        Self::new(kernel_size, None, Some(output_ratio), false)
    }
    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        if input_shape.len() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "FractionalMaxPool2d expects 4D input [N, C, H, W], got {}D: {:?}",
                input_shape.len(),
                input_shape
            )));
        }
        let (output_height, output_width) = if let Some((oh, ow)) = self.output_size {
            (oh, ow)
        } else if let Some((rh, rw)) = self.output_ratio {
            if !(0.0..=1.0).contains(&rh) || !(0.0..=1.0).contains(&rw) {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "output_ratio must be in range [0.0, 1.0], got ({}, {})",
                    rh, rw
                )));
            }
            (
                (input_shape[2] as f32 * rh) as usize,
                (input_shape[3] as f32 * rw) as usize,
            )
        } else {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Either output_size or output_ratio must be specified".to_string(),
            ));
        };
        if output_height == 0 || output_width == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Output size must be positive, got ({}, {})",
                output_height, output_width
            )));
        }
        let output_shape = [input_shape[0], input_shape[1], output_height, output_width];
        let output = zeros(&output_shape)?;
        let indices = if self.return_indices {
            Some(zeros(&output_shape)?)
        } else {
            None
        };
        Ok((output, indices))
    }
}
/// LP pooling 2D layer
pub struct LPPool2d {
    pub base: ModuleBase,
    pub norm_type: f32,
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub ceil_mode: bool,
}
impl LPPool2d {
    pub fn new(
        norm_type: f32,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            norm_type,
            kernel_size,
            stride,
            ceil_mode,
        }
    }
}
/// 2D max pooling layer
pub struct MaxPool2d {
    pub base: ModuleBase,
    pub kernel_size: (usize, usize),
    pub stride: Option<(usize, usize)>,
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub ceil_mode: bool,
}
impl MaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        dilation: (usize, usize),
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }
    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new((kernel_size, kernel_size), None, (0, 0), (1, 1), false)
    }
}
/// LP pooling 1D layer
pub struct LPPool1d {
    pub base: ModuleBase,
    pub norm_type: f32,
    pub kernel_size: usize,
    pub stride: Option<usize>,
    pub ceil_mode: bool,
}
impl LPPool1d {
    pub fn new(norm_type: f32, kernel_size: usize, stride: Option<usize>, ceil_mode: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            norm_type,
            kernel_size,
            stride,
            ceil_mode,
        }
    }
}
/// Global pooling utilities
pub struct GlobalPool;
impl GlobalPool {
    /// Global average pooling - reduces spatial dimensions to 1x1
    pub fn global_avg_pool2d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveAvgPool2d::new((Some(1), Some(1)));
        adaptive_pool.forward(input)
    }
    /// Global max pooling - reduces spatial dimensions to 1x1
    pub fn global_max_pool2d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveMaxPool2d::new((Some(1), Some(1)), false);
        adaptive_pool.forward(input)
    }
    /// Global average pooling for 1D inputs
    pub fn global_avg_pool1d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveAvgPool1d::new(1);
        adaptive_pool.forward(input)
    }
    /// Global max pooling for 1D inputs
    pub fn global_max_pool1d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveMaxPool1d::new(1, false);
        adaptive_pool.forward(input)
    }
    /// Global average pooling for 3D inputs
    pub fn global_avg_pool3d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveAvgPool3d::new((Some(1), Some(1), Some(1)));
        adaptive_pool.forward(input)
    }
    /// Global max pooling for 3D inputs
    pub fn global_max_pool3d(input: &Tensor) -> Result<Tensor> {
        let adaptive_pool = AdaptiveMaxPool3d::new((Some(1), Some(1), Some(1)), false);
        adaptive_pool.forward(input)
    }
}
/// Adaptive 1D max pooling layer
pub struct AdaptiveMaxPool1d {
    pub base: ModuleBase,
    pub output_size: usize,
    pub return_indices: bool,
}
impl AdaptiveMaxPool1d {
    pub fn new(output_size: usize, return_indices: bool) -> Self {
        Self {
            base: ModuleBase::new(),
            output_size,
            return_indices,
        }
    }
    /// Forward pass returning both output and optionally indices
    pub fn forward_with_indices(&self, input: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let binding = input.shape();
        let input_shape = binding.dims();
        let output_shape = [input_shape[0], input_shape[1], self.output_size];
        let output = zeros(&output_shape)?;
        let indices = if self.return_indices {
            Some(zeros(&output_shape)?)
        } else {
            None
        };
        Ok((output, indices))
    }
}
/// 1D max pooling layer
pub struct MaxPool1d {
    pub base: ModuleBase,
    pub kernel_size: usize,
    pub stride: Option<usize>,
    pub padding: usize,
    pub dilation: usize,
    pub ceil_mode: bool,
}
impl MaxPool1d {
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }
    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new(kernel_size, None, 0, 1, false)
    }
}
/// 3D max pooling layer
pub struct MaxPool3d {
    pub base: ModuleBase,
    pub kernel_size: (usize, usize, usize),
    pub stride: Option<(usize, usize, usize)>,
    pub padding: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub ceil_mode: bool,
}
impl MaxPool3d {
    pub fn new(
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        ceil_mode: bool,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }
    pub fn with_kernel_size(kernel_size: usize) -> Self {
        Self::new(
            (kernel_size, kernel_size, kernel_size),
            None,
            (0, 0, 0),
            (1, 1, 1),
            false,
        )
    }
}
