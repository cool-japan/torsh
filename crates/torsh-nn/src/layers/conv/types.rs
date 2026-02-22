//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Module, ModuleBase, Parameter};
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// 1D transposed convolutional layer (deconvolution)
pub struct ConvTranspose1d {
    pub(super) base: ModuleBase,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_size: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
    pub(super) output_padding: usize,
    dilation: usize,
    pub(super) groups: usize,
    pub(super) use_bias: bool,
}
impl ConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        bias: bool,
        groups: usize,
    ) -> Self {
        let mut base = ModuleBase::new();
        let weight_shape = [in_channels, out_channels / groups, kernel_size];
        let weight = crate::init::xavier_uniform(&weight_shape)
            .expect("Failed to initialize convtranspose1d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        if bias {
            let bias_tensor = zeros(&[out_channels]).expect("zeros tensor for bias should succeed");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }
        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
    pub fn with_defaults(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(in_channels, out_channels, kernel_size, 1, 0, 0, 1, true, 1)
    }
    /// Perform 1D transposed convolution
    pub(crate) fn conv_transpose1d_direct(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_length = input_shape[2];
        let out_length = (in_length - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + self.output_padding
            + 1;
        let output_shape = [batch_size, self.out_channels, out_length];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;
        for batch_idx in 0..batch_size {
            for in_ch in 0..in_channels {
                for in_x in 0..in_length {
                    let input_val = input_data
                        [batch_idx * (in_channels * in_length) + in_ch * in_length + in_x];
                    for kx in 0..self.kernel_size {
                        let out_x = in_x * self.stride + kx * self.dilation;
                        if out_x >= self.padding {
                            let actual_out_x = out_x - self.padding;
                            if actual_out_x < out_length {
                                for out_ch in 0..self.out_channels {
                                    let weight_idx = in_ch * (self.out_channels * self.kernel_size)
                                        + out_ch * self.kernel_size
                                        + kx;
                                    let output_idx = batch_idx * (self.out_channels * out_length)
                                        + out_ch * out_length
                                        + actual_out_x;
                                    output_data[output_idx] += input_val * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_vec(output_data, &output_shape)
    }
}
/// 2D convolutional layer
pub struct Conv2d {
    pub(super) base: ModuleBase,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_size: (usize, usize),
    pub(super) stride: (usize, usize),
    pub(super) padding: (usize, usize),
    pub(super) dilation: (usize, usize),
    pub(super) groups: usize,
    pub(super) use_bias: bool,
}
impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
        groups: usize,
    ) -> Self {
        let mut base = ModuleBase::new();
        let weight_shape = [
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
        let weight =
            crate::init::xavier_uniform(&weight_shape).expect("Failed to initialize conv2d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        if bias {
            let bias_tensor = zeros(&[out_channels]).expect("zeros tensor for bias should succeed");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }
        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
    pub fn with_defaults(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (0, 0),
            (1, 1),
            true,
            1,
        )
    }
    /// Perform 2D convolution using direct implementation
    pub(crate) fn conv2d_im2col(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        let out_height =
            (in_height + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let out_width =
            (in_width + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;
        let output_shape = [batch_size, self.out_channels, out_height, out_width];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;
        for batch_idx in 0..batch_size {
            for out_ch in 0..self.out_channels {
                for out_y in 0..out_height {
                    for out_x in 0..out_width {
                        let mut sum = 0.0f32;
                        for in_ch in 0..in_channels {
                            for ky in 0..self.kernel_size.0 {
                                for kx in 0..self.kernel_size.1 {
                                    let in_y = out_y * self.stride.0 + ky * self.dilation.0;
                                    let in_x = out_x * self.stride.1 + kx * self.dilation.1;
                                    if in_y >= self.padding.0 && in_x >= self.padding.1 {
                                        let actual_in_y = in_y - self.padding.0;
                                        let actual_in_x = in_x - self.padding.1;
                                        if actual_in_y < in_height && actual_in_x < in_width {
                                            let input_idx = batch_idx
                                                * (in_channels * in_height * in_width)
                                                + in_ch * (in_height * in_width)
                                                + actual_in_y * in_width
                                                + actual_in_x;
                                            let weight_idx = out_ch
                                                * (in_channels
                                                    * self.kernel_size.0
                                                    * self.kernel_size.1)
                                                + in_ch * (self.kernel_size.0 * self.kernel_size.1)
                                                + ky * self.kernel_size.1
                                                + kx;
                                            if input_idx < input_data.len()
                                                && weight_idx < weight_data.len()
                                            {
                                                sum +=
                                                    input_data[input_idx] * weight_data[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        let output_idx = batch_idx * (self.out_channels * out_height * out_width)
                            + out_ch * (out_height * out_width)
                            + out_y * out_width
                            + out_x;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
        Tensor::from_vec(output_data, &output_shape)
    }
}
/// Depthwise separable convolutional layer
///
/// A depthwise separable convolution is a factorized convolution that splits a standard
/// convolution into two separate operations:
/// 1. **Depthwise convolution**: Applies a single filter per input channel (groups = in_channels)
/// 2. **Pointwise convolution**: A 1x1 convolution that combines the outputs
///
/// This factorization dramatically reduces the number of parameters and computational cost
/// compared to standard convolutions, making it ideal for mobile and embedded applications.
///
/// # Parameter Reduction
///
/// For a standard convolution with parameters:
/// - `in_channels` = C_in
/// - `out_channels` = C_out
/// - `kernel_size` = K
///
/// **Standard Conv2d parameters**: C_out × C_in × K × K
///
/// **Depthwise Separable parameters**: C_in × K × K + C_out × C_in × 1 × 1
///
/// **Reduction ratio**: (C_in × K² + C_out × C_in) / (C_out × C_in × K²) ≈ 1/K² + 1/C_out
///
/// For typical values (K=3, C_out=64): ~8-9x parameter reduction
///
/// # Applications
///
/// - **MobileNet**: Uses depthwise separable convolutions throughout
/// - **EfficientNet**: Core building block for efficient architectures
/// - **Xception**: "Extreme" version of Inception with depthwise separable convolutions
///
/// # Example
///
/// ```ignore
/// use torsh_nn::layers::DepthwiseSeparableConv;
///
/// // Create a depthwise separable conv layer
/// let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true)?;
///
/// // Standard Conv2d equivalent would have 32 × 64 × 3 × 3 = 18,432 parameters
/// // Depthwise separable has 32 × 3 × 3 + 64 × 32 × 1 × 1 = 288 + 2,048 = 2,336 parameters
/// // Reduction: ~7.9x fewer parameters
/// ```
///
/// # References
///
/// - Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", 2017
/// - Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017
pub struct DepthwiseSeparableConv {
    pub(super) depthwise: Conv2d,
    pub(super) pointwise: Conv2d,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_size: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
}
impl DepthwiseSeparableConv {
    /// Create a new depthwise separable convolutional layer
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the depthwise convolution kernel (square)
    /// * `stride` - Stride for the depthwise convolution
    /// * `padding` - Padding for the depthwise convolution
    /// * `bias` - Whether to include bias terms
    ///
    /// # Returns
    ///
    /// A new `DepthwiseSeparableConv` layer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true)?;
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let depthwise = Conv2d::new(
            in_channels,
            in_channels,
            (kernel_size, kernel_size),
            (stride, stride),
            (padding, padding),
            (1, 1),
            bias,
            in_channels,
        );
        let pointwise = Conv2d::new(
            in_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            (1, 1),
            bias,
            1,
        );
        Self {
            depthwise,
            pointwise,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }
    /// Create with default parameters (stride=1, padding=1, bias=true)
    pub fn with_defaults(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(in_channels, out_channels, kernel_size, 1, 1, true)
    }
    /// Get the total number of parameters in this layer
    ///
    /// Returns (depthwise_params, pointwise_params, total_params)
    pub fn param_count(&self) -> (usize, usize, usize) {
        let depthwise_params = self.depthwise.parameters().len();
        let pointwise_params = self.pointwise.parameters().len();
        (
            depthwise_params,
            pointwise_params,
            depthwise_params + pointwise_params,
        )
    }
    /// Calculate the parameter reduction compared to standard Conv2d
    ///
    /// Returns the ratio: depthwise_separable_params / standard_conv_params
    pub fn parameter_reduction_ratio(&self) -> f32 {
        let standard_params =
            self.out_channels * self.in_channels * self.kernel_size * self.kernel_size;
        let depthwise_params = self.in_channels * self.kernel_size * self.kernel_size;
        let pointwise_params = self.out_channels * self.in_channels;
        let total_params = depthwise_params + pointwise_params;
        total_params as f32 / standard_params as f32
    }
    /// Get input channels
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    /// Get output channels
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    /// Get kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
    /// Get stride
    pub fn stride(&self) -> usize {
        self.stride
    }
    /// Get padding
    pub fn padding(&self) -> usize {
        self.padding
    }
}
/// 2D transposed convolutional layer (deconvolution)
pub struct ConvTranspose2d {
    pub(super) base: ModuleBase,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_size: (usize, usize),
    pub(super) stride: (usize, usize),
    pub(super) padding: (usize, usize),
    pub(super) output_padding: (usize, usize),
    dilation: (usize, usize),
    pub(super) groups: usize,
    pub(super) use_bias: bool,
}
impl ConvTranspose2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
        groups: usize,
    ) -> Self {
        let mut base = ModuleBase::new();
        let weight_shape = [
            in_channels,
            out_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
        let weight = crate::init::xavier_uniform(&weight_shape)
            .expect("Failed to initialize convtranspose2d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        if bias {
            let bias_tensor = zeros(&[out_channels]).expect("zeros tensor for bias should succeed");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }
        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
    pub fn with_defaults(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (0, 0),
            (0, 0),
            (1, 1),
            true,
            1,
        )
    }
    /// Perform 2D transposed convolution
    pub(crate) fn conv_transpose2d_direct(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        let out_height = (in_height - 1) * self.stride.0 - 2 * self.padding.0
            + self.dilation.0 * (self.kernel_size.0 - 1)
            + self.output_padding.0
            + 1;
        let out_width = (in_width - 1) * self.stride.1 - 2 * self.padding.1
            + self.dilation.1 * (self.kernel_size.1 - 1)
            + self.output_padding.1
            + 1;
        let output_shape = [batch_size, self.out_channels, out_height, out_width];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;
        for batch_idx in 0..batch_size {
            for in_ch in 0..in_channels {
                for in_y in 0..in_height {
                    for in_x in 0..in_width {
                        let input_val = input_data[batch_idx
                            * (in_channels * in_height * in_width)
                            + in_ch * (in_height * in_width)
                            + in_y * in_width
                            + in_x];
                        for ky in 0..self.kernel_size.0 {
                            for kx in 0..self.kernel_size.1 {
                                let out_y = in_y * self.stride.0 + ky * self.dilation.0;
                                let out_x = in_x * self.stride.1 + kx * self.dilation.1;
                                if out_y >= self.padding.0 && out_x >= self.padding.1 {
                                    let actual_out_y = out_y - self.padding.0;
                                    let actual_out_x = out_x - self.padding.1;
                                    if actual_out_y < out_height && actual_out_x < out_width {
                                        for out_ch in 0..self.out_channels {
                                            let weight_idx = in_ch
                                                * (self.out_channels
                                                    * self.kernel_size.0
                                                    * self.kernel_size.1)
                                                + out_ch
                                                    * (self.kernel_size.0 * self.kernel_size.1)
                                                + ky * self.kernel_size.1
                                                + kx;
                                            let output_idx = batch_idx
                                                * (self.out_channels * out_height * out_width)
                                                + out_ch * (out_height * out_width)
                                                + actual_out_y * out_width
                                                + actual_out_x;
                                            output_data[output_idx] +=
                                                input_val * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_vec(output_data, &output_shape)
    }
}
/// 3D convolutional layer
pub struct Conv3d {
    pub(super) base: ModuleBase,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_size: (usize, usize, usize),
    pub(super) stride: (usize, usize, usize),
    pub(super) padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    pub(super) groups: usize,
    pub(super) use_bias: bool,
}
impl Conv3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        bias: bool,
        groups: usize,
    ) -> Self {
        let mut base = ModuleBase::new();
        let weight_shape = [
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        ];
        let weight =
            crate::init::xavier_uniform(&weight_shape).expect("Failed to initialize conv3d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        if bias {
            let bias_tensor = zeros(&[out_channels]).expect("zeros tensor for bias should succeed");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }
        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
    pub fn with_defaults(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            true,
            1,
        )
    }
    /// Perform 3D convolution using direct implementation
    pub(crate) fn conv3d_direct(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_depth = input_shape[2];
        let in_height = input_shape[3];
        let in_width = input_shape[4];
        let out_depth =
            (in_depth + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let out_height =
            (in_height + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;
        let out_width =
            (in_width + 2 * self.padding.2 - self.dilation.2 * (self.kernel_size.2 - 1) - 1)
                / self.stride.2
                + 1;
        let output_shape = [
            batch_size,
            self.out_channels,
            out_depth,
            out_height,
            out_width,
        ];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;
        for batch_idx in 0..batch_size {
            for out_ch in 0..self.out_channels {
                for out_z in 0..out_depth {
                    for out_y in 0..out_height {
                        for out_x in 0..out_width {
                            let mut sum = 0.0f32;
                            for in_ch in 0..in_channels {
                                for kz in 0..self.kernel_size.0 {
                                    for ky in 0..self.kernel_size.1 {
                                        for kx in 0..self.kernel_size.2 {
                                            let in_z = out_z * self.stride.0 + kz * self.dilation.0;
                                            let in_y = out_y * self.stride.1 + ky * self.dilation.1;
                                            let in_x = out_x * self.stride.2 + kx * self.dilation.2;
                                            if in_z >= self.padding.0
                                                && in_y >= self.padding.1
                                                && in_x >= self.padding.2
                                            {
                                                let actual_in_z = in_z - self.padding.0;
                                                let actual_in_y = in_y - self.padding.1;
                                                let actual_in_x = in_x - self.padding.2;
                                                if actual_in_z < in_depth
                                                    && actual_in_y < in_height
                                                    && actual_in_x < in_width
                                                {
                                                    let input_idx = batch_idx
                                                        * (in_channels
                                                            * in_depth
                                                            * in_height
                                                            * in_width)
                                                        + in_ch * (in_depth * in_height * in_width)
                                                        + actual_in_z * (in_height * in_width)
                                                        + actual_in_y * in_width
                                                        + actual_in_x;
                                                    let weight_idx = out_ch
                                                        * (in_channels
                                                            * self.kernel_size.0
                                                            * self.kernel_size.1
                                                            * self.kernel_size.2)
                                                        + in_ch
                                                            * (self.kernel_size.0
                                                                * self.kernel_size.1
                                                                * self.kernel_size.2)
                                                        + kz * (self.kernel_size.1
                                                            * self.kernel_size.2)
                                                        + ky * self.kernel_size.2
                                                        + kx;
                                                    sum += input_data[input_idx]
                                                        * weight_data[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            let output_idx = batch_idx
                                * (self.out_channels * out_depth * out_height * out_width)
                                + out_ch * (out_depth * out_height * out_width)
                                + out_z * (out_height * out_width)
                                + out_y * out_width
                                + out_x;
                            output_data[output_idx] = sum;
                        }
                    }
                }
            }
        }
        Tensor::from_vec(output_data, &output_shape)
    }
}
/// 3D transposed convolutional layer (deconvolution)
pub struct ConvTranspose3d {
    pub(super) base: ModuleBase,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_size: (usize, usize, usize),
    pub(super) stride: (usize, usize, usize),
    pub(super) padding: (usize, usize, usize),
    pub(super) output_padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    pub(super) groups: usize,
    pub(super) use_bias: bool,
}
impl ConvTranspose3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        bias: bool,
        groups: usize,
    ) -> Self {
        let mut base = ModuleBase::new();
        let weight_shape = [
            in_channels,
            out_channels / groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        ];
        let weight = crate::init::xavier_uniform(&weight_shape)
            .expect("Failed to initialize convtranspose3d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        if bias {
            let bias_tensor = zeros(&[out_channels]).expect("zeros tensor for bias should succeed");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }
        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
    pub fn with_defaults(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
            true,
            1,
        )
    }
    /// Perform 3D transposed convolution
    pub(crate) fn conv_transpose3d_direct(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_depth = input_shape[2];
        let in_height = input_shape[3];
        let in_width = input_shape[4];
        let out_depth = (in_depth - 1) * self.stride.0 - 2 * self.padding.0
            + self.dilation.0 * (self.kernel_size.0 - 1)
            + self.output_padding.0
            + 1;
        let out_height = (in_height - 1) * self.stride.1 - 2 * self.padding.1
            + self.dilation.1 * (self.kernel_size.1 - 1)
            + self.output_padding.1
            + 1;
        let out_width = (in_width - 1) * self.stride.2 - 2 * self.padding.2
            + self.dilation.2 * (self.kernel_size.2 - 1)
            + self.output_padding.2
            + 1;
        let output_shape = [
            batch_size,
            self.out_channels,
            out_depth,
            out_height,
            out_width,
        ];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;
        for batch_idx in 0..batch_size {
            for in_ch in 0..in_channels {
                for in_z in 0..in_depth {
                    for in_y in 0..in_height {
                        for in_x in 0..in_width {
                            let input_val = input_data[batch_idx
                                * (in_channels * in_depth * in_height * in_width)
                                + in_ch * (in_depth * in_height * in_width)
                                + in_z * (in_height * in_width)
                                + in_y * in_width
                                + in_x];
                            for kz in 0..self.kernel_size.0 {
                                for ky in 0..self.kernel_size.1 {
                                    for kx in 0..self.kernel_size.2 {
                                        let out_z = in_z * self.stride.0 + kz * self.dilation.0;
                                        let out_y = in_y * self.stride.1 + ky * self.dilation.1;
                                        let out_x = in_x * self.stride.2 + kx * self.dilation.2;
                                        if out_z >= self.padding.0
                                            && out_y >= self.padding.1
                                            && out_x >= self.padding.2
                                        {
                                            let actual_out_z = out_z - self.padding.0;
                                            let actual_out_y = out_y - self.padding.1;
                                            let actual_out_x = out_x - self.padding.2;
                                            if actual_out_z < out_depth
                                                && actual_out_y < out_height
                                                && actual_out_x < out_width
                                            {
                                                for out_ch in 0..self.out_channels {
                                                    let weight_idx = in_ch
                                                        * (self.out_channels
                                                            * self.kernel_size.0
                                                            * self.kernel_size.1
                                                            * self.kernel_size.2)
                                                        + out_ch
                                                            * (self.kernel_size.0
                                                                * self.kernel_size.1
                                                                * self.kernel_size.2)
                                                        + kz * (self.kernel_size.1
                                                            * self.kernel_size.2)
                                                        + ky * self.kernel_size.2
                                                        + kx;
                                                    let output_idx = batch_idx
                                                        * (self.out_channels
                                                            * out_depth
                                                            * out_height
                                                            * out_width)
                                                        + out_ch
                                                            * (out_depth * out_height * out_width)
                                                        + actual_out_z * (out_height * out_width)
                                                        + actual_out_y * out_width
                                                        + actual_out_x;
                                                    output_data[output_idx] +=
                                                        input_val * weight_data[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_vec(output_data, &output_shape)
    }
}
/// 1D convolutional layer
pub struct Conv1d {
    pub(super) base: ModuleBase,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_size: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
    dilation: usize,
    pub(super) groups: usize,
    pub(super) use_bias: bool,
}
impl Conv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
        groups: usize,
    ) -> Self {
        let mut base = ModuleBase::new();
        let weight_shape = [out_channels, in_channels / groups, kernel_size];
        let weight =
            crate::init::xavier_uniform(&weight_shape).expect("Failed to initialize conv1d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));
        if bias {
            let bias_tensor = zeros(&[out_channels]).expect("zeros tensor for bias should succeed");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }
        Self {
            base,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
        }
    }
    pub fn with_defaults(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::new(in_channels, out_channels, kernel_size, 1, 0, 1, true, 1)
    }
    /// Perform 1D convolution using direct implementation
    pub(crate) fn conv1d_direct(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_length = input_shape[2];
        let out_length =
            (in_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
                + 1;
        let output_shape = [batch_size, self.out_channels, out_length];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;
        for batch_idx in 0..batch_size {
            for out_ch in 0..self.out_channels {
                for out_x in 0..out_length {
                    let mut sum = 0.0f32;
                    for in_ch in 0..in_channels {
                        for kx in 0..self.kernel_size {
                            let in_x = out_x * self.stride + kx * self.dilation;
                            if in_x >= self.padding {
                                let actual_in_x = in_x - self.padding;
                                if actual_in_x < in_length {
                                    let input_idx = batch_idx * (in_channels * in_length)
                                        + in_ch * in_length
                                        + actual_in_x;
                                    let weight_idx = out_ch * (in_channels * self.kernel_size)
                                        + in_ch * self.kernel_size
                                        + kx;
                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                    let output_idx =
                        batch_idx * (self.out_channels * out_length) + out_ch * out_length + out_x;
                    output_data[output_idx] = sum;
                }
            }
        }
        Tensor::from_vec(output_data, &output_shape)
    }
}
