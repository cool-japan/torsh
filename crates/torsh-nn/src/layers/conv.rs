//! Convolutional layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// 1D convolutional layer
pub struct Conv1d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    use_bias: bool,
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

        // Weight shape: [out_channels, in_channels/groups, kernel_size]
        let weight_shape = [out_channels, in_channels / groups, kernel_size];
        let weight =
            crate::init::xavier_uniform(&weight_shape).expect("Failed to initialize conv1d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_channels]).unwrap();
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
    fn conv1d_direct(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_length = input_shape[2];

        // Calculate output length
        let out_length =
            (in_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
                + 1;

        // Output tensor
        let output_shape = [batch_size, self.out_channels, out_length];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];

        // Get input and weight data
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // For each output channel
            for out_ch in 0..self.out_channels {
                // For each output position
                for out_x in 0..out_length {
                    let mut sum = 0.0f32;

                    // Convolution computation
                    for in_ch in 0..in_channels {
                        for kx in 0..self.kernel_size {
                            // Calculate input position with padding
                            let in_x = out_x * self.stride + kx * self.dilation;

                            // Check if position is within bounds (considering padding)
                            if in_x >= self.padding {
                                let actual_in_x = in_x - self.padding;

                                if actual_in_x < in_length {
                                    // Calculate indices
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

                    // Store result
                    let output_idx =
                        batch_idx * (self.out_channels * out_length) + out_ch * out_length + out_x;
                    output_data[output_idx] = sum;
                }
            }
        }

        Tensor::from_vec(output_data, &output_shape)
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // 1D convolution implementation
        // Input shape: [batch_size, in_channels, length]
        // Weight shape: [out_channels, in_channels/groups, kernel_size]
        // Output shape: [batch_size, out_channels, output_length]

        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Perform actual 1D convolution
        let mut output = self.conv1d_direct(input, &weight)?;

        // Add bias if present
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            output = output.add_op(&reshaped_bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// 2D convolutional layer
pub struct Conv2d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    use_bias: bool,
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

        // Weight shape: [out_channels, in_channels/groups, kernel_height, kernel_width]
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
            let bias_tensor = zeros(&[out_channels]).unwrap();
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
    fn conv2d_im2col(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        // Calculate output dimensions
        let out_height =
            (in_height + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let out_width =
            (in_width + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;

        // Output tensor
        let output_shape = [batch_size, self.out_channels, out_height, out_width];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];

        // Get input and weight data
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // For each output channel
            for out_ch in 0..self.out_channels {
                // For each output position
                for out_y in 0..out_height {
                    for out_x in 0..out_width {
                        let mut sum = 0.0f32;

                        // Convolution computation
                        for in_ch in 0..in_channels {
                            for ky in 0..self.kernel_size.0 {
                                for kx in 0..self.kernel_size.1 {
                                    // Calculate input position with padding
                                    let in_y = out_y * self.stride.0 + ky * self.dilation.0;
                                    let in_x = out_x * self.stride.1 + kx * self.dilation.1;

                                    // Check if position is within bounds (considering padding)
                                    if in_y >= self.padding.0 && in_x >= self.padding.1 {
                                        let actual_in_y = in_y - self.padding.0;
                                        let actual_in_x = in_x - self.padding.1;

                                        if actual_in_y < in_height && actual_in_x < in_width {
                                            // Calculate indices
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

                                            // Add bounds checking to prevent crashes
                                            if input_idx < input_data.len() && weight_idx < weight_data.len() {
                                                sum += input_data[input_idx] * weight_data[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Store result
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

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Basic 2D convolution implementation
        // Input shape: [batch_size, in_channels, height, width]
        // Weight shape: [out_channels, in_channels/groups, kernel_height, kernel_width]
        // Output shape: [batch_size, out_channels, output_height, output_width]

        let weight = self.base.parameters["weight"].tensor().read().clone();

        let binding = input.shape();
        let input_shape = binding.dims();

        // Defensive bounds checking - ensure we have at least 4 dimensions
        if input_shape.len() < 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                format!(
                    "Conv2d expects 4D input (batch_size, channels, height, width), got {}D: {:?}",
                    input_shape.len(),
                    input_shape
                )
            ));
        }

        let output_height =
            (input_shape[2] + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let output_width =
            (input_shape[3] + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;

        let _output_shape = [
            input_shape[0],
            self.out_channels,
            output_height,
            output_width,
        ];
        // Perform actual 2D convolution using im2col approach
        let mut output = self.conv2d_im2col(input, &weight)?;

        // Add bias if present
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
            output = output.add_op(&reshaped_bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for Conv1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv1d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("groups", &self.groups)
            .finish()
    }
}

impl std::fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("groups", &self.groups)
            .finish()
    }
}

/// 3D convolutional layer
pub struct Conv3d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    use_bias: bool,
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

        // Weight shape: [out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width]
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
            let bias_tensor = zeros(&[out_channels]).unwrap();
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
    fn conv3d_direct(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_depth = input_shape[2];
        let in_height = input_shape[3];
        let in_width = input_shape[4];

        // Calculate output dimensions
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

        // Output tensor
        let output_shape = [
            batch_size,
            self.out_channels,
            out_depth,
            out_height,
            out_width,
        ];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];

        // Get input and weight data
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // For each output channel
            for out_ch in 0..self.out_channels {
                // For each output position
                for out_z in 0..out_depth {
                    for out_y in 0..out_height {
                        for out_x in 0..out_width {
                            let mut sum = 0.0f32;

                            // Convolution computation
                            for in_ch in 0..in_channels {
                                for kz in 0..self.kernel_size.0 {
                                    for ky in 0..self.kernel_size.1 {
                                        for kx in 0..self.kernel_size.2 {
                                            // Calculate input position with padding
                                            let in_z = out_z * self.stride.0 + kz * self.dilation.0;
                                            let in_y = out_y * self.stride.1 + ky * self.dilation.1;
                                            let in_x = out_x * self.stride.2 + kx * self.dilation.2;

                                            // Check if position is within bounds (considering padding)
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
                                                    // Calculate indices
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

                            // Store result
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

impl Module for Conv3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Basic 3D convolution implementation
        // Input shape: [batch_size, in_channels, depth, height, width]
        // Weight shape: [out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width]
        // Output shape: [batch_size, out_channels, output_depth, output_height, output_width]

        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Perform actual 3D convolution
        let mut output = self.conv3d_direct(input, &weight)?;

        // Add bias if present
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias
                .unsqueeze(0)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .unsqueeze(4)?;
            output = output.add_op(&reshaped_bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for Conv3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv3d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("groups", &self.groups)
            .finish()
    }
}

/// 1D transposed convolutional layer (deconvolution)
pub struct ConvTranspose1d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    use_bias: bool,
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

        // Weight shape: [in_channels, out_channels/groups, kernel_size]
        let weight_shape = [in_channels, out_channels / groups, kernel_size];
        let weight = crate::init::xavier_uniform(&weight_shape)
            .expect("Failed to initialize convtranspose1d weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_channels]).unwrap();
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
    fn conv_transpose1d_direct(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_length = input_shape[2];

        // Calculate output length
        let out_length = (in_length - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + self.output_padding
            + 1;

        // Output tensor
        let output_shape = [batch_size, self.out_channels, out_length];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];

        // Get input and weight data
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // For each input channel
            for in_ch in 0..in_channels {
                // For each input position
                for in_x in 0..in_length {
                    let input_val = input_data
                        [batch_idx * (in_channels * in_length) + in_ch * in_length + in_x];

                    // For each kernel position
                    for kx in 0..self.kernel_size {
                        // Calculate output position
                        let out_x = in_x * self.stride + kx * self.dilation;

                        // Check if position is within bounds (considering padding)
                        if out_x >= self.padding {
                            let actual_out_x = out_x - self.padding;

                            if actual_out_x < out_length {
                                // For each output channel
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

impl Module for ConvTranspose1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Perform transposed convolution
        let mut output = self.conv_transpose1d_direct(input, &weight)?;

        // Add bias if present
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            output = output.add_op(&reshaped_bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// 2D transposed convolutional layer (deconvolution)
pub struct ConvTranspose2d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    use_bias: bool,
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

        // Weight shape: [in_channels, out_channels/groups, kernel_height, kernel_width]
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
            let bias_tensor = zeros(&[out_channels]).unwrap();
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
    fn conv_transpose2d_direct(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        // Calculate output dimensions
        let out_height = (in_height - 1) * self.stride.0 - 2 * self.padding.0
            + self.dilation.0 * (self.kernel_size.0 - 1)
            + self.output_padding.0
            + 1;
        let out_width = (in_width - 1) * self.stride.1 - 2 * self.padding.1
            + self.dilation.1 * (self.kernel_size.1 - 1)
            + self.output_padding.1
            + 1;

        // Output tensor
        let output_shape = [batch_size, self.out_channels, out_height, out_width];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];

        // Get input and weight data
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // For each input channel
            for in_ch in 0..in_channels {
                // For each input position
                for in_y in 0..in_height {
                    for in_x in 0..in_width {
                        let input_val = input_data[batch_idx
                            * (in_channels * in_height * in_width)
                            + in_ch * (in_height * in_width)
                            + in_y * in_width
                            + in_x];

                        // For each kernel position
                        for ky in 0..self.kernel_size.0 {
                            for kx in 0..self.kernel_size.1 {
                                // Calculate output position
                                let out_y = in_y * self.stride.0 + ky * self.dilation.0;
                                let out_x = in_x * self.stride.1 + kx * self.dilation.1;

                                // Check if position is within bounds (considering padding)
                                if out_y >= self.padding.0 && out_x >= self.padding.1 {
                                    let actual_out_y = out_y - self.padding.0;
                                    let actual_out_x = out_x - self.padding.1;

                                    if actual_out_y < out_height && actual_out_x < out_width {
                                        // For each output channel
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

impl Module for ConvTranspose2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Perform transposed convolution
        let mut output = self.conv_transpose2d_direct(input, &weight)?;

        // Add bias if present
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
            output = output.add_op(&reshaped_bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// 3D transposed convolutional layer (deconvolution)
pub struct ConvTranspose3d {
    base: ModuleBase,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    use_bias: bool,
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

        // Weight shape: [in_channels, out_channels/groups, kernel_depth, kernel_height, kernel_width]
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
            let bias_tensor = zeros(&[out_channels]).unwrap();
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
    fn conv_transpose3d_direct(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let input_shape_binding = input.shape();
        let input_shape = input_shape_binding.dims();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_depth = input_shape[2];
        let in_height = input_shape[3];
        let in_width = input_shape[4];

        // Calculate output dimensions
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

        // Output tensor
        let output_shape = [
            batch_size,
            self.out_channels,
            out_depth,
            out_height,
            out_width,
        ];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];

        // Get input and weight data
        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Process each sample in the batch
        for batch_idx in 0..batch_size {
            // For each input channel
            for in_ch in 0..in_channels {
                // For each input position
                for in_z in 0..in_depth {
                    for in_y in 0..in_height {
                        for in_x in 0..in_width {
                            let input_val = input_data[batch_idx
                                * (in_channels * in_depth * in_height * in_width)
                                + in_ch * (in_depth * in_height * in_width)
                                + in_z * (in_height * in_width)
                                + in_y * in_width
                                + in_x];

                            // For each kernel position
                            for kz in 0..self.kernel_size.0 {
                                for ky in 0..self.kernel_size.1 {
                                    for kx in 0..self.kernel_size.2 {
                                        // Calculate output position
                                        let out_z = in_z * self.stride.0 + kz * self.dilation.0;
                                        let out_y = in_y * self.stride.1 + ky * self.dilation.1;
                                        let out_x = in_x * self.stride.2 + kx * self.dilation.2;

                                        // Check if position is within bounds (considering padding)
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
                                                // For each output channel
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

impl Module for ConvTranspose3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Perform transposed convolution
        let mut output = self.conv_transpose3d_direct(input, &weight)?;

        // Add bias if present
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            let reshaped_bias = bias
                .unsqueeze(0)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .unsqueeze(4)?;
            output = output.add_op(&reshaped_bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl std::fmt::Debug for ConvTranspose1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvTranspose1d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("output_padding", &self.output_padding)
            .field("groups", &self.groups)
            .finish()
    }
}

impl std::fmt::Debug for ConvTranspose2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvTranspose2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("output_padding", &self.output_padding)
            .field("groups", &self.groups)
            .finish()
    }
}

impl std::fmt::Debug for ConvTranspose3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvTranspose3d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("output_padding", &self.output_padding)
            .field("groups", &self.groups)
            .finish()
    }
}
