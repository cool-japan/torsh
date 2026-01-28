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
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Conv2d expects 4D input (batch_size, channels, height, width), got {}D: {:?}",
                input_shape.len(),
                input_shape
            )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    // ========================================================================
    // Conv1d Tests
    // ========================================================================

    #[test]
    fn test_conv1d_new() {
        let conv = Conv1d::new(3, 16, 3, 1, 0, 1, true, 1);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 0);
        assert_eq!(conv.dilation, 1);
        assert_eq!(conv.groups, 1);
        assert!(conv.use_bias);
    }

    #[test]
    fn test_conv1d_with_defaults() {
        let conv = Conv1d::with_defaults(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 0);
    }

    #[test]
    fn test_conv1d_forward() -> Result<()> {
        let conv = Conv1d::with_defaults(3, 16, 3);
        let input = zeros(&[2, 3, 32])?; // batch=2, channels=3, length=32

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (32 - 3) / 1 + 1 = 30
        assert_eq!(output_shape.dims(), &[2, 16, 30]);
        Ok(())
    }

    #[test]
    fn test_conv1d_forward_with_stride() -> Result<()> {
        let conv = Conv1d::new(1, 1, 3, 2, 0, 1, false, 1);
        let input = zeros(&[1, 1, 10])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (10 - 3) / 2 + 1 = 4
        assert_eq!(output_shape.dims(), &[1, 1, 4]);
        Ok(())
    }

    #[test]
    fn test_conv1d_forward_with_padding() -> Result<()> {
        let conv = Conv1d::new(1, 1, 3, 1, 1, 1, false, 1);
        let input = zeros(&[1, 1, 10])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (10 + 2*1 - 3) / 1 + 1 = 10
        assert_eq!(output_shape.dims(), &[1, 1, 10]);
        Ok(())
    }

    #[test]
    fn test_conv1d_parameters() {
        let conv = Conv1d::with_defaults(3, 16, 3);
        let params = conv.parameters();

        // Should have weight and bias
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    #[test]
    fn test_conv1d_parameters_no_bias() {
        let conv = Conv1d::new(3, 16, 3, 1, 0, 1, false, 1);
        let params = conv.parameters();

        // Should only have weight (no bias)
        assert_eq!(params.len(), 1);
        assert!(params.contains_key("weight"));
        assert!(!params.contains_key("bias"));
    }

    #[test]
    fn test_conv1d_training_mode() {
        let mut conv = Conv1d::with_defaults(3, 16, 3);
        assert!(conv.training());

        conv.eval();
        assert!(!conv.training());

        conv.train();
        assert!(conv.training());
    }

    // ========================================================================
    // Conv2d Tests
    // ========================================================================

    #[test]
    fn test_conv2d_new() {
        let conv = Conv2d::new(3, 16, (3, 3), (1, 1), (0, 0), (1, 1), true, 1);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3));
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (0, 0));
        assert_eq!(conv.dilation, (1, 1));
        assert_eq!(conv.groups, 1);
    }

    #[test]
    fn test_conv2d_with_defaults() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3));
    }

    #[test]
    fn test_conv2d_forward() -> Result<()> {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let input = zeros(&[2, 3, 32, 32])?; // batch=2, channels=3, height=32, width=32

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (32 - 3) / 1 + 1 = 30
        assert_eq!(output_shape.dims(), &[2, 16, 30, 30]);
        Ok(())
    }

    #[test]
    fn test_conv2d_forward_with_stride() -> Result<()> {
        let conv = Conv2d::new(1, 1, (3, 3), (2, 2), (0, 0), (1, 1), false, 1);
        let input = zeros(&[1, 1, 8, 8])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (8 - 3) / 2 + 1 = 3
        assert_eq!(output_shape.dims(), &[1, 1, 3, 3]);
        Ok(())
    }

    #[test]
    fn test_conv2d_forward_with_padding() -> Result<()> {
        let conv = Conv2d::new(1, 1, (3, 3), (1, 1), (1, 1), (1, 1), false, 1);
        let input = zeros(&[1, 1, 8, 8])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (8 + 2*1 - 3) / 1 + 1 = 8 (same size with padding=1)
        assert_eq!(output_shape.dims(), &[1, 1, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_conv2d_forward_invalid_input() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let input = zeros(&[2, 3, 32]).unwrap(); // 3D input - should fail

        let result = conv.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let params = conv.parameters();

        // Should have weight and bias
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // ========================================================================
    // Conv3d Tests
    // ========================================================================

    #[test]
    fn test_conv3d_new() {
        let conv = Conv3d::new(3, 16, (3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1), true, 1);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3, 3));
        assert_eq!(conv.stride, (1, 1, 1));
        assert_eq!(conv.padding, (0, 0, 0));
    }

    #[test]
    fn test_conv3d_with_defaults() {
        let conv = Conv3d::with_defaults(3, 16, 3);
        assert_eq!(conv.in_channels, 3);
        assert_eq!(conv.out_channels, 16);
        assert_eq!(conv.kernel_size, (3, 3, 3));
    }

    #[test]
    fn test_conv3d_forward() -> Result<()> {
        let conv = Conv3d::with_defaults(1, 8, 3);
        let input = zeros(&[1, 1, 16, 16, 16])?; // batch, channels, depth, height, width

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (16 - 3) / 1 + 1 = 14
        assert_eq!(output_shape.dims(), &[1, 8, 14, 14, 14]);
        Ok(())
    }

    #[test]
    fn test_conv3d_forward_with_stride() -> Result<()> {
        let conv = Conv3d::new(1, 1, (3, 3, 3), (2, 2, 2), (0, 0, 0), (1, 1, 1), false, 1);
        let input = zeros(&[1, 1, 8, 8, 8])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (8 - 3) / 2 + 1 = 3
        assert_eq!(output_shape.dims(), &[1, 1, 3, 3, 3]);
        Ok(())
    }

    #[test]
    fn test_conv3d_parameters() {
        let conv = Conv3d::with_defaults(1, 8, 3);
        let params = conv.parameters();

        // Should have weight and bias
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // ========================================================================
    // ConvTranspose1d Tests
    // ========================================================================

    #[test]
    fn test_convtranspose1d_new() {
        let conv = ConvTranspose1d::new(16, 3, 3, 1, 0, 0, 1, true, 1);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 0);
        assert_eq!(conv.output_padding, 0);
    }

    #[test]
    fn test_convtranspose1d_with_defaults() {
        let conv = ConvTranspose1d::with_defaults(16, 3, 3);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, 3);
    }

    #[test]
    fn test_convtranspose1d_forward() -> Result<()> {
        let conv = ConvTranspose1d::with_defaults(16, 8, 4);
        let input = zeros(&[2, 16, 16])?; // batch=2, channels=16, length=16

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (16 - 1) * 1 - 2 * 0 + 1 * (4 - 1) + 0 + 1 = 15 + 3 + 1 = 19
        assert_eq!(output_shape.dims(), &[2, 8, 19]);
        Ok(())
    }

    #[test]
    fn test_convtranspose1d_forward_with_stride() -> Result<()> {
        let conv = ConvTranspose1d::new(1, 1, 4, 2, 1, 0, 1, false, 1);
        let input = zeros(&[1, 1, 8])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (8 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 0 + 1 = 14 - 2 + 3 + 1 = 16
        assert_eq!(output_shape.dims(), &[1, 1, 16]);
        Ok(())
    }

    #[test]
    fn test_convtranspose1d_parameters() {
        let conv = ConvTranspose1d::with_defaults(16, 3, 3);
        let params = conv.parameters();

        // Should have weight and bias
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // ========================================================================
    // ConvTranspose2d Tests
    // ========================================================================

    #[test]
    fn test_convtranspose2d_new() {
        let conv = ConvTranspose2d::new(16, 3, (3, 3), (1, 1), (0, 0), (0, 0), (1, 1), true, 1);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3));
        assert_eq!(conv.stride, (1, 1));
        assert_eq!(conv.padding, (0, 0));
        assert_eq!(conv.output_padding, (0, 0));
    }

    #[test]
    fn test_convtranspose2d_with_defaults() {
        let conv = ConvTranspose2d::with_defaults(16, 3, 3);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3));
    }

    #[test]
    fn test_convtranspose2d_forward() -> Result<()> {
        let conv = ConvTranspose2d::with_defaults(16, 8, 4);
        let input = zeros(&[2, 16, 8, 8])?; // batch=2, channels=16, height=8, width=8

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (8 - 1) * 1 - 2 * 0 + 1 * (4 - 1) + 0 + 1 = 7 + 3 + 1 = 11
        assert_eq!(output_shape.dims(), &[2, 8, 11, 11]);
        Ok(())
    }

    #[test]
    fn test_convtranspose2d_forward_with_stride() -> Result<()> {
        let conv = ConvTranspose2d::new(1, 1, (4, 4), (2, 2), (1, 1), (0, 0), (1, 1), false, 1);
        let input = zeros(&[1, 1, 4, 4])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (4 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 0 + 1 = 6 - 2 + 3 + 1 = 8
        assert_eq!(output_shape.dims(), &[1, 1, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_convtranspose2d_parameters() {
        let conv = ConvTranspose2d::with_defaults(16, 3, 3);
        let params = conv.parameters();

        // Should have weight and bias
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // ========================================================================
    // ConvTranspose3d Tests
    // ========================================================================

    #[test]
    fn test_convtranspose3d_new() {
        let conv = ConvTranspose3d::new(
            16,
            3,
            (3, 3, 3),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
            true,
            1,
        );
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3, 3));
        assert_eq!(conv.stride, (1, 1, 1));
        assert_eq!(conv.padding, (0, 0, 0));
        assert_eq!(conv.output_padding, (0, 0, 0));
    }

    #[test]
    fn test_convtranspose3d_with_defaults() {
        let conv = ConvTranspose3d::with_defaults(16, 3, 3);
        assert_eq!(conv.in_channels, 16);
        assert_eq!(conv.out_channels, 3);
        assert_eq!(conv.kernel_size, (3, 3, 3));
    }

    #[test]
    fn test_convtranspose3d_forward() -> Result<()> {
        let conv = ConvTranspose3d::with_defaults(8, 4, 4);
        let input = zeros(&[1, 8, 4, 4, 4])?; // batch, channels, depth, height, width

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (4 - 1) * 1 - 2 * 0 + 1 * (4 - 1) + 0 + 1 = 3 + 3 + 1 = 7
        assert_eq!(output_shape.dims(), &[1, 4, 7, 7, 7]);
        Ok(())
    }

    #[test]
    fn test_convtranspose3d_forward_with_stride() -> Result<()> {
        let conv = ConvTranspose3d::new(
            1,
            1,
            (4, 4, 4),
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            false,
            1,
        );
        let input = zeros(&[1, 1, 2, 2, 2])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // Expected: (2 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 0 + 1 = 2 - 2 + 3 + 1 = 4
        assert_eq!(output_shape.dims(), &[1, 1, 4, 4, 4]);
        Ok(())
    }

    #[test]
    fn test_convtranspose3d_parameters() {
        let conv = ConvTranspose3d::with_defaults(16, 3, 3);
        let params = conv.parameters();

        // Should have weight and bias
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // ========================================================================
    // Module Trait Tests (Common Behaviors)
    // ========================================================================

    #[test]
    fn test_module_training_modes() {
        let mut conv = Conv2d::with_defaults(3, 16, 3);

        // Default should be training mode
        assert!(conv.training());

        // Set to eval mode
        conv.set_training(false);
        assert!(!conv.training());

        // Set back to training mode
        conv.set_training(true);
        assert!(conv.training());
    }

    #[test]
    fn test_module_named_parameters() {
        let conv = Conv2d::with_defaults(3, 16, 3);
        let named_params = conv.named_parameters();

        // Should have weight and bias
        assert_eq!(named_params.len(), 2);
        assert!(named_params.contains_key("weight"));
        assert!(named_params.contains_key("bias"));
    }

    #[test]
    fn test_module_to_device() -> Result<()> {
        let mut conv = Conv2d::with_defaults(3, 16, 3);

        // Should succeed
        conv.to_device(DeviceType::Cpu)?;

        Ok(())
    }
}
