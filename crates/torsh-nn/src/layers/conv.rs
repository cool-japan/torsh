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
    depthwise: Conv2d,
    pointwise: Conv2d,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
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
        // Depthwise convolution: one filter per input channel (groups = in_channels)
        let depthwise = Conv2d::new(
            in_channels,
            in_channels, // out_channels = in_channels for depthwise
            (kernel_size, kernel_size),
            (stride, stride),
            (padding, padding),
            (1, 1), // dilation
            bias,
            in_channels, // groups = in_channels (key to depthwise convolution)
        );

        // Pointwise convolution: 1x1 convolution to combine channels
        let pointwise = Conv2d::new(
            in_channels,
            out_channels,
            (1, 1), // 1x1 kernel
            (1, 1), // stride = 1
            (0, 0), // no padding needed for 1x1
            (1, 1), // dilation
            bias,
            1, // standard convolution (no grouping)
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
        // Standard Conv2d: out_channels × in_channels × kernel_size × kernel_size
        let standard_params =
            self.out_channels * self.in_channels * self.kernel_size * self.kernel_size;

        // Depthwise separable: in_channels × kernel_size × kernel_size + out_channels × in_channels × 1 × 1
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

impl Module for DepthwiseSeparableConv {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply depthwise convolution
        let depthwise_out = self.depthwise.forward(input)?;

        // Apply pointwise convolution
        let output = self.pointwise.forward(&depthwise_out)?;

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Add depthwise parameters with "depthwise." prefix
        for (name, param) in self.depthwise.parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }

        // Add pointwise parameters with "pointwise." prefix
        for (name, param) in self.pointwise.parameters() {
            params.insert(format!("pointwise.{}", name), param);
        }

        params
    }

    fn training(&self) -> bool {
        self.depthwise.training()
    }

    fn train(&mut self) {
        self.depthwise.train();
        self.pointwise.train();
    }

    fn eval(&mut self) {
        self.depthwise.eval();
        self.pointwise.eval();
    }

    fn set_training(&mut self, training: bool) {
        self.depthwise.set_training(training);
        self.pointwise.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.depthwise.to_device(device)?;
        self.pointwise.to_device(device)?;
        Ok(())
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Add depthwise parameters with "depthwise." prefix
        for (name, param) in self.depthwise.named_parameters() {
            params.insert(format!("depthwise.{}", name), param);
        }

        // Add pointwise parameters with "pointwise." prefix
        for (name, param) in self.pointwise.named_parameters() {
            params.insert(format!("pointwise.{}", name), param);
        }

        params
    }
}

impl std::fmt::Debug for DepthwiseSeparableConv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthwiseSeparableConv")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
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

    // ========================================================================
    // DepthwiseSeparableConv Tests
    // ========================================================================

    #[test]
    fn test_depthwise_separable_conv_new() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        assert_eq!(conv.in_channels(), 32);
        assert_eq!(conv.out_channels(), 64);
        assert_eq!(conv.kernel_size(), 3);
        assert_eq!(conv.stride(), 1);
        assert_eq!(conv.padding(), 1);
    }

    #[test]
    fn test_depthwise_separable_conv_with_defaults() {
        let conv = DepthwiseSeparableConv::with_defaults(16, 32, 3);
        assert_eq!(conv.in_channels(), 16);
        assert_eq!(conv.out_channels(), 32);
        assert_eq!(conv.kernel_size(), 3);
        assert_eq!(conv.stride(), 1);
        assert_eq!(conv.padding(), 1);
    }

    #[test]
    fn test_depthwise_separable_conv_forward() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        let input = zeros(&[2, 32, 8, 8])?; // batch=2, channels=32, height=8, width=8

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // With padding=1, kernel=3, stride=1: output size = (8 + 2*1 - 3) / 1 + 1 = 8
        assert_eq!(output_shape.dims(), &[2, 64, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_kernel_size_3x3() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, false);
        let input = zeros(&[1, 16, 16, 16])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // With padding=1, kernel=3, stride=1: maintains spatial dimensions
        assert_eq!(output_shape.dims(), &[1, 32, 16, 16]);
        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_kernel_size_5x5() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(24, 48, 5, 1, 2, true);
        let input = zeros(&[1, 24, 32, 32])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // With padding=2, kernel=5, stride=1: maintains spatial dimensions
        assert_eq!(output_shape.dims(), &[1, 48, 32, 32]);
        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_stride_2() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 2, 1, true);
        let input = zeros(&[1, 32, 16, 16])?;

        let output = conv.forward(&input)?;
        let output_shape = output.shape();

        // With padding=1, kernel=3, stride=2: (16 + 2*1 - 3) / 2 + 1 = 8
        assert_eq!(output_shape.dims(), &[1, 64, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_with_bias() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, true);
        let params = conv.parameters();

        // Should have 4 parameters: depthwise.weight, depthwise.bias, pointwise.weight, pointwise.bias
        assert_eq!(params.len(), 4);
        assert!(params.contains_key("depthwise.weight"));
        assert!(params.contains_key("depthwise.bias"));
        assert!(params.contains_key("pointwise.weight"));
        assert!(params.contains_key("pointwise.bias"));

        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_without_bias() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, false);
        let params = conv.parameters();

        // Should have 2 parameters: depthwise.weight, pointwise.weight
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("depthwise.weight"));
        assert!(params.contains_key("pointwise.weight"));
        assert!(!params.contains_key("depthwise.bias"));
        assert!(!params.contains_key("pointwise.bias"));

        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_parameter_reduction() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);

        // Standard Conv2d: 64 × 32 × 3 × 3 = 18,432 parameters (weights only)
        // Depthwise: 32 × 3 × 3 = 288 parameters
        // Pointwise: 64 × 32 × 1 × 1 = 2,048 parameters
        // Total: 288 + 2,048 = 2,336 parameters
        // Reduction: 2,336 / 18,432 ≈ 0.127 (about 8x fewer parameters)

        let ratio = conv.parameter_reduction_ratio();
        assert!(
            ratio > 0.1 && ratio < 0.15,
            "Expected ratio around 0.127, got {}",
            ratio
        );
    }

    #[test]
    fn test_depthwise_separable_conv_multiple_channels() -> Result<()> {
        // Test with various channel configurations
        let configs = vec![(16, 32, 3), (32, 64, 3), (64, 128, 3), (128, 256, 3)];

        for (in_ch, out_ch, kernel) in configs {
            let conv = DepthwiseSeparableConv::new(in_ch, out_ch, kernel, 1, 1, true);
            let input = zeros(&[1, in_ch, 8, 8])?;

            let output = conv.forward(&input)?;
            let output_shape = output.shape();

            assert_eq!(output_shape.dims(), &[1, out_ch, 8, 8]);
        }

        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_batch_processing() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(24, 48, 3, 1, 1, true);

        // Test with various batch sizes
        for batch_size in [1, 2, 4, 8] {
            let input = zeros(&[batch_size, 24, 16, 16])?;
            let output = conv.forward(&input)?;
            let output_shape = output.shape();

            assert_eq!(output_shape.dims(), &[batch_size, 48, 16, 16]);
        }

        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_training_mode() {
        let mut conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);

        // Default should be training mode
        assert!(conv.training());

        // Set to eval mode
        conv.eval();
        assert!(!conv.training());

        // Set back to training mode
        conv.train();
        assert!(conv.training());

        // Test set_training
        conv.set_training(false);
        assert!(!conv.training());
    }

    #[test]
    fn test_depthwise_separable_conv_module_trait() -> Result<()> {
        let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1, true);

        // Test parameters()
        let params = conv.parameters();
        assert_eq!(params.len(), 4); // depthwise weight+bias, pointwise weight+bias

        // Test named_parameters()
        let named_params = conv.named_parameters();
        assert_eq!(named_params.len(), 4);
        assert!(named_params.contains_key("depthwise.weight"));
        assert!(named_params.contains_key("pointwise.weight"));

        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_to_device() -> Result<()> {
        let mut conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);

        // Should succeed
        conv.to_device(DeviceType::Cpu)?;

        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_shape_preservation() -> Result<()> {
        // Test that with proper padding, spatial dimensions are preserved
        let test_cases = vec![
            (3, 1), // kernel=3, padding=1
            (5, 2), // kernel=5, padding=2
        ];

        for (kernel, padding) in test_cases {
            let conv = DepthwiseSeparableConv::new(16, 32, kernel, 1, padding, true);
            let input = zeros(&[2, 16, 32, 32])?;

            let output = conv.forward(&input)?;
            let output_shape = output.shape();

            // Spatial dimensions should be preserved
            assert_eq!(output_shape.dims()[2], 32);
            assert_eq!(output_shape.dims()[3], 32);
        }

        Ok(())
    }

    #[test]
    fn test_depthwise_separable_conv_param_count() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        let (dw_params, pw_params, total_params) = conv.param_count();

        // Depthwise should have 2 params (weight + bias)
        assert_eq!(dw_params, 2);

        // Pointwise should have 2 params (weight + bias)
        assert_eq!(pw_params, 2);

        // Total should be 4
        assert_eq!(total_params, 4);
    }

    #[test]
    fn test_depthwise_separable_conv_debug_format() {
        let conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1, true);
        let debug_str = format!("{:?}", conv);

        // Should contain key information
        assert!(debug_str.contains("DepthwiseSeparableConv"));
        assert!(debug_str.contains("in_channels"));
        assert!(debug_str.contains("out_channels"));
        assert!(debug_str.contains("kernel_size"));
    }
}
