//! Convolution and signal processing operations for tensors

use crate::{FloatElement, Tensor};
use torsh_core::error::{Result, TorshError};
use torsh_core::TensorElement;

impl<T: FloatElement> Tensor<T> {
    /// 1D convolution operation
    pub fn conv1d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        // Input shape: (N, C_in, L)
        // Weight shape: (C_out, C_in/groups, kernel_size)
        // Output shape: (N, C_out, L_out)

        let input_shape_obj = self.shape();
        let input_shape = input_shape_obj.dims();
        let weight_shape_obj = weight.shape();
        let weight_shape = weight_shape_obj.dims();

        if input_shape.len() != 3 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 3D input tensor for conv1d, got {}D",
                input_shape.len()
            )));
        }

        if weight_shape.len() != 3 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 3D weight tensor for conv1d, got {}D",
                weight_shape.len()
            )));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_length = input_shape[2];

        let out_channels = weight_shape[0];
        let kernel_size = weight_shape[2];

        // Check groups
        if in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(TorshError::InvalidArgument(
                "in_channels and out_channels must be divisible by groups".to_string(),
            ));
        }

        if weight_shape[1] != in_channels / groups {
            return Err(TorshError::InvalidArgument(format!(
                "Weight tensor has wrong number of input channels: expected {}, got {}",
                in_channels / groups,
                weight_shape[1]
            )));
        }

        // Calculate output length
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let padded_length = input_length + 2 * padding;
        let output_length = (padded_length - effective_kernel) / stride + 1;

        // Initialize output
        let mut output_data =
            vec![<T as TensorElement>::zero(); batch_size * out_channels * output_length];

        // Perform convolution
        for n in 0..batch_size {
            for g in 0..groups {
                let out_ch_start = g * (out_channels / groups);
                let out_ch_end = (g + 1) * (out_channels / groups);
                let in_ch_start = g * (in_channels / groups);
                let in_ch_end = (g + 1) * (in_channels / groups);

                for oc in out_ch_start..out_ch_end {
                    for ol in 0..output_length {
                        let mut sum = <T as TensorElement>::zero();

                        for ic in in_ch_start..in_ch_end {
                            let ic_rel = ic - in_ch_start;
                            for k in 0..kernel_size {
                                let il = (ol * stride + k * dilation) as i32 - padding as i32;

                                if il >= 0 && (il as usize) < input_length {
                                    let input_idx = n * in_channels * input_length
                                        + ic * input_length
                                        + il as usize;
                                    let weight_idx = oc * (in_channels / groups) * kernel_size
                                        + ic_rel * kernel_size
                                        + k;

                                    let input_val = self.storage.get(input_idx)?;
                                    let weight_val = weight.storage.get(weight_idx)?;
                                    sum = sum + input_val * weight_val;
                                }
                            }
                        }

                        let output_idx = n * out_channels * output_length + oc * output_length + ol;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        // Create output tensor
        let mut output = Tensor::from_data(
            output_data,
            vec![batch_size, out_channels, output_length],
            self.device(),
        )?;

        // Add bias if provided
        if let Some(b) = bias {
            if b.shape().dims() != [out_channels] {
                return Err(TorshError::InvalidArgument(format!(
                    "Bias must have shape [{}], got {:?}",
                    out_channels,
                    b.shape().dims()
                )));
            }

            // For now, use element-wise addition - TODO: implement efficient broadcasting
            let bias_data = b.to_vec()?;
            let mut output_data = output.to_vec()?;

            for n in 0..batch_size {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..out_channels {
                    for ol in 0..output_length {
                        let idx = n * out_channels * output_length + oc * output_length + ol;
                        output_data[idx] = output_data[idx] + bias_data[oc];
                    }
                }
            }

            // Recreate tensor with modified data
            output = Tensor::from_data(
                output_data,
                vec![batch_size, out_channels, output_length],
                self.device(),
            )?;
        }

        // Track operation for autograd
        if self.requires_grad
            || weight.requires_grad
            || (bias.is_some() && bias.unwrap().requires_grad)
        {
            use std::sync::Arc;
            output.requires_grad = true;
            output.operation = crate::Operation::Custom(
                "conv1d".to_string(),
                vec![
                    Arc::downgrade(&Arc::new(self.clone())),
                    Arc::downgrade(&Arc::new(weight.clone())),
                ],
            );
        }

        Ok(output)
    }

    /// 2D convolution operation
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        // Input shape: (N, C_in, H, W)
        // Weight shape: (C_out, C_in/groups, kernel_h, kernel_w)
        // Output shape: (N, C_out, H_out, W_out)

        let input_shape_obj = self.shape();
        let input_shape = input_shape_obj.dims();
        let weight_shape_obj = weight.shape();
        let weight_shape = weight_shape_obj.dims();

        if input_shape.len() != 4 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 4D input tensor for conv2d, got {}D",
                input_shape.len()
            )));
        }

        if weight_shape.len() != 4 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 4D weight tensor for conv2d, got {}D",
                weight_shape.len()
            )));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let out_channels = weight_shape[0];
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        // Check groups
        if in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(TorshError::InvalidArgument(
                "in_channels and out_channels must be divisible by groups".to_string(),
            ));
        }

        if weight_shape[1] != in_channels / groups {
            return Err(TorshError::InvalidArgument(format!(
                "Weight tensor has wrong number of input channels: expected {}, got {}",
                in_channels / groups,
                weight_shape[1]
            )));
        }

        // Calculate output dimensions
        let effective_kernel_h = (kernel_height - 1) * dilation.0 + 1;
        let effective_kernel_w = (kernel_width - 1) * dilation.1 + 1;
        let padded_height = input_height + 2 * padding.0;
        let padded_width = input_width + 2 * padding.1;
        let output_height = (padded_height - effective_kernel_h) / stride.0 + 1;
        let output_width = (padded_width - effective_kernel_w) / stride.1 + 1;

        // Initialize output
        let mut output_data = vec![
            <T as TensorElement>::zero();
            batch_size * out_channels * output_height * output_width
        ];

        let self_data = self.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Perform convolution
        for n in 0..batch_size {
            for g in 0..groups {
                let out_ch_start = g * (out_channels / groups);
                let out_ch_end = (g + 1) * (out_channels / groups);
                let in_ch_start = g * (in_channels / groups);
                let in_ch_end = (g + 1) * (in_channels / groups);

                for oc in out_ch_start..out_ch_end {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let mut sum = <T as TensorElement>::zero();

                            for ic in in_ch_start..in_ch_end {
                                let ic_rel = ic - in_ch_start;
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let ih = (oh * stride.0 + kh * dilation.0) as i32
                                            - padding.0 as i32;
                                        let iw = (ow * stride.1 + kw * dilation.1) as i32
                                            - padding.1 as i32;

                                        if ih >= 0
                                            && (ih as usize) < input_height
                                            && iw >= 0
                                            && (iw as usize) < input_width
                                        {
                                            let input_idx =
                                                n * in_channels * input_height * input_width
                                                    + ic * input_height * input_width
                                                    + ih as usize * input_width
                                                    + iw as usize;
                                            let weight_idx = oc
                                                * (in_channels / groups)
                                                * kernel_height
                                                * kernel_width
                                                + ic_rel * kernel_height * kernel_width
                                                + kh * kernel_width
                                                + kw;

                                            sum = sum
                                                + self_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }

                            let output_idx = n * out_channels * output_height * output_width
                                + oc * output_height * output_width
                                + oh * output_width
                                + ow;
                            output_data[output_idx] = sum;
                        }
                    }
                }
            }
        }

        // Create output tensor
        let mut output = Tensor::from_data(
            output_data,
            vec![batch_size, out_channels, output_height, output_width],
            self.device(),
        )?;

        // Add bias if provided
        if let Some(b) = bias {
            if b.shape().dims() != [out_channels] {
                return Err(TorshError::InvalidArgument(format!(
                    "Bias must have shape [{}], got {:?}",
                    out_channels,
                    b.shape().dims()
                )));
            }

            let bias_data = b.to_vec()?;

            let mut output_data = output.to_vec()?;

            for n in 0..batch_size {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..out_channels {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let idx = n * out_channels * output_height * output_width
                                + oc * output_height * output_width
                                + oh * output_width
                                + ow;
                            output_data[idx] = output_data[idx] + bias_data[oc];
                        }
                    }
                }
            }

            // Create new output tensor with bias added
            output = Tensor::from_data(
                output_data,
                vec![batch_size, out_channels, output_height, output_width],
                self.device(),
            )?;
        }

        // Track operation for autograd
        if self.requires_grad
            || weight.requires_grad
            || (bias.is_some() && bias.unwrap().requires_grad)
        {
            use std::sync::Arc;
            output.requires_grad = true;
            output.operation = crate::Operation::Custom(
                "conv2d".to_string(),
                vec![
                    Arc::downgrade(&Arc::new(self.clone())),
                    Arc::downgrade(&Arc::new(weight.clone())),
                ],
            );
        }

        Ok(output)
    }

    /// 3D convolution operation
    pub fn conv3d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Result<Self> {
        // Input shape: (N, C_in, D, H, W)
        // Weight shape: (C_out, C_in/groups, kernel_d, kernel_h, kernel_w)
        // Output shape: (N, C_out, D_out, H_out, W_out)

        let input_shape_obj = self.shape();
        let input_shape = input_shape_obj.dims();
        let weight_shape_obj = weight.shape();
        let weight_shape = weight_shape_obj.dims();

        if input_shape.len() != 5 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 5D input tensor for conv3d, got {}D",
                input_shape.len()
            )));
        }

        if weight_shape.len() != 5 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 5D weight tensor for conv3d, got {}D",
                weight_shape.len()
            )));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_depth = input_shape[2];
        let input_height = input_shape[3];
        let input_width = input_shape[4];

        let out_channels = weight_shape[0];
        let kernel_depth = weight_shape[2];
        let kernel_height = weight_shape[3];
        let kernel_width = weight_shape[4];

        // Check groups
        if in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(TorshError::InvalidArgument(
                "in_channels and out_channels must be divisible by groups".to_string(),
            ));
        }

        if weight_shape[1] != in_channels / groups {
            return Err(TorshError::InvalidArgument(format!(
                "Weight tensor has wrong number of input channels: expected {}, got {}",
                in_channels / groups,
                weight_shape[1]
            )));
        }

        // Calculate output dimensions
        let effective_kernel_d = (kernel_depth - 1) * dilation.0 + 1;
        let effective_kernel_h = (kernel_height - 1) * dilation.1 + 1;
        let effective_kernel_w = (kernel_width - 1) * dilation.2 + 1;
        let padded_depth = input_depth + 2 * padding.0;
        let padded_height = input_height + 2 * padding.1;
        let padded_width = input_width + 2 * padding.2;
        let output_depth = (padded_depth - effective_kernel_d) / stride.0 + 1;
        let output_height = (padded_height - effective_kernel_h) / stride.1 + 1;
        let output_width = (padded_width - effective_kernel_w) / stride.2 + 1;

        // Initialize output
        let output_size = batch_size * out_channels * output_depth * output_height * output_width;
        let mut output_data = vec![<T as TensorElement>::zero(); output_size];

        let self_data = self.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Perform convolution
        for n in 0..batch_size {
            for g in 0..groups {
                let out_ch_start = g * (out_channels / groups);
                let out_ch_end = (g + 1) * (out_channels / groups);
                let in_ch_start = g * (in_channels / groups);
                let in_ch_end = (g + 1) * (in_channels / groups);

                for oc in out_ch_start..out_ch_end {
                    for od in 0..output_depth {
                        for oh in 0..output_height {
                            for ow in 0..output_width {
                                let mut sum = <T as TensorElement>::zero();

                                for ic in in_ch_start..in_ch_end {
                                    let ic_rel = ic - in_ch_start;
                                    for kd in 0..kernel_depth {
                                        for kh in 0..kernel_height {
                                            for kw in 0..kernel_width {
                                                let id = (od * stride.0 + kd * dilation.0) as i32
                                                    - padding.0 as i32;
                                                let ih = (oh * stride.1 + kh * dilation.1) as i32
                                                    - padding.1 as i32;
                                                let iw = (ow * stride.2 + kw * dilation.2) as i32
                                                    - padding.2 as i32;

                                                if id >= 0
                                                    && (id as usize) < input_depth
                                                    && ih >= 0
                                                    && (ih as usize) < input_height
                                                    && iw >= 0
                                                    && (iw as usize) < input_width
                                                {
                                                    let input_idx = n
                                                        * in_channels
                                                        * input_depth
                                                        * input_height
                                                        * input_width
                                                        + ic * input_depth
                                                            * input_height
                                                            * input_width
                                                        + id as usize * input_height * input_width
                                                        + ih as usize * input_width
                                                        + iw as usize;
                                                    let weight_idx = oc
                                                        * (in_channels / groups)
                                                        * kernel_depth
                                                        * kernel_height
                                                        * kernel_width
                                                        + ic_rel
                                                            * kernel_depth
                                                            * kernel_height
                                                            * kernel_width
                                                        + kd * kernel_height * kernel_width
                                                        + kh * kernel_width
                                                        + kw;

                                                    sum = sum
                                                        + self_data[input_idx]
                                                            * weight_data[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }

                                let output_idx =
                                    n * out_channels * output_depth * output_height * output_width
                                        + oc * output_depth * output_height * output_width
                                        + od * output_height * output_width
                                        + oh * output_width
                                        + ow;
                                output_data[output_idx] = sum;
                            }
                        }
                    }
                }
            }
        }

        // Create output tensor
        let mut output = Tensor::from_data(
            output_data,
            vec![
                batch_size,
                out_channels,
                output_depth,
                output_height,
                output_width,
            ],
            self.device(),
        )?;

        // Add bias if provided
        if let Some(b) = bias {
            if b.shape().dims() != [out_channels] {
                return Err(TorshError::InvalidArgument(format!(
                    "Bias must have shape [{}], got {:?}",
                    out_channels,
                    b.shape().dims()
                )));
            }

            let bias_data = b.to_vec()?;

            let mut output_data = output.to_vec()?;

            for n in 0..batch_size {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..out_channels {
                    for od in 0..output_depth {
                        for oh in 0..output_height {
                            for ow in 0..output_width {
                                let idx =
                                    n * out_channels * output_depth * output_height * output_width
                                        + oc * output_depth * output_height * output_width
                                        + od * output_height * output_width
                                        + oh * output_width
                                        + ow;
                                output_data[idx] = output_data[idx] + bias_data[oc];
                            }
                        }
                    }
                }
            }

            // Create new output tensor with bias added
            output = Tensor::from_data(
                output_data,
                vec![
                    batch_size,
                    out_channels,
                    output_depth,
                    output_height,
                    output_width,
                ],
                self.device(),
            )?;
        }

        // Track operation for autograd
        if self.requires_grad
            || weight.requires_grad
            || (bias.is_some() && bias.unwrap().requires_grad)
        {
            use std::sync::Arc;
            output.requires_grad = true;
            output.operation = crate::Operation::Custom(
                "conv3d".to_string(),
                vec![
                    Arc::downgrade(&Arc::new(self.clone())),
                    Arc::downgrade(&Arc::new(weight.clone())),
                ],
            );
        }

        Ok(output)
    }

    /// Depthwise 2D convolution operation
    /// Each input channel is convolved with its own kernel independently
    pub fn depthwise_conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Self> {
        // Input shape: (N, C_in, H, W)
        // Weight shape: (C_in, 1, kernel_h, kernel_w) - each channel has its own kernel
        // Output shape: (N, C_in, H_out, W_out)

        let input_shape_obj = self.shape();
        let input_shape = input_shape_obj.dims();
        let weight_shape_obj = weight.shape();
        let weight_shape = weight_shape_obj.dims();

        if input_shape.len() != 4 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 4D input tensor for depthwise_conv2d, got {}D",
                input_shape.len()
            )));
        }

        if weight_shape.len() != 4 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 4D weight tensor for depthwise_conv2d, got {}D",
                weight_shape.len()
            )));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        // For depthwise conv, weight should have shape (C_in, 1, kernel_h, kernel_w)
        if weight_shape[0] != in_channels || weight_shape[1] != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Weight tensor must have shape ({}, 1, kernel_h, kernel_w), got ({}, {}, {}, {})",
                in_channels, weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
            )));
        }

        // Calculate output dimensions
        let effective_kernel_h = (kernel_height - 1) * dilation.0 + 1;
        let effective_kernel_w = (kernel_width - 1) * dilation.1 + 1;
        let padded_height = input_height + 2 * padding.0;
        let padded_width = input_width + 2 * padding.1;
        let output_height = (padded_height - effective_kernel_h) / stride.0 + 1;
        let output_width = (padded_width - effective_kernel_w) / stride.1 + 1;

        // Initialize output
        let mut output_data = vec![
            <T as TensorElement>::zero();
            batch_size * in_channels * output_height * output_width
        ];

        let _self_data = self.to_vec()?;
        let _weight_data = weight.to_vec()?;

        // Perform depthwise convolution
        for n in 0..batch_size {
            for c in 0..in_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = <T as TensorElement>::zero();

                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih =
                                    (oh * stride.0 + kh * dilation.0) as i32 - padding.0 as i32;
                                let iw =
                                    (ow * stride.1 + kw * dilation.1) as i32 - padding.1 as i32;

                                if ih >= 0
                                    && (ih as usize) < input_height
                                    && iw >= 0
                                    && (iw as usize) < input_width
                                {
                                    let input_idx = n * in_channels * input_height * input_width
                                        + c * input_height * input_width
                                        + ih as usize * input_width
                                        + iw as usize;
                                    let weight_idx =
                                        c * kernel_height * kernel_width + kh * kernel_width + kw;

                                    let input_val = self.storage.get(input_idx)?;
                                    let weight_val = weight.storage.get(weight_idx)?;
                                    sum = sum + input_val * weight_val;
                                }
                            }
                        }

                        let output_idx = n * in_channels * output_height * output_width
                            + c * output_height * output_width
                            + oh * output_width
                            + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        // Create output tensor
        let mut output = Tensor::from_data(
            output_data,
            vec![batch_size, in_channels, output_height, output_width],
            self.device(),
        )?;

        // Add bias if provided
        if let Some(b) = bias {
            if b.shape().dims() != [in_channels] {
                return Err(TorshError::InvalidArgument(format!(
                    "Bias must have shape [{}], got {:?}",
                    in_channels,
                    b.shape().dims()
                )));
            }

            let bias_data = b.to_vec()?;

            let mut output_data = output.to_vec()?;

            for n in 0..batch_size {
                #[allow(clippy::needless_range_loop)]
                for c in 0..in_channels {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let idx = n * in_channels * output_height * output_width
                                + c * output_height * output_width
                                + oh * output_width
                                + ow;
                            output_data[idx] = output_data[idx] + bias_data[c];
                        }
                    }
                }
            }

            // Create new output tensor with bias added
            output = Tensor::from_data(
                output_data,
                vec![batch_size, in_channels, output_height, output_width],
                self.device(),
            )?;
        }

        // Track operation for autograd
        if self.requires_grad
            || weight.requires_grad
            || (bias.is_some() && bias.unwrap().requires_grad)
        {
            use std::sync::Arc;
            output.requires_grad = true;
            output.operation = crate::Operation::Custom(
                "depthwise_conv2d".to_string(),
                vec![
                    Arc::downgrade(&Arc::new(self.clone())),
                    Arc::downgrade(&Arc::new(weight.clone())),
                ],
            );
        }

        Ok(output)
    }

    /// Separable 2D convolution operation
    /// Factorized into depthwise convolution followed by pointwise (1x1) convolution
    pub fn separable_conv2d(
        &self,
        depthwise_weight: &Self,
        pointwise_weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Self> {
        // Step 1: Depthwise convolution
        let depthwise_output = self.depthwise_conv2d(
            depthwise_weight,
            None, // No bias in depthwise step
            stride,
            padding,
            dilation,
        )?;

        // Step 2: Pointwise (1x1) convolution
        let output = depthwise_output.conv2d(
            pointwise_weight,
            bias,
            (1, 1), // stride = 1 for pointwise
            (0, 0), // padding = 0 for pointwise
            (1, 1), // dilation = 1 for pointwise
            1,      // groups = 1 for pointwise
        )?;

        // Track operation for autograd
        if self.requires_grad
            || depthwise_weight.requires_grad
            || pointwise_weight.requires_grad
            || (bias.is_some() && bias.unwrap().requires_grad)
        {
            use std::sync::Arc;
            let mut tracked_output = output;
            tracked_output.requires_grad = true;
            tracked_output.operation = crate::Operation::Custom(
                "separable_conv2d".to_string(),
                vec![
                    Arc::downgrade(&Arc::new(self.clone())),
                    Arc::downgrade(&Arc::new(depthwise_weight.clone())),
                    Arc::downgrade(&Arc::new(pointwise_weight.clone())),
                ],
            );
            Ok(tracked_output)
        } else {
            Ok(output)
        }
    }

    /// Transposed (deconvolution) 2D convolution operation
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Self> {
        // Input shape: (N, C_in, H, W)
        // Weight shape: (C_in, C_out/groups, kernel_h, kernel_w)
        // Output shape: (N, C_out, H_out, W_out)

        let input_shape_obj = self.shape();
        let input_shape = input_shape_obj.dims();
        let weight_shape_obj = weight.shape();
        let weight_shape = weight_shape_obj.dims();

        if input_shape.len() != 4 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 4D input tensor for conv_transpose2d, got {}D",
                input_shape.len()
            )));
        }

        if weight_shape.len() != 4 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 4D weight tensor for conv_transpose2d, got {}D",
                weight_shape.len()
            )));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let out_channels = weight_shape[1] * groups;
        let kernel_height = weight_shape[2];
        let kernel_width = weight_shape[3];

        // Check groups
        if in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(TorshError::InvalidArgument(
                "in_channels and out_channels must be divisible by groups".to_string(),
            ));
        }

        if weight_shape[0] != in_channels {
            return Err(TorshError::InvalidArgument(format!(
                "Weight tensor has wrong number of input channels: expected {}, got {}",
                in_channels, weight_shape[0]
            )));
        }

        // Calculate output dimensions
        let effective_kernel_h = (kernel_height - 1) * dilation.0 + 1;
        let effective_kernel_w = (kernel_width - 1) * dilation.1 + 1;
        let output_height =
            (input_height - 1) * stride.0 - 2 * padding.0 + effective_kernel_h + output_padding.0;
        let output_width =
            (input_width - 1) * stride.1 - 2 * padding.1 + effective_kernel_w + output_padding.1;

        // Initialize output
        let mut output_data = vec![
            <T as TensorElement>::zero();
            batch_size * out_channels * output_height * output_width
        ];

        let self_data = self.to_vec()?;
        let weight_data = weight.to_vec()?;

        // Perform transposed convolution
        for n in 0..batch_size {
            for g in 0..groups {
                let in_ch_start = g * (in_channels / groups);
                let in_ch_end = (g + 1) * (in_channels / groups);
                let out_ch_start = g * (out_channels / groups);
                let out_ch_end = (g + 1) * (out_channels / groups);

                for ic in in_ch_start..in_ch_end {
                    for ih in 0..input_height {
                        for iw in 0..input_width {
                            let input_val = self_data[n * in_channels * input_height * input_width
                                + ic * input_height * input_width
                                + ih * input_width
                                + iw];

                            for oc in out_ch_start..out_ch_end {
                                let oc_rel = oc - out_ch_start;
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let oh = ih * stride.0 + kh * dilation.0;
                                        let ow = iw * stride.1 + kw * dilation.1;

                                        if oh >= padding.0 && ow >= padding.1 {
                                            let oh_final = oh - padding.0;
                                            let ow_final = ow - padding.1;

                                            if oh_final < output_height && ow_final < output_width {
                                                let weight_idx = ic
                                                    * (out_channels / groups)
                                                    * kernel_height
                                                    * kernel_width
                                                    + oc_rel * kernel_height * kernel_width
                                                    + kh * kernel_width
                                                    + kw;
                                                let output_idx =
                                                    n * out_channels * output_height * output_width
                                                        + oc * output_height * output_width
                                                        + oh_final * output_width
                                                        + ow_final;

                                                output_data[output_idx] = output_data[output_idx]
                                                    + input_val * weight_data[weight_idx];
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

        // Create output tensor
        let mut output = Tensor::from_data(
            output_data,
            vec![batch_size, out_channels, output_height, output_width],
            self.device(),
        )?;

        // Add bias if provided
        if let Some(b) = bias {
            if b.shape().dims() != [out_channels] {
                return Err(TorshError::InvalidArgument(format!(
                    "Bias must have shape [{}], got {:?}",
                    out_channels,
                    b.shape().dims()
                )));
            }

            let bias_data = b.to_vec()?;

            let mut output_data = output.to_vec()?;

            for n in 0..batch_size {
                #[allow(clippy::needless_range_loop)]
                for oc in 0..out_channels {
                    for oh in 0..output_height {
                        for ow in 0..output_width {
                            let idx = n * out_channels * output_height * output_width
                                + oc * output_height * output_width
                                + oh * output_width
                                + ow;
                            output_data[idx] = output_data[idx] + bias_data[oc];
                        }
                    }
                }
            }

            // Create new output tensor with bias added
            output = Tensor::from_data(
                output_data,
                vec![batch_size, out_channels, output_height, output_width],
                self.device(),
            )?;
        }

        // Track operation for autograd
        if self.requires_grad
            || weight.requires_grad
            || (bias.is_some() && bias.unwrap().requires_grad)
        {
            use std::sync::Arc;
            output.requires_grad = true;
            output.operation = crate::Operation::Custom(
                "conv_transpose2d".to_string(),
                vec![
                    Arc::downgrade(&Arc::new(self.clone())),
                    Arc::downgrade(&Arc::new(weight.clone())),
                ],
            );
        }

        Ok(output)
    }

    /// 1D cross-correlation operation
    /// Computes the cross-correlation between two 1D signals
    #[allow(clippy::needless_range_loop)]
    pub fn xcorr1d(&self, other: &Self, mode: CorrelationMode) -> Result<Self> {
        let self_shape_ref = self.shape();
        let other_shape_ref = other.shape();
        let self_shape = self_shape_ref.dims();
        let other_shape = other_shape_ref.dims();

        if self_shape.len() != 1 || other_shape.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "xcorr1d requires 1D tensors".to_string(),
            ));
        }

        let n = self_shape[0];
        let m = other_shape[0];

        let (output_size, lag_start) = match mode {
            CorrelationMode::Full => (n + m - 1, -(m as i32 - 1)),
            CorrelationMode::Valid => {
                if n < m || m < n {
                    return Err(TorshError::InvalidArgument(
                        "Valid mode requires both tensors to have the same size or one to be smaller".to_string(),
                    ));
                }
                (std::cmp::max(n, m) - std::cmp::min(n, m) + 1, 0)
            }
            CorrelationMode::Same => (n, -((m as i32 - 1) / 2)),
        };

        let mut output_data = vec![<T as TensorElement>::zero(); output_size];
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        // Compute cross-correlation: (f ★ g)[lag] = Σ_i f[i] * g[i - lag]
        for i in 0..output_size {
            let mut sum = <T as TensorElement>::zero();
            let lag = lag_start + i as i32;

            for j in 0..n {
                let other_idx = j as i32 - lag;
                if other_idx >= 0 && (other_idx as usize) < m {
                    sum = sum + self_data[j] * other_data[other_idx as usize];
                }
            }
            output_data[i] = sum;
        }

        let output = Tensor::from_data(output_data, vec![output_size], self.device())?;

        Ok(output)
    }

    /// 1D auto-correlation operation
    /// Computes the auto-correlation of a 1D signal
    pub fn autocorr1d(&self, max_lag: Option<usize>) -> Result<Self> {
        let shape_ref = self.shape();
        let shape = shape_ref.dims();
        if shape.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "autocorr1d requires 1D tensor".to_string(),
            ));
        }

        let n = shape[0];
        let max_lag = max_lag.unwrap_or(n - 1).min(n - 1);

        let self_data = self.to_vec()?;
        let mut output_data = Vec::with_capacity(max_lag + 1);

        // Directly compute auto-correlation: R[k] = Σ_n x[n] * x[n-k]
        for lag in 0..=max_lag {
            let mut sum = <T as TensorElement>::zero();

            for i in lag..n {
                sum = sum + self_data[i] * self_data[i - lag];
            }

            output_data.push(sum);
        }

        let output = Tensor::from_data(output_data, vec![max_lag + 1], self.device())?;
        Ok(output)
    }

    /// 2D cross-correlation operation
    /// Computes the 2D cross-correlation between two signals
    pub fn xcorr2d(&self, other: &Self, mode: CorrelationMode) -> Result<Self> {
        let self_shape_ref = self.shape();
        let other_shape_ref = other.shape();
        let self_shape = self_shape_ref.dims();
        let other_shape = other_shape_ref.dims();

        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "xcorr2d requires 2D tensors".to_string(),
            ));
        }

        let (h1, w1) = (self_shape[0], self_shape[1]);
        let (h2, w2) = (other_shape[0], other_shape[1]);

        let (out_h, out_w, start_h, start_w) = match mode {
            CorrelationMode::Full => (h1 + h2 - 1, w1 + w2 - 1, 0, 0),
            CorrelationMode::Valid => {
                if h1 < h2 || w1 < w2 {
                    return Err(TorshError::InvalidArgument(
                        "Valid mode requires first tensor to be larger than or equal to second"
                            .to_string(),
                    ));
                }
                (h1 - h2 + 1, w1 - w2 + 1, h2 - 1, w2 - 1)
            }
            CorrelationMode::Same => (h1, w1, (h2 - 1) / 2, (w2 - 1) / 2),
        };

        let mut output_data = vec![<T as TensorElement>::zero(); out_h * out_w];
        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        // Compute 2D cross-correlation
        for i in 0..out_h {
            for j in 0..out_w {
                let mut sum = <T as TensorElement>::zero();
                let actual_i = i + start_h;
                let actual_j = j + start_w;

                for ki in 0..h2 {
                    for kj in 0..w2 {
                        let src_i = actual_i as i32 - ki as i32;
                        let src_j = actual_j as i32 - kj as i32;

                        if src_i >= 0
                            && (src_i as usize) < h1
                            && src_j >= 0
                            && (src_j as usize) < w1
                        {
                            let self_idx = src_i as usize * w1 + src_j as usize;
                            let other_idx = ki * w2 + kj;
                            sum = sum + self_data[self_idx] * other_data[other_idx];
                        }
                    }
                }
                output_data[i * out_w + j] = sum;
            }
        }

        let output = Tensor::from_data(output_data, vec![out_h, out_w], self.device())?;
        Ok(output)
    }

    /// 1D median filter
    /// Applies a median filter with the specified window size
    pub fn median_filter1d(&self, window_size: usize) -> Result<Self> {
        let shape_ref = self.shape();
        let shape = shape_ref.dims();
        if shape.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "median_filter1d requires 1D tensor".to_string(),
            ));
        }

        if window_size == 0 || window_size % 2 == 0 {
            return Err(TorshError::InvalidArgument(
                "Window size must be odd and greater than 0".to_string(),
            ));
        }

        let n = shape[0];
        let half_window = window_size / 2;
        let mut output_data = Vec::with_capacity(n);
        let self_data = self.to_vec()?;

        for i in 0..n {
            let mut window_values = Vec::new();

            // Collect values in the window (with padding by repeating edge values)
            for j in 0..window_size {
                let idx = i as i32 + j as i32 - half_window as i32;
                let actual_idx = if idx < 0 {
                    0
                } else if idx >= n as i32 {
                    n - 1
                } else {
                    idx as usize
                };
                window_values.push(self_data[actual_idx]);
            }

            // Sort to find median
            window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            output_data.push(window_values[half_window]);
        }

        let output = Tensor::from_data(output_data, vec![n], self.device())?;
        Ok(output)
    }

    /// 2D median filter
    /// Applies a 2D median filter with the specified window size
    pub fn median_filter2d(&self, window_size: (usize, usize)) -> Result<Self> {
        let shape_ref = self.shape();
        let shape = shape_ref.dims();
        if shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "median_filter2d requires 2D tensor".to_string(),
            ));
        }

        let (window_h, window_w) = window_size;
        if window_h == 0 || window_w == 0 || window_h % 2 == 0 || window_w % 2 == 0 {
            return Err(TorshError::InvalidArgument(
                "Window dimensions must be odd and greater than 0".to_string(),
            ));
        }

        let (h, w) = (shape[0], shape[1]);
        let half_h = window_h / 2;
        let half_w = window_w / 2;
        let mut output_data = Vec::with_capacity(h * w);
        let self_data = self.to_vec()?;

        for i in 0..h {
            for j in 0..w {
                let mut window_values = Vec::new();

                // Collect values in the 2D window
                for di in 0..window_h {
                    for dj in 0..window_w {
                        let row = i as i32 + di as i32 - half_h as i32;
                        let col = j as i32 + dj as i32 - half_w as i32;

                        // Handle boundaries by clamping
                        let actual_row = row.max(0).min(h as i32 - 1) as usize;
                        let actual_col = col.max(0).min(w as i32 - 1) as usize;

                        window_values.push(self_data[actual_row * w + actual_col]);
                    }
                }

                // Sort to find median
                window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                output_data.push(window_values[window_values.len() / 2]);
            }
        }

        let output = Tensor::from_data(output_data, vec![h, w], self.device())?;
        Ok(output)
    }

    /// 1D Gaussian filter
    /// Applies a Gaussian filter with specified sigma (standard deviation)
    pub fn gaussian_filter1d(&self, sigma: f32, kernel_size: Option<usize>) -> Result<Self> {
        let tensor_shape = self.shape();
        let shape = tensor_shape.dims();
        if shape.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "gaussian_filter1d requires 1D tensor".to_string(),
            ));
        }

        if sigma <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Sigma must be positive".to_string(),
            ));
        }

        // Calculate kernel size if not provided (6 sigma rule)
        let kernel_size = kernel_size.unwrap_or(((6.0 * sigma) as usize).max(3));
        let kernel_size = if kernel_size % 2 == 0 {
            kernel_size + 1
        } else {
            kernel_size
        };

        // Generate Gaussian kernel
        let half_size = kernel_size / 2;
        let mut kernel = Vec::with_capacity(kernel_size);
        let mut sum = 0.0f32;

        for i in 0..kernel_size {
            let x = i as f32 - half_size as f32;
            let value = (-0.5 * (x / sigma).powi(2)).exp();
            kernel.push(value);
            sum += value;
        }

        // Normalize kernel
        for value in &mut kernel {
            *value /= sum;
        }

        // Create kernel tensor
        let kernel_data: Vec<T> = kernel
            .into_iter()
            .map(|v| T::from(v as f64).unwrap_or_else(|| T::from(0.0).unwrap()))
            .collect();
        let kernel_tensor = Tensor::from_data(kernel_data, vec![kernel_size], self.device())?;

        // Apply convolution (which is equivalent to correlation for symmetric kernels)
        self.xcorr1d(&kernel_tensor, CorrelationMode::Same)
    }

    /// 2D Gaussian filter
    /// Applies a 2D Gaussian filter with specified sigma values
    pub fn gaussian_filter2d(
        &self,
        sigma: (f32, f32),
        kernel_size: Option<(usize, usize)>,
    ) -> Result<Self> {
        let tensor_shape = self.shape();
        let shape = tensor_shape.dims();
        if shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "gaussian_filter2d requires 2D tensor".to_string(),
            ));
        }

        let (sigma_x, sigma_y) = sigma;
        if sigma_x <= 0.0 || sigma_y <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Sigma values must be positive".to_string(),
            ));
        }

        // Calculate kernel sizes if not provided
        let (kernel_h, kernel_w) = kernel_size.unwrap_or((
            ((6.0 * sigma_y) as usize).max(3),
            ((6.0 * sigma_x) as usize).max(3),
        ));
        let kernel_h = if kernel_h % 2 == 0 {
            kernel_h + 1
        } else {
            kernel_h
        };
        let kernel_w = if kernel_w % 2 == 0 {
            kernel_w + 1
        } else {
            kernel_w
        };

        // Generate 2D Gaussian kernel
        let half_h = kernel_h / 2;
        let half_w = kernel_w / 2;
        let mut kernel = Vec::with_capacity(kernel_h * kernel_w);
        let mut sum = 0.0f32;

        for i in 0..kernel_h {
            for j in 0..kernel_w {
                let y = i as f32 - half_h as f32;
                let x = j as f32 - half_w as f32;
                let value = (-0.5 * ((x / sigma_x).powi(2) + (y / sigma_y).powi(2))).exp();
                kernel.push(value);
                sum += value;
            }
        }

        // Normalize kernel
        for value in &mut kernel {
            *value /= sum;
        }

        // Create kernel tensor
        let kernel_data: Vec<T> = kernel
            .into_iter()
            .map(|v| T::from(v as f64).unwrap_or_else(|| T::from(0.0).unwrap()))
            .collect();
        let kernel_tensor =
            Tensor::from_data(kernel_data, vec![kernel_h, kernel_w], self.device())?;

        // Apply 2D correlation
        self.xcorr2d(&kernel_tensor, CorrelationMode::Same)
    }
}

/// Correlation modes for signal processing operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationMode {
    /// Full correlation output
    Full,
    /// Valid correlation output (no padding)
    Valid,
    /// Same size as input (with padding)
    Same,
}
