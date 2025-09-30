//! Advanced neural network operations using Metal Performance Shaders

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::{CommandBuffer, Device, NSUInteger};
use objc2::runtime::Object;
use objc2::{msg_send, sel, ClassType};

use crate::metal::{
    mps::{create_image_descriptor, MPSDataType, MPSOperation},
    MetalBuffer, MetalError, Result,
};

/// Batch normalization using MPS
pub struct MPSBatchNormalization {
    batch_norm: *mut Object,
    mean: MetalBuffer,
    variance: MetalBuffer,
    gamma: MetalBuffer,
    beta: MetalBuffer,
    num_features: usize,
    eps: f32,
    momentum: f32,
}

impl MPSBatchNormalization {
    /// Create a new batch normalization operation
    pub fn new(
        device: &Device,
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
    ) -> Result<Self> {
        unsafe {
            // Create MPS batch normalization
            let class = objc2::class!(MPSCNNBatchNormalization);
            let batch_norm: *mut Object = msg_send![class, alloc];
            let batch_norm: *mut Object = msg_send![batch_norm,
                initWithDevice: device.as_ptr() as *mut Object as *mut Object
                dataSource: std::ptr::null_mut::<Object>()
            ];

            // Set epsilon
            let _: () = msg_send![batch_norm, setEpsilon: eps as f32];

            // Create buffers for statistics
            let mean = MetalBuffer::zeros(
                &torsh_core::Shape::from(vec![num_features]),
                &torsh_core::DType::F32,
                &crate::metal::device::MetalDevice::new()?,
            )?;

            let variance = MetalBuffer::zeros(
                &torsh_core::Shape::from(vec![num_features]),
                &torsh_core::DType::F32,
                &crate::metal::device::MetalDevice::new()?,
            )?;

            // Create affine parameters if enabled
            let gamma = if affine {
                MetalBuffer::ones(
                    &torsh_core::Shape::from(vec![num_features]),
                    &torsh_core::DType::F32,
                    &crate::metal::device::MetalDevice::new()?,
                )?
            } else {
                MetalBuffer::zeros(
                    &torsh_core::Shape::from(vec![num_features]),
                    &torsh_core::DType::F32,
                    &crate::metal::device::MetalDevice::new()?,
                )?
            };

            let beta = MetalBuffer::zeros(
                &torsh_core::Shape::from(vec![num_features]),
                &torsh_core::DType::F32,
                &crate::metal::device::MetalDevice::new()?,
            )?;

            Ok(Self {
                batch_norm,
                mean,
                variance,
                gamma,
                beta,
                num_features,
                eps,
                momentum,
            })
        }
    }

    /// Forward pass
    pub fn forward(
        &mut self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        training: bool,
    ) -> Result<()> {
        unsafe {
            let input_shape = input.shape().dims();
            if input_shape.len() != 4 {
                return Err(MetalError::ShapeMismatch {
                    expected: vec![4],
                    got: vec![input_shape.len()],
                });
            }

            let [_batch, channels, height, width] = [
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ];

            if channels != self.num_features {
                return Err(MetalError::ShapeMismatch {
                    expected: vec![self.num_features],
                    got: vec![channels],
                });
            }

            // Create MPS images
            let class = objc2::class!(MPSImage);

            let input_desc = create_image_descriptor(width, height, channels, MPSDataType::Float32);
            let input_image: *mut Object = msg_send![class, alloc];
            let input_image: *mut Object = msg_send![input_image,
                initWithDevice: input.buffer().device().as_ptr() as *mut Object
                imageDescriptor: input_desc
            ];

            let output_desc =
                create_image_descriptor(width, height, channels, MPSDataType::Float32);
            let output_image: *mut Object = msg_send![class, alloc];
            let output_image: *mut Object = msg_send![output_image,
                initWithDevice: output.buffer().device().as_ptr() as *mut Object
                imageDescriptor: output_desc
            ];

            // Encode the operation
            if training {
                let _: () = msg_send![self.batch_norm,
                    encodeToCommandBuffer: command_buffer.as_ptr() as *mut Object
                    sourceImage: input_image
                    batchNormalizationState: std::ptr::null_mut::<Object>()
                    destinationImage: output_image
                ];
            } else {
                let _: () = msg_send![self.batch_norm,
                    encodeToCommandBuffer: command_buffer.as_ptr() as *mut Object
                    sourceImage: input_image
                    destinationImage: output_image
                ];
            }

            Ok(())
        }
    }
}

impl Drop for MPSBatchNormalization {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.batch_norm, release];
        }
    }
}

/// Multi-head attention using MPS matrix operations
pub struct MPSMultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    q_proj: MPSLinear,
    k_proj: MPSLinear,
    v_proj: MPSLinear,
    out_proj: MPSLinear,
    dropout_p: f32,
}

impl MPSMultiHeadAttention {
    /// Create a new multi-head attention operation
    pub fn new(
        device: &Device,
        embed_dim: usize,
        num_heads: usize,
        dropout_p: f32,
    ) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(MetalError::InvalidArgument(
                "embed_dim must be divisible by num_heads".to_string(),
            ));
        }

        let head_dim = embed_dim / num_heads;

        let q_proj = MPSLinear::new(device, embed_dim, embed_dim, true)?;
        let k_proj = MPSLinear::new(device, embed_dim, embed_dim, true)?;
        let v_proj = MPSLinear::new(device, embed_dim, embed_dim, true)?;
        let out_proj = MPSLinear::new(device, embed_dim, embed_dim, true)?;

        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            dropout_p,
        })
    }

    /// Forward pass with scaled dot-product attention
    pub fn forward(
        &self,
        command_buffer: &CommandBuffer,
        query: &MetalBuffer,
        key: &MetalBuffer,
        value: &MetalBuffer,
        output: &MetalBuffer,
        mask: Option<&MetalBuffer>,
    ) -> Result<()> {
        let seq_len = query.shape().dims()[1];
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Create output buffers for Q, K, V projections
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        let q = MetalBuffer::zeros(q_shape, &query.dtype(), &crate::metal::device::MetalDevice::new()?)?;
        let k = MetalBuffer::zeros(k_shape, &key.dtype(), &crate::metal::device::MetalDevice::new()?)?;
        let v = MetalBuffer::zeros(v_shape, &value.dtype(), &crate::metal::device::MetalDevice::new()?)?;

        // Project Q, K, V
        self.q_proj.forward(command_buffer, query, &q)?;
        self.k_proj.forward(command_buffer, key, &k)?;
        self.v_proj.forward(command_buffer, value, &v)?;

        // Reshape for multi-head (batch_size, seq_len, num_heads, head_dim)
        // Then transpose to (batch_size, num_heads, seq_len, head_dim)

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        let scores = self.scaled_dot_product_attention(command_buffer, &q, &k, &v, scale, mask)?;

        // Output projection
        self.out_proj.forward(command_buffer, &scores, output)?;

        Ok(())
    }

    fn scaled_dot_product_attention(
        &self,
        command_buffer: &CommandBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        scale: f32,
        mask: Option<&MetalBuffer>,
    ) -> Result<MetalBuffer> {
        // Get shapes and validate dimensions
        let q_shape = q.shape().dims();
        let k_shape = k.shape().dims();
        let v_shape = v.shape().dims();

        if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
            return Err(crate::metal::error::MetalError::ShapeMismatch {
                expected: vec![2],
                got: vec![q_shape.len(), k_shape.len(), v_shape.len()],
            });
        }

        let seq_len = q_shape[q_shape.len() - 2];
        let k_seq_len = k_shape[k_shape.len() - 2];

        // Step 1: Q @ K^T
        // Create matmul for Q @ K^T with K transposed
        let qk_matmul = crate::metal::mps::MPSMatMul::new(
            &q.buffer().device().to_owned(),
            q,
            k,
            None,
            1.0,   // alpha
            0.0,   // beta
            false, // transpose_a (Q)
            true,  // transpose_b (K^T)
        )?;

        // Get intermediate result buffer for attention scores
        let scores_shape = torsh_core::Shape::from(vec![seq_len, k_seq_len]);
        let mut scores = MetalBuffer::zeros(
            &scores_shape,
            &torsh_core::DType::F32,
            &crate::metal::device::MetalDevice::new()?,
        )?;

        // Encode Q @ K^T operation
        qk_matmul.encode_matmul(command_buffer, q, k)?;
        scores = qk_matmul.output().clone();

        // Step 2: Scale by the provided scale factor (usually 1/sqrt(head_dim))
        // For simplicity, assuming scale is already 1/sqrt(head_dim)
        if scale != 1.0 {
            // Create a simple element-wise scaling operation
            // In a full implementation, this would use MPS element-wise operations
            // For now, we'll apply the scaling factor in the matrix multiplication
            let scaled_matmul = crate::metal::mps::MPSMatMul::new(
                &q.buffer().device().to_owned(),
                q,
                k,
                None,
                scale, // alpha (scaling factor)
                0.0,   // beta
                false, // transpose_a (Q)
                true,  // transpose_b (K^T)
            )?;
            scaled_matmul.encode_matmul(command_buffer, q, k)?;
            scores = scaled_matmul.output().clone();
        }

        // Step 3: Apply mask if provided
        if let Some(_mask_buffer) = mask {
            // In a full implementation, this would add the mask to the scores
            // Masked positions would be set to -inf before softmax
            // For now, we'll skip masking in this simplified implementation
        }

        // Step 4: Apply Softmax
        let softmax = crate::metal::mps::MPSActivation::new(
            &q.buffer().device().to_owned(),
            crate::metal::mps::ActivationType::Softmax,
        )?;

        let mut attention_weights = MetalBuffer::zeros(
            &scores.shape(),
            &torsh_core::DType::F32,
            &crate::metal::device::MetalDevice::new()?,
        )?;

        // Apply softmax to get attention weights
        softmax.apply(command_buffer, &scores, &attention_weights)?;

        // Step 5: Attention weights @ V
        let output_matmul = crate::metal::mps::MPSMatMul::new(
            &q.buffer().device().to_owned(),
            &attention_weights,
            v,
            None,
            1.0,   // alpha
            0.0,   // beta
            false, // transpose_a (attention weights)
            false, // transpose_b (V)
        )?;

        // Create output buffer with same shape as input Q
        let output = MetalBuffer::zeros(
            q.shape(),
            &torsh_core::DType::F32,
            &crate::metal::device::MetalDevice::new()?,
        )?;

        // Encode final matrix multiplication
        output_matmul.encode_matmul(command_buffer, &attention_weights, v)?;

        // Return the result from the final matrix multiplication
        Ok(output_matmul.output().clone())
    }
}

/// Linear layer using MPS matrix multiplication
pub struct MPSLinear {
    weight: MetalBuffer,
    bias: Option<MetalBuffer>,
    in_features: usize,
    out_features: usize,
}

impl MPSLinear {
    /// Create a new linear layer
    pub fn new(
        device: &Device,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self> {
        // Initialize weight with Xavier/Glorot uniform
        let bound = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight = MetalBuffer::rand(
            &torsh_core::Shape::from(vec![out_features, in_features]),
            &torsh_core::DType::F32,
            &crate::metal::device::MetalDevice::new()?,
        )?;

        let bias_buffer = if bias {
            Some(MetalBuffer::zeros(
                &torsh_core::Shape::from(vec![out_features]),
                &torsh_core::DType::F32,
                &crate::metal::device::MetalDevice::new()?,
            )?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_buffer,
            in_features,
            out_features,
        })
    }

    /// Forward pass: output = input @ weight^T + bias
    pub fn forward(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        // Use MPS matrix multiplication
        let matmul = crate::metal::mps::MPSMatMul::new(
            &input.buffer().device().to_owned(),
            input,
            &self.weight,
            self.bias.as_ref(),
            1.0,                                         // alpha
            if self.bias.is_some() { 1.0 } else { 0.0 }, // beta
            false,                                       // transpose_a
            true,                                        // transpose_b (transpose weight)
        )?;

        matmul.encode_matmul(command_buffer, input, &self.weight)?;

        Ok(())
    }

    /// Forward pass with allocated output buffer
    pub fn forward_with_output(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
    ) -> Result<MetalBuffer> {
        // Create output buffer
        let input_shape = input.shape().dims();
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(self.out_features);

        let output = MetalBuffer::zeros(
            &torsh_core::Shape::from(output_shape),
            &torsh_core::DType::F32,
            &crate::metal::device::MetalDevice::new()?,
        )?;

        self.forward(command_buffer, input, &output)?;
        Ok(output)
    }
}

/// Optimized convolution with various algorithms
pub struct MPSOptimizedConv2d {
    conv: *mut Object,
    algorithm: ConvolutionAlgorithm,
    params: Conv2dParams,
    device: Device,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvolutionAlgorithm {
    /// Direct convolution
    Direct,
    /// Winograd algorithm for 3x3 convolutions
    Winograd,
    /// FFT-based convolution for large kernels
    FFT,
    /// Im2Col + GEMM
    Im2ColGemm,
}

#[derive(Debug, Clone)]
pub struct Conv2dParams {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub stride_height: usize,
    pub stride_width: usize,
    pub padding_height: usize,
    pub padding_width: usize,
    pub dilation_height: usize,
    pub dilation_width: usize,
    pub groups: usize,
}

impl MPSOptimizedConv2d {
    /// Create optimized convolution with automatic algorithm selection
    pub fn new(
        device: &Device,
        params: Conv2dParams,
        weights: &MetalBuffer,
        bias: Option<&MetalBuffer>,
        auto_select_algorithm: bool,
    ) -> Result<Self> {
        let algorithm = if auto_select_algorithm {
            Self::select_optimal_algorithm(&params)
        } else {
            ConvolutionAlgorithm::Direct
        };

        unsafe {
            let conv = match algorithm {
                ConvolutionAlgorithm::Winograd => {
                    Self::create_winograd_conv(device, &params, weights, bias)?
                }
                ConvolutionAlgorithm::FFT => Self::create_fft_conv(device, &params, weights, bias)?,
                _ => Self::create_direct_conv(device, &params, weights, bias)?,
            };

            Ok(Self {
                conv,
                algorithm,
                params,
                device: device.clone(),
            })
        }
    }

    fn select_optimal_algorithm(params: &Conv2dParams) -> ConvolutionAlgorithm {
        // Heuristics for algorithm selection
        if params.kernel_height == 3
            && params.kernel_width == 3
            && params.stride_height == 1
            && params.stride_width == 1
        {
            ConvolutionAlgorithm::Winograd
        } else if params.kernel_height >= 7 && params.kernel_width >= 7 {
            ConvolutionAlgorithm::FFT
        } else {
            ConvolutionAlgorithm::Direct
        }
    }

    unsafe fn create_direct_conv(
        device: &Device,
        params: &Conv2dParams,
        _weights: &MetalBuffer,
        _bias: Option<&MetalBuffer>,
    ) -> Result<*mut Object> {
        let class = objc2::class!(MPSCNNConvolution);
        let conv: *mut Object = msg_send![class, alloc];

        // Create convolution descriptor
        let desc_class = objc2::class!(MPSCNNConvolutionDescriptor);
        let desc: *mut Object = msg_send![desc_class, alloc];
        let desc: *mut Object = msg_send![desc, init];

        let _: () = msg_send![desc, setKernelHeight: params.kernel_height as NSUInteger];
        let _: () = msg_send![desc, setKernelWidth: params.kernel_width as NSUInteger];
        let _: () = msg_send![desc, setInputFeatureChannels: params.in_channels as NSUInteger];
        let _: () = msg_send![desc, setOutputFeatureChannels: params.out_channels as NSUInteger];
        let _: () = msg_send![desc, setStrideInPixelsX: params.stride_width as NSUInteger];
        let _: () = msg_send![desc, setStrideInPixelsY: params.stride_height as NSUInteger];

        let conv: *mut Object = msg_send![conv,
            initWithDevice: device.as_ptr() as *mut Object
            convolutionDescriptor: desc
            kernelWeights: std::ptr::null::<f32>()
            biasTerms: std::ptr::null::<f32>()
            flags: 0 as NSUInteger
        ];

        Ok(conv)
    }

    unsafe fn create_winograd_conv(
        device: &Device,
        params: &Conv2dParams,
        _weights: &MetalBuffer,
        _bias: Option<&MetalBuffer>,
    ) -> Result<*mut Object> {
        // For Winograd, we'd use a specialized implementation
        // This is simplified - real implementation would use Winograd transforms
        Self::create_direct_conv(device, params, _weights, _bias)
    }

    unsafe fn create_fft_conv(
        device: &Device,
        params: &Conv2dParams,
        _weights: &MetalBuffer,
        _bias: Option<&MetalBuffer>,
    ) -> Result<*mut Object> {
        // For FFT convolution, we'd use frequency domain operations
        // This is simplified - real implementation would use FFT
        Self::create_direct_conv(device, params, _weights, _bias)
    }

    /// Encode the optimized convolution
    pub fn encode(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        match self.algorithm {
            ConvolutionAlgorithm::Winograd => self.encode_winograd(command_buffer, input, output),
            ConvolutionAlgorithm::FFT => self.encode_fft(command_buffer, input, output),
            _ => self.encode_direct(command_buffer, input, output),
        }
    }

    fn encode_direct(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        unsafe {
            // Create MPS images for input and output
            let input_shape = input.shape().dims();
            let output_shape = output.shape().dims();

            let class = objc2::class!(MPSImage);

            let input_desc = create_image_descriptor(
                input_shape[3],
                input_shape[2],
                input_shape[1],
                MPSDataType::Float32,
            );
            let input_image: *mut Object = msg_send![class, alloc];
            let input_image: *mut Object = msg_send![input_image,
                initWithDevice: self.device.as_ptr() as *mut Object
                imageDescriptor: input_desc
            ];

            let output_desc = create_image_descriptor(
                output_shape[3],
                output_shape[2],
                output_shape[1],
                MPSDataType::Float32,
            );
            let output_image: *mut Object = msg_send![class, alloc];
            let output_image: *mut Object = msg_send![output_image,
                initWithDevice: self.device.as_ptr() as *mut Object
                imageDescriptor: output_desc
            ];

            // Encode the convolution
            let _: () = msg_send![self.conv,
                encodeToCommandBuffer: command_buffer.as_ptr() as *mut Object
                sourceImage: input_image
                destinationImage: output_image
            ];

            Ok(())
        }
    }

    fn encode_winograd(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        // Winograd algorithm implementation would go here
        // For now, fall back to direct
        self.encode_direct(command_buffer, input, output)
    }

    fn encode_fft(
        &self,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> Result<()> {
        // FFT convolution implementation would go here
        // For now, fall back to direct
        self.encode_direct(command_buffer, input, output)
    }
}

impl Drop for MPSOptimizedConv2d {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.conv, release];
        }
    }
}

/// High-performance fused operations
pub struct MPSFusedOps;

impl MPSFusedOps {
    /// Fused convolution + batch norm + activation
    pub fn conv_bn_activation(
        device: &Device,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        conv_params: &Conv2dParams,
        conv_weights: &MetalBuffer,
        conv_bias: Option<&MetalBuffer>,
        bn_weight: &MetalBuffer,
        bn_bias: &MetalBuffer,
        bn_mean: &MetalBuffer,
        bn_var: &MetalBuffer,
        activation: ActivationType,
        output: &MetalBuffer,
    ) -> Result<()> {
        unsafe {
            // Create fused operation
            let class = objc2::class!(MPSCNNConvolution);
            // ... implementation would create a custom fused kernel
            // This is simplified for now

            Ok(())
        }
    }

    /// Fused matrix multiplication + bias + activation
    pub fn linear_bias_activation(
        device: &Device,
        command_buffer: &CommandBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        bias: Option<&MetalBuffer>,
        activation: ActivationType,
        output: &MetalBuffer,
    ) -> Result<()> {
        // Implementation would create a fused GEMM operation
        // This is simplified for now
        Ok(())
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    ReLU6,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    LeakyReLU(f32),
    ELU(f32),
}

impl ActivationType {
    /// Get the MPS neuron type constant
    pub fn to_mps_neuron_type(&self) -> u32 {
        match self {
            ActivationType::ReLU => 1,         // MPSCNNNeuronTypeReLU
            ActivationType::ReLU6 => 4,        // MPSCNNNeuronTypeReLU6
            ActivationType::Sigmoid => 3,      // MPSCNNNeuronTypeSigmoid
            ActivationType::Tanh => 2,         // MPSCNNNeuronTypeTanH
            ActivationType::LeakyReLU(_) => 5, // MPSCNNNeuronTypeLeakyReLU
            ActivationType::ELU(_) => 6,       // MPSCNNNeuronTypeELU
            _ => 1,                            // Default to ReLU for unsupported types
        }
    }

    /// Get activation parameter (a)
    pub fn get_param_a(&self) -> f32 {
        match self {
            ActivationType::LeakyReLU(alpha) => *alpha,
            ActivationType::ELU(alpha) => *alpha,
            _ => 0.0,
        }
    }
}
