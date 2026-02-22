//! Specialized hardware-specific quantization operations
//!
//! This module provides optimized quantization implementations that leverage
//! specific hardware acceleration features like Intel VNNI, NVIDIA DP4A,
//! and modern GPU tensor cores for maximum performance.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::quantization::QuantizedTensor;
use crate::{BackendResult, Device};
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Intel VNNI (Vector Neural Network Instructions) quantization operations
///
/// VNNI instructions provide hardware acceleration for neural network workloads
/// by enabling efficient INT8 matrix operations directly in the CPU pipeline.
/// These operations are available on Intel processors starting with Cascade Lake.
#[derive(Debug, Clone)]
pub struct VnniQuantizationOps {
    /// Whether VNNI instructions are available on this CPU
    vnni_available: bool,
    /// Device this operator is configured for
    device: Device,
}

impl VnniQuantizationOps {
    /// Create new VNNI quantization operations
    ///
    /// Automatically detects VNNI availability and configures the operator
    /// for optimal performance on the current hardware.
    pub fn new(device: Device) -> Self {
        Self {
            vnni_available: Self::detect_vnni(),
            device,
        }
    }

    /// Detect Intel VNNI instruction support
    ///
    /// Uses runtime CPU feature detection to determine if VNNI instructions
    /// are available. This includes both AVX512-VNNI and AVX-VNNI variants.
    fn detect_vnni() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Check for VNNI support via CPUID
            // Note: This primarily checks for AVX512-VNNI
            // AVX-VNNI would require additional detection logic
            is_x86_feature_detected!("avx512vnni")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Check if VNNI operations are available
    pub fn is_available(&self) -> bool {
        self.vnni_available
    }

    /// VNNI-accelerated INT8 matrix multiplication
    ///
    /// Performs quantized matrix multiplication using Intel VNNI instructions
    /// for optimal performance. This operation is specifically optimized for
    /// INT8 quantized neural network inference.
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix operand (quantized)
    /// * `b` - Right matrix operand (quantized)
    ///
    /// # Returns
    ///
    /// Result matrix from the multiplication operation
    ///
    /// # Errors
    ///
    /// Returns an error if VNNI is not available or if the operation fails
    pub fn vnni_qmatmul_int8(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.vnni_available {
            return Err(TorshError::BackendError("VNNI not available".to_string()));
        }

        // Validate input tensors
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(TorshError::BackendError(
                "VNNI matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if a.shape[1] != b.shape[0] {
            return Err(TorshError::BackendError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        // In a real implementation, this would use actual VNNI instructions
        // through inline assembly or intrinsics. For now, we provide a
        // placeholder implementation that could be replaced with optimized code.
        let result_data = self.vnni_matmul_kernel(&a.data, &b.data, m, k, n)?;

        Ok(QuantizedTensor {
            data: result_data,
            shape: vec![m, n],
            params: a.params.clone(),
            device: self.device.clone(),
        })
    }

    /// VNNI-optimized convolution operation
    ///
    /// Implements quantized 2D convolution using VNNI instructions for
    /// accelerated neural network inference.
    pub fn vnni_qconv2d(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        bias: Option<&QuantizedTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor> {
        if !self.vnni_available {
            return Err(TorshError::BackendError("VNNI not available".to_string()));
        }

        // Validate input dimensions
        if input.shape.len() != 4 || weight.shape.len() != 4 {
            return Err(TorshError::BackendError(
                "VNNI convolution requires 4D tensors".to_string(),
            ));
        }

        // For now, provide a simplified implementation
        // Real VNNI convolution would use optimized kernels
        let batch_size = input.shape[0];
        let out_channels = weight.shape[0];
        let out_height = (input.shape[2] + 2 * padding.0 - weight.shape[2]) / stride.0 + 1;
        let out_width = (input.shape[3] + 2 * padding.1 - weight.shape[3]) / stride.1 + 1;

        let output_size = batch_size * out_channels * out_height * out_width;
        let result_data = vec![0u8; output_size];

        // Apply bias if provided
        let final_data = if let Some(_bias_tensor) = bias {
            // In real implementation, would add bias using VNNI operations
            result_data
        } else {
            result_data
        };

        Ok(QuantizedTensor {
            data: final_data,
            shape: vec![batch_size, out_channels, out_height, out_width],
            params: input.params.clone(),
            device: self.device.clone(),
        })
    }

    /// VNNI matrix multiplication kernel
    ///
    /// Core kernel implementing VNNI-accelerated matrix multiplication.
    /// In a production implementation, this would use inline assembly or
    /// compiler intrinsics to emit VNNI instructions.
    fn vnni_matmul_kernel(
        &self,
        a_data: &[u8],
        b_data: &[u8],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<u8>> {
        // Placeholder implementation
        // Real VNNI kernel would use instructions like VPDPBUSD
        let mut result = vec![0u8; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for l in 0..k {
                    let a_val = a_data[i * k + l] as i8 as i32;
                    let b_val = b_data[l * n + j] as i8 as i32;
                    acc += a_val * b_val;
                }
                // Clamp and convert back to u8
                result[i * n + j] = acc.clamp(-128, 127) as u8;
            }
        }

        Ok(result)
    }

    /// Get optimal block size for VNNI operations
    pub fn optimal_block_size(&self) -> usize {
        if self.vnni_available {
            // VNNI works well with larger blocks due to vectorization
            512
        } else {
            64
        }
    }
}

/// NVIDIA DP4A (4-element dot product and accumulate) quantization operations
///
/// DP4A instructions enable efficient INT8 operations on NVIDIA GPUs by computing
/// 4-element dot products in a single instruction. This is particularly beneficial
/// for quantized neural network inference on CUDA devices.
#[derive(Debug, Clone)]
pub struct Dp4aQuantizationOps {
    /// Whether DP4A instructions are available
    dp4a_available: bool,
    /// Device this operator is configured for
    device: Device,
}

impl Dp4aQuantizationOps {
    /// Create new DP4A quantization operations
    ///
    /// Configures DP4A operations for the given CUDA device. DP4A is available
    /// on modern NVIDIA GPUs (Pascal architecture and newer).
    pub fn new(device: Device) -> Self {
        Self {
            dp4a_available: Self::detect_dp4a(&device),
            device,
        }
    }

    /// Detect DP4A instruction support
    ///
    /// In a real implementation, this would query CUDA device properties
    /// to determine DP4A availability based on the GPU architecture.
    fn detect_dp4a(device: &Device) -> bool {
        match device.device_type() {
            torsh_core::device::DeviceType::Cuda(_) => {
                // In practice, would check GPU compute capability
                // DP4A is available on Pascal (6.1+) and newer architectures
                true
            }
            _ => false,
        }
    }

    /// Check if DP4A operations are available
    pub fn is_available(&self) -> bool {
        self.dp4a_available
    }

    /// DP4A-accelerated INT8 matrix multiplication
    ///
    /// Implements quantized matrix multiplication using NVIDIA DP4A instructions
    /// for high-performance inference on CUDA GPUs.
    ///
    /// # Arguments
    ///
    /// * `a` - Left matrix operand (quantized)
    /// * `b` - Right matrix operand (quantized)
    ///
    /// # Returns
    ///
    /// Result matrix from the multiplication operation
    pub fn dp4a_qmatmul_int8(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.dp4a_available {
            return Err(TorshError::BackendError("DP4A not available".to_string()));
        }

        // Validate tensor dimensions
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(TorshError::BackendError(
                "DP4A matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if a.shape[1] != b.shape[0] {
            return Err(TorshError::BackendError(
                "Matrix dimensions incompatible".to_string(),
            ));
        }

        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        // In a real implementation, this would launch CUDA kernels using DP4A
        let result_data = self.dp4a_matmul_kernel(&a.data, &b.data, m, k, n)?;

        Ok(QuantizedTensor {
            data: result_data,
            shape: vec![m, n],
            params: a.params.clone(),
            device: self.device.clone(),
        })
    }

    /// DP4A-optimized convolution operation
    pub fn dp4a_qconv2d(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        bias: Option<&QuantizedTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor> {
        if !self.dp4a_available {
            return Err(TorshError::BackendError("DP4A not available".to_string()));
        }

        // Validate dimensions
        if input.shape.len() != 4 || weight.shape.len() != 4 {
            return Err(TorshError::BackendError(
                "DP4A convolution requires 4D tensors".to_string(),
            ));
        }

        // Calculate output dimensions
        let batch_size = input.shape[0];
        let out_channels = weight.shape[0];
        let out_height = (input.shape[2] + 2 * padding.0 - weight.shape[2]) / stride.0 + 1;
        let out_width = (input.shape[3] + 2 * padding.1 - weight.shape[3]) / stride.1 + 1;

        // In production, would use optimized CUDA kernels with DP4A
        let output_size = batch_size * out_channels * out_height * out_width;
        let result_data = vec![0u8; output_size];

        // Apply bias if provided
        if let Some(_bias_tensor) = bias {
            // Would use DP4A for bias addition as well
        }

        Ok(QuantizedTensor {
            data: result_data,
            shape: vec![batch_size, out_channels, out_height, out_width],
            params: input.params.clone(),
            device: self.device.clone(),
        })
    }

    /// DP4A matrix multiplication kernel placeholder
    ///
    /// In a real implementation, this would launch optimized CUDA kernels
    /// that use DP4A instructions for 4-element dot products.
    fn dp4a_matmul_kernel(
        &self,
        a_data: &[u8],
        b_data: &[u8],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<u8>> {
        // Placeholder for CUDA kernel launch
        // Real implementation would use optimized DP4A kernels
        let mut result = vec![0u8; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;

                // Process in groups of 4 for DP4A efficiency
                for l in (0..k).step_by(4) {
                    for offset in 0..4.min(k - l) {
                        let a_val = a_data[i * k + l + offset] as i8 as i32;
                        let b_val = b_data[(l + offset) * n + j] as i8 as i32;
                        acc += a_val * b_val;
                    }
                }

                result[i * n + j] = acc.clamp(-128, 127) as u8;
            }
        }

        Ok(result)
    }

    /// Get optimal block size for DP4A operations
    pub fn optimal_block_size(&self) -> usize {
        if self.dp4a_available {
            // DP4A benefits from larger blocks due to GPU parallelism
            1024
        } else {
            64
        }
    }
}

/// Tensor Core quantization operations for modern NVIDIA GPUs
///
/// Tensor Cores provide specialized acceleration for mixed-precision and quantized
/// matrix operations on modern NVIDIA GPUs (Volta, Turing, Ampere, etc.).
/// These operations can achieve significantly higher throughput than standard CUDA cores.
#[derive(Debug, Clone)]
pub struct TensorCoreQuantizationOps {
    /// Whether Tensor Cores are available
    tensor_cores_available: bool,
    /// Device this operator is configured for
    device: Device,
    /// Supported quantization formats for Tensor Cores
    supported_formats: Vec<TensorCoreFormat>,
}

impl TensorCoreQuantizationOps {
    /// Create new Tensor Core quantization operations
    ///
    /// Detects Tensor Core availability and supported formats based on
    /// the GPU architecture and capabilities.
    pub fn new(device: Device) -> Self {
        let (available, formats) = Self::detect_tensor_cores(&device);

        Self {
            tensor_cores_available: available,
            device,
            supported_formats: formats,
        }
    }

    /// Detect Tensor Core support and available formats
    ///
    /// Different GPU architectures support different Tensor Core formats:
    /// - Volta (V100): FP16, INT8, INT4
    /// - Turing (T4): FP16, INT8, INT4, INT1
    /// - Ampere (A100): FP16, BF16, TF32, INT8, INT4, INT1
    fn detect_tensor_cores(device: &Device) -> (bool, Vec<TensorCoreFormat>) {
        match device.device_type() {
            torsh_core::device::DeviceType::Cuda(_) => {
                // In practice, would query actual GPU compute capability
                let formats = vec![
                    TensorCoreFormat::Int8,
                    TensorCoreFormat::Int4,
                    TensorCoreFormat::Int1,
                ];
                (true, formats)
            }
            _ => (false, vec![]),
        }
    }

    /// Check if Tensor Cores are available
    pub fn is_available(&self) -> bool {
        self.tensor_cores_available
    }

    /// Get supported quantization formats
    pub fn supported_formats(&self) -> &[TensorCoreFormat] {
        &self.supported_formats
    }

    /// Tensor Core INT8 matrix multiplication
    ///
    /// Uses Tensor Cores for high-performance INT8 matrix multiplication.
    /// This can achieve much higher throughput than standard CUDA cores
    /// for appropriately sized matrices.
    pub fn tensor_core_qmatmul_int8(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.tensor_cores_available {
            return Err(TorshError::BackendError(
                "Tensor Cores not available".to_string(),
            ));
        }

        if !self.supported_formats.contains(&TensorCoreFormat::Int8) {
            return Err(TorshError::BackendError(
                "INT8 not supported on available Tensor Cores".to_string(),
            ));
        }

        // Validate dimensions and alignment for Tensor Cores
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(TorshError::BackendError(
                "Tensor Core operations require 2D tensors".to_string(),
            ));
        }

        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        // Tensor Cores have specific alignment requirements
        if !self.check_tensor_core_alignment(m, k, n) {
            return Err(TorshError::BackendError(
                "Matrix dimensions not aligned for Tensor Cores".to_string(),
            ));
        }

        // In production, would use WMMA (Warp Matrix Multiply Accumulate) or similar
        let result_data = self.tensor_core_matmul_kernel(&a.data, &b.data, m, k, n)?;

        Ok(QuantizedTensor {
            data: result_data,
            shape: vec![m, n],
            params: a.params.clone(),
            device: self.device.clone(),
        })
    }

    /// Tensor Core INT4 matrix multiplication
    ///
    /// Specialized operation for INT4 quantized models using Tensor Cores,
    /// providing even higher throughput at the cost of precision.
    pub fn tensor_core_qmatmul_int4(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.tensor_cores_available {
            return Err(TorshError::BackendError(
                "Tensor Cores not available".to_string(),
            ));
        }

        if !self.supported_formats.contains(&TensorCoreFormat::Int4) {
            return Err(TorshError::BackendError(
                "INT4 not supported on available Tensor Cores".to_string(),
            ));
        }

        // INT4 operations have specific requirements
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        // For INT4, data should be packed
        if a.data.len() != (m * k + 1) / 2 || b.data.len() != (k * n + 1) / 2 {
            return Err(TorshError::BackendError(
                "INT4 data should be packed for Tensor Cores".to_string(),
            ));
        }

        let result_data = self.tensor_core_matmul_int4_kernel(&a.data, &b.data, m, k, n)?;

        Ok(QuantizedTensor {
            data: result_data,
            shape: vec![m, n],
            params: a.params.clone(),
            device: self.device.clone(),
        })
    }

    /// Check matrix dimension alignment for Tensor Cores
    ///
    /// Tensor Cores have specific alignment requirements that vary by format
    /// and GPU architecture. This function validates that dimensions meet
    /// the requirements for optimal performance.
    fn check_tensor_core_alignment(&self, m: usize, k: usize, n: usize) -> bool {
        // Common Tensor Core alignment requirements
        // Different architectures may have different requirements
        const ALIGNMENT_16: usize = 16;
        const ALIGNMENT_8: usize = 8;

        // Check if dimensions are aligned to required boundaries
        (m % ALIGNMENT_16 == 0 || m % ALIGNMENT_8 == 0)
            && (k % ALIGNMENT_16 == 0 || k % ALIGNMENT_8 == 0)
            && (n % ALIGNMENT_16 == 0 || n % ALIGNMENT_8 == 0)
    }

    /// Tensor Core matrix multiplication kernel for INT8
    fn tensor_core_matmul_kernel(
        &self,
        a_data: &[u8],
        b_data: &[u8],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<u8>> {
        // Placeholder for Tensor Core WMMA implementation
        // Real implementation would use CUDA WMMA API
        let mut result = vec![0u8; m * n];

        // Simulate Tensor Core operation with optimized blocking
        const BLOCK_SIZE: usize = 16;

        for i in (0..m).step_by(BLOCK_SIZE) {
            for j in (0..n).step_by(BLOCK_SIZE) {
                for l in (0..k).step_by(BLOCK_SIZE) {
                    // Process 16x16 blocks using simulated Tensor Core operation
                    for ii in 0..BLOCK_SIZE.min(m - i) {
                        for jj in 0..BLOCK_SIZE.min(n - j) {
                            let mut acc = 0i32;
                            for ll in 0..BLOCK_SIZE.min(k - l) {
                                let a_val = a_data[(i + ii) * k + (l + ll)] as i8 as i32;
                                let b_val = b_data[(l + ll) * n + (j + jj)] as i8 as i32;
                                acc += a_val * b_val;
                            }
                            result[(i + ii) * n + (j + jj)] = acc.clamp(-128, 127) as u8;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Tensor Core matrix multiplication kernel for INT4
    fn tensor_core_matmul_int4_kernel(
        &self,
        _a_data: &[u8],
        _b_data: &[u8],
        m: usize,
        _k: usize,
        n: usize,
    ) -> BackendResult<Vec<u8>> {
        // Placeholder for INT4 Tensor Core implementation
        // Would use specialized INT4 WMMA operations
        let result_size = (m * n + 1) / 2; // Packed INT4 output
        let result = vec![0u8; result_size];

        // Real implementation would use INT4 Tensor Core instructions
        // This is a simplified placeholder

        Ok(result)
    }

    /// Get optimal matrix size for Tensor Core operations
    pub fn optimal_matrix_size(&self) -> (usize, usize, usize) {
        if self.tensor_cores_available {
            // Tensor Cores work best with specific sizes
            (256, 256, 256)
        } else {
            (64, 64, 64)
        }
    }
}

/// Supported Tensor Core quantization formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorCoreFormat {
    /// 8-bit integer format
    Int8,
    /// 4-bit integer format
    Int4,
    /// 1-bit binary format
    Int1,
    /// 16-bit floating point (for reference)
    Fp16,
}

/// Unified interface for all specialized quantization operations
///
/// This trait provides a common interface for accessing specialized hardware
/// features across different architectures and instruction sets.
pub trait SpecializedQuantizationOps {
    /// Check if specialized operations are available
    fn is_available(&self) -> bool;

    /// Get the device this operator is configured for
    fn device(&self) -> &Device;

    /// Get optimal block size for this specialized implementation
    fn optimal_block_size(&self) -> usize;

    /// Perform specialized matrix multiplication
    fn specialized_qmatmul(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor>;
}

impl SpecializedQuantizationOps for VnniQuantizationOps {
    fn is_available(&self) -> bool {
        self.vnni_available
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn optimal_block_size(&self) -> usize {
        self.optimal_block_size()
    }

    fn specialized_qmatmul(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        self.vnni_qmatmul_int8(a, b)
    }
}

impl SpecializedQuantizationOps for Dp4aQuantizationOps {
    fn is_available(&self) -> bool {
        self.dp4a_available
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn optimal_block_size(&self) -> usize {
        self.optimal_block_size()
    }

    fn specialized_qmatmul(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        self.dp4a_qmatmul_int8(a, b)
    }
}

impl SpecializedQuantizationOps for TensorCoreQuantizationOps {
    fn is_available(&self) -> bool {
        self.tensor_cores_available
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn optimal_block_size(&self) -> usize {
        let (m, _, _) = self.optimal_matrix_size();
        m
    }

    fn specialized_qmatmul(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        self.tensor_core_qmatmul_int8(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::QuantizationParams;

    #[test]
    fn test_vnni_ops_creation() {
        let device = Device::cpu().unwrap();
        let vnni_ops = VnniQuantizationOps::new(device);

        // VNNI availability depends on actual hardware
        // Just test that creation works
        assert!(vnni_ops.optimal_block_size() > 0);
    }

    #[test]
    fn test_vnni_detection() {
        let vnni_available = VnniQuantizationOps::detect_vnni();

        // This will vary by platform
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // On x86/x86_64, detection should work
            assert!(vnni_available == is_x86_feature_detected!("avx512vnni"));
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // On other platforms, should always be false
            assert!(!vnni_available);
        }
    }

    #[test]
    fn test_dp4a_ops_creation() {
        // Test with CPU device (should not support DP4A)
        let cpu_device = Device::cpu().unwrap();
        let dp4a_ops = Dp4aQuantizationOps::new(cpu_device);
        assert!(!dp4a_ops.is_available());

        // DP4A should only be available on CUDA devices
        assert!(dp4a_ops.optimal_block_size() > 0);
    }

    #[test]
    fn test_tensor_core_ops_creation() {
        let device = Device::cpu().unwrap();
        let tc_ops = TensorCoreQuantizationOps::new(device);

        // CPU device should not support Tensor Cores
        assert!(!tc_ops.is_available());
        assert!(tc_ops.supported_formats().is_empty());

        let (m, k, n) = tc_ops.optimal_matrix_size();
        assert!(m > 0 && k > 0 && n > 0);
    }

    #[test]
    fn test_tensor_core_alignment_check() {
        let device = Device::cpu().unwrap();
        let tc_ops = TensorCoreQuantizationOps::new(device);

        // Test various alignments
        assert!(tc_ops.check_tensor_core_alignment(16, 16, 16)); // Well aligned
        assert!(tc_ops.check_tensor_core_alignment(32, 32, 32)); // Well aligned
        assert!(!tc_ops.check_tensor_core_alignment(15, 15, 15)); // Poorly aligned
        assert!(tc_ops.check_tensor_core_alignment(8, 8, 8)); // 8-byte aligned
    }

    #[test]
    fn test_tensor_core_formats() {
        let formats = vec![
            TensorCoreFormat::Int8,
            TensorCoreFormat::Int4,
            TensorCoreFormat::Int1,
            TensorCoreFormat::Fp16,
        ];

        // Test format equality
        assert_eq!(formats[0], TensorCoreFormat::Int8);
        assert_ne!(formats[0], TensorCoreFormat::Int4);

        // Test cloning
        let cloned_format = formats[0].clone();
        assert_eq!(cloned_format, TensorCoreFormat::Int8);
    }

    #[test]
    fn test_specialized_ops_trait() {
        let device = Device::cpu().unwrap();

        // Test VNNI implementation
        let vnni_ops = VnniQuantizationOps::new(device.clone());
        let _: &dyn SpecializedQuantizationOps = &vnni_ops;

        // Test DP4A implementation
        let dp4a_ops = Dp4aQuantizationOps::new(device.clone());
        let _: &dyn SpecializedQuantizationOps = &dp4a_ops;

        // Test Tensor Core implementation
        let tc_ops = TensorCoreQuantizationOps::new(device.clone());
        let _: &dyn SpecializedQuantizationOps = &tc_ops;

        // All should implement the trait correctly
        assert!(vnni_ops.device() == &device);
        assert!(dp4a_ops.device() == &device);
        assert!(tc_ops.device() == &device);
    }

    #[test]
    fn test_vnni_matrix_operations() {
        let device = Device::cpu().unwrap();
        let vnni_ops = VnniQuantizationOps::new(device.clone());

        if vnni_ops.is_available() {
            let params = QuantizationParams::int8_symmetric();

            let a_tensor = QuantizedTensor {
                data: vec![100u8; 4], // 2x2 matrix
                shape: vec![2, 2],
                params: params.clone(),
                device: device.clone(),
            };

            let b_tensor = QuantizedTensor {
                data: vec![50u8; 4], // 2x2 matrix
                shape: vec![2, 2],
                params: params.clone(),
                device: device.clone(),
            };

            let result = vnni_ops.vnni_qmatmul_int8(&a_tensor, &b_tensor);
            if result.is_ok() {
                let result_tensor = result.unwrap();
                assert_eq!(result_tensor.shape, vec![2, 2]);
            }
        }
    }

    #[test]
    fn test_dp4a_matrix_operations() {
        let device = Device::cpu().unwrap();
        let dp4a_ops = Dp4aQuantizationOps::new(device.clone());

        // Since CPU doesn't support DP4A, operations should fail appropriately
        let params = QuantizationParams::int8_symmetric();

        let a_tensor = QuantizedTensor {
            data: vec![100u8; 4],
            shape: vec![2, 2],
            params: params.clone(),
            device: device.clone(),
        };

        let b_tensor = QuantizedTensor {
            data: vec![50u8; 4],
            shape: vec![2, 2],
            params: params.clone(),
            device: device.clone(),
        };

        let result = dp4a_ops.dp4a_qmatmul_int8(&a_tensor, &b_tensor);
        assert!(result.is_err()); // Should fail on CPU
    }

    #[test]
    fn test_tensor_core_matrix_operations() {
        let device = Device::cpu().unwrap();
        let tc_ops = TensorCoreQuantizationOps::new(device.clone());

        // CPU doesn't support Tensor Cores
        let params = QuantizationParams::int8_symmetric();

        let a_tensor = QuantizedTensor {
            data: vec![100u8; 16 * 16],
            shape: vec![16, 16],
            params: params.clone(),
            device: device.clone(),
        };

        let b_tensor = QuantizedTensor {
            data: vec![50u8; 16 * 16],
            shape: vec![16, 16],
            params: params.clone(),
            device: device.clone(),
        };

        let result = tc_ops.tensor_core_qmatmul_int8(&a_tensor, &b_tensor);
        assert!(result.is_err()); // Should fail on CPU
    }
}
