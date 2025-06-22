//! CPU Kernel Implementation

use crate::optimized_kernels;
use torsh_backends::kernel::{KernelHandle, KernelMetadata};
use torsh_backends::{
    BackendError, BackendResult, Buffer, Device, Kernel, KernelDescriptor, KernelLaunchConfig,
};
use torsh_core::error::{Result, TorshError};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// CPU kernel function signature
pub type CpuKernelFn = fn(&[&Buffer], &[u8], &KernelLaunchConfig) -> Result<()>;

/// CPU kernel implementation
pub struct CpuKernel {
    name: String,
}

impl CpuKernel {
    /// Create a new CPU kernel from a descriptor
    pub fn new(descriptor: &KernelDescriptor) -> BackendResult<Self> {
        Ok(Self {
            name: descriptor.name.clone(),
        })
    }

    /// Create a CPU kernel and return an abstract Kernel
    pub fn new_kernel(device: Device, descriptor: &KernelDescriptor) -> BackendResult<Kernel> {
        let _cpu_kernel = Self::new(descriptor)?;

        let handle = KernelHandle::Generic {
            handle: Box::new("CPU kernel placeholder".to_string()),
        };

        let metadata = KernelMetadata {
            compile_time_ms: 1.0,
            binary_size: 0,
            registers_per_thread: None,
            shared_memory_usage: None,
            max_workgroup_size: Some((u32::MAX, 1, 1)),
            compiler_version: "CPU Backend".to_string(),
            warnings: Vec::new(),
            performance_hints: vec!["Use SIMD for better performance".to_string()],
        };

        let kernel = Kernel::new(
            0,
            device,
            descriptor.name.clone(),
            descriptor.clone(),
            handle,
            metadata,
        );

        Ok(kernel)
    }

    /// Get the kernel name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Helper function to get CPU buffer as f32 slice
    fn get_cpu_buffer_f32(buffer: &Buffer) -> Result<&[f32]> {
        match &buffer.handle {
            torsh_backends::buffer::BufferHandle::Cpu { ptr, size } => unsafe {
                Ok(std::slice::from_raw_parts(*ptr as *const f32, size / 4))
            },
            _ => Err(TorshError::InvalidArgument(
                "Buffer is not CPU buffer".to_string(),
            )),
        }
    }

    /// Helper function to get CPU buffer as mutable f32 slice
    fn get_cpu_buffer_f32_mut(buffer: &Buffer) -> Result<&mut [f32]> {
        match &buffer.handle {
            torsh_backends::buffer::BufferHandle::Cpu { ptr, size } => unsafe {
                Ok(std::slice::from_raw_parts_mut(*ptr as *mut f32, size / 4))
            },
            _ => Err(TorshError::InvalidArgument(
                "Buffer is not CPU buffer".to_string(),
            )),
        }
    }

    /// Execute the kernel
    pub async fn execute(
        &self,
        _buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> BackendResult<()> {
        Err(BackendError::ComputeError {
            reason: "CPU kernel execution not yet implemented".to_string(),
        }
        .into())
    }

    /// Get kernel function by name
    pub fn get_kernel_fn(descriptor: &KernelDescriptor) -> Result<CpuKernelFn> {
        let kernel_fn: CpuKernelFn = match descriptor.name.as_str() {
            "add" => Self::kernel_add,
            "mul" => Self::kernel_mul,
            "sub" => Self::kernel_sub,
            "div" => Self::kernel_div,
            "relu" => Self::kernel_relu,
            "sigmoid" => Self::kernel_sigmoid,
            "tanh" => Self::kernel_tanh,
            "matmul" => Self::kernel_matmul,
            "dot" => Self::kernel_dot,
            "sum" => Self::kernel_sum,
            "mean" => Self::kernel_mean,
            _ => {
                return Err(TorshError::InvalidArgument(format!(
                    "Unsupported kernel: {}",
                    descriptor.name
                )))
            }
        };

        Ok(kernel_fn)
    }

    // Built-in kernel implementations

    /// Element-wise addition kernel
    fn kernel_add(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Add kernel requires 3 buffers".to_string(),
            ));
        }

        let a = Self::get_cpu_buffer_f32(buffers[0])?;
        let b = Self::get_cpu_buffer_f32(buffers[1])?;
        let result = Self::get_cpu_buffer_f32_mut(buffers[2])?;

        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::InvalidArgument(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Use optimized parallel addition
        optimized_kernels::parallel_ops::parallel_elementwise(a, b, result, |x, y| x + y);

        Ok(())
    }

    /// Element-wise multiplication kernel
    fn kernel_mul(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Mul kernel requires 3 buffers".to_string(),
            ));
        }

        let a = Self::get_cpu_buffer_f32(buffers[0])?;
        let b = Self::get_cpu_buffer_f32(buffers[1])?;
        let result = Self::get_cpu_buffer_f32_mut(buffers[2])?;

        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::InvalidArgument(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Use optimized parallel multiplication
        optimized_kernels::parallel_ops::parallel_elementwise(a, b, result, |x, y| x * y);

        Ok(())
    }

    /// Element-wise subtraction kernel
    fn kernel_sub(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Sub kernel requires 3 buffers".to_string(),
            ));
        }

        let a = Self::get_cpu_buffer_f32(buffers[0])?;
        let b = Self::get_cpu_buffer_f32(buffers[1])?;
        let result = Self::get_cpu_buffer_f32_mut(buffers[2])?;

        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::InvalidArgument(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Use optimized parallel subtraction
        optimized_kernels::parallel_ops::parallel_elementwise(a, b, result, |x, y| x - y);

        Ok(())
    }

    /// Element-wise division kernel
    fn kernel_div(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Div kernel requires 3 buffers".to_string(),
            ));
        }

        let a = Self::get_cpu_buffer_f32(buffers[0])?;
        let b = Self::get_cpu_buffer_f32(buffers[1])?;
        let result = Self::get_cpu_buffer_f32_mut(buffers[2])?;

        if a.len() != b.len() || a.len() != result.len() {
            return Err(TorshError::InvalidArgument(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Use optimized parallel division
        optimized_kernels::parallel_ops::parallel_elementwise(a, b, result, |x, y| x / y);

        Ok(())
    }

    /// ReLU activation kernel
    fn kernel_relu(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "ReLU kernel requires 2 buffers".to_string(),
            ));
        }

        let input = Self::get_cpu_buffer_f32(buffers[0])?;
        let output = Self::get_cpu_buffer_f32_mut(buffers[1])?;

        if input.len() != output.len() {
            return Err(TorshError::InvalidArgument(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Use optimized parallel ReLU
        optimized_kernels::parallel_ops::parallel_unary(input, output, |x| x.max(0.0));

        Ok(())
    }

    /// Sigmoid activation kernel
    fn kernel_sigmoid(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Sigmoid kernel requires 2 buffers".to_string(),
            ));
        }

        let input = Self::get_cpu_buffer_f32(buffers[0])?;
        let output = Self::get_cpu_buffer_f32_mut(buffers[1])?;

        if input.len() != output.len() {
            return Err(TorshError::InvalidArgument(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Use optimized parallel Sigmoid
        optimized_kernels::parallel_ops::parallel_unary(input, output, |x| {
            1.0 / (1.0 + (-x).exp())
        });

        Ok(())
    }

    /// Tanh activation kernel
    fn kernel_tanh(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Tanh kernel requires 2 buffers".to_string(),
            ));
        }

        let input = Self::get_cpu_buffer_f32(buffers[0])?;
        let output = Self::get_cpu_buffer_f32_mut(buffers[1])?;

        if input.len() != output.len() {
            return Err(TorshError::InvalidArgument(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Use optimized parallel Tanh
        optimized_kernels::parallel_ops::parallel_unary(input, output, |x| x.tanh());

        Ok(())
    }

    /// Matrix multiplication kernel using BLAS
    fn kernel_matmul(
        buffers: &[&Buffer],
        uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Matmul kernel requires 3 buffers".to_string(),
            ));
        }

        let a = Self::get_cpu_buffer_f32(buffers[0])?;
        let b = Self::get_cpu_buffer_f32(buffers[1])?;
        let result = Self::get_cpu_buffer_f32_mut(buffers[2])?;

        // Parse matrix dimensions from uniform data
        // Expected format: [m: u32, n: u32, k: u32, transpose_a: u8, transpose_b: u8]
        if uniform_data.len() < 14 {
            return Err(TorshError::InvalidArgument(
                "Insufficient uniform data for matmul (need m, n, k, transpose_a, transpose_b)"
                    .to_string(),
            ));
        }

        let m = u32::from_le_bytes([
            uniform_data[0],
            uniform_data[1],
            uniform_data[2],
            uniform_data[3],
        ]) as usize;
        let n = u32::from_le_bytes([
            uniform_data[4],
            uniform_data[5],
            uniform_data[6],
            uniform_data[7],
        ]) as usize;
        let k = u32::from_le_bytes([
            uniform_data[8],
            uniform_data[9],
            uniform_data[10],
            uniform_data[11],
        ]) as usize;
        let transpose_a = uniform_data[12] != 0;
        let transpose_b = uniform_data[13] != 0;

        // Use optimized matrix multiplication
        optimized_kernels::optimized_matmul(a, b, result, m, n, k, transpose_a, transpose_b)
    }

    /// Dot product kernel using BLAS
    fn kernel_dot(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Dot kernel requires 3 buffers".to_string(),
            ));
        }

        let a = Self::get_cpu_buffer_f32(buffers[0])?;
        let b = Self::get_cpu_buffer_f32(buffers[1])?;
        let result = Self::get_cpu_buffer_f32_mut(buffers[2])?;

        if result.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Output buffer should have size 1 for dot product".to_string(),
            ));
        }

        // Use optimized dot product
        result[0] = optimized_kernels::optimized_dot(a, b)?;

        Ok(())
    }

    /// Sum reduction kernel
    fn kernel_sum(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Sum kernel requires 2 buffers".to_string(),
            ));
        }

        let input = Self::get_cpu_buffer_f32(buffers[0])?;
        let output = Self::get_cpu_buffer_f32_mut(buffers[1])?;

        if output.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Output buffer should have size 1 for sum reduction".to_string(),
            ));
        }

        // Use optimized parallel sum
        output[0] = optimized_kernels::parallel_ops::parallel_sum(input);

        Ok(())
    }

    /// Mean reduction kernel
    fn kernel_mean(
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _launch_config: &KernelLaunchConfig,
    ) -> Result<()> {
        if buffers.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Mean kernel requires 2 buffers".to_string(),
            ));
        }

        let input = Self::get_cpu_buffer_f32(buffers[0])?;
        let output = Self::get_cpu_buffer_f32_mut(buffers[1])?;

        if output.len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Output buffer should have size 1 for mean reduction".to_string(),
            ));
        }

        // Use optimized parallel mean
        output[0] = optimized_kernels::parallel_ops::parallel_mean(input);

        Ok(())
    }
}

// Extension trait for Kernel to work with CPU kernels
pub trait KernelCpuExt {
    fn is_cpu(&self) -> bool;
}

impl KernelCpuExt for Kernel {
    fn is_cpu(&self) -> bool {
        matches!(self.handle, KernelHandle::Generic { .. })
    }
}

/// CPU kernel executor for managing kernel execution
pub struct CpuKernelExecutor;

impl CpuKernelExecutor {
    /// Create a new CPU kernel executor
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuKernelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_backends::buffer::{BufferDescriptor, BufferHandle, BufferUsage};
    use torsh_backends::kernel::{KernelLanguage, KernelSource};
    use torsh_core::device::DeviceType;

    #[test]
    fn test_cpu_kernel_creation() {
        let descriptor = KernelDescriptor::new(
            "add".to_string(),
            KernelSource::Source {
                code: "// CPU add kernel".to_string(),
                language: KernelLanguage::Custom("CPU".to_string()),
            },
        );

        let kernel = CpuKernel::new(&descriptor).unwrap();
        assert_eq!(kernel.name(), "add");
    }
}
