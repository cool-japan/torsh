//! CUDA kernels for tensor operations

pub mod neural_ops;
pub mod reduction_ops;
pub mod tensor_ops;

use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;

/// Launch configuration for CUDA kernels
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory: usize,
}

impl LaunchConfig {
    /// Create 1D launch configuration
    pub fn new_1d(total_threads: usize, block_size: u32) -> Self {
        let grid_size = ((total_threads as u32 + block_size - 1) / block_size, 1, 1);
        Self {
            grid_size,
            block_size: (block_size, 1, 1),
            shared_memory: 0,
        }
    }

    /// Create 2D launch configuration
    pub fn new_2d(width: usize, height: usize, block_x: u32, block_y: u32) -> Self {
        let grid_x = (width as u32 + block_x - 1) / block_x;
        let grid_y = (height as u32 + block_y - 1) / block_y;

        Self {
            grid_size: (grid_x, grid_y, 1),
            block_size: (block_x, block_y, 1),
            shared_memory: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_memory(mut self, shared_memory: usize) -> Self {
        self.shared_memory = shared_memory;
        self
    }

    /// Get optimal block size for 1D operations
    pub fn optimal_block_size_1d(device_props: &crate::device::DeviceProperties) -> u32 {
        // Use warp size as base, typically 32 for NVIDIA GPUs
        let warp_size = device_props.warp_size;

        // Choose block size based on compute capability
        if device_props.compute_capability >= 75 {
            512.min(device_props.max_threads_per_block)
        } else if device_props.compute_capability >= 50 {
            256.min(device_props.max_threads_per_block)
        } else {
            128.min(device_props.max_threads_per_block)
        }
    }

    /// Get optimal block size for 2D operations
    pub fn optimal_block_size_2d(device_props: &crate::device::DeviceProperties) -> (u32, u32) {
        let total_threads = Self::optimal_block_size_1d(device_props);

        // For 2D operations, use square blocks when possible
        let side = (total_threads as f32).sqrt() as u32;
        let side = (side / device_props.warp_size) * device_props.warp_size;

        if side * side <= device_props.max_threads_per_block {
            (side, side)
        } else {
            // Fall back to rectangular blocks
            (
                device_props.warp_size,
                total_threads / device_props.warp_size,
            )
        }
    }
}

/// Kernel function trait
pub trait KernelFunction<Args> {
    fn launch(&self, config: &LaunchConfig, stream: &CudaStream, args: Args) -> CudaResult<()>;
}

/// Generic kernel launcher
pub struct Kernel<F, Args> {
    function: F,
    _phantom: std::marker::PhantomData<Args>,
}

impl<F, Args> Kernel<F, Args> {
    pub fn new(function: F) -> Self {
        Self {
            function,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, Args> KernelFunction<Args> for Kernel<F, Args>
where
    F: Fn(&LaunchConfig, &CudaStream, Args) -> CudaResult<()>,
{
    fn launch(&self, config: &LaunchConfig, stream: &CudaStream, args: Args) -> CudaResult<()> {
        (self.function)(config, stream, args)
    }
}

/// Kernel registry for compiled kernels
pub struct KernelRegistry {
    module: cust::Module,
}

impl KernelRegistry {
    /// Load kernels from PTX
    pub fn load_from_ptx(ptx: &str) -> CudaResult<Self> {
        let module = cust::Module::load_from_string(ptx)?;
        Ok(Self { module })
    }

    /// Get kernel function by name
    pub fn get_function(&self, name: &str) -> CudaResult<cust::Function> {
        Ok(self.module.get_function(name)?)
    }

    /// Launch kernel by name
    pub fn launch_kernel(
        &self,
        name: &str,
        config: &LaunchConfig,
        stream: &CudaStream,
        args: &[&dyn cust::DeviceCopy],
    ) -> CudaResult<()> {
        let func = self.get_function(name)?;

        unsafe {
            cust::launch!(
                func<<<config.grid_size, config.block_size, config.shared_memory, stream.raw()>>>(
                    args.as_ptr()
                )
            )?;
        }

        Ok(())
    }
}

/// Common kernel arguments
#[derive(Debug)]
pub struct ElementwiseArgs<T> {
    pub input: cust::DevicePointer<T>,
    pub output: cust::DevicePointer<T>,
    pub size: usize,
}

#[derive(Debug)]
pub struct BinaryArgs<T> {
    pub lhs: cust::DevicePointer<T>,
    pub rhs: cust::DevicePointer<T>,
    pub output: cust::DevicePointer<T>,
    pub size: usize,
}

#[derive(Debug)]
pub struct ReductionArgs<T> {
    pub input: cust::DevicePointer<T>,
    pub output: cust::DevicePointer<T>,
    pub size: usize,
    pub axis: i32,
}

#[derive(Debug)]
pub struct MatmulArgs<T> {
    pub a: cust::DevicePointer<T>,
    pub b: cust::DevicePointer<T>,
    pub c: cust::DevicePointer<T>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config_1d() {
        let config = LaunchConfig::new_1d(1000, 256);
        assert_eq!(config.grid_size, (4, 1, 1)); // (1000 + 256 - 1) / 256 = 4
        assert_eq!(config.block_size, (256, 1, 1));
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::new_2d(1024, 768, 16, 16);
        assert_eq!(config.grid_size, (64, 48, 1)); // 1024/16 = 64, 768/16 = 48
        assert_eq!(config.block_size, (16, 16, 1));
    }

    #[test]
    fn test_optimal_block_sizes() {
        let props = crate::device::DeviceProperties {
            name: "Test GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            compute_capability: 75,
            multiprocessor_count: 80,
            max_threads_per_block: 1024,
            warp_size: 32,
        };

        let block_1d = LaunchConfig::optimal_block_size_1d(&props);
        assert_eq!(block_1d, 512);

        let (block_x, block_y) = LaunchConfig::optimal_block_size_2d(&props);
        assert!(block_x * block_y <= props.max_threads_per_block);
    }
}
