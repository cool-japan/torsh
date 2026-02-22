//! CUDA kernel compiler.

use super::common_types::{GeneratedKernel, KernelOperation, KernelSpec};
use crate::error::BackendError;

/// CUDA kernel compiler
pub struct CudaCompiler {
    #[allow(dead_code)]
    nvcc_path: Option<String>,
}

impl CudaCompiler {
    pub fn new() -> Self {
        Self {
            nvcc_path: Self::find_nvcc(),
        }
    }

    fn find_nvcc() -> Option<String> {
        let paths = ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc", "nvcc"];
        for path in &paths {
            if std::process::Command::new(path)
                .arg("--version")
                .output()
                .is_ok()
            {
                return Some(path.to_string());
            }
        }
        None
    }

    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        let start_time = std::time::Instant::now();
        let source_code = match &spec.operation {
            KernelOperation::ElementwiseAdd => self.generate_cuda_elementwise_add(&spec)?,
            KernelOperation::ElementwiseMul => self.generate_cuda_elementwise_mul(&spec)?,
            KernelOperation::MatrixMultiply { m, n, k } => {
                self.generate_cuda_matmul(&spec, *m, *n, *k)?
            }
            _ => {
                return Err(BackendError::NotImplemented(format!(
                    "CUDA kernel generation not implemented for {:?}",
                    spec.operation
                )));
            }
        };
        let compilation_time = start_time.elapsed().as_millis() as u64;
        Ok(GeneratedKernel {
            source_code,
            entry_point: "kernel_main".to_string(),
            compiled_binary: None,
            spec,
            compilation_time_ms: compilation_time,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        })
    }

    fn generate_cuda_elementwise_add(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_c_type();
        let _block_size = spec.workgroup_size.unwrap_or((256, 1, 1)).0;
        let source = format!(
            r#"
extern "C" __global__ void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input_a[idx] + input_b[idx];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cuda_elementwise_mul(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_c_type();
        let source = format!(
            r#"
extern "C" __global__ void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input_a[idx] * input_b[idx];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cuda_matmul(
        &self,
        spec: &KernelSpec,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_c_type();
        let tile_size = 16;
        let source = format!(
            r#"
#define TILE_SIZE {tile_size}

extern "C" __global__ void kernel_main(
    const {data_type}* __restrict__ A,
    const {data_type}* __restrict__ B,
    {data_type}* __restrict__ C,
    int M, int N, int K
) {{
    __shared__ {data_type} tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ {data_type} tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    {data_type} sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {{
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        
        if (a_row < M && a_col < K) {{
            tile_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        }} else {{
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }}
        
        if (b_row < K && b_col < N) {{
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        }} else {{
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }}
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {{
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }}
        
        __syncthreads();
    }}
    
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#,
            data_type = data_type,
            tile_size = tile_size
        );
        Ok(source)
    }
}
