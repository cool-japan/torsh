//! CPU kernel compiler with SIMD optimization.

use super::common_types::{
    CpuSimdSupport, GeneratedKernel, KernelDataType, KernelOperation, KernelSpec, ReductionOp,
};
use crate::error::BackendError;

/// CPU kernel compiler with SIMD optimization
pub struct CpuCompiler {
    compiler_available: bool,
    pub(crate) simd_support: CpuSimdSupport,
}

impl CpuCompiler {
    pub fn new() -> Self {
        Self {
            compiler_available: Self::check_compiler_availability(),
            simd_support: Self::detect_simd_support(),
        }
    }

    fn check_compiler_availability() -> bool {
        let compilers = ["gcc", "clang", "cl", "icc"];
        for compiler in &compilers {
            if std::process::Command::new(compiler)
                .arg("--version")
                .output()
                .is_ok()
            {
                return true;
            }
        }
        false
    }

    fn detect_simd_support() -> CpuSimdSupport {
        let mut support = CpuSimdSupport {
            sse2: false,
            sse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            neon: false,
        };
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "sse2")]
            {
                support.sse2 = true;
            }
            #[cfg(target_feature = "sse3")]
            {
                support.sse3 = true;
            }
            #[cfg(target_feature = "sse4.1")]
            {
                support.sse4_1 = true;
            }
            #[cfg(target_feature = "sse4.2")]
            {
                support.sse4_2 = true;
            }
            #[cfg(target_feature = "avx")]
            {
                support.avx = true;
            }
            #[cfg(target_feature = "avx2")]
            {
                support.avx2 = true;
            }
            #[cfg(target_feature = "avx512f")]
            {
                support.avx512f = true;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_feature = "neon")]
            {
                support.neon = true;
            }
        }
        support
    }

    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        if !self.compiler_available {
            return Err(BackendError::BackendError(
                "No C compiler available for CPU kernel generation".to_string(),
            ));
        }
        let start_time = std::time::Instant::now();
        let source_code = match &spec.operation {
            KernelOperation::ElementwiseAdd => self.generate_cpu_elementwise_add(&spec)?,
            KernelOperation::ElementwiseMul => self.generate_cpu_elementwise_mul(&spec)?,
            KernelOperation::ElementwiseDiv => self.generate_cpu_elementwise_div(&spec)?,
            KernelOperation::ElementwiseSub => self.generate_cpu_elementwise_sub(&spec)?,
            KernelOperation::MatrixMultiply { m, n, k } => {
                self.generate_cpu_matmul(&spec, *m, *n, *k)?
            }
            KernelOperation::ReLU => self.generate_cpu_relu(&spec)?,
            KernelOperation::GELU => self.generate_cpu_gelu(&spec)?,
            KernelOperation::Softmax { dim } => self.generate_cpu_softmax(&spec, *dim)?,
            KernelOperation::Transpose { dims } => self.generate_cpu_transpose(&spec, dims)?,
            KernelOperation::Reduction { op, dim } => {
                self.generate_cpu_reduction(&spec, op, *dim)?
            }
            _ => {
                return Err(BackendError::NotImplemented(format!(
                    "CPU kernel generation not implemented for {:?}",
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

    fn generate_cpu_elementwise_add(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let (vector_includes, vector_ops) =
            self.generate_simd_operations(&spec.operation, data_type)?;
        let source = format!(
            r#"
{vector_includes}
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    {vector_ops}

    // Fallback scalar implementation
    #pragma omp parallel for
    for (size_t i = vector_end; i < size; ++i) {{
        output[i] = input_a[i] + input_b[i];
    }}
}}
"#,
            data_type = data_type,
            vector_includes = vector_includes,
            vector_ops = vector_ops
        );
        Ok(source)
    }

    fn generate_cpu_elementwise_mul(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let (vector_includes, vector_ops) =
            self.generate_simd_operations(&spec.operation, data_type)?;
        let source = format!(
            r#"
{vector_includes}
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    {vector_ops}

    // Fallback scalar implementation
    #pragma omp parallel for
    for (size_t i = vector_end; i < size; ++i) {{
        output[i] = input_a[i] * input_b[i];
    }}
}}
"#,
            data_type = data_type,
            vector_includes = vector_includes,
            vector_ops = vector_ops
        );
        Ok(source)
    }

    fn generate_cpu_elementwise_div(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = input_a[i] / input_b[i];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cpu_elementwise_sub(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = input_a[i] - input_b[i];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cpu_matmul(
        &self,
        spec: &KernelSpec,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ A,
    const {data_type}* __restrict__ B,
    {data_type}* __restrict__ C,
    size_t M, size_t N, size_t K
) {{
    const size_t BLOCK_SIZE = 64;

    #pragma omp parallel for collapse(2)
    for (size_t bi = 0; bi < M; bi += BLOCK_SIZE) {{
        for (size_t bj = 0; bj < N; bj += BLOCK_SIZE) {{
            for (size_t bk = 0; bk < K; bk += BLOCK_SIZE) {{
                size_t i_max = (bi + BLOCK_SIZE < M) ? bi + BLOCK_SIZE : M;
                size_t j_max = (bj + BLOCK_SIZE < N) ? bj + BLOCK_SIZE : N;
                size_t k_max = (bk + BLOCK_SIZE < K) ? bk + BLOCK_SIZE : K;

                for (size_t i = bi; i < i_max; ++i) {{
                    for (size_t j = bj; j < j_max; ++j) {{
                        {data_type} sum = C[i * N + j];
                        for (size_t k = bk; k < k_max; ++k) {{
                            sum += A[i * K + k] * B[k * N + j];
                        }}
                        C[i * N + j] = sum;
                    }}
                }}
            }}
        }}
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cpu_relu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let source = format!(
            r#"
#include <omp.h>
#include <algorithm>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = std::max(input[i], ({data_type})0.0);
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cpu_gelu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let source = format!(
            r#"
#include <omp.h>
#include <cmath>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    const {data_type} sqrt_2_over_pi = 0.7978845608028654;
    const {data_type} a = 0.044715;

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        const {data_type} x = input[i];
        const {data_type} inner = sqrt_2_over_pi * (x + a * x * x * x);
        output[i] = 0.5 * x * (1.0 + std::tanh(inner));
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cpu_softmax(&self, spec: &KernelSpec, _dim: usize) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let source = format!(
            r#"
#include <omp.h>
#include <cmath>
#include <algorithm>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    // Find maximum for numerical stability
    {data_type} max_val = input[0];
    for (size_t i = 1; i < size; ++i) {{
        max_val = std::max(max_val, input[i]);
    }}

    // Compute exponentials
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = std::exp(input[i] - max_val);
    }}

    // Compute sum
    {data_type} sum = 0.0;
    for (size_t i = 0; i < size; ++i) {{
        sum += output[i];
    }}

    // Normalize
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = output[i] / sum;
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cpu_transpose(
        &self,
        spec: &KernelSpec,
        _dims: &[usize],
    ) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t rows,
    size_t cols
) {{
    const size_t BLOCK_SIZE = 32;

    #pragma omp parallel for collapse(2)
    for (size_t bi = 0; bi < rows; bi += BLOCK_SIZE) {{
        for (size_t bj = 0; bj < cols; bj += BLOCK_SIZE) {{
            size_t i_max = (bi + BLOCK_SIZE < rows) ? bi + BLOCK_SIZE : rows;
            size_t j_max = (bj + BLOCK_SIZE < cols) ? bj + BLOCK_SIZE : cols;

            for (size_t i = bi; i < i_max; ++i) {{
                for (size_t j = bj; j < j_max; ++j) {{
                    output[j * rows + i] = input[i * cols + j];
                }}
            }}
        }}
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_cpu_reduction(
        &self,
        spec: &KernelSpec,
        op: &ReductionOp,
        _dim: Option<usize>,
    ) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let (init_value, _combine_op) = match op {
            ReductionOp::Sum => ("0.0", "+"),
            ReductionOp::Max => match data_type {
                "float" => ("-std::numeric_limits<float>::infinity()", "std::max"),
                "double" => ("-std::numeric_limits<double>::infinity()", "std::max"),
                _ => ("0.0", "std::max"),
            },
            ReductionOp::Min => match data_type {
                "float" => ("std::numeric_limits<float>::infinity()", "std::min"),
                "double" => ("std::numeric_limits<double>::infinity()", "std::min"),
                _ => ("0.0", "std::min"),
            },
            ReductionOp::Mean => ("0.0", "+"),
            ReductionOp::Product => ("1.0", "*"),
        };
        let source = format!(
            r#"
#include <omp.h>
#include <limits>
#include <algorithm>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    {data_type} result = {init_value};

    #pragma omp parallel for reduction({combine_op}:result)
    for (size_t i = 0; i < size; ++i) {{
        {reduction_expr}
    }}

    {post_process}
    output[0] = result;
}}
"#,
            data_type = data_type,
            init_value = init_value,
            combine_op = match op {
                ReductionOp::Sum | ReductionOp::Mean => "+",
                ReductionOp::Product => "*",
                _ => "+",
            },
            reduction_expr = match op {
                ReductionOp::Sum | ReductionOp::Mean | ReductionOp::Product =>
                    "result = result {} input[i];".replace(
                        "{}",
                        match op {
                            ReductionOp::Sum | ReductionOp::Mean => "+",
                            ReductionOp::Product => "*",
                            _ => "+",
                        },
                    ),
                ReductionOp::Max => "result = std::max(result, input[i]);".to_string(),
                ReductionOp::Min => "result = std::min(result, input[i]);".to_string(),
            },
            post_process = match op {
                ReductionOp::Mean => "result = result / size;",
                _ => "",
            }
        );
        Ok(source)
    }

    fn generate_simd_operations(
        &self,
        operation: &KernelOperation,
        data_type: &str,
    ) -> Result<(String, String), BackendError> {
        let includes = if self.simd_support.avx2 {
            "#include <immintrin.h>"
        } else if self.simd_support.sse2 {
            "#include <emmintrin.h>"
        } else if self.simd_support.neon {
            "#include <arm_neon.h>"
        } else {
            ""
        };
        let vector_ops = match (operation, data_type) {
            (KernelOperation::ElementwiseAdd, "float") if self.simd_support.avx2 => {
                r#"
    size_t vector_end = (size / 8) * 8;

    for (size_t i = 0; i < vector_end; i += 8) {
        __m256 a = _mm256_load_ps(&input_a[i]);
        __m256 b = _mm256_load_ps(&input_b[i]);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_store_ps(&output[i], result);
    }
"#
            }
            (KernelOperation::ElementwiseMul, "float") if self.simd_support.avx2 => {
                r#"
    size_t vector_end = (size / 8) * 8;

    for (size_t i = 0; i < vector_end; i += 8) {
        __m256 a = _mm256_load_ps(&input_a[i]);
        __m256 b = _mm256_load_ps(&input_b[i]);
        __m256 result = _mm256_mul_ps(a, b);
        _mm256_store_ps(&output[i], result);
    }
"#
            }
            (KernelOperation::ElementwiseAdd, "float") if self.simd_support.sse2 => {
                r#"
    size_t vector_end = (size / 4) * 4;

    for (size_t i = 0; i < vector_end; i += 4) {
        __m128 a = _mm_load_ps(&input_a[i]);
        __m128 b = _mm_load_ps(&input_b[i]);
        __m128 result = _mm_add_ps(a, b);
        _mm_store_ps(&output[i], result);
    }
"#
            }
            (KernelOperation::ElementwiseMul, "float") if self.simd_support.sse2 => {
                r#"
    size_t vector_end = (size / 4) * 4;

    for (size_t i = 0; i < vector_end; i += 4) {
        __m128 a = _mm_load_ps(&input_a[i]);
        __m128 b = _mm_load_ps(&input_b[i]);
        __m128 result = _mm_mul_ps(a, b);
        _mm_store_ps(&output[i], result);
    }
"#
            }
            _ => "size_t vector_end = 0; // No vectorization available",
        };
        Ok((includes.to_string(), vector_ops.to_string()))
    }

    pub(crate) fn cpu_type(&self, data_type: KernelDataType) -> Result<&'static str, BackendError> {
        match data_type {
            KernelDataType::F32 => Ok("float"),
            KernelDataType::F64 => Ok("double"),
            KernelDataType::I32 => Ok("int32_t"),
            KernelDataType::I64 => Ok("int64_t"),
            KernelDataType::U32 => Ok("uint32_t"),
            KernelDataType::U64 => Ok("uint64_t"),
            KernelDataType::F16 => Err(BackendError::NotImplemented(
                "F16 not supported in CPU kernels without specific compiler extensions".to_string(),
            )),
            KernelDataType::BF16 => Err(BackendError::NotImplemented(
                "BF16 not supported in CPU kernels".to_string(),
            )),
        }
    }
}
