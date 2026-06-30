//! CPU FFT implementation backed by the `oxifft` engine.
//!
//! All complex data is stored in the backend [`Buffer`] as interleaved
//! single-precision floats (`re, im, re, im, ...`). Each `f32` occupies 4 bytes,
//! so a complex element occupies 8 bytes and a buffer holds
//! `buffer.size / 8` complex values.

use crate::cpu::buffer::BufferCpuExt;
use crate::fft::{FftDirection, FftExecutor, FftNormalization, FftOps, FftPlan, FftType};
use crate::{BackendResult, Buffer, Device};
use oxifft::{Complex, Direction, Flags, Plan};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Number of bytes occupied by a single interleaved complex `f32` value.
const COMPLEX_F32_BYTES: usize = 2 * core::mem::size_of::<f32>();

/// CPU FFT operations implementation
#[derive(Clone, Debug)]
pub struct CpuFftOps {
    /// Cache for FFT executors
    executor_cache: Arc<Mutex<HashMap<String, Arc<CpuFftExecutor>>>>,
    /// Number of threads to use for parallel FFT
    num_threads: usize,
}

impl CpuFftOps {
    /// Create a new CPU FFT operations instance
    pub fn new(num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| rayon::current_num_threads());

        Self {
            executor_cache: Arc::new(Mutex::new(HashMap::new())),
            num_threads,
        }
    }

    /// Get or create an FFT executor for the given plan
    fn get_or_create_executor(&self, plan: &FftPlan) -> BackendResult<Arc<CpuFftExecutor>> {
        let mut cache = self
            .executor_cache
            .lock()
            .expect("lock should not be poisoned");

        if let Some(executor) = cache.get(&plan.id) {
            return Ok(executor.clone());
        }

        let executor = Arc::new(CpuFftExecutor::new(plan.clone(), self.num_threads)?);
        cache.insert(plan.id.clone(), executor.clone());
        Ok(executor)
    }

    /// Compute the normalization factor for an FFT of length `n`.
    ///
    /// This mirrors the semantics of [`FftNormalization`]:
    /// - [`FftNormalization::None`] never scales.
    /// - [`FftNormalization::Backward`] scales the inverse transform by `1/n`
    ///   (the forward transform is left unscaled), matching the convention used
    ///   by NumPy/PyTorch and the higher-level `torsh-functional` spectral code.
    /// - [`FftNormalization::Ortho`] scales both directions by `1/sqrt(n)`.
    fn normalization_factor(
        size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> f32 {
        let n = size as f32;
        match normalization {
            FftNormalization::None => 1.0,
            FftNormalization::Backward => {
                if matches!(direction, FftDirection::Inverse) {
                    1.0 / n
                } else {
                    1.0
                }
            }
            FftNormalization::Ortho => 1.0 / n.sqrt(),
        }
    }

    /// Read an interleaved complex `f32` buffer into a `Vec<Complex<f32>>`.
    ///
    /// The buffer must be a CPU buffer and hold at least `count` complex
    /// elements (`count * 8` bytes). An honest error is returned when the
    /// underlying memory is not accessible rather than fabricating data.
    fn read_complex_buffer(buffer: &Buffer, count: usize) -> BackendResult<Vec<Complex<f32>>> {
        if !buffer.is_cpu() {
            return Err(torsh_core::error::TorshError::BackendError(
                "FFT: buffer data not accessible for CPU compute".to_string(),
            ));
        }

        let required_bytes = count * COMPLEX_F32_BYTES;
        if required_bytes > buffer.size {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "FFT: buffer holds {} bytes but {} are required for {} complex elements",
                buffer.size, required_bytes, count
            )));
        }

        let ptr = buffer.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "FFT: buffer data not accessible for CPU compute".to_string(),
            )
        })?;

        let mut values = Vec::with_capacity(count);
        // SAFETY: `ptr` references at least `required_bytes` of initialized
        // CPU memory (validated above) laid out as interleaved `f32` pairs.
        unsafe {
            let floats = std::slice::from_raw_parts(ptr as *const f32, count * 2);
            for pair in floats.chunks_exact(2) {
                values.push(Complex::new(pair[0], pair[1]));
            }
        }

        Ok(values)
    }

    /// Write a slice of complex values back into an interleaved `f32` buffer,
    /// scaling every component by `scale` as it is written.
    fn write_complex_buffer(
        buffer: &Buffer,
        values: &[Complex<f32>],
        scale: f32,
    ) -> BackendResult<()> {
        if !buffer.is_cpu() {
            return Err(torsh_core::error::TorshError::BackendError(
                "FFT: buffer data not accessible for CPU compute".to_string(),
            ));
        }

        let required_bytes = values.len() * COMPLEX_F32_BYTES;
        if required_bytes > buffer.size {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "FFT: output buffer holds {} bytes but {} are required",
                buffer.size, required_bytes
            )));
        }

        let ptr = buffer.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "FFT: buffer data not accessible for CPU compute".to_string(),
            )
        })?;

        // SAFETY: `ptr` references at least `required_bytes` of CPU memory
        // (validated above) laid out as interleaved `f32` pairs.
        unsafe {
            let floats = std::slice::from_raw_parts_mut(ptr as *mut f32, values.len() * 2);
            for (slot, value) in floats.chunks_exact_mut(2).zip(values.iter()) {
                slot[0] = value.re * scale;
                slot[1] = value.im * scale;
            }
        }

        Ok(())
    }

    /// Execute a batched 1D complex-to-complex FFT.
    ///
    /// `input`/`output` contain `batch * size` interleaved complex `f32`
    /// elements. Each contiguous run of `size` elements is transformed
    /// independently using a single shared [`Plan`].
    fn execute_dft_1d(
        &self,
        input: &Buffer,
        output: &Buffer,
        size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        if size == 0 {
            return Err(torsh_core::error::TorshError::BackendError(
                "FFT: transform size must be greater than zero".to_string(),
            ));
        }

        if input.size != output.size {
            return Err(torsh_core::error::TorshError::BackendError(
                "Input and output buffers must have the same size".to_string(),
            ));
        }

        let total_complex = input.size / COMPLEX_F32_BYTES;
        if total_complex == 0 || !total_complex.is_multiple_of(size) {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "FFT: buffer holds {} complex elements which is not a multiple of size {}",
                total_complex, size
            )));
        }
        let batch = total_complex / size;

        let oxi_direction = match direction {
            FftDirection::Forward => Direction::Forward,
            FftDirection::Inverse => Direction::Backward,
        };

        let plan = Plan::<f32>::dft_1d(size, oxi_direction, Flags::ESTIMATE).ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(format!(
                "FFT: failed to create plan for size {}",
                size
            ))
        })?;

        let scale = Self::normalization_factor(size, direction, normalization);

        let input_values = Self::read_complex_buffer(input, total_complex)?;
        let mut output_values = vec![Complex::<f32>::zero(); total_complex];

        for b in 0..batch {
            let start = b * size;
            let end = start + size;
            plan.execute(&input_values[start..end], &mut output_values[start..end]);
        }

        Self::write_complex_buffer(output, &output_values, scale)
    }
}

#[async_trait::async_trait]
impl FftOps for CpuFftOps {
    async fn create_fft_plan(
        &self,
        _device: &Device,
        plan: &FftPlan,
    ) -> BackendResult<Box<dyn FftExecutor>> {
        let executor = self.get_or_create_executor(plan)?;
        Ok(Box::new((*executor).clone()))
    }

    async fn fft_1d(
        &self,
        _device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // Real complex-to-complex DFT via the oxifft engine. Buffers hold
        // interleaved complex `f32` data; batches are transformed in place.
        self.execute_dft_1d(input, output, size, direction, normalization)
    }

    async fn fft_2d(
        &self,
        _device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: (usize, usize),
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        let (n0, n1) = size;
        if n0 == 0 || n1 == 0 {
            return Err(torsh_core::error::TorshError::BackendError(
                "FFT: 2D transform dimensions must be greater than zero".to_string(),
            ));
        }

        if input.size != output.size {
            return Err(torsh_core::error::TorshError::BackendError(
                "Input and output buffers must have the same size".to_string(),
            ));
        }

        let plane = n0 * n1;
        let total_complex = input.size / COMPLEX_F32_BYTES;
        if total_complex == 0 || !total_complex.is_multiple_of(plane) {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "FFT: buffer holds {} complex elements which is not a multiple of {}x{}",
                total_complex, n0, n1
            )));
        }
        let batch = total_complex / plane;

        let oxi_direction = match direction {
            FftDirection::Forward => Direction::Forward,
            FftDirection::Inverse => Direction::Backward,
        };

        let plan =
            Plan::<f32>::dft_2d(n0, n1, oxi_direction, Flags::ESTIMATE).ok_or_else(|| {
                torsh_core::error::TorshError::BackendError(format!(
                    "FFT: failed to create 2D plan for {}x{}",
                    n0, n1
                ))
            })?;

        // 2D normalization divides by the total number of points (n0 * n1).
        let scale = Self::normalization_factor(plane, direction, normalization);

        let input_values = Self::read_complex_buffer(input, total_complex)?;
        let mut output_values = vec![Complex::<f32>::zero(); total_complex];

        for b in 0..batch {
            let start = b * plane;
            let end = start + plane;
            plan.execute(&input_values[start..end], &mut output_values[start..end]);
        }

        Self::write_complex_buffer(output, &output_values, scale)
    }

    async fn fft_3d(
        &self,
        _device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: (usize, usize, usize),
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        let (n0, n1, n2) = size;
        if n0 == 0 || n1 == 0 || n2 == 0 {
            return Err(torsh_core::error::TorshError::BackendError(
                "FFT: 3D transform dimensions must be greater than zero".to_string(),
            ));
        }

        if input.size != output.size {
            return Err(torsh_core::error::TorshError::BackendError(
                "Input and output buffers must have the same size".to_string(),
            ));
        }

        let volume = n0 * n1 * n2;
        let total_complex = input.size / COMPLEX_F32_BYTES;
        if total_complex == 0 || !total_complex.is_multiple_of(volume) {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "FFT: buffer holds {} complex elements which is not a multiple of {}x{}x{}",
                total_complex, n0, n1, n2
            )));
        }
        let batch = total_complex / volume;

        let oxi_direction = match direction {
            FftDirection::Forward => Direction::Forward,
            FftDirection::Inverse => Direction::Backward,
        };

        let plan =
            Plan::<f32>::dft_3d(n0, n1, n2, oxi_direction, Flags::ESTIMATE).ok_or_else(|| {
                torsh_core::error::TorshError::BackendError(format!(
                    "FFT: failed to create 3D plan for {}x{}x{}",
                    n0, n1, n2
                ))
            })?;

        // 3D normalization divides by the total number of points (n0 * n1 * n2).
        let scale = Self::normalization_factor(volume, direction, normalization);

        let input_values = Self::read_complex_buffer(input, total_complex)?;
        let mut output_values = vec![Complex::<f32>::zero(); total_complex];

        for b in 0..batch {
            let start = b * volume;
            let end = start + volume;
            plan.execute(&input_values[start..end], &mut output_values[start..end]);
        }

        Self::write_complex_buffer(output, &output_values, scale)
    }

    async fn fft_batch(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        _batch_size: usize,
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to 1D FFT
        let fft_size = size.first().copied().unwrap_or(1);
        self.fft_1d(device, input, output, fft_size, direction, normalization)
            .await
    }

    async fn rfft(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        direction: FftDirection,
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to complex FFT
        let fft_size = size.first().copied().unwrap_or(1);
        self.fft_1d(device, input, output, fft_size, direction, normalization)
            .await
    }

    async fn irfft(
        &self,
        device: &Device,
        input: &Buffer,
        output: &Buffer,
        size: &[usize],
        normalization: FftNormalization,
    ) -> BackendResult<()> {
        // For now, delegate to complex FFT with inverse direction
        let fft_size = size.first().copied().unwrap_or(1);
        self.fft_1d(
            device,
            input,
            output,
            fft_size,
            FftDirection::Inverse,
            normalization,
        )
        .await
    }

    fn supports_fft(&self) -> bool {
        true
    }

    fn get_optimal_fft_sizes(&self, min_size: usize, max_size: usize) -> Vec<usize> {
        let mut sizes = Vec::new();

        // Add power-of-2 sizes
        let mut size = 1;
        while size < min_size {
            size *= 2;
        }
        while size <= max_size {
            sizes.push(size);
            size *= 2;
        }

        // Add sizes with small prime factors (2, 3, 5, 7)
        for n in (min_size..=max_size).step_by(2) {
            if is_smooth_number(n, &[2, 3, 5, 7]) {
                sizes.push(n);
            }
        }

        sizes.sort_unstable();
        sizes.dedup();
        sizes
    }
}

/// CPU FFT executor
#[derive(Clone, Debug)]
pub struct CpuFftExecutor {
    plan: FftPlan,
    fft_ops: Arc<CpuFftOps>,
}

impl CpuFftExecutor {
    pub fn new(plan: FftPlan, num_threads: usize) -> BackendResult<Self> {
        if !plan.is_valid() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Invalid FFT plan".to_string(),
            ));
        }

        let fft_ops = Arc::new(CpuFftOps::new(Some(num_threads)));

        Ok(Self { plan, fft_ops })
    }
}

#[async_trait::async_trait]
impl FftExecutor for CpuFftExecutor {
    async fn execute(&self, device: &Device, input: &Buffer, output: &Buffer) -> BackendResult<()> {
        match self.plan.fft_type {
            FftType::C2C => {
                self.fft_ops
                    .fft_1d(
                        device,
                        input,
                        output,
                        self.plan.dimensions[0],
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::C2C2D => {
                self.fft_ops
                    .fft_2d(
                        device,
                        input,
                        output,
                        (self.plan.dimensions[0], self.plan.dimensions[1]),
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::C2C3D => {
                self.fft_ops
                    .fft_3d(
                        device,
                        input,
                        output,
                        (
                            self.plan.dimensions[0],
                            self.plan.dimensions[1],
                            self.plan.dimensions[2],
                        ),
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::R2C | FftType::R2C2D | FftType::R2C3D => {
                self.fft_ops
                    .rfft(
                        device,
                        input,
                        output,
                        &self.plan.dimensions,
                        self.plan.direction,
                        self.plan.normalization,
                    )
                    .await
            }
            FftType::C2R | FftType::C2R2D | FftType::C2R3D => {
                self.fft_ops
                    .irfft(
                        device,
                        input,
                        output,
                        &self.plan.dimensions,
                        self.plan.normalization,
                    )
                    .await
            }
        }
    }

    fn plan(&self) -> &FftPlan {
        &self.plan
    }

    fn memory_requirements(&self) -> usize {
        self.plan.input_buffer_size() + self.plan.output_buffer_size()
    }

    fn is_valid(&self) -> bool {
        self.plan.is_valid()
    }
}

/// Check if a number is smooth (has only small prime factors)
fn is_smooth_number(n: usize, primes: &[usize]) -> bool {
    let mut n = n;
    for &prime in primes {
        while n.is_multiple_of(prime) {
            n /= prime;
        }
    }
    n == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{BufferDescriptor, BufferUsage, MemoryLocation};
    use crate::cpu::buffer::CpuBuffer;

    /// Build a CPU [`Buffer`] holding the given interleaved complex `f32` data.
    fn complex_buffer(device: &Device, values: &[(f32, f32)]) -> Buffer {
        let mut bytes = Vec::with_capacity(values.len() * COMPLEX_F32_BYTES);
        for &(re, im) in values {
            bytes.extend_from_slice(&re.to_le_bytes());
            bytes.extend_from_slice(&im.to_le_bytes());
        }

        let descriptor = BufferDescriptor {
            size: bytes.len(),
            usage: BufferUsage::STORAGE_READ_WRITE,
            location: MemoryLocation::Host,
            dtype: None,
            shape: None,
            initial_data: None,
            alignment: None,
            zero_init: false,
        };

        let buffer = CpuBuffer::new_buffer(device.clone(), &descriptor)
            .expect("failed to allocate CPU complex buffer");
        let cpu = buffer
            .as_cpu_buffer()
            .expect("generic handle should expose a CpuBuffer");
        cpu.write_bytes(&bytes, 0)
            .expect("failed to write complex data into buffer");
        buffer
    }

    /// Read interleaved complex `f32` data back out of a CPU [`Buffer`].
    fn read_complex(buffer: &Buffer) -> Vec<(f32, f32)> {
        let count = buffer.size / COMPLEX_F32_BYTES;
        let ptr = buffer.as_cpu_ptr().expect("buffer should be CPU-backed");
        let mut out = Vec::with_capacity(count);
        // SAFETY: buffer holds `count` interleaved complex `f32` values.
        unsafe {
            let floats = std::slice::from_raw_parts(ptr as *const f32, count * 2);
            for pair in floats.chunks_exact(2) {
                out.push((pair[0], pair[1]));
            }
        }
        out
    }

    fn test_device() -> Device {
        crate::cpu::device::CpuDevice::new(0, 1)
            .expect("CPU device creation should succeed")
            .to_device()
    }

    #[test]
    fn test_cpu_fft_ops_creation() {
        let fft_ops = CpuFftOps::new(Some(2));
        assert!(fft_ops.supports_fft());
    }

    /// An impulse (delta) signal transforms to a flat spectrum: every bin must
    /// equal 1. This proves a real DFT ran instead of the old identity copy,
    /// which would have left the buffer as [1, 0, 0, 0].
    #[tokio::test]
    async fn test_fft_1d_delta_is_not_identity() {
        let fft_ops = CpuFftOps::new(Some(1));
        let device = test_device();
        let size = 8;

        let mut input_vals = vec![(0.0f32, 0.0f32); size];
        input_vals[0] = (1.0, 0.0); // delta at t = 0

        let input = complex_buffer(&device, &input_vals);
        let output = complex_buffer(&device, &vec![(0.0f32, 0.0f32); size]);

        fft_ops
            .fft_1d(
                &device,
                &input,
                &output,
                size,
                FftDirection::Forward,
                FftNormalization::None,
            )
            .await
            .expect("forward FFT should succeed");

        let result = read_complex(&output);

        // Spectrum of a delta is flat: every bin is (1, 0).
        for (re, im) in &result {
            assert!(
                (re - 1.0).abs() < 1e-4,
                "expected flat real spectrum, got {re}"
            );
            assert!(
                im.abs() < 1e-4,
                "expected zero imaginary spectrum, got {im}"
            );
        }

        // The transform must NOT be an identity copy of the input.
        assert_ne!(
            result, input_vals,
            "FFT output must differ from input (no identity copy)"
        );
    }

    /// The DC bin of an all-ones signal equals N (sum of inputs), and the
    /// remaining bins are zero. This is impossible for the old copy-based stub.
    #[tokio::test]
    async fn test_fft_1d_dc_component() {
        let fft_ops = CpuFftOps::new(Some(1));
        let device = test_device();
        let size = 8;

        let input_vals = vec![(1.0f32, 0.0f32); size];
        let input = complex_buffer(&device, &input_vals);
        let output = complex_buffer(&device, &vec![(0.0f32, 0.0f32); size]);

        fft_ops
            .fft_1d(
                &device,
                &input,
                &output,
                size,
                FftDirection::Forward,
                FftNormalization::None,
            )
            .await
            .expect("forward FFT should succeed");

        let result = read_complex(&output);
        assert!(
            (result[0].0 - size as f32).abs() < 1e-4,
            "DC bin should equal N"
        );
        assert!(result[0].1.abs() < 1e-4);
        for (re, im) in &result[1..] {
            assert!(re.abs() < 1e-4, "non-DC real part should vanish, got {re}");
            assert!(
                im.abs() < 1e-4,
                "non-DC imaginary part should vanish, got {im}"
            );
        }
    }

    /// Forward followed by backward (with 1/N normalization on the inverse)
    /// must recover the original signal.
    #[tokio::test]
    async fn test_fft_1d_roundtrip() {
        let fft_ops = CpuFftOps::new(Some(1));
        let device = test_device();
        let size = 8;

        let input_vals: Vec<(f32, f32)> = (0..size)
            .map(|i| (i as f32, (i as f32) * 0.5 - 1.0))
            .collect();
        let input = complex_buffer(&device, &input_vals);
        let spectrum = complex_buffer(&device, &vec![(0.0f32, 0.0f32); size]);
        let recovered = complex_buffer(&device, &vec![(0.0f32, 0.0f32); size]);

        fft_ops
            .fft_1d(
                &device,
                &input,
                &spectrum,
                size,
                FftDirection::Forward,
                FftNormalization::None,
            )
            .await
            .expect("forward FFT should succeed");

        fft_ops
            .fft_1d(
                &device,
                &spectrum,
                &recovered,
                size,
                FftDirection::Inverse,
                FftNormalization::Backward,
            )
            .await
            .expect("inverse FFT should succeed");

        let result = read_complex(&recovered);
        for ((re, im), (ore, oim)) in result.iter().zip(input_vals.iter()) {
            assert!(
                (re - ore).abs() < 1e-3,
                "real roundtrip mismatch: {re} vs {ore}"
            );
            assert!(
                (im - oim).abs() < 1e-3,
                "imag roundtrip mismatch: {im} vs {oim}"
            );
        }
    }

    /// A 2D FFT of a 2x2 all-ones block puts the entire energy in the DC bin.
    #[tokio::test]
    async fn test_fft_2d_dc_component() {
        let fft_ops = CpuFftOps::new(Some(1));
        let device = test_device();

        let input_vals = vec![(1.0f32, 0.0f32); 4];
        let input = complex_buffer(&device, &input_vals);
        let output = complex_buffer(&device, &vec![(0.0f32, 0.0f32); 4]);

        fft_ops
            .fft_2d(
                &device,
                &input,
                &output,
                (2, 2),
                FftDirection::Forward,
                FftNormalization::None,
            )
            .await
            .expect("2D FFT should succeed");

        let result = read_complex(&output);
        assert!((result[0].0 - 4.0).abs() < 1e-4, "2D DC bin should equal 4");
        for (re, im) in &result[1..] {
            assert!(re.abs() < 1e-4 && im.abs() < 1e-4);
        }
    }

    /// Mismatched buffer sizes must produce an honest error, never silent
    /// fabricated output.
    #[tokio::test]
    async fn test_fft_1d_size_mismatch_errors() {
        let fft_ops = CpuFftOps::new(Some(1));
        let device = test_device();

        let input = complex_buffer(&device, &vec![(1.0f32, 0.0f32); 4]);
        let output = complex_buffer(&device, &vec![(0.0f32, 0.0f32); 8]);

        let result = fft_ops
            .fft_1d(
                &device,
                &input,
                &output,
                4,
                FftDirection::Forward,
                FftNormalization::None,
            )
            .await;
        assert!(result.is_err(), "size mismatch must error, not fabricate");
    }

    #[test]
    fn test_smooth_numbers() {
        assert!(is_smooth_number(8, &[2, 3, 5, 7])); // 2^3
        assert!(is_smooth_number(12, &[2, 3, 5, 7])); // 2^2 * 3
        assert!(is_smooth_number(30, &[2, 3, 5, 7])); // 2 * 3 * 5
        assert!(!is_smooth_number(11, &[2, 3, 5, 7])); // Prime
        assert!(!is_smooth_number(13, &[2, 3, 5, 7])); // Prime
    }

    #[test]
    fn test_optimal_fft_sizes() {
        let fft_ops = CpuFftOps::new(Some(1));
        let sizes = fft_ops.get_optimal_fft_sizes(100, 1000);

        assert!(!sizes.is_empty());
        assert!(sizes.iter().all(|&size| size >= 100 && size <= 1000));

        // Check that sizes are sorted
        let mut sorted_sizes = sizes.clone();
        sorted_sizes.sort_unstable();
        assert_eq!(sizes, sorted_sizes);
    }

    #[test]
    fn test_fft_plan_validation() {
        let plan = FftPlan::new(
            FftType::C2C,
            vec![1024],
            1,
            torsh_core::dtype::DType::F32,
            torsh_core::dtype::DType::F32,
            FftDirection::Forward,
            FftNormalization::None,
        );

        assert!(plan.is_valid());

        let executor = CpuFftExecutor::new(plan, 1);
        assert!(executor.is_ok());
    }
}
