//! Hardware Acceleration Backend Implementations
//!
//! This module contains additional hardware accelerator backend implementations
//! extracted from hardware_acceleration.rs to keep file sizes manageable.
//! Currently houses the WebGPU accelerator backend.

use super::*;

/// WebGPU accelerator implementation for browser deployment
#[derive(Debug)]
pub struct WebGpuAccelerator {
    initialized: bool,
    devices: Vec<HardwareDevice>,
    config: Option<AccelerationConfig>,
}

impl WebGpuAccelerator {
    pub fn new() -> Self {
        Self {
            initialized: false,
            devices: Vec::new(),
            config: None,
        }
    }

    fn detect_webgpu_devices(&self) -> AutogradResult<Vec<HardwareDevice>> {
        // No real WebGPU/wgpu adapter is wired into this crate. Returning a
        // fabricated "WebGPU Default Adapter" with invented memory/TFLOPS would
        // misrepresent the available hardware, so we report no devices. Real
        // detection requires a wgpu-backed adapter request.
        Ok(Vec::new())
    }

    fn is_webgpu_available(&self) -> bool {
        // The `webgpu` feature / wasm target currently only enables a software
        // simulation with no real wgpu backend, so this accelerator cannot drive
        // a GPU. Report unavailable so callers do not dispatch simulated work.
        false
    }
}

/// Build an honest "WebGPU backend not available" error for an operation that
/// would otherwise have to fabricate a result. No real wgpu adapter is wired
/// in, so all device operations are unsupported.
fn webgpu_backend_unavailable(operation: &str) -> AutogradError {
    AutogradError::gradient_computation(
        format!("webgpu::{operation}"),
        "WebGPU backend not available: no real wgpu adapter is wired into \
         torsh-autograd, so this operation cannot be performed on hardware.",
    )
}

impl HardwareAccelerator for WebGpuAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::WebGPU
    }

    fn is_available(&self) -> bool {
        self.is_webgpu_available()
    }

    fn get_devices(&self) -> AutogradResult<Vec<HardwareDevice>> {
        if self.initialized {
            Ok(self.devices.clone())
        } else {
            self.detect_webgpu_devices()
        }
    }

    fn initialize(&mut self, config: &AccelerationConfig) -> AutogradResult<()> {
        if !self.is_available() {
            return Err(AutogradError::gradient_computation(
                "webgpu_availability",
                "WebGPU not available on this system",
            ));
        }

        self.devices = self.detect_webgpu_devices()?;
        self.config = Some(config.clone());
        self.initialized = true;

        tracing::info!(
            "WebGPU accelerator initialized with {} devices",
            self.devices.len()
        );
        Ok(())
    }

    fn shutdown(&mut self) -> AutogradResult<()> {
        self.initialized = false;
        self.devices.clear();
        self.config = None;
        tracing::info!("WebGPU accelerator shutdown");
        Ok(())
    }

    fn allocate_memory(
        &self,
        _device_id: u32,
        size: usize,
    ) -> AutogradResult<HardwareMemoryHandle> {
        Err(webgpu_backend_unavailable(&format!(
            "allocate_memory({size} bytes)"
        )))
    }

    fn deallocate_memory(&self, _handle: HardwareMemoryHandle) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("deallocate_memory"))
    }

    fn copy_to_device(&self, _data: &[f64], _handle: &HardwareMemoryHandle) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("copy_to_device"))
    }

    fn copy_from_device(
        &self,
        _handle: &HardwareMemoryHandle,
        _data: &mut [f64],
    ) -> AutogradResult<()> {
        // Previously filled the output buffer with a synthetic ramp (`i * 0.15`).
        Err(webgpu_backend_unavailable("copy_from_device"))
    }

    fn accelerated_add(
        &self,
        _device_id: u32,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        _size: usize,
    ) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("accelerated_add"))
    }

    fn accelerated_mul(
        &self,
        _device_id: u32,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        _size: usize,
    ) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("accelerated_mul"))
    }

    fn accelerated_matmul(
        &self,
        _device_id: u32,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("accelerated_matmul"))
    }

    fn accelerated_conv2d(
        &self,
        _device_id: u32,
        _input: &HardwareMemoryHandle,
        _kernel: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        _params: &Conv2DParams,
    ) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("accelerated_conv2d"))
    }

    fn accelerated_backward_add(
        &self,
        _device_id: u32,
        _grad_output: &HardwareMemoryHandle,
        _grad_a: &HardwareMemoryHandle,
        _grad_b: &HardwareMemoryHandle,
        _size: usize,
    ) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("accelerated_backward_add"))
    }

    fn accelerated_backward_mul(
        &self,
        _device_id: u32,
        _grad_output: &HardwareMemoryHandle,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _grad_a: &HardwareMemoryHandle,
        _grad_b: &HardwareMemoryHandle,
        _size: usize,
    ) -> AutogradResult<()> {
        Err(webgpu_backend_unavailable("accelerated_backward_mul"))
    }

    fn get_device_stats(&self, _device_id: u32) -> AutogradResult<DeviceStats> {
        // Previously returned hard-coded integrated-GPU-shaped statistics.
        Err(webgpu_backend_unavailable("get_device_stats"))
    }

    fn benchmark_operation(
        &self,
        _device_id: u32,
        operation: &str,
        _size: usize,
    ) -> AutogradResult<f64> {
        // Previously returned `sleep`-derived timings as benchmark results.
        Err(webgpu_backend_unavailable(&format!(
            "benchmark_operation({operation})"
        )))
    }
}

impl Default for WebGpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}
