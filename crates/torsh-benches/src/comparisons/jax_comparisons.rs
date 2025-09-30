//! JAX comparison benchmarks
//!
//! This module provides JAX benchmark implementations for comparing
//! ToRSh performance with JAX operations.

/// JAX benchmark runner
#[cfg(feature = "jax")]
pub struct JAXBenchRunner {
    python_initialized: bool,
    jax_available: bool,
    device: String,
}

#[cfg(feature = "jax")]
impl JAXBenchRunner {
    pub fn new() -> Self {
        let mut runner = Self {
            python_initialized: false,
            jax_available: false,
            device: "cpu".to_string(),
        };

        if let Err(e) = runner.initialize_python() {
            eprintln!("Warning: Failed to initialize Python/JAX: {}", e);
        }

        runner
    }

    /// Initialize Python interpreter and check JAX availability
    fn initialize_python(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "jax")]
        {
            use pyo3::prelude::*;

            pyo3::prepare_freethreaded_python();

            Python::with_gil(|py| -> PyResult<()> {
                // Try to import JAX
                match py.import("jax") {
                    Ok(jax) => {
                        self.jax_available = true;
                        self.python_initialized = true;

                        // Check available devices
                        let devices = jax.call_method0("devices")?;
                        let device_list: Vec<String> = devices.extract()?;

                        // Check for GPU devices
                        let has_gpu = device_list
                            .iter()
                            .any(|d| d.contains("gpu") || d.contains("GPU"));

                        if has_gpu {
                            self.device = "gpu".to_string();
                            println!("JAX GPU available, using GPU for benchmarks");
                        } else {
                            println!("JAX CPU only, using CPU for benchmarks");
                        }

                        Ok(())
                    }
                    Err(e) => {
                        eprintln!("JAX not available: {}", e);
                        Err(e)
                    }
                }
            })?;
        }

        Ok(())
    }

    /// Check if JAX is available
    pub fn is_jax_available(&self) -> bool {
        self.jax_available
    }

    /// Run JAX tensor operation benchmark
    #[cfg(feature = "jax")]
    pub fn benchmark_jax_operation(
        &self,
        operation: &str,
        size: usize,
        iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if !self.jax_available {
            return Err("JAX not available".into());
        }

        use pyo3::prelude::*;

        Python::with_gil(|py| -> PyResult<f64> {
            let jax = py.import("jax")?;
            let jnp = py.import("jax.numpy")?;
            let time_module = py.import("time")?;

            // Create the benchmark script as a string
            let benchmark_code = format!(
                r#"
import jax
import jax.numpy as jnp
import time
import numpy as np

# Set up device
device = '{}'
if device == 'gpu':
    # Try to use GPU if available
    try:
        # This will fail if no GPU, falling back to CPU
        jax.device_put(jnp.array([1.0]), jax.devices('gpu')[0])
    except:
        device = 'cpu'

jax.config.update('jax_platform_name', device)
key = jax.random.PRNGKey(42)  # For reproducibility

def benchmark_operation():
    if '{}' == 'matmul':
        key1, key2 = jax.random.split(key)
        a = jax.random.normal(key1, ({}, {}))
        b = jax.random.normal(key2, ({}, {}))

        # Compile the function
        matmul_fn = jax.jit(jnp.matmul)

        # Warmup
        for _ in range(5):
            _ = matmul_fn(a, b).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = matmul_fn(a, b)
            result.block_until_ready()  # Ensure computation is complete

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'add':
        key1, key2 = jax.random.split(key)
        a = jax.random.normal(key1, ({},))
        b = jax.random.normal(key2, ({},))

        # Compile the function
        add_fn = jax.jit(jnp.add)

        # Warmup
        for _ in range(10):
            _ = add_fn(a, b).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = add_fn(a, b)
            result.block_until_ready()

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'conv2d':
        batch_size = 16
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        input_size = {}

        # Create input and kernel (JAX uses NHWC format)
        key1, key2 = jax.random.split(key)
        x = jax.random.normal(key1, (batch_size, input_size, input_size, in_channels))
        kernel = jax.random.normal(key2, (kernel_size, kernel_size, in_channels, out_channels))

        # Define convolution function
        def conv_fn(x, kernel):
            return jax.lax.conv_general_dilated(
                x, kernel,
                window_strides=(1, 1),
                padding='SAME',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )

        # Compile the function
        compiled_conv = jax.jit(conv_fn)

        # Warmup
        for _ in range(5):
            _ = compiled_conv(x, kernel).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = compiled_conv(x, kernel)
            result.block_until_ready()

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'relu':
        key1 = jax.random.split(key)[0]
        x = jax.random.normal(key1, ({},))

        # Compile the function
        relu_fn = jax.jit(jnp.maximum, static_argnums=())

        # Warmup
        for _ in range(10):
            _ = relu_fn(x, 0.0).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = relu_fn(x, 0.0)
            result.block_until_ready()

        end_time = time.time()

        return (end_time - start_time) / {}

    else:
        raise ValueError(f"Unknown operation: {}")

avg_time = benchmark_operation()
avg_time
"#,
                self.device,
                operation,
                size,
                size, // matmul shapes
                size,
                size,
                iterations,
                iterations, // matmul iterations
                operation,
                size, // add shapes
                size,
                iterations,
                iterations, // add iterations
                operation,
                size, // conv2d input size
                iterations,
                iterations, // conv2d iterations
                operation,
                size, // relu shapes
                iterations,
                iterations, // relu iterations
                operation   // error case
            );

            // Execute the benchmark code
            let result: f64 = py.eval(&benchmark_code, None, None)?.extract()?;
            Ok(result)
        })
        .map_err(|e| e.into())
    }

    #[cfg(not(feature = "jax"))]
    pub fn benchmark_jax_operation(
        &self,
        _operation: &str,
        _size: usize,
        _iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        Err("JAX feature not enabled".into())
    }
}

#[cfg(feature = "jax")]
impl Default for JAXBenchRunner {
    fn default() -> Self {
        Self::new()
    }
}