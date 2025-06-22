//! Python bindings for ToRSh via PyO3

use pyo3::prelude::*;
use pyo3::Bound;

mod tensor;
mod module;
mod optimizer;
mod functional;
mod utils;

pub use tensor::PyTensor;
pub use module::{PyModule, PyLinear};
pub use optimizer::{PyOptimizer, PySGD, PyAdam};
pub use functional::*;
pub use utils::*;

/// Initialize the Python module
#[pymodule]
fn torsh(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // Add tensor class
    m.add_class::<PyTensor>()?;
    
    // Add neural network modules
    m.add_class::<PyLinear>()?;
    
    // Add optimizers
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    
    // Add functional operations
    let functional = pyo3::types::PyModule::new_bound(m.py(), "functional")?;
    functional.add_function(wrap_pyfunction!(relu, m)?)?;
    functional.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    functional.add_function(wrap_pyfunction!(tanh, m)?)?;
    functional.add_function(wrap_pyfunction!(softmax, m)?)?;
    functional.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    functional.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_submodule(&functional)?;
    
    // Add utility functions
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    
    // Add constants
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add device information
    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_device_count, m)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    
    #[test]
    fn test_module_creation() {
        Python::with_gil(|py| {
            let module = pyo3::types::PyModule::new_bound(py, "test_torsh").unwrap();
            let result = torsh(&module);
            assert!(result.is_ok());
        });
    }
}