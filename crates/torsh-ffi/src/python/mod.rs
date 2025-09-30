//! Python bindings for ToRSh via PyO3

use pyo3::prelude::*;

mod dataloader;
mod functional;
mod module;
mod optimizer;
pub mod tensor;
mod utils;

pub use dataloader::{PyDataLoader, PyDataLoaderBuilder, PyRandomDataLoader};
pub use functional::*;
pub use module::{PyLinear, PyModule};
pub use optimizer::{PyAdam, PyOptimizer, PySGD};
pub use tensor::PyTensor;
pub use utils::*;

// Re-export integration modules
// TEMPORARILY DISABLED DUE TO PyO3 API COMPATIBILITY ISSUES
// pub use crate::jupyter_widgets::{
//     DataExplorationWidget, JupyterWidgets, TensorVisualizationWidget, TrainingMonitorWidget,
// };
pub use crate::pandas_support::{DataAnalysisResult, PandasSupport, TorshDataFrame, TorshSeries};
// TEMPORARILY DISABLED DUE TO PyO3 API COMPATIBILITY ISSUES
// pub use crate::plotting_utilities::{PlotResult, PlottingUtilities, StatPlotConfig};
pub use crate::scipy_integration::{
    LinalgResult, OptimizationResult, SciPyIntegration, SignalResult,
};

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

    // Add data loaders
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyRandomDataLoader>()?;
    m.add_class::<PyDataLoaderBuilder>()?;

    // Add functional operations
    // Add functional operations directly to main module
    m.add_function(wrap_pyfunction!(functional::relu, m)?)?;
    m.add_function(wrap_pyfunction!(functional::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(functional::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(functional::softmax, m)?)?;
    m.add_function(wrap_pyfunction!(functional::cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(functional::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(functional::binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(functional::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(functional::log_softmax, m)?)?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(utils::tensor, m)?)?;
    m.add_function(wrap_pyfunction!(utils::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(utils::ones, m)?)?;
    m.add_function(wrap_pyfunction!(utils::randn, m)?)?;
    m.add_function(wrap_pyfunction!(utils::rand, m)?)?;
    m.add_function(wrap_pyfunction!(utils::eye, m)?)?;
    m.add_function(wrap_pyfunction!(utils::full, m)?)?;
    m.add_function(wrap_pyfunction!(utils::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(utils::arange, m)?)?;
    m.add_function(wrap_pyfunction!(utils::stack, m)?)?;
    m.add_function(wrap_pyfunction!(utils::cat, m)?)?;
    m.add_function(wrap_pyfunction!(utils::from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(utils::to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(utils::manual_seed, m)?)?;

    // Add dataloader functions
    m.add_function(wrap_pyfunction!(dataloader::create_dataloader, m)?)?;
    m.add_function(wrap_pyfunction!(dataloader::create_dataset_from_array, m)?)?;

    // Register custom exception types
    #[cfg(feature = "python")]
    crate::error::python_exceptions::register_exceptions(m)?;
    m.add_function(wrap_pyfunction!(dataloader::get_dataloader_info, m)?)?;
    m.add_function(wrap_pyfunction!(dataloader::benchmark_dataloader, m)?)?;

    // Add integration utilities classes
    m.add_class::<SciPyIntegration>()?;
    m.add_class::<OptimizationResult>()?;
    m.add_class::<LinalgResult>()?;
    m.add_class::<SignalResult>()?;

    m.add_class::<PandasSupport>()?;
    m.add_class::<TorshDataFrame>()?;
    m.add_class::<TorshSeries>()?;
    m.add_class::<DataAnalysisResult>()?;

    // TEMPORARILY DISABLED DUE TO PyO3 API COMPATIBILITY ISSUES
    // m.add_class::<PlottingUtilities>()?;
    // m.add_class::<PlotResult>()?;
    // TEMPORARILY DISABLED DUE TO PyO3 API COMPATIBILITY ISSUES
    // m.add_class::<StatPlotConfig>()?;

    // m.add_class::<JupyterWidgets>()?;
    // m.add_class::<TensorVisualizationWidget>()?;
    // m.add_class::<TrainingMonitorWidget>()?;
    // m.add_class::<DataExplorationWidget>()?;

    // Create submodules for integration utilities
    let scipy_utils = crate::scipy_integration::create_scipy_utilities(m.py())?;
    m.add("scipy", scipy_utils)?;

    let pandas_utils = crate::pandas_support::create_pandas_utilities(m.py())?;
    m.add("pandas", pandas_utils)?;

    // TEMPORARILY DISABLED DUE TO PyO3 API COMPATIBILITY ISSUES
    // let plotting_utils = crate::plotting_utilities::create_plotting_utilities(m.py())?;
    // m.add("plotting", plotting_utils)?;

    // let jupyter_utils = crate::jupyter_widgets::create_jupyter_utilities(m.py())?;
    // m.add("jupyter", jupyter_utils)?;

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
            let module = pyo3::types::PyModule::new(py, "test_torsh").unwrap();
            let result = torsh(&module);
            assert!(result.is_ok());
        });
    }
}
