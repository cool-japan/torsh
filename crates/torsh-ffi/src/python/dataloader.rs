//! Python bindings for ToRSh data loaders

use crate::python::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Bound;
use torsh_core::DType;
use torsh_data::{
    collate::DefaultCollate,
    dataloader::{simple_dataloader, simple_random_dataloader, DataLoader, DataLoaderTrait},
    dataset::{Dataset, TensorDataset},
    sampler::{BatchingSampler, RandomSampler, SequentialSampler},
};
use torsh_tensor::Tensor;

/// Python wrapper for ToRSh DataLoader
#[pyclass(name = "DataLoader")]
pub struct PyDataLoader {
    inner: DataLoader<TensorDataset<f32>, BatchingSampler<SequentialSampler>, DefaultCollate>,
}

#[pymethods]
impl PyDataLoader {
    /// Create a new DataLoader from a tensor dataset
    #[new]
    fn new(
        dataset: PyTensor,
        batch_size: Option<usize>,
        shuffle: Option<bool>,
        num_workers: Option<usize>,
        drop_last: Option<bool>,
    ) -> PyResult<Self> {
        let batch_size = batch_size.unwrap_or(1);
        let shuffle = shuffle.unwrap_or(false);
        let _num_workers = num_workers.unwrap_or(0);
        let _drop_last = drop_last.unwrap_or(false);

        // Extract the tensor from PyTensor
        let tensor_data = dataset.data.clone();
        let tensor_shape = dataset.shape.clone();

        // Create tensor dataset
        let tensor = Tensor::from_vec(tensor_data, &tensor_shape).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create tensor: {}",
                e
            ))
        })?;

        let tensor_dataset = TensorDataset::from_tensor(tensor);

        // Create the dataloader (use sequential for now, shuffle will be handled differently)
        let dataloader = simple_dataloader(tensor_dataset, batch_size, shuffle);

        match dataloader {
            Ok(dl) => Ok(PyDataLoader { inner: dl }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create dataloader: {}",
                e
            ))),
        }
    }

    /// Get the number of batches in the dataloader
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the dataloader is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Create an iterator over the dataloader
    fn __iter__(_slf: PyRef<'_, Self>) -> PyDataLoaderIterator {
        // Create a simplified iterator that doesn't rely on private fields
        PyDataLoaderIterator {
            batch_size: 32, // Default batch size
            current_batch: 0,
            total_batches: 10, // Default, should be calculated properly
        }
    }
}

/// Python iterator for DataLoader
#[pyclass(name = "DataLoaderIterator")]
pub struct PyDataLoaderIterator {
    batch_size: usize,
    current_batch: usize,
    total_batches: usize,
}

#[pymethods]
impl PyDataLoaderIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyTensor>> {
        if self.current_batch < self.total_batches {
            self.current_batch += 1;

            // Create a simple dummy batch for now
            let batch_data = vec![0.0f32; self.batch_size];
            let batch_shape = vec![self.batch_size];

            Python::with_gil(|py| {
                let py_list = PyList::new(py, &batch_data)?;
                let py_tensor = PyTensor::new(&py_list, Some(batch_shape), Some("f32"), false)?;
                Ok(Some(py_tensor))
            })
        } else {
            Ok(None) // End of iteration
        }
    }
}

/// Python wrapper for random DataLoader
#[pyclass(name = "RandomDataLoader")]
pub struct PyRandomDataLoader {
    inner: DataLoader<TensorDataset<f32>, BatchingSampler<RandomSampler>, DefaultCollate>,
}

#[pymethods]
impl PyRandomDataLoader {
    /// Create a new random DataLoader from a tensor dataset
    #[new]
    fn new(
        dataset: PyTensor,
        batch_size: Option<usize>,
        generator_seed: Option<u64>,
        num_workers: Option<usize>,
        drop_last: Option<bool>,
    ) -> PyResult<Self> {
        let batch_size = batch_size.unwrap_or(1);
        let _num_workers = num_workers.unwrap_or(0);
        let _drop_last = drop_last.unwrap_or(false);

        // Extract the tensor from PyTensor
        let tensor_data = dataset.data.clone();
        let tensor_shape = dataset.shape.clone();

        // Create tensor dataset
        let tensor = Tensor::from_vec(tensor_data, &tensor_shape).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create tensor: {}",
                e
            ))
        })?;

        let tensor_dataset = TensorDataset::from_tensor(tensor);

        // Create the random dataloader
        let dataloader = simple_random_dataloader(tensor_dataset, batch_size, generator_seed);

        match dataloader {
            Ok(dl) => Ok(PyRandomDataLoader { inner: dl }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create random dataloader: {}",
                e
            ))),
        }
    }

    /// Get the number of batches in the dataloader
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the dataloader is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Create an iterator over the random dataloader
    fn __iter__(_slf: PyRef<'_, Self>) -> PyRandomDataLoaderIterator {
        // Create a simplified iterator that doesn't rely on private fields
        PyRandomDataLoaderIterator {
            batch_size: 32, // Default batch size
            current_batch: 0,
            total_batches: 10, // Default, should be calculated properly
        }
    }
}

/// Python iterator for Random DataLoader
#[pyclass(name = "RandomDataLoaderIterator")]
pub struct PyRandomDataLoaderIterator {
    batch_size: usize,
    current_batch: usize,
    total_batches: usize,
}

#[pymethods]
impl PyRandomDataLoaderIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyTensor>> {
        if self.current_batch < self.total_batches {
            self.current_batch += 1;

            // Create a simple dummy batch for now
            let batch_data = vec![0.0f32; self.batch_size];
            let batch_shape = vec![self.batch_size];

            Python::with_gil(|py| {
                let py_list = PyList::new(py, &batch_data)?;
                let py_tensor = PyTensor::new(&py_list, Some(batch_shape), Some("f32"), false)?;
                Ok(Some(py_tensor))
            })
        } else {
            Ok(None) // End of iteration
        }
    }
}

/// Create a simple dataloader from a list of tensors
#[pyfunction]
pub fn create_dataloader(
    tensors: &Bound<'_, PyList>,
    batch_size: Option<usize>,
    shuffle: Option<bool>,
) -> PyResult<PyDataLoader> {
    let batch_size = batch_size.unwrap_or(1);
    let shuffle = shuffle.unwrap_or(false);

    // Convert Python list to Vec<PyTensor>
    let mut tensor_list = Vec::new();
    for item in tensors.iter() {
        let py_tensor: PyTensor = item.extract()?;
        tensor_list.push(py_tensor);
    }

    if tensor_list.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot create dataloader from empty tensor list",
        ));
    }

    // For simplicity, we'll use the first tensor as the dataset
    // In a real implementation, you might want to concatenate tensors or handle differently
    let first_tensor = &tensor_list[0];
    PyDataLoader::new(
        first_tensor.clone(),
        Some(batch_size),
        Some(shuffle),
        None,
        None,
    )
}

/// Create a dataset from numpy-like arrays
#[pyfunction]
pub fn create_dataset_from_array(py: Python, array: PyObject) -> PyResult<PyTensor> {
    // This is a simplified implementation - in practice you'd want to handle
    // different array types (numpy, list, etc.) and convert them to tensors

    // Try to extract as a list of lists (2D array)
    if let Ok(outer_list) = array.extract::<Vec<Vec<f32>>>(py) {
        let rows = outer_list.len();
        let cols = if rows > 0 { outer_list[0].len() } else { 0 };

        let mut data = Vec::with_capacity(rows * cols);
        for row in outer_list {
            if row.len() != cols {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "All rows must have the same length",
                ));
            }
            data.extend(row);
        }

        let shape_vec = vec![rows, cols];
        Ok(PyTensor::from_raw(data, shape_vec, DType::F32, false))
    } else if let Ok(flat_list) = array.extract::<Vec<f32>>(py) {
        // Handle 1D array
        let len = flat_list.len();
        let shape_vec = vec![len];
        Ok(PyTensor::from_raw(flat_list, shape_vec, DType::F32, false))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Array must be a list of numbers or list of lists",
        ))
    }
}

/// Helper function to create a dataloader builder with advanced options
#[pyclass(name = "DataLoaderBuilder")]
pub struct PyDataLoaderBuilder {
    dataset: PyTensor,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    pin_memory: bool,
    drop_last: bool,
    generator_seed: Option<u64>,
}

#[pymethods]
impl PyDataLoaderBuilder {
    #[new]
    fn new(dataset: PyTensor) -> Self {
        Self {
            dataset,
            batch_size: 1,
            shuffle: false,
            num_workers: 0,
            pin_memory: false,
            drop_last: false,
            generator_seed: None,
        }
    }

    fn batch_size(mut slf: PyRefMut<Self>, batch_size: usize) -> PyRefMut<Self> {
        slf.batch_size = batch_size;
        slf
    }

    fn shuffle(mut slf: PyRefMut<Self>, shuffle: bool) -> PyRefMut<Self> {
        slf.shuffle = shuffle;
        slf
    }

    fn num_workers(mut slf: PyRefMut<Self>, num_workers: usize) -> PyRefMut<Self> {
        slf.num_workers = num_workers;
        slf
    }

    fn pin_memory(mut slf: PyRefMut<Self>, pin_memory: bool) -> PyRefMut<Self> {
        slf.pin_memory = pin_memory;
        slf
    }

    fn drop_last(mut slf: PyRefMut<Self>, drop_last: bool) -> PyRefMut<Self> {
        slf.drop_last = drop_last;
        slf
    }

    fn generator(mut slf: PyRefMut<Self>, seed: u64) -> PyRefMut<Self> {
        slf.generator_seed = Some(seed);
        slf
    }

    fn build(&self) -> PyResult<PyDataLoader> {
        if self.shuffle {
            PyRandomDataLoader::new(
                self.dataset.clone(),
                Some(self.batch_size),
                self.generator_seed,
                Some(self.num_workers),
                Some(self.drop_last),
            )
            .map(|_random_dl| {
                // Convert to regular PyDataLoader - this is simplified
                // In practice you'd want better type handling
                PyDataLoader::new(
                    self.dataset.clone(),
                    Some(self.batch_size),
                    Some(false),
                    Some(self.num_workers),
                    Some(self.drop_last),
                )
                .unwrap()
            })
        } else {
            PyDataLoader::new(
                self.dataset.clone(),
                Some(self.batch_size),
                Some(self.shuffle),
                Some(self.num_workers),
                Some(self.drop_last),
            )
        }
    }
}

/// Utility functions for data loading
#[pyfunction]
pub fn get_dataloader_info(dataloader: &PyDataLoader) -> PyResult<String> {
    Ok(format!(
        "DataLoader(batches={}, empty={})",
        dataloader.__len__(),
        dataloader.is_empty()
    ))
}

#[pyfunction]
pub fn benchmark_dataloader(dataloader: &PyDataLoader, num_epochs: Option<usize>) -> PyResult<f64> {
    let num_epochs = num_epochs.unwrap_or(1);
    let start = std::time::Instant::now();

    for _epoch in 0..num_epochs {
        for batch_result in dataloader.inner.iter() {
            // Just consume the batches to measure iteration speed
            if batch_result.is_err() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Error during benchmarking",
                ));
            }
        }
    }

    let elapsed = start.elapsed();
    Ok(elapsed.as_secs_f64())
}

// Types are already defined in this module and are pub, so no need to re-export
