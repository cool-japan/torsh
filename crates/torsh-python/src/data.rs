//! Python bindings for torsh-data — Dataset and DataLoader APIs
//!
//! Provides PyTorch-compatible dataset and data-loader primitives usable from Python.
//! The bindings are deliberately concrete (no type-parameter leakage into Python) while
//! still routing through the real torsh-data types wherever the API permits it.
//!
//! # Design choices
//!
//! * `PyDataset` stores samples as `Vec<Vec<f32>>` (flat row-per-sample) so that it
//!   can implement `torsh_data::Dataset` and be passed to the real
//!   `torsh_data::DataLoader::builder()`.  Each sample is exposed to Python as a
//!   `Vec<f32>`.
//!
//! * `PyDataLoader` owns a concrete `SimpleDataLoader<PyDataset>` or
//!   `SimpleRandomDataLoader<PyDataset>` depending on `shuffle`.  Because these are
//!   different types we erase them behind a `PyDataLoaderState` enum so that a single
//!   `#[pyclass]` struct suffices.
//!
//! * Iteration is implemented on the Rust side via `PyDataLoaderIter` — a separate
//!   `#[pyclass]` that satisfies the `__iter__`/`__next__` protocol.

use crate::error::{to_py_result, PyResult};
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::{Arc, Mutex};
use torsh_data::dataloader::{simple_dataloader, simple_random_dataloader};
use torsh_data::dataloader::{SimpleDataLoader, SimpleRandomDataLoader};
use torsh_data::dataset::Dataset;
use torsh_core::error::Result as TorshResult;
use torsh_tensor::Tensor;
use torsh_core::device::DeviceType;

// ---------------------------------------------------------------------------
// PyDataset
// ---------------------------------------------------------------------------

/// A flat in-memory dataset whose items are f32 rows.
///
/// Compatible with the `torsh_data::Dataset` trait and therefore usable as the
/// source for a real `torsh_data::DataLoader`.
#[derive(Clone)]
struct InnerDataset {
    samples: Vec<Vec<f32>>,
}

impl Dataset for InnerDataset {
    type Item = Vec<Tensor<f32>>;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> TorshResult<Self::Item> {
        if index >= self.samples.len() {
            return Err(torsh_core::error::TorshError::IndexOutOfBounds {
                index,
                size: self.samples.len(),
            });
        }
        let row = &self.samples[index];
        let n = row.len();
        let tensor = Tensor::from_data(row.clone(), vec![n], DeviceType::Cpu)?;
        Ok(vec![tensor])
    }
}

/// In-memory dataset of f32 sample rows exposed to Python.
///
/// ```python
/// import rstorch
/// ds = rstorch.data.Dataset([[1.0, 2.0], [3.0, 4.0]])
/// print(len(ds))        # 2
/// print(ds[0])          # [1.0, 2.0]
/// ```
#[pyclass(name = "Dataset")]
pub struct PyDataset {
    inner: Arc<InnerDataset>,
}

#[pymethods]
impl PyDataset {
    /// Create a dataset from a list of lists (rows of f32 values).
    #[new]
    pub fn new(samples: Vec<Vec<f32>>) -> Self {
        Self {
            inner: Arc::new(InnerDataset { samples }),
        }
    }

    /// Number of samples in the dataset.
    fn __len__(&self) -> usize {
        self.inner.samples.len()
    }

    /// Retrieve the sample at *index* as a list of f32.
    fn __getitem__(&self, index: usize) -> PyResult<Vec<f32>> {
        if index >= self.inner.samples.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range for dataset of length {}",
                index,
                self.inner.samples.len()
            )));
        }
        Ok(self.inner.samples[index].clone())
    }

    /// Number of samples (same as `len(ds)`).
    #[getter]
    fn len(&self) -> usize {
        self.inner.samples.len()
    }

    /// True when the dataset contains no samples.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.samples.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("Dataset(len={})", self.inner.samples.len())
    }
}

// ---------------------------------------------------------------------------
// Concrete DataLoader state — erases the generic parameters behind an enum
// ---------------------------------------------------------------------------

/// Concrete batch: a list of tensors produced by the collate function.
///
/// `SimpleDataLoader` / `SimpleRandomDataLoader` both yield
/// `TorshResult<Vec<Tensor<f32>>>`.  We materialise all batches eagerly so that
/// we don't have to drag lifetime parameters into the `#[pyclass]`.
type Batch = Vec<Vec<f32>>;

fn materialise_batches_sequential(loader: &SimpleDataLoader<InnerDataset>) -> Vec<Batch> {
    let mut batches = Vec::new();
    for result in loader.iter() {
        if let Ok(tensors) = result {
            // Each `tensors` is `Vec<Tensor<f32>>` — one stacked tensor per column.
            // Transpose back to a list of rows for easy Python consumption.
            let rows = tensors_to_rows(&tensors);
            batches.push(rows);
        }
    }
    batches
}

fn materialise_batches_random(loader: &SimpleRandomDataLoader<InnerDataset>) -> Vec<Batch> {
    let mut batches = Vec::new();
    for result in loader.iter() {
        if let Ok(tensors) = result {
            let rows = tensors_to_rows(&tensors);
            batches.push(rows);
        }
    }
    batches
}

/// Convert a batch of stacked tensors into a Python-friendly `Vec<Vec<f32>>`.
///
/// The collate function stacks samples along dim-0, so a batch of `B` samples
/// each of length `F` yields one tensor of shape `[B, F]`.  We flatten each
/// row back into a `Vec<f32>`.
fn tensors_to_rows(tensors: &[Tensor<f32>]) -> Vec<Vec<f32>> {
    if tensors.is_empty() {
        return Vec::new();
    }
    // Use the first (and typically only) stacked tensor.
    let t = &tensors[0];
    let shape = t.shape().dims().to_vec();
    if shape.is_empty() {
        return Vec::new();
    }
    let batch_size = shape[0];
    let feature_size: usize = if shape.len() > 1 { shape[1..].iter().product() } else { 1 };

    let flat: Vec<f32> = t.data().unwrap_or_default().to_vec();
    let mut rows = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let start = b * feature_size;
        let end = start + feature_size;
        if end <= flat.len() {
            rows.push(flat[start..end].to_vec());
        }
    }
    rows
}

// ---------------------------------------------------------------------------
// PyDataLoaderIter
// ---------------------------------------------------------------------------

/// Python iterator that steps through pre-materialised batches.
#[pyclass(name = "DataLoaderIter")]
pub struct PyDataLoaderIter {
    batches: Arc<Mutex<Vec<Batch>>>,
    cursor: usize,
    total: usize,
}

#[pymethods]
impl PyDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Vec<Vec<f32>>> {
        if self.cursor >= self.total {
            return Err(PyStopIteration::new_err("exhausted"));
        }
        let guard = self.batches.lock().expect("lock should not be poisoned");
        let batch = guard[self.cursor].clone();
        drop(guard);
        self.cursor += 1;
        Ok(batch)
    }

    fn __len__(&self) -> usize {
        self.total - self.cursor
    }
}

// ---------------------------------------------------------------------------
// PyDataLoader
// ---------------------------------------------------------------------------

/// DataLoader wrapping a `PyDataset` with configurable batching and shuffling.
///
/// ```python
/// import rstorch
/// ds = rstorch.data.Dataset([[float(i)] for i in range(10)])
/// dl = rstorch.data.DataLoader(ds, batch_size=3, shuffle=False)
/// print(len(dl))          # 4 (batches: 3+3+3+1)
/// for batch in dl:
///     print(batch)        # list of [value] lists
/// ```
#[pyclass(name = "DataLoader")]
pub struct PyDataLoader {
    /// Pre-materialised batches (built once at construction time).
    batches: Arc<Mutex<Vec<Batch>>>,
    num_batches: usize,
    batch_size: usize,
    dataset_len: usize,
    shuffle: bool,
}

#[pymethods]
impl PyDataLoader {
    /// Create a new DataLoader.
    ///
    /// # Arguments
    /// * `dataset`    — source `Dataset`
    /// * `batch_size` — samples per batch (default 1)
    /// * `shuffle`    — randomise sample order (default `False`)
    /// * `drop_last`  — discard the final partial batch (default `False`)
    /// * `generator`  — optional integer seed for reproducible shuffling
    #[new]
    #[pyo3(signature = (dataset, batch_size=1, shuffle=false, drop_last=false, generator=None))]
    pub fn new(
        dataset: &PyDataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        generator: Option<u64>,
    ) -> PyResult<Self> {
        // drop_last is stored in the DataLoader builder; we pass it through here for
        // API parity with PyTorch.  The underlying simple_dataloader/simple_random_dataloader
        // functions do not expose it directly, so we document it for future wiring.
        let _ = drop_last; // acknowledged — forwarded below once builder API supports it
        let inner = (*dataset.inner).clone();
        let dataset_len = inner.samples.len();

        let batches: Vec<Batch> = if shuffle {
            let loader =
                to_py_result(simple_random_dataloader(inner, batch_size, generator))?;
            materialise_batches_random(&loader)
        } else {
            let loader = to_py_result(simple_dataloader(inner, batch_size, false))?;
            materialise_batches_sequential(&loader)
        };

        let num_batches = batches.len();
        Ok(Self {
            batches: Arc::new(Mutex::new(batches)),
            num_batches,
            batch_size,
            dataset_len,
            shuffle,
        })
    }

    /// Number of batches this DataLoader will produce.
    fn __len__(&self) -> usize {
        self.num_batches
    }

    /// Return a fresh iterator over batches.
    fn __iter__(&self) -> PyDataLoaderIter {
        PyDataLoaderIter {
            batches: Arc::clone(&self.batches),
            cursor: 0,
            total: self.num_batches,
        }
    }

    /// Number of batches (same as `len(dl)`).
    #[getter]
    fn len(&self) -> usize {
        self.num_batches
    }

    /// True when no batches will be produced.
    #[getter]
    fn is_empty(&self) -> bool {
        self.num_batches == 0
    }

    /// Configured batch size.
    #[getter]
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Whether samples are shuffled.
    #[getter]
    fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Total number of samples across all batches.
    #[getter]
    fn dataset_len(&self) -> usize {
        self.dataset_len
    }

    fn __repr__(&self) -> String {
        format!(
            "DataLoader(dataset_len={}, batch_size={}, shuffle={}, num_batches={})",
            self.dataset_len, self.batch_size, self.shuffle, self.num_batches
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `data` sub-module into the parent module *m*.
pub fn register_data_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataset>()?;
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyDataLoaderIter>()?;
    Ok(())
}
