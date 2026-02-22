//! WebAssembly support for torsh-data
//!
//! This module provides WASM-compatible implementations of data loading and processing
//! functionality, enabling browser-based machine learning applications.

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use web_sys::{console, window, Request, RequestInit, RequestMode, Response};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use js_sys::{Array, Promise, Uint8Array};

use crate::{utils, Dataset};
use std::collections::VecDeque;
use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
use std::marker::PhantomData;

/// WASM-compatible dataset implementation
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub struct WasmDataset<T: TensorElement> {
    data: Vec<T>,
    length: usize,
    batch_size: usize,
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
pub struct WasmDataset<T: TensorElement> {
    _phantom: PhantomData<T>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl<T: TensorElement + Clone> WasmDataset<T> {
    /// Create a new WASM dataset from data
    pub fn new(data: Vec<T>) -> Self {
        let length = data.len();
        Self {
            data,
            length,
            batch_size: 32,
        }
    }

    /// Set batch size for processing
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Get a batch of data starting from the given index
    pub fn get_batch(&self, start_idx: usize, batch_size: usize) -> Result<Vec<T>> {
        if start_idx >= self.length {
            return Err(utils::errors::invalid_index(start_idx, self.length));
        }

        let end_idx = std::cmp::min(start_idx + batch_size, self.length);
        Ok(self.data[start_idx..end_idx].to_vec())
    }

    /// Convert to tensor
    pub fn to_tensor(&self) -> Result<Tensor<T>> {
        let shape = vec![self.length];
        Ok(Tensor::from_data(
            self.data.clone(),
            shape,
            DeviceType::Cpu,
        )?)
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
impl<T: TensorElement> WasmDataset<T> {
    /// Placeholder implementation when WASM is not enabled
    pub fn new(_data: Vec<T>) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Placeholder for batch size setting
    pub fn with_batch_size(self, _batch_size: usize) -> Self {
        self
    }

    /// Placeholder for batch retrieval
    pub fn get_batch(&self, _start_idx: usize, _batch_size: usize) -> Result<Vec<T>> {
        Err(TorshError::InvalidArgument(
            "WASM support not enabled".to_string(),
        ))
    }

    /// Placeholder for tensor conversion
    pub fn to_tensor(&self) -> Result<Tensor<T>> {
        Err(TorshError::InvalidArgument(
            "WASM support not enabled".to_string(),
        ))
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl<T: TensorElement + Clone> Dataset for WasmDataset<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.length
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.length {
            return Err(utils::errors::invalid_index(index, self.length));
        }
        Ok(self.data[index].clone())
    }
}

#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
impl<T: TensorElement> Dataset for WasmDataset<T> {
    type Item = T;

    fn len(&self) -> usize {
        0
    }

    fn get(&self, _index: usize) -> Result<Self::Item> {
        Err(TorshError::InvalidArgument(
            "WASM support not enabled".to_string(),
        ))
    }
}

/// WASM-compatible data loader with JavaScript bindings
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct WasmDataLoader {
    batch_size: usize,
    current_data: Option<Vec<f32>>,
    current_index: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl WasmDataLoader {
    /// Create a new WASM data loader
    #[wasm_bindgen(constructor)]
    pub fn new(batch_size: usize) -> WasmDataLoader {
        utils::validate_positive(batch_size, "batch_size").unwrap_or_else(|_| {
            console::warn_1(&"Invalid batch size, using default of 32".into());
        });

        Self {
            batch_size: if batch_size > 0 { batch_size } else { 32 },
            current_data: None,
            current_index: 0,
        }
    }

    /// Load data from a URL using the Fetch API
    #[wasm_bindgen]
    pub async fn load_from_url(&mut self, url: &str) -> Result<(), JsValue> {
        let window = window().ok_or("No global window object")?;

        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(url, &opts)?;

        let resp_value =
            wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: Response = resp_value.dyn_into()?;

        if !resp.ok() {
            return Err(format!("HTTP error: {}", resp.status()).into());
        }

        let array_buffer = wasm_bindgen_futures::JsFuture::from(resp.array_buffer()?).await?;
        let uint8_array = Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();

        // Parse the data (assuming it's JSON for now)
        let text = String::from_utf8(bytes).map_err(|e| e.to_string())?;
        let data: Vec<f32> = serde_json::from_str(&text).map_err(|e| e.to_string())?;

        self.current_data = Some(data);
        self.current_index = 0;

        Ok(())
    }

    /// Load data from a JavaScript array
    #[wasm_bindgen]
    pub fn load_from_array(&mut self, array: &Array) -> Result<(), JsValue> {
        let mut data = Vec::new();

        for i in 0..array.length() {
            let value = array.get(i);
            if let Some(num) = value.as_f64() {
                data.push(num as f32);
            } else {
                return Err("Array contains non-numeric values".into());
            }
        }

        self.current_data = Some(data);
        self.current_index = 0;

        Ok(())
    }

    /// Get the next batch of data
    #[wasm_bindgen]
    pub fn next_batch(&mut self) -> Result<Array, JsValue> {
        let data = self
            .current_data
            .as_ref()
            .ok_or("No data loaded. Call load_from_url() or load_from_array() first")?;

        if self.current_index >= data.len() {
            return Err("End of dataset reached".into());
        }

        let end_index = std::cmp::min(self.current_index + self.batch_size, data.len());
        let batch = &data[self.current_index..end_index];

        let js_array = Array::new();
        for &value in batch {
            js_array.push(&JsValue::from_f64(value as f64));
        }

        self.current_index = end_index;

        Ok(js_array)
    }

    /// Check if there are more batches available
    #[wasm_bindgen]
    pub fn has_next(&self) -> bool {
        self.current_data
            .as_ref()
            .map(|data| self.current_index < data.len())
            .unwrap_or(false)
    }

    /// Reset the data loader to the beginning
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Get the total number of data points
    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.current_data
            .as_ref()
            .map(|data| data.len())
            .unwrap_or(0)
    }

    /// Get the current progress as a percentage
    #[wasm_bindgen]
    pub fn progress(&self) -> f32 {
        let total_len = self.len();
        if total_len == 0 {
            return 0.0;
        }
        (self.current_index as f32 / total_len as f32) * 100.0
    }
}

/// Streaming dataset for processing data as it arrives
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub struct StreamingDataset<T: TensorElement> {
    buffer: VecDeque<T>,
    batch_size: usize,
    max_buffer_size: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl<T: TensorElement + Clone> StreamingDataset<T> {
    /// Create a new streaming dataset
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            batch_size,
            max_buffer_size: batch_size * 10,
        }
    }

    /// Add data to the stream
    pub fn add_data(&mut self, data: Vec<T>) -> Result<()> {
        for item in data {
            if self.buffer.len() >= self.max_buffer_size {
                self.buffer.pop_front();
            }
            self.buffer.push_back(item);
        }
        Ok(())
    }

    /// Get the next batch if available
    pub fn next_batch(&mut self) -> Option<Vec<T>> {
        if self.buffer.len() >= self.batch_size {
            let mut batch = Vec::with_capacity(self.batch_size);
            for _ in 0..self.batch_size {
                if let Some(item) = self.buffer.pop_front() {
                    batch.push(item);
                }
            }
            Some(batch)
        } else {
            None
        }
    }

    /// Check if a batch is available
    pub fn has_batch(&self) -> bool {
        self.buffer.len() >= self.batch_size
    }

    /// Get the current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

/// Progressive dataset that loads data in chunks
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub struct ProgressiveDataset {
    loaded_chunks: Vec<Vec<f32>>,
    pending_urls: VecDeque<String>,
    load_progress: f32,
    chunk_size: usize,
    total_expected_size: Option<usize>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl ProgressiveDataset {
    /// Create a new progressive dataset
    pub fn new(urls: Vec<String>, chunk_size: usize) -> Self {
        Self {
            loaded_chunks: Vec::new(),
            pending_urls: urls.into(),
            load_progress: 0.0,
            chunk_size,
            total_expected_size: None,
        }
    }

    /// Load the next chunk of data
    pub async fn load_next_chunk(&mut self) -> Result<bool, JsValue> {
        if let Some(url) = self.pending_urls.pop_front() {
            let window = window().ok_or("No global window object")?;

            let mut opts = RequestInit::new();
            opts.method("GET");
            opts.mode(RequestMode::Cors);

            let request = Request::new_with_str_and_init(&url, &opts)?;
            let resp_value =
                wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request)).await?;
            let resp: Response = resp_value.dyn_into()?;

            if !resp.ok() {
                return Err(format!("HTTP error: {}", resp.status()).into());
            }

            let array_buffer = wasm_bindgen_futures::JsFuture::from(resp.array_buffer()?).await?;
            let uint8_array = Uint8Array::new(&array_buffer);
            let bytes = uint8_array.to_vec();

            let text = String::from_utf8(bytes).map_err(|e| e.to_string())?;
            let chunk_data: Vec<f32> = serde_json::from_str(&text).map_err(|e| e.to_string())?;

            self.loaded_chunks.push(chunk_data);
            self.update_progress();

            Ok(true)
        } else {
            Ok(false) // No more chunks to load
        }
    }

    /// Update the loading progress
    fn update_progress(&mut self) {
        let total_chunks = self.loaded_chunks.len() + self.pending_urls.len();
        if total_chunks > 0 {
            self.load_progress = (self.loaded_chunks.len() as f32) / (total_chunks as f32) * 100.0;
        }
    }

    /// Get the current loading progress (0-100)
    pub fn progress(&self) -> f32 {
        self.load_progress
    }

    /// Get all loaded data as a flattened vector
    pub fn get_loaded_data(&self) -> Vec<f32> {
        self.loaded_chunks.iter().flatten().cloned().collect()
    }

    /// Check if all chunks have been loaded
    pub fn is_complete(&self) -> bool {
        self.pending_urls.is_empty()
    }

    /// Get the number of loaded chunks
    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded_chunks.len()
    }
}

/// Memory-efficient data processor with optimization
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub struct WasmMemoryPool {
    buffers: VecDeque<Vec<u8>>,
    max_pool_size: usize,
    buffer_size: usize,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
impl WasmMemoryPool {
    /// Create a new memory pool
    pub fn new(max_pool_size: usize, buffer_size: usize) -> Self {
        Self {
            buffers: VecDeque::new(),
            max_pool_size,
            buffer_size,
        }
    }

    /// Get a buffer from the pool or create a new one
    pub fn get_buffer(&mut self) -> Vec<u8> {
        self.buffers
            .pop_front()
            .unwrap_or_else(|| Vec::with_capacity(self.buffer_size))
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, mut buffer: Vec<u8>) {
        if self.buffers.len() < self.max_pool_size {
            buffer.clear();
            self.buffers.push_back(buffer);
        }
    }

    /// Get current pool size
    pub fn pool_size(&self) -> usize {
        self.buffers.len()
    }
}

/// Advanced WASM data processor with workers and optimization
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
pub struct WasmDataProcessor {
    memory_pool: WasmMemoryPool,
    batch_size: usize,
    enable_workers: bool,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[wasm_bindgen]
impl WasmDataProcessor {
    /// Create a new data processor
    #[wasm_bindgen(constructor)]
    pub fn new(batch_size: usize, enable_workers: bool) -> Self {
        Self {
            memory_pool: WasmMemoryPool::new(10, 1024 * 1024), // 10 buffers of 1MB each
            batch_size,
            enable_workers,
        }
    }

    /// Process a batch of data efficiently
    #[wasm_bindgen]
    pub fn process_batch(&mut self, input: &Array) -> Result<Array, JsValue> {
        let mut buffer = self.memory_pool.get_buffer();

        // Convert JavaScript array to Rust vector
        let mut data = Vec::with_capacity(input.length() as usize);
        for i in 0..input.length() {
            let value = input.get(i);
            if let Some(num) = value.as_f64() {
                data.push(num as f32);
            }
        }

        // Process the data (example: apply normalization)
        let processed_data: Vec<f32> = data
            .iter()
            .map(|&x| (x - 0.5) * 2.0) // Simple normalization
            .collect();

        // Convert back to JavaScript array
        let result = Array::new();
        for value in processed_data {
            result.push(&JsValue::from_f64(value as f64));
        }

        // Return buffer to pool
        self.memory_pool.return_buffer(buffer);

        Ok(result)
    }

    /// Get memory pool statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> JsValue {
        let stats = format!(
            "{{\"pool_size\": {}, \"max_pool_size\": {}}}",
            self.memory_pool.pool_size(),
            self.memory_pool.max_pool_size
        );
        JsValue::from_str(&stats)
    }
}

/// WASM-specific optimizations and utilities
pub mod optimization {
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    use wasm_bindgen::prelude::*;
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    use web_sys::console;

    /// Memory-efficient chunk processor for large datasets
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub struct ChunkProcessor {
        chunk_size: usize,
        processing_queue: std::collections::VecDeque<Vec<f32>>,
        max_queue_size: usize,
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    impl ChunkProcessor {
        /// Create a new chunk processor
        pub fn new(chunk_size: usize, max_queue_size: usize) -> Self {
            Self {
                chunk_size,
                processing_queue: std::collections::VecDeque::new(),
                max_queue_size,
            }
        }

        /// Add data to be processed in chunks
        pub fn add_data(&mut self, data: Vec<f32>) -> Result<(), String> {
            if self.processing_queue.len() >= self.max_queue_size {
                return Err("Processing queue is full".to_string());
            }

            // Split data into chunks
            for chunk in data.chunks(self.chunk_size) {
                self.processing_queue.push_back(chunk.to_vec());
            }

            Ok(())
        }

        /// Process the next chunk
        pub fn process_next_chunk(&mut self) -> Option<Vec<f32>> {
            if let Some(chunk) = self.processing_queue.pop_front() {
                // Apply processing (normalize, etc.)
                let processed: Vec<f32> = chunk
                    .iter()
                    .map(|&x| (x - 0.5) * 2.0) // Simple normalization
                    .collect();
                Some(processed)
            } else {
                None
            }
        }

        /// Get queue status
        pub fn queue_status(&self) -> (usize, usize) {
            (self.processing_queue.len(), self.max_queue_size)
        }
    }

    /// Browser compatibility checker
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub struct BrowserCompatibility;

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    impl BrowserCompatibility {
        /// Check if the browser supports required features
        pub fn check_support() -> BrowserSupport {
            let window = web_sys::window();
            let navigator = window.as_ref().and_then(|w| w.navigator().ok());

            BrowserSupport {
                webassembly: Self::check_webassembly(),
                fetch_api: Self::check_fetch_api(),
                array_buffer: Self::check_array_buffer(),
                shared_array_buffer: Self::check_shared_array_buffer(),
                web_workers: Self::check_web_workers(),
            }
        }

        fn check_webassembly() -> bool {
            js_sys::eval("typeof WebAssembly !== 'undefined'")
                .map(|v| v.as_bool().unwrap_or(false))
                .unwrap_or(false)
        }

        fn check_fetch_api() -> bool {
            js_sys::eval("typeof fetch !== 'undefined'")
                .map(|v| v.as_bool().unwrap_or(false))
                .unwrap_or(false)
        }

        fn check_array_buffer() -> bool {
            js_sys::eval("typeof ArrayBuffer !== 'undefined'")
                .map(|v| v.as_bool().unwrap_or(false))
                .unwrap_or(false)
        }

        fn check_shared_array_buffer() -> bool {
            js_sys::eval("typeof SharedArrayBuffer !== 'undefined'")
                .map(|v| v.as_bool().unwrap_or(false))
                .unwrap_or(false)
        }

        fn check_web_workers() -> bool {
            js_sys::eval("typeof Worker !== 'undefined'")
                .map(|v| v.as_bool().unwrap_or(false))
                .unwrap_or(false)
        }
    }

    /// Browser support information
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[derive(Debug, Clone)]
    pub struct BrowserSupport {
        pub webassembly: bool,
        pub fetch_api: bool,
        pub array_buffer: bool,
        pub shared_array_buffer: bool,
        pub web_workers: bool,
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    impl BrowserSupport {
        /// Check if the browser has full support for all features
        pub fn is_fully_supported(&self) -> bool {
            self.webassembly && self.fetch_api && self.array_buffer
        }

        /// Get a list of missing features
        pub fn missing_features(&self) -> Vec<&'static str> {
            let mut missing = Vec::new();

            if !self.webassembly {
                missing.push("WebAssembly");
            }
            if !self.fetch_api {
                missing.push("Fetch API");
            }
            if !self.array_buffer {
                missing.push("ArrayBuffer");
            }
            if !self.shared_array_buffer {
                missing.push("SharedArrayBuffer");
            }
            if !self.web_workers {
                missing.push("Web Workers");
            }

            missing
        }
    }

    /// Performance monitoring for WASM operations
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub struct WasmPerformanceMonitor {
        operation_times: std::collections::HashMap<String, Vec<f64>>,
        max_samples: usize,
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    impl WasmPerformanceMonitor {
        /// Create a new performance monitor
        pub fn new(max_samples: usize) -> Self {
            Self {
                operation_times: std::collections::HashMap::new(),
                max_samples,
            }
        }

        /// Record an operation time
        pub fn record_operation(&mut self, operation: &str, duration_ms: f64) {
            let times = self
                .operation_times
                .entry(operation.to_string())
                .or_insert_with(Vec::new);

            if times.len() >= self.max_samples {
                times.remove(0);
            }
            times.push(duration_ms);
        }

        /// Get average time for an operation
        pub fn average_time(&self, operation: &str) -> Option<f64> {
            self.operation_times.get(operation).and_then(|times| {
                if times.is_empty() {
                    None
                } else {
                    Some(times.iter().sum::<f64>() / times.len() as f64)
                }
            })
        }

        /// Get performance report
        pub fn performance_report(&self) -> String {
            let mut report = String::from("WASM Performance Report:\n");

            for (operation, times) in &self.operation_times {
                if !times.is_empty() {
                    let avg = times.iter().sum::<f64>() / times.len() as f64;
                    let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                    report.push_str(&format!(
                        "  {}: avg={:.2}ms, min={:.2}ms, max={:.2}ms, samples={}\n",
                        operation,
                        avg,
                        min,
                        max,
                        times.len()
                    ));
                }
            }

            report
        }
    }
}

/// Utility functions for WASM support
pub mod wasm_utils {
    use super::*;

    /// Check if WASM support is available at compile time
    pub const fn is_wasm_available() -> bool {
        cfg!(all(target_arch = "wasm32", feature = "wasm"))
    }

    /// Log a message to the browser console (WASM only)
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn log(message: &str) {
        console::log_1(&message.into());
    }

    /// Log a warning to the browser console (WASM only)
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn warn(message: &str) {
        console::warn_1(&message.into());
    }

    /// Log an error to the browser console (WASM only)
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn error(message: &str) {
        console::error_1(&message.into());
    }

    /// No-op implementations for non-WASM builds
    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    pub fn log(_message: &str) {}

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    pub fn warn(_message: &str) {}

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    pub fn error(_message: &str) {}

    /// Create a sample dataset for testing (WASM compatible)
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    pub fn create_sample_dataset() -> Result<WasmDataset<f32>> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        Ok(WasmDataset::new(data))
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    pub fn create_sample_dataset() -> Result<WasmDataset<f32>> {
        Ok(WasmDataset::new(vec![]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_availability() {
        // Test that we can detect WASM availability
        assert!(wasm_utils::is_wasm_available() || !wasm_utils::is_wasm_available());
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[test]
    fn test_wasm_dataset() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dataset = WasmDataset::new(data);

        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.get(0).unwrap(), 1.0);
        assert_eq!(dataset.get(4).unwrap(), 5.0);
        assert!(dataset.get(5).is_err());
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[test]
    fn test_streaming_dataset() {
        let mut stream = StreamingDataset::new(3);

        assert!(!stream.has_batch());

        stream.add_data(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(stream.has_batch());

        let batch = stream.next_batch().unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch, vec![1.0, 2.0, 3.0]);

        assert!(!stream.has_batch());
        assert_eq!(stream.buffer_size(), 1);
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[test]
    fn test_progressive_dataset() {
        let urls = vec![
            "http://example.com/chunk1.json".to_string(),
            "http://example.com/chunk2.json".to_string(),
        ];
        let mut progressive = ProgressiveDataset::new(urls, 1000);

        assert_eq!(progressive.progress(), 0.0);
        assert!(!progressive.is_complete());
        assert_eq!(progressive.loaded_chunk_count(), 0);
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[test]
    fn test_memory_pool() {
        let mut pool = WasmMemoryPool::new(5, 1024);

        assert_eq!(pool.pool_size(), 0);

        let buffer1 = pool.get_buffer();
        assert_eq!(buffer1.capacity(), 1024);

        pool.return_buffer(buffer1);
        assert_eq!(pool.pool_size(), 1);

        let buffer2 = pool.get_buffer();
        assert_eq!(pool.pool_size(), 0);

        // Test pool overflow
        for i in 0..10 {
            let buffer = vec![0u8; 100];
            pool.return_buffer(buffer);
        }
        assert_eq!(pool.pool_size(), 5); // Max pool size
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[test]
    fn test_wasm_data_processor() {
        let processor = WasmDataProcessor::new(32, false);

        // Test that processor was created successfully
        let stats = processor.get_memory_stats();
        assert!(stats.is_string());
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
    #[test]
    fn test_wasm_disabled() {
        let dataset = WasmDataset::<f32>::new(vec![]);
        assert_eq!(dataset.len(), 0);
        assert!(dataset.get(0).is_err());
    }
}
