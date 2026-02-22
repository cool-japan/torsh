# WebAssembly Support Plan for torsh-data

## Overview

This document outlines the plan for adding WebAssembly (WASM) support to the torsh-data crate, enabling browser-based data processing capabilities. This would allow data loading and preprocessing operations to run efficiently in web browsers, opening up new possibilities for client-side machine learning and data analysis.

## Current State Analysis

### Compatible Components
1. **Basic Dataset Operations**: Most dataset operations are pure Rust and should work in WASM
2. **Transform Pipeline**: Mathematical transforms and data manipulation should be compatible
3. **Collate Functions**: Tensor collation and batching operations
4. **Sampling**: Random and sequential sampling algorithms
5. **Text Processing**: Tokenization, vocabulary management, and text transforms

### Potential Challenges
1. **File System Access**: Browser security restrictions limit file system access
2. **Threading**: WASM has limited threading support compared to native Rust
3. **Memory Management**: Different memory model in WASM environment
4. **Network I/O**: Different patterns for data fetching in browsers
5. **Random Number Generation**: May need alternative entropy sources

## Implementation Strategy

### Phase 1: Core WASM Compatibility

#### 1.1 Feature Flag Setup
```toml
# Add to Cargo.toml
[features]
wasm = ["wasm-bindgen", "js-sys", "web-sys"]
default = []

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = "0.3"
getrandom = { version = "0.2", features = ["js"] }
```

#### 1.2 Conditional Compilation
- Use `#[cfg(target_arch = "wasm32")]` for WASM-specific code
- Use `#[cfg(not(target_arch = "wasm32"))]` for native-only features
- Create WASM-compatible alternatives for problematic dependencies

#### 1.3 Basic Dataset Support
```rust
#[cfg(target_arch = "wasm32")]
pub struct WasmDataset<T> {
    data: Vec<T>,
    length: usize,
}

#[cfg(target_arch = "wasm32")]
impl<T> Dataset<T> for WasmDataset<T> {
    fn len(&self) -> usize {
        self.length
    }
    
    fn get(&self, index: usize) -> Result<T> {
        // WASM-specific implementation
    }
}
```

### Phase 2: Browser Integration

#### 2.1 Web APIs Integration
- **Fetch API**: For loading data from URLs
- **File API**: For handling user-uploaded files
- **Worker API**: For background processing
- **IndexedDB**: For client-side data persistence

#### 2.2 JavaScript Bindings
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmDataLoader {
    inner: DataLoader<f32>,
}

#[wasm_bindgen]
impl WasmDataLoader {
    #[wasm_bindgen(constructor)]
    pub fn new(batch_size: usize) -> WasmDataLoader {
        // Create WASM-compatible DataLoader
    }
    
    #[wasm_bindgen]
    pub async fn load_from_url(&mut self, url: &str) -> Result<(), JsValue> {
        // Load data using Fetch API
    }
    
    #[wasm_bindgen]
    pub fn next_batch(&mut self) -> Result<js_sys::Array, JsValue> {
        // Return batch as JavaScript array
    }
}
```

#### 2.3 Async/Await Support
```rust
#[cfg(target_arch = "wasm32")]
pub async fn load_remote_dataset(url: &str) -> Result<Box<dyn Dataset<f32>>> {
    use web_sys::window;
    
    let window = window().unwrap();
    let response = JsFuture::from(window.fetch_with_str(url)).await?;
    let response: Response = response.dyn_into()?;
    
    // Process response data
    let data = parse_response_data(response).await?;
    Ok(Box::new(WasmDataset::new(data)))
}
```

### Phase 3: Performance Optimization

#### 3.1 Memory Management
- Use `wee_alloc` for smaller WASM binary size
- Implement memory pooling for frequent allocations
- Optimize garbage collection patterns

#### 3.2 Parallel Processing
```rust
#[cfg(all(target_arch = "wasm32", feature = "web-workers"))]
pub struct WasmWorkerPool {
    workers: Vec<Worker>,
    task_queue: VecDeque<Task>,
}

impl WasmWorkerPool {
    pub fn new(num_workers: usize) -> Self {
        // Create web workers for parallel processing
    }
    
    pub async fn process_batch(&self, batch: Batch) -> Result<ProcessedBatch> {
        // Distribute work across web workers
    }
}
```

#### 3.3 Streaming Data Processing
```rust
#[cfg(target_arch = "wasm32")]
pub struct StreamingDataset {
    stream: ReadableStream,
    buffer: VecDeque<DataPoint>,
}

impl StreamingDataset {
    pub async fn from_stream(stream: ReadableStream) -> Self {
        // Create dataset from streaming source
    }
    
    pub async fn next(&mut self) -> Option<DataPoint> {
        // Get next data point from stream
    }
}
```

### Phase 4: Advanced Features

#### 4.1 Client-Side Caching
```rust
#[cfg(target_arch = "wasm32")]
pub struct IndexedDBCache {
    db: IdbDatabase,
}

impl IndexedDBCache {
    pub async fn store_dataset(&self, key: &str, dataset: &[u8]) -> Result<()> {
        // Store dataset in IndexedDB
    }
    
    pub async fn load_dataset(&self, key: &str) -> Result<Vec<u8>> {
        // Load dataset from IndexedDB
    }
}
```

#### 4.2 Progressive Loading
```rust
#[cfg(target_arch = "wasm32")]
pub struct ProgressiveDataset {
    loaded_chunks: Vec<DataChunk>,
    pending_urls: VecDeque<String>,
    load_progress: f32,
}

impl ProgressiveDataset {
    pub async fn load_next_chunk(&mut self) -> Result<()> {
        // Load next chunk progressively
    }
    
    pub fn progress(&self) -> f32 {
        self.load_progress
    }
}
```

## Browser Compatibility Matrix

| Feature | Chrome | Firefox | Safari | Edge | Notes |
|---------|--------|---------|--------|------|-------|
| Basic WASM | ✓ | ✓ | ✓ | ✓ | Universally supported |
| WASM Threads | ✓ | ✓ | Partial | ✓ | Safari has limitations |
| Bulk Memory | ✓ | ✓ | ✓ | ✓ | Required for optimization |
| SIMD | ✓ | ✓ | ✗ | ✓ | Safari support coming |
| Shared Memory | ✓ | ✓ | ✗ | ✓ | Limited by browser policy |

## Security Considerations

### 1. Memory Safety
- Leverage Rust's memory safety guarantees
- Avoid unsafe code blocks in WASM builds
- Use bounds checking for all array access

### 2. Data Privacy
- Implement client-side encryption for sensitive data
- Avoid sending private data to servers unnecessarily
- Use secure random number generation

### 3. Origin Policies
- Respect CORS policies for cross-origin requests
- Implement proper CSP headers for WASM modules
- Handle same-origin restrictions appropriately

## Testing Strategy

### 1. Unit Tests
```rust
#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use wasm_bindgen_test::*;
    
    #[wasm_bindgen_test]
    fn test_wasm_dataset() {
        let dataset = WasmDataset::new(vec![1, 2, 3, 4, 5]);
        assert_eq!(dataset.len(), 5);
    }
    
    #[wasm_bindgen_test]
    async fn test_async_loading() {
        let dataset = load_remote_dataset("test_data.json").await.unwrap();
        assert!(dataset.len() > 0);
    }
}
```

### 2. Integration Tests
- Test in real browser environments
- Verify performance characteristics
- Test memory usage patterns
- Validate cross-browser compatibility

### 3. Performance Benchmarks
```rust
#[cfg(target_arch = "wasm32")]
mod wasm_benchmarks {
    use web_sys::console;
    
    pub fn benchmark_data_loading() {
        let start = js_sys::Date::now();
        // Perform data loading operation
        let end = js_sys::Date::now();
        console::log_1(&format!("Loading took: {}ms", end - start).into());
    }
}
```

## Example Usage

### JavaScript Integration
```javascript
import init, { WasmDataLoader } from './pkg/torsh_data.js';

async function loadData() {
    await init();
    
    const loader = new WasmDataLoader(32); // batch size
    await loader.load_from_url('/api/dataset');
    
    while (true) {
        try {
            const batch = loader.next_batch();
            // Process batch in JavaScript
            await processBatch(batch);
        } catch (e) {
            break; // End of dataset
        }
    }
}
```

### Web Worker Usage
```javascript
// worker.js
import { WasmDataProcessor } from './pkg/torsh_data.js';

self.onmessage = async function(e) {
    const processor = new WasmDataProcessor();
    const result = await processor.process(e.data);
    self.postMessage(result);
};
```

## Migration Path

### 1. Gradual Feature Introduction
- Start with basic dataset operations
- Add transforms and sampling
- Implement advanced features incrementally

### 2. Backward Compatibility
- Maintain existing APIs for native platforms
- Use feature flags to enable WASM-specific code
- Provide fallbacks for unsupported operations

### 3. Documentation Updates
- Create WASM-specific examples
- Update API documentation with browser considerations
- Provide migration guide for existing users

## Timeline and Milestones

### Milestone 1: Basic WASM Support (4 weeks)
- [ ] Set up WASM build infrastructure
- [ ] Implement basic dataset operations
- [ ] Create JavaScript bindings
- [ ] Add unit tests

### Milestone 2: Browser Integration (6 weeks)
- [ ] Implement web API integrations
- [ ] Add async data loading
- [ ] Create streaming support
- [ ] Browser compatibility testing

### Milestone 3: Performance Optimization (4 weeks)
- [ ] Memory management optimization
- [ ] Parallel processing with web workers
- [ ] Caching mechanisms
- [ ] Performance benchmarking

### Milestone 4: Production Ready (2 weeks)
- [ ] Security audit
- [ ] Documentation completion
- [ ] Example applications
- [ ] Release preparation

## Success Metrics

1. **Performance**: WASM performance within 80% of native Rust performance
2. **Compatibility**: Support for 95% of core torsh-data features
3. **Browser Support**: Works in all major browsers (Chrome, Firefox, Safari, Edge)
4. **Bundle Size**: WASM binary under 2MB compressed
5. **Memory Usage**: Efficient memory usage patterns in browser environment

## Conclusion

WebAssembly support for torsh-data would enable powerful client-side data processing capabilities, opening up new possibilities for machine learning applications in browsers. The proposed implementation strategy provides a clear path from basic compatibility to production-ready browser integration while maintaining the performance and safety characteristics that make Rust an excellent choice for data processing workloads.