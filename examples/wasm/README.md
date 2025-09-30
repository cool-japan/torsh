# ToRSh WASM Deployment Example

This example demonstrates how to deploy ToRSh models to WebAssembly for running machine learning inference directly in the browser.

## Features

- 🎨 **Digit Classification**: Draw digits and classify them using a CNN
- 💭 **Sentiment Analysis**: Analyze text sentiment (positive/negative/neutral)  
- ⚡ **Performance Benchmarking**: Test inference speed in the browser
- 🔧 **Image Processing**: Canvas data preprocessing utilities
- 📊 **Model Information**: Display model architecture and parameters

## Prerequisites

- Rust (with `wasm32-unknown-unknown` target)
- wasm-pack
- A local web server (e.g., Python's http.server)

## Setup

1. Install the WASM target for Rust:
```bash
rustup target add wasm32-unknown-unknown
```

2. Install wasm-pack:
```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

## Building

From this directory (`examples/wasm/`), run:

```bash
# Build the WASM module
wasm-pack build --target web --out-dir pkg

# For optimized production build
wasm-pack build --target web --out-dir pkg --release
```

This will create a `pkg/` directory with the generated WASM module and JavaScript bindings.

## Running

1. Start a local web server in this directory:
```bash
# Using Python 3
python3 -m http.server 8000

# Or using Node.js http-server
npx http-server -p 8000
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
wasm/
├── Cargo.toml          # Rust dependencies and configuration
├── src/
│   └── lib.rs          # Rust/WASM implementation
├── index.html          # Web interface
├── main.js             # JavaScript glue code
├── pkg/                # Generated WASM output (after build)
└── README.md           # This file
```

## Implementation Details

### Models

1. **DigitClassifier**: A small CNN for MNIST-style digit classification
   - Input: 28x28 grayscale image
   - Architecture: Conv2d → ReLU → Conv2d → ReLU → FC → ReLU → FC
   - Output: 10-class probabilities

2. **SentimentAnalyzer**: Simple feedforward network for text sentiment
   - Input: Tokenized text (max 20 tokens)
   - Architecture: Embedding → FC → ReLU → FC → ReLU → FC
   - Output: 3-class (negative/neutral/positive)

### JavaScript API

The WASM module exports several classes and functions:

```javascript
// Create a digit classifier
const classifier = new DigitClassifier();

// Get model information
const info = classifier.model_info();

// Make a prediction
const prediction = classifier.predict(imageData);

// Process canvas data
const grayscale = ImageProcessor.process_canvas_data(rgbaData, width, height);

// Run performance benchmark
const results = Benchmark.inference_speed_test(iterations);
```

### Optimization Tips

1. **Binary Size**: Use `opt-level = "z"` and `lto = true` in Cargo.toml
2. **Loading Time**: Consider lazy-loading models or splitting into multiple WASM modules
3. **Memory Usage**: Reuse tensors when possible to reduce allocations
4. **Performance**: Use SIMD instructions when available (requires browser support)

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (14+)
- Mobile browsers: Touch events supported for drawing

## Troubleshooting

### CORS Errors
If you see CORS errors, make sure you're serving the files through a web server, not opening index.html directly.

### Module Not Found
Ensure you've run `wasm-pack build` and the `pkg/` directory exists.

### Performance Issues
- Check browser console for errors
- Ensure you're using a release build
- Try reducing model size or input dimensions

## Extending the Example

To add your own models:

1. Define the model structure in `src/lib.rs`
2. Add `#[wasm_bindgen]` attributes to expose it to JavaScript
3. Implement the forward pass and any preprocessing
4. Update the JavaScript interface in `main.js`
5. Add UI elements in `index.html`

## Performance Considerations

On a modern laptop (M1 MacBook Air), expect:
- Digit classification: ~5-10ms per inference
- Sentiment analysis: ~2-5ms per inference
- Loading time: ~100-200ms for WASM module

## Future Improvements

- [ ] Add WebGL/WebGPU backend for GPU acceleration
- [ ] Implement model quantization for smaller size
- [ ] Add more preprocessing utilities (augmentation, normalization)
- [ ] Support for loading external model weights
- [ ] Web Workers for non-blocking inference
- [ ] Progressive Web App (PWA) support

## License

This example is part of the ToRSh project and follows the same license.