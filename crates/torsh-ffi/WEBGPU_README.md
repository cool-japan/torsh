# ToRSh WebGPU Acceleration

Hardware-accelerated deep learning in web browsers using WebGPU. This module enables GPU-accelerated tensor operations and neural network training directly in the browser with near-native performance.

## 🚀 Features

- **GPU Acceleration**: Leverage WebGPU for hardware-accelerated tensor operations
- **WGSL Compute Shaders**: Optimized compute shaders for common operations
- **Automatic Fallback**: Falls back to CPU when WebGPU unavailable
- **Memory Pooling**: Efficient GPU buffer management
- **Browser Compatibility**: Works on Chrome 113+, Edge 113+, Firefox 113+ (experimental)

## 📦 Installation

### NPM

```bash
npm install torsh-wasm
```

### Yarn

```bash
yarn add torsh-wasm
```

### CDN

```html
<script type="module">
  import init, * as torsh from 'https://cdn.jsdelivr.net/npm/torsh-wasm/torsh_wasm.js';
  await init();
</script>
```

## 🎯 Quick Start

### Basic Usage

```javascript
import init, * as torsh from 'torsh-wasm';

// Initialize ToRSh with WebGPU support
await init();
const gpu = await torsh.webgpu.init();

if (gpu.isSupported()) {
  console.log('WebGPU is available!');

  // Create GPU tensors
  const x = torsh.Tensor.randn([1024, 1024], { device: 'gpu' });
  const y = torsh.Tensor.randn([1024, 1024], { device: 'gpu' });

  // GPU-accelerated operations
  const result = x.matmul(y);  // Runs on GPU

  console.log('Result shape:', result.shape());
}
```

### TypeScript Support

Full TypeScript definitions are available:

```typescript
import init, { Tensor, Linear, Adam, webgpu } from 'torsh-wasm';

await init();
const device = await webgpu.init();

const x: Tensor = Tensor.randn([100, 10], { device: 'gpu' });
const layer: Linear = new Linear(10, 5, true, { device: 'gpu' });
const output: Tensor = layer.forward(x);
```

## 🧠 Neural Network Example

```javascript
import init, * as torsh from 'torsh-wasm';

await init();
await torsh.webgpu.init();

// Define model
const model = new torsh.Sequential([
  new torsh.Linear(784, 128, true, { device: 'gpu' }),
  new torsh.ReLU(),
  new torsh.Linear(128, 64, true, { device: 'gpu' }),
  new torsh.ReLU(),
  new torsh.Linear(64, 10, true, { device: 'gpu' })
]);

// Training setup
const optimizer = new torsh.Adam(model.parameters(), {
  lr: 0.001,
  beta1: 0.9,
  beta2: 0.999
});

// Training loop
for (let epoch = 0; epoch < 10; epoch++) {
  // Forward pass (GPU-accelerated)
  const output = model.forward(input);
  const loss = torsh.loss.crossEntropy(output, target);

  // Backward pass (GPU-accelerated)
  loss.backward();

  // Update parameters (GPU-accelerated)
  optimizer.step();
  optimizer.zeroGrad();

  console.log(`Epoch ${epoch}, Loss: ${loss.data()[0]}`);
}
```

## ⚡ Performance Comparison

Typical speedup over CPU (matrix multiplication, 1024×1024):

| Hardware | Speedup |
|----------|---------|
| NVIDIA RTX 3080 | 30-50x |
| NVIDIA GTX 1660 | 15-25x |
| Intel Integrated GPU | 3-8x |
| AMD Radeon RX 6800 | 25-40x |

### Benchmark Example

```javascript
const size = 1024;
const x = torsh.Tensor.randn([size, size], { device: 'gpu' });
const y = torsh.Tensor.randn([size, size], { device: 'gpu' });

// GPU benchmark
torsh.profiler.start();
const resultGpu = x.matmul(y);
const gpuTime = torsh.profiler.stop();

// CPU benchmark
const xCpu = x.to('cpu');
const yCpu = y.to('cpu');
torsh.profiler.start();
const resultCpu = xCpu.matmul(yCpu);
const cpuTime = torsh.profiler.stop();

console.log(`GPU: ${gpuTime.totalTime}ms`);
console.log(`CPU: ${cpuTime.totalTime}ms`);
console.log(`Speedup: ${(cpuTime.totalTime / gpuTime.totalTime).toFixed(1)}x`);
```

## 🎨 Supported Operations

### Basic Operations (GPU-Accelerated)

- **Element-wise**: `add`, `sub`, `mul`, `div`, `pow`
- **Reductions**: `sum`, `mean`, `max`, `min`
- **Linear Algebra**: `matmul`, `transpose`

### Neural Network Operations (GPU-Accelerated)

- **Activations**: `relu`, `sigmoid`, `tanh`, `softmax`
- **Loss Functions**: `mse`, `binaryCrossEntropy`, `crossEntropy`
- **Layers**: `Linear`, `Sequential`
- **Optimizers**: `Adam`, `SGD`

## 🖥️ Browser Compatibility

| Browser | Version | Status | Notes |
|---------|---------|--------|-------|
| Chrome | 113+ | ✅ Stable | Full WebGPU support |
| Edge | 113+ | ✅ Stable | Full WebGPU support |
| Firefox | 113+ | ⚠️ Experimental | Enable `dom.webgpu.enabled` in `about:config` |
| Safari | 17+ | 🚧 Preview | WebGPU in Technology Preview |
| Opera | 99+ | ✅ Stable | Based on Chromium |

### Enabling WebGPU in Firefox

1. Type `about:config` in the address bar
2. Search for `dom.webgpu.enabled`
3. Set to `true`
4. Restart browser

## 💾 Memory Management

WebGPU buffers are automatically managed, but you can optimize memory usage:

```javascript
// Create tensor
const tensor = torsh.Tensor.randn([1000, 1000], { device: 'gpu' });

// Use tensor...

// Explicitly free GPU memory
tensor.free();

// Check memory usage
const usage = torsh.webgpu.getMemoryUsage();
console.log(`GPU Memory: ${usage.allocated / 1024 / 1024} MB`);
console.log(`Peak Usage: ${usage.peak / 1024 / 1024} MB`);
console.log(`Cached Buffers: ${usage.cachedBuffers}`);
```

## 🔧 Advanced Configuration

### Device Selection

```javascript
// Check if WebGPU is available
if (torsh.webgpu.isSupported()) {
  // Get device info
  const device = await torsh.webgpu.init();
  const info = device.info();

  console.log('GPU Vendor:', info.vendor);
  console.log('Architecture:', info.architecture);
  console.log('Max Buffer Size:', info.maxBufferSize / 1024 / 1024, 'MB');
  console.log('F16 Support:', info.supportsF16);
}
```

### Profiling

```javascript
// Profile operations
const result = torsh.profiler.profile('my_operation', () => {
  const x = torsh.Tensor.randn([512, 512], { device: 'gpu' });
  const y = torsh.Tensor.randn([512, 512], { device: 'gpu' });
  return x.matmul(y);
});

const stats = torsh.profiler.stop();
console.log('GPU Time:', stats.gpuTime, 'ms');
console.log('CPU Time:', stats.cpuTime, 'ms');
console.log('Operations/sec:', stats.opsPerSec);
```

## 🚀 Edge Deployment

### Cloudflare Workers

```javascript
// worker.js
import init, { Tensor, Linear } from './torsh_wasm.js';
import wasmModule from './torsh_wasm_bg.wasm';

export default {
  async fetch(request, env, ctx) {
    // Initialize WASM
    await init(wasmModule);

    // Run inference
    const input = Tensor.randn([1, 10], { device: 'cpu' });
    const model = new Linear(10, 5);
    const output = model.forward(input);

    return new Response(JSON.stringify({
      output: output.data()
    }));
  }
};
```

### Vercel Edge Functions

```javascript
// api/inference.js
import init, { Sequential, Linear, ReLU } from 'torsh-wasm';

export const config = {
  runtime: 'edge',
};

export default async function handler(request) {
  await init();

  const model = new Sequential([
    new Linear(784, 128),
    new ReLU(),
    new Linear(128, 10)
  ]);

  // ... inference logic ...

  return new Response(JSON.stringify({ result }));
}
```

## 📊 Performance Tips

1. **Batch Operations**: Combine multiple operations to reduce overhead
2. **Memory Pooling**: Reuse buffers when possible (automatic)
3. **Device Management**: Keep tensors on GPU to avoid transfers
4. **Shader Cache**: Shaders are automatically cached (use `clearShaderCache()` if needed)

```javascript
// Good: Keep tensors on GPU
const x = Tensor.randn([1000, 1000], { device: 'gpu' });
const y = Tensor.randn([1000, 1000], { device: 'gpu' });
const z = x.matmul(y).relu();  // All on GPU

// Bad: Unnecessary transfers
const x = Tensor.randn([1000, 1000], { device: 'cpu' });
const y = x.to('gpu');  // Transfer to GPU
const z = y.matmul(x.to('gpu')).to('cpu');  // Multiple transfers
```

## 🐛 Troubleshooting

### WebGPU Not Available

```javascript
if (!torsh.webgpu.isSupported()) {
  console.error('WebGPU not available. Possible reasons:');
  console.error('1. Browser does not support WebGPU');
  console.error('2. GPU drivers need updating');
  console.error('3. WebGPU is disabled in browser settings');

  // Fallback to CPU
  const x = Tensor.randn([100, 100], { device: 'cpu' });
}
```

### Out of Memory

```javascript
try {
  const large_tensor = Tensor.zeros([10000, 10000], { device: 'gpu' });
} catch (error) {
  console.error('GPU out of memory:', error);

  // Clear cached buffers
  torsh.webgpu.clearShaderCache();

  // Or use CPU
  const tensor = Tensor.zeros([10000, 10000], { device: 'cpu' });
}
```

### Performance Issues

```javascript
// Check device info
const device = await torsh.webgpu.init();
const info = device.info();

if (info.maxBufferSize < 256 * 1024 * 1024) {
  console.warn('GPU has limited memory, consider smaller batches');
}

// Monitor memory usage
setInterval(() => {
  const usage = torsh.webgpu.getMemoryUsage();
  console.log(`Memory: ${usage.allocated / 1024 / 1024} MB`);
}, 1000);
```

## 📚 API Documentation

Full API documentation is available in the `webgpu.d.ts` TypeScript definitions file.

## 🤝 Contributing

Contributions are welcome! Please see the main ToRSh repository for contribution guidelines.

## 📄 License

Same license as the main ToRSh project.

## 🔗 Links

- [ToRSh Repository](https://github.com/cool-japan/torsh)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [WGSL Spec](https://gpuweb.github.io/gpuweb/wgsl/)
- [Examples](./examples/)

## 💬 Support

For issues and questions, please use the GitHub issue tracker.
