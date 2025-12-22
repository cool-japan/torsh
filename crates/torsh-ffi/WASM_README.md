# ToRSh WASM - WebAssembly Deep Learning

Run deep learning models directly in browsers and on edge platforms with ToRSh's WebAssembly bindings.

## 🚀 Features

- **Browser Compatibility**: Run ML models in Chrome, Firefox, Safari, Edge
- **Edge Deployment**: Cloudflare Workers, Vercel Edge, AWS Lambda@Edge
- **PyTorch-like API**: Familiar tensor operations and neural network API
- **Memory Efficient**: Optimized for WASM heap constraints
- **Type-Safe**: Complete TypeScript type definitions
- **Zero Dependencies**: No runtime dependencies except the WASM module

## 📦 Installation

### npm / yarn
```bash
npm install torsh-wasm
# or
yarn add torsh-wasm
```

### CDN
```html
<script type="module">
  import init, * as torsh from 'https://cdn.jsdelivr.net/npm/torsh-wasm/torsh_wasm.js';
  await init();
  // Use torsh here
</script>
```

## 🎯 Quick Start

### Browser Example
```javascript
import init, { Tensor, Sequential, Linear, Adam } from 'torsh-wasm';

// Initialize WASM module
await init();

// Create tensors
const x = Tensor.randn([100, 10]);
const y = Tensor.ones([100, 1]);

// Build model
const model = new Sequential([
  new Linear(10, 64),
  'relu',
  new Linear(64, 1),
  'sigmoid'
]);

// Training
const optimizer = new Adam(model.parameters().length, { lr: 0.001 });

for (let epoch = 0; epoch < 100; epoch++) {
  const output = model.forward(x);
  const loss = torsh.loss.binary_cross_entropy(output, y);

  // Update weights (simplified)
  console.log(`Epoch ${epoch}, Loss: ${loss}`);
}
```

### Node.js Example
```javascript
const torsh = require('torsh-wasm');

async function main() {
  await torsh.default();

  const tensor = torsh.Tensor.randn([5, 5]);
  const result = tensor.matmul(tensor.transpose());

  console.log('Result shape:', result.shape());
  console.log('Result data:', result.data());
}

main();
```

### Deno Example
```typescript
import init, { Tensor } from "https://deno.land/x/torsh_wasm/mod.ts";

await init();

const x = Tensor.randn([10, 10]);
const y = x.add(Tensor.ones([10, 10]));
console.log(y.mean());
```

## 🌐 Edge Deployment

### Cloudflare Workers
```javascript
import { Tensor, Sequential } from './torsh_wasm.js';

export default {
  async fetch(request) {
    const model = new Sequential([/* ... */]);
    const input = Tensor.from_json(await request.json());
    const output = model.forward(input);

    return new Response(JSON.stringify({
      prediction: output.data()
    }));
  }
}
```

See `examples/wasm_cloudflare_worker.js` for complete example.

### Vercel Edge Functions
```typescript
import { Tensor } from 'torsh-wasm';
import type { NextRequest } from 'next/server';

export const config = { runtime: 'edge' };

export default async function handler(req: NextRequest) {
  const { features } = await req.json();
  const tensor = Tensor.from_array(features, [1, 10]);
  // Process tensor...

  return new Response(JSON.stringify({ result: tensor.data() }));
}
```

## 📚 API Reference

### Tensor Operations

```typescript
// Creation
Tensor.zeros(shape: number[]): Tensor
Tensor.ones(shape: number[]): Tensor
Tensor.randn(shape: number[]): Tensor  // Normal distribution
Tensor.rand(shape: number[]): Tensor   // Uniform [0, 1)

// Operations
tensor.add(other: Tensor): Tensor
tensor.sub(other: Tensor): Tensor
tensor.mul(other: Tensor): Tensor
tensor.div(other: Tensor): Tensor
tensor.matmul(other: Tensor): Tensor

// Scalar operations
tensor.add_scalar(value: number): Tensor
tensor.mul_scalar(value: number): Tensor

// Shape manipulation
tensor.reshape(newShape: number[]): Tensor
tensor.transpose(): Tensor  // 2D only

// Activations
tensor.relu(): Tensor
tensor.sigmoid(): Tensor
tensor.tanh(): Tensor

// Reductions
tensor.sum(): number
tensor.mean(): number
tensor.max(): number
tensor.min(): number

// Properties
tensor.shape(): number[]
tensor.data(): Float32Array
tensor.numel(): number

// Cleanup
tensor.free(): void  // Important: call when done!
```

### Neural Network Layers

```typescript
// Linear layer
const layer = new Linear(
  in_features: number,
  out_features: number,
  bias: boolean = true
);

// Sequential container
const model = new Sequential([
  new Linear(10, 64),
  'relu',
  new Linear(64, 32),
  'sigmoid',
  new Linear(32, 1)
]);

// Forward pass
const output = model.forward(input: Tensor): Tensor

// Get parameters
const params = model.parameters(): Tensor[]
```

### Optimizers

```typescript
// Adam optimizer
const optimizer = new Adam(
  numParams: number,
  config?: {
    learning_rate?: number,  // default: 0.001
    beta1?: number,          // default: 0.9
    beta2?: number,          // default: 0.999
    epsilon?: number,        // default: 1e-8
    weight_decay?: number    // default: 0.0
  }
);

// Optimization step
optimizer.step(parameters: Tensor[], gradients: Float32Array[]): void

// Reset gradients
optimizer.zero_grad(): void
```

### Loss Functions

```typescript
import { loss } from 'torsh-wasm';

// Mean Squared Error
loss.mse_loss(predictions: Tensor, targets: Tensor): number

// Binary Cross Entropy
loss.binary_cross_entropy(predictions: Tensor, targets: Tensor): number

// Cross Entropy (multi-class)
loss.cross_entropy(predictions: Tensor, targets: Tensor): number
```

## 💡 Usage Tips

### Memory Management
WASM has limited memory. Always free tensors when done:

```javascript
function processData(input) {
  const x = Tensor.from_array(input, [10, 10]);
  const y = x.matmul(x.transpose());

  const result = y.data();

  // Free memory
  x.free();
  y.free();

  return result;
}
```

### Performance
- Batch operations when possible
- Reuse tensors instead of creating new ones
- Use typed arrays for data transfer
- Profile with browser DevTools

### Error Handling
```javascript
try {
  const result = tensor1.matmul(tensor2);
} catch (error) {
  console.error('Operation failed:', error.message);
  // Handle error...
}
```

## 📊 Benchmarks

Performance on typical operations (2.4 GHz Intel Core i9):

| Operation | Size | Time |
|-----------|------|------|
| Add | 1000×1000 | 12.3 ms |
| MatMul | 1000×1000 | 234.5 ms |
| ReLU | 1000×1000 | 6.1 ms |
| Sigmoid | 1000×1000 | 42.3 ms |

## 🔧 Building from Source

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM module
cd crates/torsh-ffi
wasm-pack build --target web --out-dir pkg

# Use in browser
cp pkg/* your-web-app/static/
```

## 📖 Examples

See the `examples/` directory for complete examples:

- `wasm_browser_demo.html` - Interactive browser demo
- `wasm_cloudflare_worker.js` - Cloudflare Workers edge deployment
- `wasm_training_example.js` - Complete training loop
- `wasm_inference_server.js` - Node.js inference server

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.

## 🔗 Links

- [ToRSh Documentation](https://docs.rs/torsh)
- [GitHub Repository](https://github.com/cool-japan/torsh)
- [Examples](https://github.com/cool-japan/torsh/tree/main/crates/torsh-ffi/examples)
- [Issue Tracker](https://github.com/cool-japan/torsh/issues)
