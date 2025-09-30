# ToRSh Node.js Bindings

[![npm version](https://badge.fury.io/js/%40torsh%2Fcore.svg)](https://www.npmjs.com/package/@torsh/core)
[![Node.js CI](https://github.com/torsh-team/torsh/workflows/Node.js%20CI/badge.svg)](https://github.com/torsh-team/torsh/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package provides Node.js/TypeScript bindings for the ToRSh deep learning framework. It allows JavaScript and TypeScript developers to leverage ToRSh's high-performance tensor operations and neural network capabilities in Node.js applications.

## Features

- **High Performance**: Direct integration with ToRSh's Rust backend via N-API
- **TypeScript Support**: Full TypeScript definitions with type safety
- **Familiar API**: PyTorch-inspired API design for ease of use
- **Memory Efficient**: Automatic memory management with proper cleanup
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Zero Dependencies**: Minimal runtime dependencies

## Installation

### Prerequisites

- Node.js 14.0.0 or later
- npm or yarn
- Rust toolchain (for building from source)

### Install from npm

```bash
npm install @torsh/core
```

### Build from Source

```bash
git clone https://github.com/torsh-team/torsh.git
cd torsh/crates/torsh-ffi/nodejs
npm install
npm run build
```

## Quick Start

### JavaScript

```javascript
const { Tensor } = require('@torsh/core');

// Create tensors
const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
const b = Tensor.ones(2, 3);

// Perform operations
const c = a.add(b);
const d = a.matmul(b.transpose());

console.log('Result:', c.toString());
console.log('Shape:', c.shape());
```

### TypeScript

```typescript
import { Tensor, nn } from '@torsh/core';

// Create tensors with type safety
const input: Tensor = Tensor.tensor([[1, 2], [3, 4]]);
const weight: Tensor = Tensor.randn(2, 3);
const bias: Tensor = Tensor.zeros(3);

// Neural network operations
const output: Tensor = nn.linear(input, weight, bias);
const activated: Tensor = output.relu();

console.log('Output shape:', activated.shape());
```

## API Reference

### Tensor Creation

```typescript
// From arrays
const t1 = Tensor.tensor([[1, 2], [3, 4]]);
const t2 = Tensor.tensor([1, 2, 3, 4, 5]);

// Special tensors
const zeros = Tensor.zeros(3, 3);
const ones = Tensor.ones(2, 4);
const random = Tensor.randn(10, 10);
const identity = Tensor.eye(5);

// Sequences
const linspace = Tensor.linspace(0, 10, 11);  // [0, 1, 2, ..., 10]
const range = Tensor.arange(0, 10, 2);        // [0, 2, 4, 6, 8]
```

### Basic Operations

```typescript
const a = Tensor.tensor([[1, 2], [3, 4]]);
const b = Tensor.tensor([[5, 6], [7, 8]]);

// Element-wise operations
const add = a.add(b);        // Addition
const sub = a.sub(b);        // Subtraction
const mul = a.mul(b);        // Element-wise multiplication
const div = a.div(b);        // Element-wise division

// Matrix operations
const matmul = a.matmul(b);  // Matrix multiplication
const transpose = a.transpose(); // Transpose

// Scalar operations
const scalarAdd = a.add(10);
const scalarMul = a.mul(2.5);
```

### Shape Operations

```typescript
const x = Tensor.randn(2, 3, 4);

// Shape information
console.log(x.shape());      // [2, 3, 4]
console.log(x.size(1));      // 3
console.log(x.numel());      // 24
console.log(x.ndim());       // 3

// Reshape
const reshaped = x.reshape(6, 4);
const flattened = x.reshape(24);

// Transpose
const transposed = x.transpose(0, 2); // Swap dims 0 and 2

// Reductions
const sum = x.sum();              // Sum all elements
const sumDim = x.sum(1);          // Sum along dimension 1
const mean = x.mean();            // Mean of all elements
const meanDim = x.mean(2, true);  // Mean along dim 2, keep dims
```

### Neural Network Operations

```typescript
// Activation functions
const x = Tensor.tensor([[-1, 0, 1], [2, -2, 3]]);

const relu = x.relu();           // ReLU activation
const sigmoid = x.sigmoid();     // Sigmoid activation
const tanh = x.tanh();          // Tanh activation
const softmax = x.softmax(-1);   // Softmax along last dimension

// Loss functions
const prediction = Tensor.randn(10, 3).softmax(-1);
const target = Tensor.zeros(10, 3);

const mse = nn.mseLoss(prediction, target);
const ce = nn.crossEntropyLoss(prediction, target);
```

### Building Neural Networks

```typescript
import { Tensor, nn } from '@torsh/core';

class SimpleNN {
  private W1: Tensor;
  private b1: Tensor;
  private W2: Tensor;
  private b2: Tensor;

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    this.W1 = Tensor.randn(inputSize, hiddenSize);
    this.b1 = Tensor.zeros(hiddenSize);
    this.W2 = Tensor.randn(hiddenSize, outputSize);
    this.b2 = Tensor.zeros(outputSize);
  }

  forward(x: Tensor): Tensor {
    const h = nn.linear(x, this.W1, this.b1).relu();
    return nn.linear(h, this.W2, this.b2);
  }

  parameters(): Tensor[] {
    return [this.W1, this.b1, this.W2, this.b2];
  }
}

// Usage
const model = new SimpleNN(784, 128, 10);
const input = Tensor.randn(32, 784);  // Batch of 32
const output = model.forward(input);
```

### Data Processing

```typescript
// Batch processing
function createBatches(data: Tensor, batchSize: number): Tensor[] {
  const batches: Tensor[] = [];
  const numSamples = data.size(0) as number;
  
  for (let i = 0; i < numSamples; i += batchSize) {
    const endIdx = Math.min(i + batchSize, numSamples);
    const batch = data.narrow(0, i, endIdx - i);
    batches.push(batch);
  }
  
  return batches;
}

// Normalization
function normalize(tensor: Tensor): Tensor {
  const mean = tensor.mean();
  const std = tensor.std();
  return tensor.sub(mean).div(std);
}

// Data augmentation (example)
function randomFlip(tensor: Tensor): Tensor {
  if (Math.random() > 0.5) {
    return tensor.flip(-1); // Flip horizontally
  }
  return tensor;
}
```

### Training Example

```typescript
import { Tensor, nn, optim, utils } from '@torsh/core';

// Set random seed for reproducibility
utils.manualSeed(42);

// Create model and data
const model = new SimpleNN(2, 64, 3);
const X = Tensor.randn(100, 2);
const y = Tensor.randint(0, 3, [100]);

// Training loop
const learningRate = 0.01;
const epochs = 100;

for (let epoch = 0; epoch < epochs; epoch++) {
  // Forward pass
  const predictions = model.forward(X);
  const loss = nn.crossEntropyLoss(predictions, y);
  
  // Backward pass (simplified - would use autograd in practice)
  const grads = computeGradients(loss, model.parameters());
  
  // Update parameters
  optim.sgdStep(model.parameters(), grads, learningRate);
  
  if (epoch % 10 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${loss.data()[0]}`);
  }
}
```

### Advanced Features

#### GPU Operations (when available)

```typescript
import { utils } from '@torsh/core';

if (utils.cudaAvailable()) {
  console.log('CUDA devices:', utils.cudaDeviceCount());
  
  const a = Tensor.randn(1000, 1000).cuda();
  const b = Tensor.randn(1000, 1000).cuda();
  const c = a.matmul(b); // Computed on GPU
  
  const result = c.cpu(); // Move back to CPU
}
```

#### Tensor Serialization

```typescript
import { utils } from '@torsh/core';

// Save tensor
const tensor = Tensor.randn(100, 100);
utils.saveTensor(tensor, 'model_weights.pt');

// Load tensor
const loadedTensor = utils.loadTensor('model_weights.pt');
console.log('Loaded shape:', loadedTensor.shape());
```

#### Performance Optimization

```typescript
// Use tensor operations efficiently
function efficientMatmul(a: Tensor, b: Tensor): Tensor {
  // Avoid multiple small operations
  return a.matmul(b);
}

// Batch operations when possible
function batchedOperations(tensors: Tensor[]): Tensor {
  // Stack tensors for batch processing
  const stacked = Tensor.stack(tensors, 0);
  return stacked.sum(1); // Reduce along batch dimension
}

// Reuse tensors to avoid allocations
let workspace = Tensor.zeros(1000, 1000);
function reuseWorkspace(input: Tensor): Tensor {
  workspace = workspace.copy_(input); // In-place copy
  return workspace.relu_(); // In-place ReLU
}
```

## Integration Examples

### Express.js API Server

```typescript
import express from 'express';
import { Tensor } from '@torsh/core';

const app = express();
app.use(express.json());

// Load pre-trained model (pseudocode)
const model = loadModel('model.pt');

app.post('/predict', (req, res) => {
  try {
    const inputData = req.body.data;
    const inputTensor = Tensor.tensor(inputData);
    const prediction = model.forward(inputTensor);
    
    res.json({
      prediction: prediction.data(),
      shape: prediction.shape()
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => {
  console.log('ML API server running on port 3000');
});
```

### Real-time Data Processing

```typescript
import { EventEmitter } from 'events';
import { Tensor, nn } from '@torsh/core';

class RealTimeProcessor extends EventEmitter {
  private buffer: number[][] = [];
  private batchSize = 32;
  private model: SimpleNN;

  constructor() {
    super();
    this.model = new SimpleNN(10, 64, 3);
  }

  addData(data: number[]): void {
    this.buffer.push(data);
    
    if (this.buffer.length >= this.batchSize) {
      this.processBatch();
    }
  }

  private processBatch(): void {
    const batch = Tensor.tensor(this.buffer.splice(0, this.batchSize));
    const predictions = this.model.forward(batch);
    
    this.emit('predictions', predictions.data());
  }
}

// Usage
const processor = new RealTimeProcessor();
processor.on('predictions', (results) => {
  console.log('Batch predictions:', results);
});

// Simulate real-time data
setInterval(() => {
  const data = Array.from({ length: 10 }, () => Math.random());
  processor.addData(data);
}, 100);
```

### Electron Desktop App

```typescript
// In main process
import { app, BrowserWindow, ipcMain } from 'electron';
import { Tensor } from '@torsh/core';

ipcMain.handle('ml-predict', async (event, inputData) => {
  const input = Tensor.tensor(inputData);
  const result = performMLInference(input);
  return result.data();
});

// In renderer process
import { ipcRenderer } from 'electron';

async function predict(data: number[][]): Promise<number[]> {
  return await ipcRenderer.invoke('ml-predict', data);
}
```

## Performance Tips

1. **Use Batch Operations**: Process multiple samples together
2. **Minimize Data Conversion**: Keep operations in tensor space
3. **Reuse Tensors**: Avoid frequent allocations
4. **GPU When Available**: Move large computations to GPU
5. **Profile Your Code**: Use Node.js profiling tools

## Error Handling

```typescript
import { Tensor } from '@torsh/core';

try {
  const a = Tensor.tensor([[1, 2], [3, 4]]);
  const b = Tensor.tensor([[1, 2, 3]]); // Wrong shape
  const c = a.matmul(b); // This will throw
} catch (error) {
  if (error.message.includes('shape mismatch')) {
    console.log('Tensor shapes are incompatible');
  } else {
    console.log('Other tensor error:', error.message);
  }
}
```

## Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run specific test file
npm test -- basic_operations.test.js
```

## Debugging

```typescript
// Enable debug mode
process.env.TORSH_DEBUG = '1';

// Tensor information
const tensor = Tensor.randn(3, 4);
console.log('Shape:', tensor.shape());
console.log('Data type:', tensor.dtype());
console.log('Device:', tensor.device());
console.log('Requires grad:', tensor.requiresGrad());
```

## Building from Source

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/torsh-team/torsh.git
cd torsh/crates/torsh-ffi/nodejs

# Install Node.js dependencies
npm install

# Build native module
npm run build:native

# Build TypeScript
npm run build:js

# Run tests
npm test
```

## Troubleshooting

### Common Issues

1. **Module Not Found**:
   ```
   Error: Cannot find module '@torsh/core'
   ```
   - Ensure the package is properly installed: `npm install @torsh/core`
   - Check Node.js version compatibility

2. **Native Module Loading**:
   ```
   Error: The specified module could not be found
   ```
   - Rebuild the native module: `npm run build:native`
   - Check for missing system dependencies

3. **Shape Mismatch Errors**:
   ```
   Error: tensor shape mismatch
   ```
   - Verify tensor dimensions before operations
   - Use `tensor.shape()` to debug tensor sizes

4. **Memory Issues**:
   ```
   Error: out of memory
   ```
   - Reduce batch sizes
   - Enable garbage collection: `global.gc()`

## Contributing

Contributions are welcome! Please see the main ToRSh repository for contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- [Documentation](https://torsh.dev/docs)
- [Issues](https://github.com/torsh-team/torsh/issues)
- [Discussions](https://github.com/torsh-team/torsh/discussions)
- [Discord](https://discord.gg/torsh)

## Related Projects

- [ToRSh Core](https://github.com/torsh-team/torsh) - Main ToRSh framework
- [ToRSh Python](https://github.com/torsh-team/torsh-python) - Python bindings
- [ToRSh Models](https://github.com/torsh-team/torsh-models) - Pre-trained models