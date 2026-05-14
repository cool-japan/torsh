/**
 * ToRSh Node.js Bindings
 *
 * This module provides Node.js/TypeScript bindings for the ToRSh deep learning framework.
 * It offers a high-level, tensor-based API similar to PyTorch for JavaScript/TypeScript developers.
 */

import bindings from 'bindings';

// Load the native module
const native = bindings('torsh_native');

/**
 * Data types supported by ToRSh tensors
 */
export type DType = 'float32' | 'float64' | 'int32' | 'int64' | 'uint8' | 'bool';

/**
 * Device types for tensor computation
 */
export type Device = 'cpu' | 'cuda' | 'metal' | 'webgpu';

/**
 * Tensor shape as array of dimensions
 */
export type Shape = number[];

/**
 * Nested array data structure for tensor creation
 */
export type TensorData = number | number[] | number[][] | number[][][] | number[][][][];

/**
 * Internal tensor handle (opaque pointer to native tensor)
 */
interface TensorHandle {
  readonly _brand: 'TensorHandle';
}

/**
 * ToRSh Tensor class
 *
 * Represents a multi-dimensional array with support for automatic differentiation,
 * GPU acceleration, and neural network operations.
 */
export class Tensor {
  private readonly handle: TensorHandle;

  private constructor(handle: TensorHandle) {
    this.handle = handle;
  }

  /** @internal Create a Tensor from a raw native handle */
  static fromHandle(handle: TensorHandle): Tensor {
    return new Tensor(handle);
  }

  /**
   * Create a tensor from nested array data
   * @param data - Nested array of numbers
   * @returns New tensor instance
   */
  static tensor(data: TensorData): Tensor {
    const handle = native.createTensor(data);
    return new Tensor(handle);
  }

  /**
   * Create a tensor filled with zeros
   * @param dims - Dimensions of the tensor
   * @returns New tensor filled with zeros
   */
  static zeros(...dims: number[]): Tensor {
    const handle = native.zeros(...dims);
    return new Tensor(handle);
  }

  /**
   * Create a tensor filled with ones
   * @param dims - Dimensions of the tensor
   * @returns New tensor filled with ones
   */
  static ones(...dims: number[]): Tensor {
    const handle = native.ones(...dims);
    return new Tensor(handle);
  }

  /**
   * Create a tensor with random normal distribution
   * @param dims - Dimensions of the tensor
   * @returns New tensor with random values
   */
  static randn(...dims: number[]): Tensor {
    const handle = native.randn(...dims);
    return new Tensor(handle);
  }

  /**
   * Create an identity matrix
   * @param n - Size of the square matrix
   * @returns Identity matrix tensor
   */
  static eye(n: number): Tensor {
    const handle = native.eye(n);
    return new Tensor(handle);
  }

  /**
   * Create a tensor with linearly spaced values
   * @param start - Starting value
   * @param end - Ending value
   * @param steps - Number of steps
   * @returns Tensor with linearly spaced values
   */
  static linspace(start: number, end: number, steps: number): Tensor {
    const handle = native.linspace(start, end, steps);
    return new Tensor(handle);
  }

  /**
   * Get the shape of the tensor
   * @returns Array of dimensions
   */
  shape(): Shape {
    return native.getShape(this.handle);
  }

  /**
   * Get the size of a specific dimension
   * @param dim - Dimension index
   * @returns Size of the dimension
   */
  size(dim?: number): number | Shape | undefined {
    const shape = this.shape();
    return dim !== undefined ? shape[dim] : shape;
  }

  /**
   * Get the total number of elements in the tensor
   * @returns Total number of elements
   */
  numel(): number {
    return this.shape().reduce((acc, dim) => acc * dim, 1);
  }

  /**
   * Get the number of dimensions
   * @returns Number of dimensions
   */
  ndim(): number {
    return this.shape().length;
  }

  /**
   * Get the tensor data as a flat array
   * @returns Flat array of tensor values
   */
  data(): number[] {
    return native.getData(this.handle);
  }

  /**
   * Element-wise addition
   * @param other - Tensor or scalar to add
   * @returns New tensor with the result
   */
  add(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const handle = native.addScalar(this.handle, other);
      return new Tensor(handle);
    } else {
      const handle = native.add(this.handle, other.handle);
      return new Tensor(handle);
    }
  }

  /**
   * Element-wise subtraction
   * @param other - Tensor or scalar to subtract
   * @returns New tensor with the result
   */
  sub(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const handle = native.subScalar(this.handle, other);
      return new Tensor(handle);
    } else {
      const handle = native.sub(this.handle, other.handle);
      return new Tensor(handle);
    }
  }

  /**
   * Element-wise multiplication
   * @param other - Tensor or scalar to multiply
   * @returns New tensor with the result
   */
  mul(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const handle = native.mulScalar(this.handle, other);
      return new Tensor(handle);
    } else {
      const handle = native.multiply(this.handle, other.handle);
      return new Tensor(handle);
    }
  }

  /**
   * Element-wise division
   * @param other - Tensor or scalar to divide by
   * @returns New tensor with the result
   */
  div(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const handle = native.divScalar(this.handle, other);
      return new Tensor(handle);
    } else {
      const handle = native.divide(this.handle, other.handle);
      return new Tensor(handle);
    }
  }

  /**
   * Matrix multiplication
   * @param other - Tensor to multiply with
   * @returns New tensor with the result
   */
  matmul(other: Tensor): Tensor {
    const handle = native.matmul(this.handle, other.handle);
    return new Tensor(handle);
  }

  /**
   * Transpose the tensor (swaps the last two dimensions for 2-D tensors)
   * @param _dim1 - First dimension to swap (ignored; 2-D only for now)
   * @param _dim2 - Second dimension to swap (ignored; 2-D only for now)
   * @returns New tensor with transposed dimensions
   */
  transpose(_dim1?: number, _dim2?: number): Tensor {
    const handle = native.transpose(this.handle);
    return new Tensor(handle);
  }

  /**
   * Reshape the tensor
   * @param dims - New dimensions
   * @returns New tensor with reshaped dimensions
   */
  reshape(...dims: number[]): Tensor {
    const handle = native.reshape(this.handle, dims);
    return new Tensor(handle);
  }

  /**
   * ReLU activation function
   * @returns New tensor with ReLU applied
   */
  relu(): Tensor {
    const handle = native.relu(this.handle);
    return new Tensor(handle);
  }

  /**
   * Sigmoid activation function
   * @returns New tensor with sigmoid applied
   */
  sigmoid(): Tensor {
    const handle = native.sigmoid(this.handle);
    return new Tensor(handle);
  }

  /**
   * Tanh activation function
   * @returns New tensor with tanh applied
   */
  tanh(): Tensor {
    const handle = native.tanh(this.handle);
    return new Tensor(handle);
  }

  /**
   * Softmax activation function
   * @param dim - Dimension to apply softmax along (default 0)
   * @returns New tensor with softmax applied
   */
  softmax(dim?: number): Tensor {
    const handle = native.softmax(this.handle, dim ?? 0);
    return new Tensor(handle);
  }

  /**
   * Sum reduction
   * @param dim - Dimension to sum along (if undefined, sum all)
   * @param _keepdim - Whether to keep the dimension (ignored for now)
   * @returns New tensor with sum
   */
  sum(dim?: number, _keepdim?: boolean): Tensor {
    const handle = dim !== undefined
      ? native.sum(this.handle, dim)
      : native.sum(this.handle);
    return new Tensor(handle);
  }

  /**
   * Mean reduction
   * @param dim - Dimension to average along (if undefined, mean of all)
   * @param _keepdim - Whether to keep the dimension (ignored for now)
   * @returns New tensor with mean
   */
  mean(dim?: number, _keepdim?: boolean): Tensor {
    const handle = dim !== undefined
      ? native.mean(this.handle, dim)
      : native.mean(this.handle);
    return new Tensor(handle);
  }

  /**
   * Convert tensor to string representation
   * @returns String representation of the tensor
   */
  toString(): string {
    const shape = this.shape();
    const shapeStr = shape.join('x');

    if (this.numel() <= 20) {
      const data = this.data();
      const dataStr = data.map(x => x.toFixed(4)).join(', ');
      return `Tensor(${shapeStr})[${dataStr}]`;
    } else {
      return `Tensor(${shapeStr})[${this.numel()} elements]`;
    }
  }

  /**
   * Clone the tensor
   * @returns New tensor with copied data
   */
  clone(): Tensor {
    const handle = native.clone(this.handle);
    return new Tensor(handle);
  }

  /**
   * Detach the tensor from the computation graph
   * @returns New tensor detached from autograd
   */
  detach(): Tensor {
    const handle = native.detach(this.handle);
    return new Tensor(handle);
  }
}

/**
 * Neural network utilities
 */
export namespace nn {
  /**
   * Linear (fully connected) layer
   * @param input - Input tensor
   * @param weight - Weight matrix
   * @param bias - Optional bias vector
   * @returns Output tensor
   */
  export function linear(input: Tensor, weight: Tensor, bias?: Tensor): Tensor {
    let output = input.matmul(weight);
    if (bias) {
      output = output.add(bias);
    }
    return output;
  }

  /**
   * 2D Convolution layer
   * @param input - Input tensor (N, C, H, W)
   * @param weight - Convolution kernels (Out, In, kH, kW)
   * @param bias - Optional bias
   * @param stride - Stride for convolution
   * @param padding - Padding for convolution
   * @returns Output tensor
   */
  export function conv2d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride: number = 1,
    padding: number = 0
  ): Tensor {
    const biasHandle = bias
      ? (bias as unknown as { handle: TensorHandle }).handle
      : null;
    const handle = native.conv2d(
      (input as unknown as { handle: TensorHandle }).handle,
      (weight as unknown as { handle: TensorHandle }).handle,
      biasHandle,
      stride,
      padding
    );
    return Tensor.fromHandle(handle);
  }

  /**
   * Mean Squared Error loss
   * @param prediction - Predicted values
   * @param target - Target values
   * @returns MSE loss
   */
  export function mseLoss(prediction: Tensor, target: Tensor): Tensor {
    const diff = prediction.sub(target);
    return diff.mul(diff).mean();
  }

  /**
   * Cross entropy loss
   * @param prediction - Predicted probabilities
   * @param target - Target class indices or one-hot encoded
   * @returns Cross entropy loss
   */
  export function crossEntropyLoss(prediction: Tensor, target: Tensor): Tensor {
    const handle = native.crossEntropyLoss(
      (prediction as unknown as { handle: TensorHandle }).handle,
      (target as unknown as { handle: TensorHandle }).handle
    );
    return Tensor.fromHandle(handle);
  }
}

/**
 * Optimization utilities
 */
export namespace optim {
  /**
   * SGD optimizer step
   * @param params - Parameters to update
   * @param grads - Gradients for parameters
   * @param lr - Learning rate
   */
  export function sgdStep(params: Tensor[], grads: Tensor[], lr: number): void {
    const paramHandles = params.map(
      p => (p as unknown as { handle: TensorHandle }).handle
    );
    const gradHandles = grads.map(
      g => (g as unknown as { handle: TensorHandle }).handle
    );
    native.sgdStep(paramHandles, gradHandles, lr);
  }

  /**
   * Adam optimizer step
   * @param params - Parameters to update
   * @param grads - Gradients for parameters
   * @param m - First moment estimates
   * @param v - Second moment estimates
   * @param lr - Learning rate
   * @param beta1 - First moment decay rate
   * @param beta2 - Second moment decay rate
   * @param eps - Small constant for numerical stability
   * @param step - Current step number
   */
  export function adamStep(
    params: Tensor[],
    grads: Tensor[],
    m: Tensor[],
    v: Tensor[],
    lr: number = 0.001,
    beta1: number = 0.9,
    beta2: number = 0.999,
    eps: number = 1e-8,
    step: number = 1
  ): void {
    const paramHandles = params.map(
      p => (p as unknown as { handle: TensorHandle }).handle
    );
    const gradHandles = grads.map(
      g => (g as unknown as { handle: TensorHandle }).handle
    );
    const mHandles = m.map(
      mi => (mi as unknown as { handle: TensorHandle }).handle
    );
    const vHandles = v.map(
      vi => (vi as unknown as { handle: TensorHandle }).handle
    );
    native.adamStep(paramHandles, gradHandles, mHandles, vHandles, lr, beta1, beta2, eps, step);
  }
}

/**
 * Utility functions
 */
export namespace utils {
  /**
   * Set random seed for reproducibility
   * @param seed - Random seed value
   */
  export function manualSeed(seed: number): void {
    native.manualSeed(seed);
  }

  /**
   * Check if CUDA is available
   * @returns True if CUDA is available
   */
  export function cudaAvailable(): boolean {
    return native.cudaAvailable();
  }

  /**
   * Get number of CUDA devices
   * @returns Number of CUDA devices
   */
  export function cudaDeviceCount(): number {
    return native.cudaDeviceCount();
  }

  /**
   * Save tensor to file
   * @param tensor - Tensor to save
   * @param filename - File path
   */
  export function saveTensor(tensor: Tensor, filename: string): void {
    // Access private handle via type assertion; the field is opaque to JS callers.
    native.saveTensor((tensor as unknown as { handle: TensorHandle }).handle, filename);
  }

  /**
   * Load tensor from file
   * @param filename - File path
   * @returns Loaded tensor
   */
  export function loadTensor(filename: string): Tensor {
    const handle = native.loadTensor(filename) as TensorHandle;
    return Tensor.fromHandle(handle);
  }
}

// Export the main Tensor class and namespaces
export { Tensor as default };

// Re-export everything for convenience
export * from './index';
