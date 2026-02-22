/**
 * ToRSh WebAssembly TypeScript Type Definitions
 *
 * Comprehensive type definitions for ToRSh WASM bindings enabling
 * type-safe deep learning in TypeScript/JavaScript environments.
 *
 * @module torsh-wasm
 * @version 0.1.0-alpha.2
 */

/**
 * Tensor shape type - array of dimension sizes
 */
export type Shape = number[];

/**
 * Tensor data type - flat array of numbers
 */
export type TensorData = Float32Array | number[];

/**
 * WASM Tensor class for multi-dimensional arrays
 *
 * Provides PyTorch-like tensor operations with automatic differentiation support.
 *
 * @example
 * ```typescript
 * // Create tensors
 * const x = Tensor.randn([2, 3]);
 * const y = Tensor.ones([3, 4]);
 *
 * // Operations
 * const z = x.matmul(y);
 * const sum = z.sum();
 * ```
 */
export class Tensor {
    /**
     * Create a new tensor from data and shape
     * @param data - Flat array of tensor values
     * @param shape - Dimensions of the tensor
     */
    constructor(data: TensorData, shape: Shape);

    /**
     * Create a tensor filled with zeros
     * @param shape - Dimensions of the tensor
     */
    static zeros(shape: Shape): Tensor;

    /**
     * Create a tensor filled with ones
     * @param shape - Dimensions of the tensor
     */
    static ones(shape: Shape): Tensor;

    /**
     * Create a tensor with random values from normal distribution N(0,1)
     * @param shape - Dimensions of the tensor
     */
    static randn(shape: Shape): Tensor;

    /**
     * Create a tensor with random uniform values [0, 1)
     * @param shape - Dimensions of the tensor
     */
    static rand(shape: Shape): Tensor;

    /**
     * Get tensor shape
     */
    shape(): Shape;

    /**
     * Get tensor data as Float32Array
     */
    data(): Float32Array;

    /**
     * Get number of elements in tensor
     */
    numel(): number;

    /**
     * Enable gradient tracking for this tensor
     * @param requiresGrad - Whether to track gradients
     */
    requires_grad_(requiresGrad: boolean): Tensor;

    /**
     * Check if gradient tracking is enabled
     */
    requires_grad(): boolean;

    /**
     * Reshape tensor to new dimensions
     * @param newShape - New tensor dimensions
     */
    reshape(newShape: Shape): Tensor;

    /**
     * Transpose 2D tensor
     */
    transpose(): Tensor;

    /**
     * Matrix multiplication
     * @param other - Tensor to multiply with
     */
    matmul(other: Tensor): Tensor;

    /**
     * Element-wise addition
     * @param other - Tensor to add
     */
    add(other: Tensor): Tensor;

    /**
     * Element-wise subtraction
     * @param other - Tensor to subtract
     */
    sub(other: Tensor): Tensor;

    /**
     * Element-wise multiplication
     * @param other - Tensor to multiply
     */
    mul(other: Tensor): Tensor;

    /**
     * Element-wise division
     * @param other - Tensor to divide by
     */
    div(other: Tensor): Tensor;

    /**
     * Add scalar value to all elements
     * @param scalar - Value to add
     */
    add_scalar(scalar: number): Tensor;

    /**
     * Multiply all elements by scalar
     * @param scalar - Value to multiply by
     */
    mul_scalar(scalar: number): Tensor;

    /**
     * ReLU activation function
     */
    relu(): Tensor;

    /**
     * Sigmoid activation function
     */
    sigmoid(): Tensor;

    /**
     * Tanh activation function
     */
    tanh(): Tensor;

    /**
     * Sum all elements
     */
    sum(): number;

    /**
     * Mean of all elements
     */
    mean(): number;

    /**
     * Maximum element value
     */
    max(): number;

    /**
     * Minimum element value
     */
    min(): number;

    /**
     * Convert tensor to JSON string
     */
    to_json(): string;

    /**
     * Create tensor from JSON string
     * @param json - JSON representation of tensor
     */
    static from_json(json: string): Tensor;

    /**
     * Free tensor memory (call when done to avoid memory leaks)
     */
    free(): void;
}

/**
 * Linear (fully-connected) layer
 *
 * Applies linear transformation: y = xW^T + b
 *
 * @example
 * ```typescript
 * const layer = new Linear(10, 5, true);
 * const input = Tensor.randn([1, 10]);
 * const output = layer.forward(input);
 * ```
 */
export class Linear {
    /**
     * Create a new linear layer
     * @param inFeatures - Size of input features
     * @param outFeatures - Size of output features
     * @param bias - Whether to include bias term
     */
    constructor(inFeatures: number, outFeatures: number, bias: boolean);

    /**
     * Forward pass through the layer
     * @param input - Input tensor
     */
    forward(input: Tensor): Tensor;

    /**
     * Get layer parameters (weights and bias)
     */
    parameters(): Tensor[];

    /**
     * Free layer memory
     */
    free(): void;
}

/**
 * Layer types for Sequential model
 */
export type Layer = Linear | 'relu' | 'sigmoid' | 'tanh';

/**
 * Sequential container for neural network layers
 *
 * @example
 * ```typescript
 * const model = new Sequential([
 *   new Linear(784, 128),
 *   'relu',
 *   new Linear(128, 10)
 * ]);
 *
 * const output = model.forward(input);
 * const params = model.parameters();
 * ```
 */
export class Sequential {
    /**
     * Create a new sequential model
     * @param layers - Array of layers
     */
    constructor(layers: Layer[]);

    /**
     * Forward pass through all layers
     * @param input - Input tensor
     */
    forward(input: Tensor): Tensor;

    /**
     * Get all trainable parameters
     */
    parameters(): Tensor[];

    /**
     * Free model memory
     */
    free(): void;
}

/**
 * Adam optimizer configuration
 */
export interface AdamConfig {
    /** Learning rate (default: 0.001) */
    learning_rate?: number;
    /** Beta1 parameter (default: 0.9) */
    beta1?: number;
    /** Beta2 parameter (default: 0.999) */
    beta2?: number;
    /** Epsilon for numerical stability (default: 1e-8) */
    epsilon?: number;
    /** Weight decay (L2 regularization) (default: 0.0) */
    weight_decay?: number;
}

/**
 * Adam optimizer
 *
 * Implements Adam algorithm with optional weight decay.
 *
 * @example
 * ```typescript
 * const model = new Sequential([...]);
 * const optimizer = new Adam(model.parameters().length, {
 *   learning_rate: 0.001,
 *   weight_decay: 0.01
 * });
 *
 * // Training loop
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   const output = model.forward(input);
 *   const loss = mse_loss(output, target);
 *
 *   // Compute gradients (simplified)
 *   const grads = compute_gradients(loss);
 *
 *   // Update parameters
 *   optimizer.step(model.parameters(), grads);
 * }
 * ```
 */
export class Adam {
    /**
     * Create a new Adam optimizer
     * @param numParams - Number of parameter tensors
     * @param config - Optimizer configuration
     */
    constructor(numParams: number, config?: AdamConfig);

    /**
     * Perform optimization step
     * @param parameters - Model parameters to update
     * @param gradients - Gradients for each parameter
     */
    step(parameters: Tensor[], gradients: Float32Array[]): void;

    /**
     * Reset optimizer state
     */
    zero_grad(): void;

    /**
     * Free optimizer memory
     */
    free(): void;
}

/**
 * Loss functions namespace
 */
export namespace loss {
    /**
     * Mean Squared Error loss
     * @param predictions - Predicted values
     * @param targets - Target values
     */
    export function mse_loss(predictions: Tensor, targets: Tensor): number;

    /**
     * Binary Cross Entropy loss
     * @param predictions - Predicted probabilities
     * @param targets - Target labels (0 or 1)
     */
    export function binary_cross_entropy(predictions: Tensor, targets: Tensor): number;

    /**
     * Cross Entropy loss for multi-class classification
     * @param predictions - Predicted probabilities
     * @param targets - Target one-hot encoded labels
     */
    export function cross_entropy(predictions: Tensor, targets: Tensor): number;
}

/**
 * Utility functions namespace
 */
export namespace utils {
    /**
     * Memory usage statistics
     */
    export interface MemoryStats {
        /** Number of tensors tracked */
        tensor_count: number;
        /** Total number of elements across all tensors */
        total_elements: number;
        /** Total bytes used by tensors */
        bytes_used: number;
    }

    /**
     * Convert Float32Array to tensor
     * @param data - Array data
     * @param shape - Tensor shape
     */
    export function float32array_to_tensor(data: Float32Array, shape: Shape): Tensor;

    /**
     * Serialize tensor to bytes
     * @param tensor - Tensor to serialize
     */
    export function tensor_to_bytes(tensor: Tensor): Uint8Array;

    /**
     * Deserialize tensor from bytes
     * @param bytes - Serialized tensor data
     */
    export function tensor_from_bytes(bytes: Uint8Array): Tensor;

    /**
     * Get memory usage statistics
     * @param tensors - Array of tensors to analyze
     */
    export function get_memory_stats(tensors: Tensor[]): MemoryStats;
}

/**
 * Initialize the WASM module
 *
 * Must be called before using any ToRSh functionality.
 *
 * @example
 * ```typescript
 * import init, * as torsh from 'torsh-wasm';
 *
 * async function main() {
 *   await init();
 *   const tensor = torsh.Tensor.randn([10, 10]);
 *   console.log('Created tensor with shape:', tensor.shape());
 * }
 *
 * main();
 * ```
 */
export default function init(input?: string | URL | Request): Promise<void>;
