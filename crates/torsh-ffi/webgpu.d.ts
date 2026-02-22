/**
 * ToRSh WebGPU Type Definitions
 *
 * TypeScript type definitions for ToRSh WebGPU-accelerated WASM module.
 * Provides type safety and IDE autocompletion for GPU-accelerated deep learning.
 *
 * @module torsh-webgpu
 * @version 0.1.0-alpha.2
 */

/**
 * WebGPU device information and capabilities
 */
export interface WebGpuInfo {
    /** GPU vendor (e.g., "NVIDIA", "AMD", "Intel") */
    vendor: string;

    /** GPU architecture (e.g., "Ampere", "RDNA2") */
    architecture: string;

    /** Maximum buffer size in bytes */
    maxBufferSize: number;

    /** Maximum compute workgroups per dimension */
    maxComputeWorkgroups: number;

    /** Maximum workgroup size */
    maxWorkgroupSize: number;

    /** WebGPU API version */
    apiVersion: string;

    /** Whether f16 (half precision) is supported */
    supportsF16: boolean;

    /** Whether shader-f16 extension is available */
    supportsShaderF16: boolean;
}

/**
 * GPU memory usage statistics
 */
export interface GpuMemoryUsage {
    /** Total GPU memory allocated in bytes */
    allocated: number;

    /** Peak GPU memory usage in bytes */
    peak: number;

    /** Number of active buffers */
    bufferCount: number;

    /** Number of cached buffers in pool */
    cachedBuffers: number;
}

/**
 * WebGPU device handle
 */
export class WebGpuDevice {
    /**
     * Create a new WebGPU device
     */
    constructor();

    /**
     * Initialize WebGPU device with detected capabilities
     * @param info GPU capabilities detected from browser
     */
    initialize(info: WebGpuInfo): Promise<void>;

    /**
     * Check if WebGPU is supported in the current environment
     * @returns true if WebGPU is available, false otherwise
     */
    isSupported(): boolean;

    /**
     * Get device information and capabilities
     * @returns Device information
     */
    info(): WebGpuInfo;

    /**
     * Get current GPU memory usage
     * @returns Memory usage statistics
     */
    memoryUsage(): GpuMemoryUsage;

    /**
     * Clear all cached shaders
     */
    clearShaderCache(): void;

    /**
     * Get number of cached shaders
     * @returns Number of cached shaders
     */
    shaderCacheSize(): number;
}

/**
 * Tensor device type
 */
export type DeviceType = 'cpu' | 'gpu' | 'cuda';

/**
 * Tensor data type
 */
export type DType = 'f32' | 'f64' | 'i32' | 'i64' | 'u8' | 'bool';

/**
 * Tensor creation options
 */
export interface TensorOptions {
    /** Device to create tensor on (default: 'cpu') */
    device?: DeviceType;

    /** Data type (default: 'f32') */
    dtype?: DType;

    /** Whether to track gradients (default: false) */
    requiresGrad?: boolean;
}

/**
 * GPU-accelerated tensor
 */
export class Tensor {
    /**
     * Create tensor from data
     * @param data Flat array of tensor data
     * @param shape Tensor shape
     * @param options Tensor options
     */
    constructor(data: number[] | Float32Array | Float64Array, shape: number[], options?: TensorOptions);

    /**
     * Create tensor filled with zeros
     * @param shape Tensor shape
     * @param options Tensor options
     */
    static zeros(shape: number[], options?: TensorOptions): Tensor;

    /**
     * Create tensor filled with ones
     * @param shape Tensor shape
     * @param options Tensor options
     */
    static ones(shape: number[], options?: TensorOptions): Tensor;

    /**
     * Create tensor with random normal distribution
     * @param shape Tensor shape
     * @param options Tensor options
     */
    static randn(shape: number[], options?: TensorOptions): Tensor;

    /**
     * Create tensor with random uniform distribution
     * @param shape Tensor shape
     * @param options Tensor options
     */
    static rand(shape: number[], options?: TensorOptions): Tensor;

    /**
     * Get tensor shape
     */
    shape(): number[];

    /**
     * Get tensor data type
     */
    dtype(): DType;

    /**
     * Get tensor device
     */
    device(): DeviceType;

    /**
     * Get tensor data as JavaScript array
     */
    data(): number[];

    /**
     * Move tensor to specified device
     * @param device Target device
     */
    to(device: DeviceType): Tensor;

    /**
     * Element-wise addition (GPU-accelerated when on GPU)
     * @param other Tensor to add
     */
    add(other: Tensor): Tensor;

    /**
     * Element-wise subtraction
     * @param other Tensor to subtract
     */
    sub(other: Tensor): Tensor;

    /**
     * Element-wise multiplication (GPU-accelerated)
     * @param other Tensor to multiply
     */
    mul(other: Tensor): Tensor;

    /**
     * Element-wise division
     * @param other Tensor to divide by
     */
    div(other: Tensor): Tensor;

    /**
     * Matrix multiplication (GPU-accelerated with tiled algorithm)
     * @param other Tensor to multiply with
     */
    matmul(other: Tensor): Tensor;

    /**
     * Transpose tensor
     */
    transpose(): Tensor;

    /**
     * Reshape tensor
     * @param shape New shape
     */
    reshape(shape: number[]): Tensor;

    /**
     * ReLU activation (GPU-accelerated)
     */
    relu(): Tensor;

    /**
     * Sigmoid activation
     */
    sigmoid(): Tensor;

    /**
     * Tanh activation
     */
    tanh(): Tensor;

    /**
     * Softmax activation (GPU-accelerated, numerically stable)
     * @param dim Dimension to apply softmax (default: -1)
     */
    softmax(dim?: number): Tensor;

    /**
     * Sum reduction (GPU-accelerated)
     * @param dim Dimension to reduce (optional)
     */
    sum(dim?: number): Tensor;

    /**
     * Mean reduction (GPU-accelerated)
     * @param dim Dimension to reduce (optional)
     */
    mean(dim?: number): Tensor;

    /**
     * Maximum value
     * @param dim Dimension to reduce (optional)
     */
    max(dim?: number): Tensor;

    /**
     * Minimum value
     * @param dim Dimension to reduce (optional)
     */
    min(dim?: number): Tensor;

    /**
     * Compute gradients (backward pass)
     */
    backward(): void;

    /**
     * Get gradient tensor
     */
    grad(): Tensor | null;

    /**
     * Zero gradients
     */
    zeroGrad(): void;

    /**
     * Free GPU memory immediately
     */
    free(): void;
}

/**
 * Neural network module options
 */
export interface ModuleOptions {
    /** Device to create module on (default: 'cpu') */
    device?: DeviceType;
}

/**
 * Linear layer (fully connected)
 */
export class Linear {
    /**
     * Create linear layer
     * @param inFeatures Number of input features
     * @param outFeatures Number of output features
     * @param bias Whether to include bias (default: true)
     * @param options Module options
     */
    constructor(inFeatures: number, outFeatures: number, bias?: boolean, options?: ModuleOptions);

    /**
     * Forward pass (GPU-accelerated matrix multiplication)
     * @param input Input tensor
     */
    forward(input: Tensor): Tensor;

    /**
     * Get trainable parameters
     */
    parameters(): Tensor[];

    /**
     * Move module to device
     * @param device Target device
     */
    to(device: DeviceType): void;
}

/**
 * ReLU activation layer
 */
export class ReLU {
    /**
     * Forward pass (GPU-accelerated)
     * @param input Input tensor
     */
    forward(input: Tensor): Tensor;
}

/**
 * Sigmoid activation layer
 */
export class Sigmoid {
    /**
     * Forward pass
     * @param input Input tensor
     */
    forward(input: Tensor): Tensor;
}

/**
 * Softmax activation layer
 */
export class Softmax {
    /**
     * Create softmax layer
     * @param dim Dimension to apply softmax (default: -1)
     */
    constructor(dim?: number);

    /**
     * Forward pass (GPU-accelerated, numerically stable)
     * @param input Input tensor
     */
    forward(input: Tensor): Tensor;
}

/**
 * Sequential container for neural network layers
 */
export class Sequential {
    /**
     * Create sequential container
     * @param layers Array of layers
     * @param options Module options
     */
    constructor(layers: Array<Linear | ReLU | Sigmoid | Softmax>, options?: ModuleOptions);

    /**
     * Forward pass through all layers (GPU-accelerated)
     * @param input Input tensor
     */
    forward(input: Tensor): Tensor;

    /**
     * Get all trainable parameters
     */
    parameters(): Tensor[];

    /**
     * Move all layers to device
     * @param device Target device
     */
    to(device: DeviceType): void;
}

/**
 * Optimizer options
 */
export interface OptimizerOptions {
    /** Learning rate */
    lr: number;
}

/**
 * Adam optimizer options
 */
export interface AdamOptions extends OptimizerOptions {
    /** Beta1 parameter (default: 0.9) */
    beta1?: number;

    /** Beta2 parameter (default: 0.999) */
    beta2?: number;

    /** Epsilon for numerical stability (default: 1e-8) */
    epsilon?: number;

    /** Weight decay (default: 0) */
    weightDecay?: number;
}

/**
 * Adam optimizer (GPU-accelerated parameter updates)
 */
export class Adam {
    /**
     * Create Adam optimizer
     * @param parameters Trainable parameters
     * @param options Optimizer options
     */
    constructor(parameters: Tensor[], options: AdamOptions);

    /**
     * Perform optimization step (GPU-accelerated)
     */
    step(): void;

    /**
     * Zero all gradients
     */
    zeroGrad(): void;
}

/**
 * SGD optimizer options
 */
export interface SgdOptions extends OptimizerOptions {
    /** Momentum (default: 0) */
    momentum?: number;

    /** Weight decay (default: 0) */
    weightDecay?: number;
}

/**
 * SGD optimizer
 */
export class SGD {
    /**
     * Create SGD optimizer
     * @param parameters Trainable parameters
     * @param options Optimizer options
     */
    constructor(parameters: Tensor[], options: SgdOptions);

    /**
     * Perform optimization step
     */
    step(): void;

    /**
     * Zero all gradients
     */
    zeroGrad(): void;
}

/**
 * Loss functions
 */
export namespace loss {
    /**
     * Mean Squared Error loss
     * @param prediction Predicted values
     * @param target Target values
     */
    export function mse(prediction: Tensor, target: Tensor): Tensor;

    /**
     * Binary Cross Entropy loss
     * @param prediction Predicted values
     * @param target Target values
     */
    export function binaryCrossEntropy(prediction: Tensor, target: Tensor): Tensor;

    /**
     * Cross Entropy loss
     * @param prediction Predicted logits
     * @param target Target class indices
     */
    export function crossEntropy(prediction: Tensor, target: Tensor): Tensor;
}

/**
 * WebGPU utilities
 */
export namespace webgpu {
    /**
     * Initialize WebGPU device
     * @returns WebGPU device instance
     */
    export function init(): Promise<WebGpuDevice>;

    /**
     * Check if WebGPU is supported
     * @returns true if WebGPU is available
     */
    export function isSupported(): boolean;

    /**
     * Get current GPU memory usage
     * @returns Memory usage statistics
     */
    export function getMemoryUsage(): GpuMemoryUsage;

    /**
     * Clear shader cache
     */
    export function clearShaderCache(): void;
}

/**
 * Performance profiling
 */
export namespace profiler {
    /**
     * Start profiling
     */
    export function start(): void;

    /**
     * Stop profiling and get results
     * @returns Profiling results
     */
    export function stop(): ProfileResults;

    /**
     * Profile a function
     * @param name Profile name
     * @param fn Function to profile
     */
    export function profile<T>(name: string, fn: () => T): T;
}

/**
 * Profiling results
 */
export interface ProfileResults {
    /** Total execution time in milliseconds */
    totalTime: number;

    /** GPU time in milliseconds */
    gpuTime: number;

    /** CPU time in milliseconds */
    cpuTime: number;

    /** Number of operations */
    operationCount: number;

    /** Operations per second */
    opsPerSec: number;

    /** Memory allocated in bytes */
    memoryAllocated: number;
}

/**
 * Initialize ToRSh WASM module
 * @returns Promise that resolves when initialization is complete
 */
export default function init(): Promise<void>;

/**
 * Get ToRSh version
 */
export function version(): string;
