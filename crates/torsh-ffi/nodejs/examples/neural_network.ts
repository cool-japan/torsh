/**
 * Neural Network Example (TypeScript)
 * 
 * This example demonstrates building and training a simple neural network using ToRSh
 */

import { Tensor, nn, optim, utils } from '@torsh/core';

// Set random seed for reproducibility
// utils.manualSeed(42);

console.log('ToRSh Node.js Examples - Neural Network (TypeScript)');
console.log('====================================================');

/**
 * Simple dataset generator for classification
 */
function generateSpiralData(samplesPerClass: number, numClasses: number): { X: number[][], y: number[] } {
  const X: number[][] = [];
  const y: number[] = [];
  
  for (let classNum = 0; classNum < numClasses; classNum++) {
    for (let i = 0; i < samplesPerClass; i++) {
      const idx = i / samplesPerClass;
      const radius = idx;
      const theta = classNum * 4 + idx * 4 + Math.random() * 0.2;
      
      const x = radius * Math.cos(theta);
      const yCoord = radius * Math.sin(theta);
      
      X.push([x, yCoord]);
      y.push(classNum);
    }
  }
  
  return { X, y };
}

/**
 * Convert class indices to one-hot encoding
 */
function createOneHot(labels: number[], numClasses: number): number[][] {
  const oneHot: number[][] = [];
  
  for (const label of labels) {
    const row = new Array(numClasses).fill(0);
    row[label] = 1;
    oneHot.push(row);
  }
  
  return oneHot;
}

/**
 * Simple neural network class
 */
class SimpleNeuralNetwork {
  private W1: Tensor;
  private b1: Tensor;
  private W2: Tensor;
  private b2: Tensor;
  private W3: Tensor;
  private b3: Tensor;

  constructor(inputSize: number, hidden1Size: number, hidden2Size: number, outputSize: number) {
    // Xavier/Glorot initialization
    const scale1 = Math.sqrt(2.0 / (inputSize + hidden1Size));
    const scale2 = Math.sqrt(2.0 / (hidden1Size + hidden2Size));
    const scale3 = Math.sqrt(2.0 / (hidden2Size + outputSize));

    // Initialize weights with random values (simplified - would use proper random initialization)
    this.W1 = this.initWeight(inputSize, hidden1Size, scale1);
    this.b1 = Tensor.zeros(hidden1Size);
    
    this.W2 = this.initWeight(hidden1Size, hidden2Size, scale2);
    this.b2 = Tensor.zeros(hidden2Size);
    
    this.W3 = this.initWeight(hidden2Size, outputSize, scale3);
    this.b3 = Tensor.zeros(outputSize);
  }

  private initWeight(rows: number, cols: number, scale: number): Tensor {
    // Create random-like initialization (simplified)
    const data: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Box-Muller transform for normal distribution (simplified)
        const u1 = Math.random();
        const u2 = Math.random();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row.push(normal * scale);
      }
      data.push(row);
    }
    return Tensor.tensor(data);
  }

  /**
   * Forward pass through the network
   */
  forward(x: Tensor): Tensor {
    // Layer 1: Linear + ReLU
    const z1 = nn.linear(x, this.W1, this.b1);
    const a1 = z1.relu();
    
    // Layer 2: Linear + ReLU
    const z2 = nn.linear(a1, this.W2, this.b2);
    const a2 = z2.relu();
    
    // Layer 3: Linear + Softmax
    const z3 = nn.linear(a2, this.W3, this.b3);
    // Note: In practice, you'd apply softmax here
    // const output = z3.softmax(-1);
    const output = z3; // Simplified for this example
    
    return output;
  }

  /**
   * Get all trainable parameters
   */
  parameters(): Tensor[] {
    return [this.W1, this.b1, this.W2, this.b2, this.W3, this.b3];
  }
}

/**
 * Calculate accuracy from predictions and targets
 */
function calculateAccuracy(predictions: Tensor, targets: number[]): number {
  const predData = predictions.data();
  const batchSize = targets.length;
  const numClasses = predData.length / batchSize;
  
  let correct = 0;
  
  for (let i = 0; i < batchSize; i++) {
    let maxIdx = 0;
    let maxVal = predData[i * numClasses];
    
    for (let j = 1; j < numClasses; j++) {
      if (predData[i * numClasses + j] > maxVal) {
        maxVal = predData[i * numClasses + j];
        maxIdx = j;
      }
    }
    
    if (maxIdx === targets[i]) {
      correct++;
    }
  }
  
  return correct / batchSize;
}

/**
 * Main training function
 */
async function trainNeuralNetwork(): Promise<void> {
  console.log('\n1. Generating Dataset:');
  
  // Generate synthetic spiral dataset
  const { X: XData, y: yData } = generateSpiralData(100, 3);
  const X = Tensor.tensor(XData);
  const yOneHot = Tensor.tensor(createOneHot(yData, 3));
  
  console.log(`Dataset: X shape ${X.shape().join('x')}, y shape ${yOneHot.shape().join('x')}`);
  
  console.log('\n2. Creating Neural Network:');
  
  // Create network: 2 -> 128 -> 64 -> 3
  const network = new SimpleNeuralNetwork(2, 128, 64, 3);
  console.log('Network architecture: 2 -> 128 -> 64 -> 3');
  
  console.log('\n3. Training Loop:');
  
  const learningRate = 0.01;
  const epochs = 50; // Reduced for demonstration
  const printEvery = 10;
  
  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Forward pass
    const predictions = network.forward(X);
    
    // Calculate loss (simplified MSE for demonstration)
    const loss = nn.mseLoss(predictions, yOneHot);
    
    if (epoch % printEvery === 0) {
      const accuracy = calculateAccuracy(predictions, yData);
      console.log(`Epoch ${epoch}, Loss: ${loss.data()[0].toFixed(6)}, Accuracy: ${(accuracy * 100).toFixed(2)}%`);
    }
    
    // Note: In a real implementation, you would:
    // 1. Compute gradients through backpropagation
    // 2. Update parameters using an optimizer
    // For this demonstration, we're skipping the actual training updates
  }
  
  console.log('\n4. Final Evaluation:');
  
  // Generate test data
  const { X: XTestData, y: yTestData } = generateSpiralData(50, 3);
  const XTest = Tensor.tensor(XTestData);
  
  const testPredictions = network.forward(XTest);
  const testAccuracy = calculateAccuracy(testPredictions, yTestData);
  
  console.log(`Test Accuracy: ${(testAccuracy * 100).toFixed(2)}%`);
  
  console.log('\n5. Sample Predictions:');
  
  const predData = testPredictions.data();
  const numClasses = 3;
  const sampleSize = Math.min(10, yTestData.length);
  
  for (let i = 0; i < sampleSize; i++) {
    const input = XTestData[i];
    const trueClass = yTestData[i];
    
    let predClass = 0;
    let maxProb = predData[i * numClasses];
    
    for (let j = 1; j < numClasses; j++) {
      if (predData[i * numClasses + j] > maxProb) {
        maxProb = predData[i * numClasses + j];
        predClass = j;
      }
    }
    
    console.log(`Input: (${input[0].toFixed(2)}, ${input[1].toFixed(2)}), True: ${trueClass}, Pred: ${predClass}, Confidence: ${maxProb.toFixed(3)}`);
  }
}

/**
 * Demonstration of tensor operations for ML
 */
function demonstrateMlOperations(): void {
  console.log('\n6. ML Operations Demonstration:');
  
  try {
    // Batch normalization simulation
    const input = Tensor.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    console.log('Input batch:', input.toString());
    
    // Simulate normalization (mean subtraction)
    const mean = Tensor.tensor([4, 5, 6]); // Would compute actual mean in practice
    // const normalized = input.sub(mean); // Would implement proper broadcasting
    
    // Activation functions
    const activationInput = Tensor.tensor([[-2, -1, 0, 1, 2]]);
    const reluOutput = activationInput.relu();
    console.log('ReLU output:', reluOutput.toString());
    
    // Matrix operations for linear layers
    const inputFeatures = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
    const weights = Tensor.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);
    const linearOutput = inputFeatures.matmul(weights);
    console.log('Linear layer output:', linearOutput.toString());
    
    console.log('ML operations demonstration completed!');
    
  } catch (error) {
    console.error('Error in ML operations:', error);
  }
}

/**
 * Performance benchmarking
 */
function benchmarkOperations(): void {
  console.log('\n7. Performance Benchmarking:');
  
  try {
    const sizes = [10, 50, 100];
    
    for (const size of sizes) {
      const iterations = 100;
      const start = Date.now();
      
      for (let i = 0; i < iterations; i++) {
        const a = Tensor.ones(size, size);
        const b = Tensor.ones(size, size);
        const c = a.add(b);
        const d = c.mul(a);
        const e = d.matmul(b);
      }
      
      const end = Date.now();
      const duration = end - start;
      const opsPerSec = (iterations * 4) / (duration / 1000); // 4 operations per iteration
      
      console.log(`Size ${size}x${size}: ${iterations} iterations in ${duration}ms (${opsPerSec.toFixed(2)} ops/sec)`);
    }
    
    console.log('Performance benchmarking completed!');
    
  } catch (error) {
    console.error('Error in benchmarking:', error);
  }
}

/**
 * Main execution function
 */
async function main(): Promise<void> {
  try {
    await trainNeuralNetwork();
    demonstrateMlOperations();
    benchmarkOperations();
    
    console.log('\n====================================================');
    console.log('Neural network example completed successfully!');
    console.log('Note: This example uses simplified implementations for demonstration.');
    console.log('In practice, you would implement proper backpropagation and optimizers.');
    
  } catch (error) {
    console.error('Error in main execution:', error);
  }
}

// Export functions for use as module
export {
  generateSpiralData,
  createOneHot,
  SimpleNeuralNetwork,
  calculateAccuracy,
  trainNeuralNetwork,
  demonstrateMlOperations,
  benchmarkOperations,
  main
};

// Run main function if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
}