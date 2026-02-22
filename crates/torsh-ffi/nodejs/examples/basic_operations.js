/**
 * Basic ToRSh Operations Example (JavaScript)
 * 
 * This example demonstrates fundamental tensor operations using ToRSh in Node.js
 */

const { Tensor } = require('@torsh/core');

console.log('ToRSh Node.js Examples - Basic Operations');
console.log('=========================================');

async function basicOperationsExample() {
  try {
    // 1. Creating tensors
    console.log('\n1. Creating Tensors:');
    
    // From nested arrays
    const a = Tensor.tensor([[1, 2, 3], [4, 5, 6]]);
    console.log('Tensor a:', a.toString());
    console.log('Shape:', a.shape());
    
    const b = Tensor.tensor([[7, 8, 9], [10, 11, 12]]);
    console.log('Tensor b:', b.toString());
    
    // Special tensors
    const zeros = Tensor.zeros(3, 3);
    const ones = Tensor.ones(2, 4);
    
    console.log('Zeros tensor:', zeros.toString());
    console.log('Ones tensor:', ones.toString());
    
    // 2. Basic arithmetic
    console.log('\n2. Basic Arithmetic:');
    
    // Element-wise operations
    const c = a.add(b);
    console.log('a + b =', c.toString());
    
    const d = a.mul(b);
    console.log('a * b (element-wise) =', d.toString());
    
    // 3. Matrix operations
    console.log('\n3. Matrix Operations:');
    
    // Create square matrices for matrix multiplication
    const m1 = Tensor.tensor([[1, 2], [3, 4]]);
    const m2 = Tensor.tensor([[5, 6], [7, 8]]);
    
    console.log('Matrix m1:', m1.toString());
    console.log('Matrix m2:', m2.toString());
    
    const matmulResult = m1.matmul(m2);
    console.log('m1 @ m2 =', matmulResult.toString());
    
    // 4. Activation functions
    console.log('\n4. Activation Functions:');
    
    const x = Tensor.tensor([[-2, -1, 0, 1, 2], [-0.5, 0, 0.5, 1.5, 2.5]]);
    console.log('Input tensor:', x.toString());
    
    const reluResult = x.relu();
    console.log('ReLU result:', reluResult.toString());
    
    // 5. Tensor information
    console.log('\n5. Tensor Information:');
    
    const info = Tensor.zeros(3, 4, 5);
    console.log('Tensor shape:', info.shape());
    console.log('Number of elements:', info.numel());
    console.log('Number of dimensions:', info.ndim());
    console.log('Size of dimension 1:', info.size(1));
    
    console.log('\nBasic operations completed successfully!');
    
  } catch (error) {
    console.error('Error during basic operations:', error.message);
  }
}

// Simple neural network forward pass example
function neuralNetworkExample() {
  console.log('\n6. Simple Neural Network Forward Pass:');
  
  try {
    // Input data (batch_size=4, features=3)
    const input = Tensor.tensor([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
      [10.0, 11.0, 12.0]
    ]);
    
    // Weight matrix (input_features=3, output_features=2)
    const weight = Tensor.tensor([
      [0.1, 0.2],
      [0.3, 0.4],
      [0.5, 0.6]
    ]);
    
    // Bias vector (output_features=2)
    const bias = Tensor.tensor([0.1, 0.2]);
    
    console.log('Input:', input.toString());
    console.log('Weight:', weight.toString());
    console.log('Bias:', bias.toString());
    
    // Linear transformation: output = input @ weight + bias
    const linear = input.matmul(weight).add(bias);
    console.log('Linear output:', linear.toString());
    
    // Apply ReLU activation
    const activated = linear.relu();
    console.log('After ReLU:', activated.toString());
    
    console.log('Neural network forward pass completed!');
    
  } catch (error) {
    console.error('Error during neural network example:', error.message);
  }
}

// Data processing example
function dataProcessingExample() {
  console.log('\n7. Data Processing Example:');
  
  try {
    // Simulate batch processing
    const batchSize = 8;
    const features = 10;
    
    // Create random-like data (would be actual data in practice)
    const data = [];
    for (let i = 0; i < batchSize; i++) {
      const row = [];
      for (let j = 0; j < features; j++) {
        row.push(Math.random() * 2 - 1); // Random values between -1 and 1
      }
      data.push(row);
    }
    
    const batch = Tensor.tensor(data);
    console.log('Batch tensor shape:', batch.shape());
    console.log('Batch data sample:', batch.toString());
    
    // Normalization (simplified - would use proper mean/std in practice)
    // Simulate applying a simple transformation
    const processed = batch.add(Tensor.ones(...batch.shape()).mul(0.1));
    console.log('Processed batch shape:', processed.shape());
    
    console.log('Data processing completed!');
    
  } catch (error) {
    console.error('Error during data processing:', error.message);
  }
}

// Performance timing example
function performanceExample() {
  console.log('\n8. Performance Timing Example:');
  
  try {
    const start = Date.now();
    
    // Perform multiple operations
    const size = 100;
    let result = Tensor.ones(size, size);
    
    for (let i = 0; i < 10; i++) {
      const temp = Tensor.ones(size, size);
      result = result.add(temp);
    }
    
    const end = Date.now();
    const duration = end - start;
    
    console.log(`Performed 10 additions on ${size}x${size} tensors in ${duration}ms`);
    console.log('Final result shape:', result.shape());
    console.log('Performance test completed!');
    
  } catch (error) {
    console.error('Error during performance test:', error.message);
  }
}

// Run all examples
async function runAllExamples() {
  await basicOperationsExample();
  neuralNetworkExample();
  dataProcessingExample();
  performanceExample();
  
  console.log('\n========================================');
  console.log('All examples completed successfully!');
  console.log('ToRSh Node.js bindings are working properly.');
}

// Export for use as module
module.exports = {
  basicOperationsExample,
  neuralNetworkExample,
  dataProcessingExample,
  performanceExample,
  runAllExamples
};

// Run examples if this file is executed directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}