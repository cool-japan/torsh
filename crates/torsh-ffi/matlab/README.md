# ToRSh MATLAB Interface

This directory contains MATLAB bindings for the ToRSh deep learning framework. The interface allows MATLAB users to create and manipulate ToRSh tensors directly from MATLAB code using MEX functions.

## Features

- **Seamless Integration**: Use ToRSh tensors as MATLAB objects with familiar syntax
- **High Performance**: Leverages Rust's performance while maintaining MATLAB ease of use
- **Complete Tensor Operations**: Element-wise arithmetic, matrix multiplication, reshaping, and more
- **Neural Network Support**: Built-in activation functions (ReLU, sigmoid, tanh, softmax)
- **Memory Efficient**: Automatic memory management with proper cleanup

## Requirements

- MATLAB R2018a or later
- Rust toolchain (1.70.0 or later)
- MATLAB MEX compiler setup
- C/C++ compiler compatible with MATLAB

## Installation

1. **Setup MEX Compiler**:
   ```matlab
   mex -setup
   ```

2. **Build the Interface**:
   ```matlab
   cd matlab
   build_mex
   ```

3. **Test Installation**:
   ```matlab
   A = TorshTensor([1, 2; 3, 4]);
   disp(A);
   ```

## Quick Start

### Creating Tensors

```matlab
% From MATLAB arrays
A = TorshTensor([1, 2, 3; 4, 5, 6]);

% Using static methods
B = TorshTensor.zeros(3, 3);
C = TorshTensor.ones(2, 4);
D = TorshTensor.randn(10, 20);
E = TorshTensor.eye(5);
```

### Basic Operations

```matlab
% Element-wise operations
A = TorshTensor([1, 2; 3, 4]);
B = TorshTensor([5, 6; 7, 8]);

C = A + B;        % Addition
D = A - B;        % Subtraction
E = A .* B;       % Element-wise multiplication
F = A ./ B;       % Element-wise division

% Matrix operations
G = A * B;        % Matrix multiplication
H = A.transpose(); % Transpose
```

### Scalar Operations

```matlab
A = TorshTensor([1, 2; 3, 4]);

B = A + 5;        % Add scalar
C = A * 2;        % Multiply by scalar
D = A / 3;        % Divide by scalar
```

### Neural Network Operations

```matlab
A = TorshTensor([-1, 0, 1; 2, -2, 3]);

% Activation functions
relu_out = A.relu();           % ReLU: max(0, x)
sigmoid_out = A.sigmoid();     % Sigmoid: 1/(1+exp(-x))
tanh_out = A.tanh();          % Tanh
softmax_out = A.softmax(-1);   % Softmax along last dimension
```

### Reduction Operations

```matlab
A = TorshTensor([1, 2, 3; 4, 5, 6]);

% Sum operations
total_sum = A.sum();           % Sum all elements
row_sum = A.sum(1);           % Sum along rows
col_sum = A.sum(2);           % Sum along columns

% Mean operations
total_mean = A.mean();         % Mean of all elements
row_mean = A.mean(1);         % Mean along rows
```

### Reshaping and Views

```matlab
A = TorshTensor.randn(2, 3, 4);

% Reshape
B = A.reshape(6, 4);          % Reshape to 6x4
C = A.reshape(24);            % Flatten to 1D

% Transpose specific dimensions
D = A.transpose(0, 2);        % Swap first and third dimensions
```

### Accessing Data

```matlab
A = TorshTensor([1, 2; 3, 4]);

% Get MATLAB array
data = A.data();              % Returns: [1, 2; 3, 4]

% Get tensor properties
dims = A.size();              % Get dimensions
n_elements = A.numel();       % Number of elements
n_dims = A.ndims();          % Number of dimensions
```

## Advanced Usage

### Memory Management

The `TorshTensor` class automatically manages memory through MATLAB's handle class system. Tensors are automatically cleaned up when they go out of scope.

```matlab
function result = heavy_computation()
    % Large tensors are automatically cleaned up
    A = TorshTensor.randn(1000, 1000);
    B = TorshTensor.randn(1000, 1000);
    result = A * B;
    % A and B are automatically freed when function exits
end
```

### Performance Tips

1. **Avoid Frequent Data Conversion**: Keep operations in ToRSh tensors rather than converting to MATLAB arrays frequently.

2. **Use In-Place Operations When Possible**: Some operations can reuse memory:
   ```matlab
   A = TorshTensor.randn(100, 100);
   A = A.relu();  % Efficient
   ```

3. **Batch Operations**: Combine multiple operations:
   ```matlab
   % Efficient
   result = A.relu().sum(1).softmax(-1);
   
   % Less efficient
   temp1 = A.relu();
   temp2 = temp1.sum(1);
   result = temp2.softmax(-1);
   ```

### Error Handling

```matlab
try
    A = TorshTensor([1, 2, 3]);
    B = TorshTensor([4, 5]);
    C = A * B;  % This will fail due to shape mismatch
catch ME
    fprintf('Error: %s\n', ME.message);
end
```

## Integration with MATLAB Ecosystem

### Plotting

```matlab
% Generate data with ToRSh
x = TorshTensor.linspace(0, 2*pi, 100);
y = x.sin();

% Convert to MATLAB for plotting
x_data = x.data();
y_data = y.data();

plot(x_data, y_data);
title('Sine Wave Generated with ToRSh');
```

### Machine Learning Workflow

```matlab
% Load data (example with random data)
X_train = TorshTensor.randn(1000, 784);  % 1000 samples, 784 features
y_train = TorshTensor.randn(1000, 10);   % 10 classes

% Simple linear transformation
W = TorshTensor.randn(784, 128);
b = TorshTensor.zeros(128);

% Forward pass
hidden = (X_train * W + b).relu();
```

## Troubleshooting

### Common Issues

1. **MEX Compilation Fails**:
   - Ensure MEX compiler is properly setup: `mex -setup`
   - Check that Rust is installed and `cargo` is in PATH
   - Verify C/C++ compiler compatibility

2. **Runtime Errors**:
   - Check tensor dimensions for operations
   - Ensure sufficient memory for large tensors
   - Verify MATLAB version compatibility

3. **Performance Issues**:
   - Use appropriate tensor sizes (not too small for overhead)
   - Minimize data conversion between MATLAB and ToRSh
   - Consider using GPU operations if available

### Getting Help

- Check MATLAB Command Window for detailed error messages
- Use `dbstop if error` for debugging
- Verify installation with the test suite: `build_mex()` includes basic tests

## Examples

See the `examples/` directory for complete examples including:
- Basic tensor operations
- Neural network layers
- Optimization algorithms
- Data processing pipelines

## API Reference

### TorshTensor Class Methods

#### Constructor
- `TorshTensor(data)` - Create tensor from MATLAB array

#### Static Methods
- `TorshTensor.zeros(dims...)` - Create zero tensor
- `TorshTensor.ones(dims...)` - Create ones tensor
- `TorshTensor.randn(dims...)` - Create random normal tensor
- `TorshTensor.eye(n, m)` - Create identity matrix

#### Instance Methods
- `data()` - Get MATLAB array
- `size(dim)` - Get dimensions
- `numel()` - Number of elements
- `ndims()` - Number of dimensions
- `relu()` - ReLU activation
- `sigmoid()` - Sigmoid activation
- `tanh()` - Tanh activation
- `softmax(dim)` - Softmax activation
- `sum(dim, keepdim)` - Sum reduction
- `mean(dim, keepdim)` - Mean reduction
- `reshape(dims...)` - Reshape tensor
- `transpose(dim1, dim2)` - Transpose dimensions

#### Operators
- `+`, `-`, `.*`, `./` - Element-wise operations
- `*` - Matrix multiplication
- Support for scalar operations

## License

This MATLAB interface is part of the ToRSh project and follows the same license terms.