# ToRSh Lua Bindings

This directory contains Lua bindings for the ToRSh deep learning framework. The bindings allow Lua scripts to create and manipulate ToRSh tensors using Lua's simple and elegant syntax.

## Features

- **Native Lua Integration**: Use ToRSh tensors as first-class Lua objects
- **Familiar Syntax**: Lua-style operations with metamethods for natural tensor arithmetic
- **High Performance**: Direct integration with ToRSh's Rust backend for optimal performance
- **Comprehensive API**: Full tensor operations, neural network functions, and utilities
- **Lightweight**: Minimal overhead while maintaining full functionality

## Requirements

- Lua 5.1, 5.2, 5.3, or 5.4
- LuaJIT (recommended for performance)
- Rust toolchain (1.70.0 or later)
- C compiler compatible with Lua

## Installation

### Method 1: Build from Source

1. **Build the Lua module**:
   ```bash
   cd torsh-ffi
   cargo build --release --features lua
   ```

2. **Copy the library**:
   ```bash
   # Linux/macOS
   cp target/release/libtorsh_ffi.so torsh_core.so
   
   # macOS
   cp target/release/libtorsh_ffi.dylib torsh_core.so
   
   # Windows
   cp target/release/torsh_ffi.dll torsh_core.dll
   ```

3. **Set up Lua path**:
   ```bash
   export LUA_CPATH="./?.so;$LUA_CPATH"
   export LUA_PATH="./lua/?.lua;$LUA_PATH"
   ```

### Method 2: LuaRocks (Future)

```bash
luarocks install torsh
```

## Quick Start

### Basic Usage

```lua
local torsh = require('torsh')

-- Create tensors
local a = torsh.tensor({{1, 2, 3}, {4, 5, 6}})
local b = torsh.ones(2, 3)

-- Perform operations
local c = a + b
local d = a * b
local e = torsh.matmul(a, b:transpose())

-- Print results
print("a + b =", c)
print("Shape:", table.concat(c:shape(), "x"))
```

### Creating Tensors

```lua
-- From Lua tables
local t1 = torsh.tensor({{1, 2}, {3, 4}})
local t2 = torsh.tensor({1, 2, 3, 4, 5})

-- Special tensors
local zeros = torsh.zeros(3, 3)
local ones = torsh.ones(2, 4)
local random = torsh.randn(10, 10)
local identity = torsh.eye(5)

-- Ranges and sequences
local linspace = torsh.linspace(0, 10, 11)  -- [0, 1, 2, ..., 10]
local range = torsh.arange(0, 10, 2)        -- [0, 2, 4, 6, 8]
```

### Basic Operations

```lua
local a = torsh.tensor({{1, 2}, {3, 4}})
local b = torsh.tensor({{5, 6}, {7, 8}})

-- Element-wise operations
local add = a + b        -- Addition
local sub = a - b        -- Subtraction
local mul = a * b        -- Element-wise multiplication
local div = a / b        -- Element-wise division

-- Matrix operations
local matmul = torsh.matmul(a, b)  -- Matrix multiplication
local transpose = a:transpose()     -- Transpose

-- Scalar operations
local scalar_add = a + 10
local scalar_mul = a * 2.5
```

### Neural Network Operations

```lua
-- Activation functions
local x = torsh.tensor({{-1, 0, 1}, {2, -2, 3}})

local relu = x:relu()           -- ReLU activation
local sigmoid = x:sigmoid()     -- Sigmoid activation
local tanh_out = x:tanh()      -- Tanh activation
local softmax = x:softmax(-1)   -- Softmax along last dimension

-- Loss functions
local pred = torsh.randn(10, 3):softmax(-1)
local target = torsh.zeros(10, 3)
-- Set some targets to 1.0 for demonstration

local mse = torsh.mse_loss(pred, target)
local ce = torsh.cross_entropy_loss(pred, target)
```

### Shape Operations

```lua
local x = torsh.randn(2, 3, 4)

-- Reshape
local reshaped = x:reshape(6, 4)
local flattened = x:reshape(24)

-- Transpose
local transposed = x:transpose(0, 2)  -- Swap dimensions 0 and 2

-- Reductions
local sum_all = x:sum()          -- Sum all elements
local sum_dim = x:sum(1)         -- Sum along dimension 1
local mean_all = x:mean()        -- Mean of all elements
local mean_dim = x:mean(2, true) -- Mean along dim 2, keep dimensions
```

### Advanced Usage

#### Building a Neural Network

```lua
-- Define network parameters
local function init_layer(input_size, output_size)
    local scale = math.sqrt(2.0 / (input_size + output_size))
    local weight = torsh.randn(input_size, output_size) * scale
    local bias = torsh.zeros(output_size)
    return weight, bias
end

-- Create layers
local W1, b1 = init_layer(784, 128)
local W2, b2 = init_layer(128, 64)
local W3, b3 = init_layer(64, 10)

-- Forward pass function
local function forward(x)
    local h1 = torsh.linear(x, W1, b1):relu()
    local h2 = torsh.linear(h1, W2, b2):relu()
    local output = torsh.linear(h2, W3, b3):softmax(-1)
    return output
end

-- Training step
local function train_step(x, y, lr)
    local pred = forward(x)
    local loss = torsh.cross_entropy_loss(pred, y)
    
    -- Simplified gradient descent (pseudo-code)
    -- In practice, you'd implement proper backpropagation
    local grads = compute_gradients(loss, {W1, b1, W2, b2, W3, b3})
    torsh.sgd_step({W1, b1, W2, b2, W3, b3}, grads, lr)
    
    return loss
end
```

#### Data Processing

```lua
-- Load and process data
local function load_dataset(filename)
    local data = torsh.load_tensor(filename)
    return data
end

-- Normalize data
local function normalize(tensor)
    local mean = tensor:mean()
    local std = tensor:std()
    return (tensor - mean) / std
end

-- Create data batches
local function create_batches(data, labels, batch_size)
    local n_samples = data:size(0)
    local batches = {}
    
    for i = 1, n_samples, batch_size do
        local end_idx = math.min(i + batch_size - 1, n_samples)
        local batch_data = data:narrow(0, i - 1, end_idx - i + 1)
        local batch_labels = labels:narrow(0, i - 1, end_idx - i + 1)
        table.insert(batches, {batch_data, batch_labels})
    end
    
    return batches
end
```

#### GPU Operations (if available)

```lua
-- Check GPU availability
if torsh.cuda_available() then
    print("CUDA is available")
    
    -- Move tensors to GPU
    local a = torsh.randn(1000, 1000)
    local a_gpu = torsh.to_device(a, "cuda")
    
    -- Perform operations on GPU
    local b_gpu = a_gpu * a_gpu
    local c_gpu = torsh.matmul(a_gpu, b_gpu)
    
    -- Move back to CPU if needed
    local c_cpu = torsh.to_device(c_gpu, "cpu")
else
    print("Using CPU")
end
```

### Debugging and Utilities

```lua
-- Print tensor information
local function print_tensor_info(tensor, name)
    print(string.format("%s: shape=%s, dtype=%s, device=%s", 
        name or "Tensor",
        table.concat(tensor:shape(), "x"),
        torsh.dtype(tensor),
        torsh.device(tensor)))
end

-- Save and load tensors
torsh.save_tensor(my_tensor, "model_weights.pt")
local loaded_tensor = torsh.load_tensor("model_weights.pt")

-- Set random seed for reproducibility
torsh.manual_seed(42)
```

## API Reference

### Core Functions

#### Tensor Creation
- `torsh.tensor(data)` - Create tensor from Lua table
- `torsh.zeros(...)` - Create zero tensor with given dimensions
- `torsh.ones(...)` - Create ones tensor
- `torsh.randn(...)` - Create random normal tensor
- `torsh.eye(n, m)` - Create identity matrix
- `torsh.linspace(start, stop, steps)` - Create linearly spaced tensor
- `torsh.arange(start, stop, step)` - Create range tensor

#### Tensor Operations
- `tensor:shape()` - Get tensor dimensions
- `tensor:size(dim)` - Get size of specific dimension
- `tensor:numel()` - Get total number of elements
- `tensor:ndim()` - Get number of dimensions
- `tensor:data()` - Get data as Lua table
- `tensor:reshape(...)` - Reshape tensor
- `tensor:transpose(dim1, dim2)` - Transpose dimensions

#### Arithmetic Operations
- `+`, `-`, `*`, `/` - Element-wise operations (with broadcasting)
- `torsh.matmul(a, b)` - Matrix multiplication
- Scalar operations supported with all operators

#### Activation Functions
- `tensor:relu()` - ReLU activation
- `tensor:sigmoid()` - Sigmoid activation
- `tensor:tanh()` - Tanh activation
- `tensor:softmax(dim)` - Softmax activation

#### Reduction Operations
- `tensor:sum(dim, keepdim)` - Sum reduction
- `tensor:mean(dim, keepdim)` - Mean reduction

#### Utility Functions
- `torsh.save_tensor(tensor, filename)` - Save tensor to file
- `torsh.load_tensor(filename)` - Load tensor from file
- `torsh.manual_seed(seed)` - Set random seed
- `torsh.cuda_available()` - Check CUDA availability
- `torsh.to_device(tensor, device)` - Move tensor to device

### Error Handling

```lua
-- Wrap tensor operations in pcall for error handling
local success, result = pcall(function()
    return torsh.matmul(a, b)
end)

if not success then
    print("Error:", result)
else
    print("Result:", result)
end
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_operations.lua` - Fundamental tensor operations
- `neural_network.lua` - Building and training neural networks
- `data_processing.lua` - Data loading and preprocessing
- `optimization.lua` - Optimization algorithms

## Performance Tips

1. **Use LuaJIT**: LuaJIT provides significant performance improvements over standard Lua
2. **Batch Operations**: Combine multiple operations to reduce overhead
3. **GPU Usage**: Move large tensors to GPU for computation-intensive operations
4. **Memory Management**: Lua's garbage collector handles tensor cleanup automatically

## Integration with Other Libraries

### With OpenResty/Nginx

```lua
-- In OpenResty/Nginx Lua context
local torsh = require('torsh')

local function predict_handler()
    local input_data = get_request_data()
    local input_tensor = torsh.tensor(input_data)
    local prediction = model_forward(input_tensor)
    return prediction:data()
end
```

### With Love2D Game Engine

```lua
-- In Love2D
local torsh = require('torsh')

function love.load()
    -- Initialize ML model for game AI
    model_weights = torsh.load_tensor("game_ai_model.pt")
end

function love.update(dt)
    local game_state = get_game_state()
    local state_tensor = torsh.tensor(game_state)
    local action = model_predict(state_tensor)
    apply_ai_action(action)
end
```

## Troubleshooting

### Common Issues

1. **Module Not Found**:
   ```
   module 'torsh_core' not found
   ```
   - Ensure the compiled library is in LUA_CPATH
   - Check that the library has the correct name and extension

2. **Memory Issues**:
   ```
   not enough memory
   ```
   - Use smaller batch sizes
   - Enable garbage collection: `collectgarbage()`

3. **Shape Mismatch**:
   ```
   tensor shape mismatch
   ```
   - Check tensor dimensions before operations
   - Use `tensor:shape()` to debug tensor sizes

### Debugging

```lua
-- Enable debug mode
_G.TORSH_DEBUG = true

-- Print tensor information
local function debug_tensor(t, name)
    if _G.TORSH_DEBUG then
        print(string.format("DEBUG: %s = %s", name, tostring(t)))
    end
end
```

## Contributing

Contributions are welcome! Please see the main ToRSh repository for contribution guidelines.

## License

This Lua interface is part of the ToRSh project and follows the same license terms.