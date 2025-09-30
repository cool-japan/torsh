# torsh-ffi

Foreign Function Interface for ToRSh, providing C and Python bindings.

## Overview

This crate enables ToRSh to be used from other programming languages through:

- **C API**: Complete C bindings for all ToRSh functionality
- **Python Integration**: PyTorch-compatible Python API
- **Language Bindings**: Support for additional languages via C FFI
- **Memory Safety**: Safe interop with proper error handling

## C API Usage

### Basic Example

```c
#include <torsh.h>

int main() {
    // Initialize ToRSh
    torsh_init();
    
    // Create tensors
    float data[] = {1.0, 2.0, 3.0, 4.0};
    int64_t shape[] = {2, 2};
    torsh_tensor_t* tensor = torsh_tensor_from_array(data, shape, 2, TORSH_F32);
    
    // Perform operations
    torsh_tensor_t* result = torsh_add(tensor, tensor);
    
    // Get data back
    float* result_data = torsh_tensor_data_ptr(result);
    
    // Cleanup
    torsh_tensor_free(tensor);
    torsh_tensor_free(result);
    torsh_cleanup();
    
    return 0;
}
```

### Neural Network Example

```c
// Create a simple model
torsh_module_t* model = torsh_sequential_new();
torsh_sequential_add(model, torsh_linear_new(784, 128, true));
torsh_sequential_add(model, torsh_relu_new());
torsh_sequential_add(model, torsh_linear_new(128, 10, true));

// Forward pass
torsh_tensor_t* output = torsh_module_forward(model, input);

// Create optimizer
torsh_optimizer_t* optimizer = torsh_adam_new(
    torsh_module_parameters(model),
    0.001,  // learning rate
    0.9,    // beta1
    0.999,  // beta2
    1e-8    // epsilon
);

// Training step
torsh_optimizer_zero_grad(optimizer);
torsh_backward(loss, false);
torsh_optimizer_step(optimizer);
```

## Python API

### Installation

```python
# From source
python setup.py install

# Or via pip (when available)
pip install torsh
```

### Basic Usage

```python
import torsh

# Tensor operations
x = torsh.tensor([[1, 2], [3, 4]], dtype=torsh.float32)
y = torsh.tensor([[5, 6], [7, 8]], dtype=torsh.float32)
z = torsh.matmul(x, y)

# Autograd
x = torsh.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.0])

# Neural networks
import torsh.nn as nn
import torsh.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
optimizer = torsh.optim.Adam(model.parameters(), lr=0.001)
```

### PyTorch Compatibility

```python
# Convert between PyTorch and ToRSh
import torch
import torsh

# PyTorch to ToRSh
torch_tensor = torch.randn(10, 20)
torsh_tensor = torsh.from_pytorch(torch_tensor)

# ToRSh to PyTorch
torch_tensor = torsh_tensor.to_pytorch()

# Share memory (zero-copy)
shared = torsh.as_tensor(torch_tensor)
```

## Error Handling

### C API

```c
torsh_error_t* error = NULL;
torsh_tensor_t* result = torsh_matmul_with_error(a, b, &error);

if (error != NULL) {
    const char* msg = torsh_error_message(error);
    fprintf(stderr, "Error: %s\n", msg);
    torsh_error_free(error);
    return -1;
}
```

### Python API

```python
try:
    result = torsh.matmul(a, b)
except torsh.ShapeMismatchError as e:
    print(f"Shape error: {e}")
except torsh.OutOfMemoryError as e:
    print(f"Memory error: {e}")
```

## Memory Management

### C API

```c
// Reference counting
torsh_tensor_retain(tensor);  // Increment ref count
torsh_tensor_release(tensor); // Decrement ref count

// Manual memory management
void* data = torsh_malloc(1024);
torsh_free(data);

// Memory pools
torsh_memory_pool_t* pool = torsh_memory_pool_new(1 << 20);  // 1MB
torsh_tensor_t* tensor = torsh_tensor_from_pool(pool, shape, ndim, dtype);
torsh_memory_pool_free(pool);  // Frees all tensors in pool
```

### Python API

Memory is managed automatically through Python's garbage collector.

## Building Language Bindings

### Ruby

```ruby
require 'ffi'

module ToRSh
  extend FFI::Library
  ffi_lib 'torsh'
  
  attach_function :torsh_init, [], :void
  attach_function :torsh_tensor_from_array, [:pointer, :pointer, :int, :int], :pointer
  # ... more bindings
end
```

### Java (JNI)

```java
public class ToRSh {
    static {
        System.loadLibrary("torsh_jni");
    }
    
    public static native void init();
    public static native long tensorFromArray(float[] data, long[] shape);
    // ... more methods
}
```

## Safety Considerations

- All C API functions validate inputs
- Null pointer checks on all pointer arguments
- Thread-safe operations where applicable
- Proper error propagation
- Memory leak prevention through RAII in bindings

## Performance

The FFI layer adds minimal overhead:
- Zero-copy tensor creation where possible
- Efficient data transfer mechanisms
- Batched operations to reduce FFI calls
- Optional async operations

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.