#!/usr/bin/env python3
"""
Example demonstrating basic tensor operations with ToRSh Python bindings.

This script shows how to use ToRSh tensors in a PyTorch-compatible way.
"""

import sys
import os

# Add the python package to the path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torsh
import numpy as np

def main():
    print("ToRSh Python Bindings - Tensor Operations Example")
    print("=" * 50)
    
    # Create tensors
    print("\n1. Creating tensors:")
    
    # From Python lists
    a = torsh.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"Tensor from list: {a}")
    print(f"Shape: {a.shape}, dtype: {a.dtype}, device: {a.device}")
    
    # From NumPy arrays
    np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = torsh.tensor(np_array)
    print(f"Tensor from NumPy: {b}")
    
    # Creation functions
    zeros = torsh.zeros([2, 3])
    ones = torsh.ones([2, 3])
    randn = torsh.randn([2, 3])
    
    print(f"Zeros: {zeros}")
    print(f"Ones: {ones}")
    print(f"Random normal: {randn}")
    
    # Basic arithmetic
    print("\n2. Basic arithmetic operations:")
    
    x = torsh.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torsh.tensor([[2.0, 3.0], [1.0, 2.0]])
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x - y = {x - y}")
    print(f"x * y = {x * y}")
    print(f"x / y = {x / y}")
    
    # Matrix operations
    print("\n3. Matrix operations:")
    
    m1 = torsh.tensor([[1.0, 2.0], [3.0, 4.0]])
    m2 = torsh.tensor([[2.0, 0.0], [1.0, 3.0]])
    
    print(f"m1 = {m1}")
    print(f"m2 = {m2}")
    print(f"m1 @ m2 = {m1 @ m2}")
    print(f"m1.transpose(0, 1) = {m1.transpose(0, 1)}")
    
    # Shape operations
    print("\n4. Shape operations:")
    
    t = torsh.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    print(f"Original tensor: {t}")
    print(f"Shape: {t.shape}")
    
    reshaped = t.reshape([4, 2])
    print(f"Reshaped [4, 2]: {reshaped}")
    
    flattened = t.flatten()
    print(f"Flattened: {flattened}")
    
    squeezed = torsh.tensor([[[1.0, 2.0, 3.0]]]).squeeze(0)
    print(f"Squeezed: {squeezed}")
    
    unsqueezed = torsh.tensor([1.0, 2.0, 3.0]).unsqueeze(0)
    print(f"Unsqueezed: {unsqueezed}")
    
    # Indexing
    print("\n5. Indexing operations:")
    
    data = torsh.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    print(f"Original: {data}")
    print(f"First row: {data[0]}")
    
    # Convert to NumPy
    print("\n6. NumPy interoperability:")
    
    torch_tensor = torsh.tensor([[1.0, 2.0], [3.0, 4.0]])
    numpy_array = torch_tensor.numpy()
    print(f"ToRSh tensor: {torch_tensor}")
    print(f"NumPy array: {numpy_array}")
    print(f"NumPy array type: {type(numpy_array)}")
    
    # Convert back to list
    python_list = torch_tensor.tolist()
    print(f"Python list: {python_list}")
    
    # Device operations (if CUDA is available)
    print("\n7. Device operations:")
    
    cpu_tensor = torsh.tensor([1.0, 2.0, 3.0])
    print(f"CPU tensor: {cpu_tensor}")
    print(f"Device: {cpu_tensor.device}")
    
    # Try CUDA (will fall back to CPU if not available)
    try:
        if torsh.cuda_is_available():
            cuda_tensor = cpu_tensor.cuda()
            print(f"CUDA tensor: {cuda_tensor}")
            print(f"Device: {cuda_tensor.device}")
            
            # Transfer back to CPU
            back_to_cpu = cuda_tensor.cpu()
            print(f"Back to CPU: {back_to_cpu}")
        else:
            print("CUDA not available")
    except Exception as e:
        print(f"CUDA operation failed: {e}")

if __name__ == "__main__":
    main()