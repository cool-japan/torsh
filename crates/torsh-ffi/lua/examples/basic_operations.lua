#!/usr/bin/env lua

-- Basic ToRSh Tensor Operations Example
-- This script demonstrates fundamental tensor operations in Lua

local torsh = require('torsh')

print("ToRSh Lua Examples - Basic Operations")
print("=====================================")

-- 1. Creating tensors
print("\n1. Creating Tensors:")

-- From nested tables
local a = torsh.tensor({{1, 2, 3}, {4, 5, 6}})
print("Created tensor a from nested table:")
torsh.print_tensor(a, "a")

-- Special tensors
local zeros = torsh.zeros(3, 3)
local ones = torsh.ones(2, 4)
local randn = torsh.randn(2, 3)

print("\nSpecial tensors:")
torsh.print_tensor(zeros, "zeros")
torsh.print_tensor(ones, "ones")
torsh.print_tensor(randn, "randn")

-- 2. Basic arithmetic
print("\n2. Basic Arithmetic:")

local b = torsh.tensor({{7, 8, 9}, {10, 11, 12}})
torsh.print_tensor(b, "b")

-- Element-wise operations
local c = a + b
local d = a - b
local e = a * b
local f = a / b

print("\nArithmetic results:")
torsh.print_tensor(c, "a + b")
torsh.print_tensor(d, "a - b")
torsh.print_tensor(e, "a * b (element-wise)")
torsh.print_tensor(f, "a / b")

-- 3. Matrix operations
print("\n3. Matrix Operations:")

-- Create square matrices for matrix multiplication
local m1 = torsh.tensor({{1, 2}, {3, 4}})
local m2 = torsh.tensor({{5, 6}, {7, 8}})

torsh.print_tensor(m1, "m1")
torsh.print_tensor(m2, "m2")

local matmul_result = torsh.matmul(m1, m2)
torsh.print_tensor(matmul_result, "matmul(m1, m2)")

-- Transpose
local transposed = m1:transpose()
torsh.print_tensor(transposed, "m1 transposed")

-- 4. Shape operations
print("\n4. Shape Operations:")

local x = torsh.randn(2, 3, 4)
torsh.print_tensor(x, "original x")

-- Reshape
local reshaped = x:reshape(6, 4)
torsh.print_tensor(reshaped, "x reshaped to 6x4")

local flattened = x:reshape(24)
torsh.print_tensor(flattened, "x flattened")

-- 5. Reduction operations
print("\n5. Reduction Operations:")

local data = torsh.tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})
torsh.print_tensor(data, "data")

local sum_all = data:sum()
local sum_rows = data:sum(0)  -- Sum along rows
local sum_cols = data:sum(1)  -- Sum along columns

torsh.print_tensor(sum_all, "sum all")
torsh.print_tensor(sum_rows, "sum rows")
torsh.print_tensor(sum_cols, "sum cols")

local mean_all = data:mean()
local mean_rows = data:mean(0)
local mean_cols = data:mean(1)

torsh.print_tensor(mean_all, "mean all")
torsh.print_tensor(mean_rows, "mean rows")
torsh.print_tensor(mean_cols, "mean cols")

-- 6. Scalar operations
print("\n6. Scalar Operations:")

local s = torsh.tensor({{1, -2, 3}, {-4, 5, -6}})
torsh.print_tensor(s, "original")

local s_plus_10 = s + 10
local s_times_2 = s * 2
local s_div_3 = s / 3

torsh.print_tensor(s_plus_10, "s + 10")
torsh.print_tensor(s_times_2, "s * 2")
torsh.print_tensor(s_div_3, "s / 3")

-- 7. Activation functions
print("\n7. Activation Functions:")

local activations = torsh.tensor({{-2, -1, 0, 1, 2}, {-0.5, 0, 0.5, 1.5, 2.5}})
torsh.print_tensor(activations, "input")

local relu = activations:relu()
local sigmoid = activations:sigmoid()
local tanh_result = activations:tanh()

torsh.print_tensor(relu, "ReLU")
torsh.print_tensor(sigmoid, "Sigmoid")
torsh.print_tensor(tanh_result, "Tanh")

-- Softmax (applied to each row)
local softmax_result = activations:softmax(-1)
torsh.print_tensor(softmax_result, "Softmax")

-- 8. Indexing and slicing (if implemented)
print("\n8. Tensor Information:")

local info_tensor = torsh.randn(3, 4, 5)
print("Tensor shape:", table.concat(info_tensor:shape(), ", "))
print("Number of elements:", info_tensor:numel())
print("Number of dimensions:", info_tensor:ndim())

-- 9. Random number generation
print("\n9. Random Number Generation:")

-- Set seed for reproducibility
torsh.manual_seed(42)

local random1 = torsh.randn(2, 3)
local random2 = torsh.randn(2, 3)

torsh.print_tensor(random1, "random1 (seed=42)")

-- Reset seed
torsh.manual_seed(42)
local random3 = torsh.randn(2, 3)
torsh.print_tensor(random3, "random3 (seed=42 again)")

print("\nrandom1 and random3 should be identical!")

-- 10. Linear algebra utilities
print("\n10. Linear Algebra:")

-- Identity matrix
local eye = torsh.eye(4)
torsh.print_tensor(eye, "4x4 identity")

-- Linear space
local linspace = torsh.linspace(0, 10, 11)
torsh.print_tensor(linspace, "linspace(0, 10, 11)")

-- Range
local range = torsh.arange(0, 10, 2)
torsh.print_tensor(range, "arange(0, 10, 2)")

print("\nBasic operations example completed!")
print("All operations executed successfully.")