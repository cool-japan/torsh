-- ToRSh Lua Module
-- High-level Lua interface for ToRSh tensors

local torsh = require('torsh_core')  -- This would be the compiled C module

-- Extend the core module with additional Lua functionality
local M = {}

-- Copy core functions
for k, v in pairs(torsh) do
    M[k] = v
end

-- Tensor class with metatable
local Tensor = {}
Tensor.__index = Tensor

-- Metamethods for tensor operations
function Tensor:__add(other)
    if type(other) == "number" then
        return self:add_scalar(other)
    else
        return torsh.add(self, other)
    end
end

function Tensor:__sub(other)
    if type(other) == "number" then
        return self:sub_scalar(other)
    else
        return torsh.sub(self, other)
    end
end

function Tensor:__mul(other)
    if type(other) == "number" then
        return self:mul_scalar(other)
    else
        return torsh.mul(self, other)
    end
end

function Tensor:__div(other)
    if type(other) == "number" then
        return self:div_scalar(other)
    else
        return torsh.div(self, other)
    end
end

function Tensor:__tostring()
    local shape = self:shape()
    local shape_str = table.concat(shape, "x")
    return string.format("Tensor(%s)", shape_str)
end

-- Instance methods
function Tensor:size(dim)
    local shape = self:shape()
    if dim then
        return shape[dim + 1]  -- Lua is 1-indexed
    else
        return shape
    end
end

function Tensor:numel()
    local shape = self:shape()
    local count = 1
    for _, dim in ipairs(shape) do
        count = count * dim
    end
    return count
end

function Tensor:ndim()
    return #self:shape()
end

function Tensor:reshape(...)
    local dims = {...}
    return torsh.reshape(self, dims)
end

function Tensor:transpose(dim1, dim2)
    dim1 = dim1 or 0
    dim2 = dim2 or 1
    return torsh.transpose(self, dim1, dim2)
end

function Tensor:sum(dim, keepdim)
    if dim then
        return torsh.sum_dim(self, dim, keepdim or false)
    else
        return torsh.sum_all(self)
    end
end

function Tensor:mean(dim, keepdim)
    if dim then
        return torsh.mean_dim(self, dim, keepdim or false)
    else
        return torsh.mean_all(self)
    end
end

-- Activation functions
function Tensor:relu()
    return torsh.relu(self)
end

function Tensor:sigmoid()
    return torsh.sigmoid(self)
end

function Tensor:tanh()
    return torsh.tanh(self)
end

function Tensor:softmax(dim)
    dim = dim or -1
    return torsh.softmax(self, dim)
end

-- Enhanced creation functions
function M.tensor(data)
    local t = torsh.tensor(data)
    return setmetatable(t, Tensor)
end

function M.zeros(...)
    local t = torsh.zeros(...)
    return setmetatable(t, Tensor)
end

function M.ones(...)
    local t = torsh.ones(...)
    return setmetatable(t, Tensor)
end

function M.randn(...)
    local t = torsh.randn(...)
    return setmetatable(t, Tensor)
end

function M.eye(n, m)
    m = m or n
    local t = torsh.eye(n, m)
    return setmetatable(t, Tensor)
end

function M.linspace(start, stop, steps)
    steps = steps or 50
    local t = torsh.linspace(start, stop, steps)
    return setmetatable(t, Tensor)
end

function M.arange(start, stop, step)
    step = step or 1
    local t = torsh.arange(start, stop, step)
    return setmetatable(t, Tensor)
end

-- Linear algebra functions
function M.matmul(a, b)
    local t = torsh.matmul(a, b)
    return setmetatable(t, Tensor)
end

function M.dot(a, b)
    return M.matmul(a, b)
end

-- Statistical functions
function M.cat(tensors, dim)
    dim = dim or 0
    local t = torsh.cat(tensors, dim)
    return setmetatable(t, Tensor)
end

function M.stack(tensors, dim)
    dim = dim or 0
    local t = torsh.stack(tensors, dim)
    return setmetatable(t, Tensor)
end

-- Utility functions
function M.save_tensor(tensor, filename)
    return torsh.save(tensor, filename)
end

function M.load_tensor(filename)
    local t = torsh.load(filename)
    return setmetatable(t, Tensor)
end

-- Neural network utilities
function M.linear(input, weight, bias)
    local output = M.matmul(input, weight)
    if bias then
        output = output + bias
    end
    return output
end

function M.conv2d(input, weight, bias, stride, padding)
    stride = stride or 1
    padding = padding or 0
    local t = torsh.conv2d(input, weight, bias, stride, padding)
    return setmetatable(t, Tensor)
end

-- Loss functions
function M.mse_loss(prediction, target)
    local diff = prediction - target
    return (diff * diff):mean()
end

function M.cross_entropy_loss(prediction, target)
    local log_softmax = prediction:softmax(-1):log()
    return -(log_softmax * target):sum(-1):mean()
end

-- Optimizer functions
function M.sgd_step(params, grads, lr)
    lr = lr or 0.01
    for i, param in ipairs(params) do
        local grad = grads[i]
        param:sub_(grad * lr)
    end
end

function M.adam_step(params, grads, m, v, lr, beta1, beta2, eps, step)
    lr = lr or 0.001
    beta1 = beta1 or 0.9
    beta2 = beta2 or 0.999
    eps = eps or 1e-8
    
    for i, param in ipairs(params) do
        local grad = grads[i]
        
        -- Update biased first moment estimate
        m[i] = m[i] * beta1 + grad * (1 - beta1)
        
        -- Update biased second raw moment estimate
        v[i] = v[i] * beta2 + (grad * grad) * (1 - beta2)
        
        -- Compute bias-corrected first moment estimate
        local m_hat = m[i] / (1 - beta1^step)
        
        -- Compute bias-corrected second raw moment estimate
        local v_hat = v[i] / (1 - beta2^step)
        
        -- Update parameters
        param:sub_(m_hat / (v_hat:sqrt() + eps) * lr)
    end
end

-- Pretty printing for tensors
function M.print_tensor(tensor, name)
    name = name or "Tensor"
    print(string.format("%s: %s", name, tostring(tensor)))
    
    local data = tensor:data()
    local shape = tensor:shape()
    
    if tensor:numel() <= 20 then
        local str_data = {}
        for i, v in ipairs(data) do
            table.insert(str_data, string.format("%.4f", v))
        end
        print("Data: [" .. table.concat(str_data, ", ") .. "]")
    else
        print("(data not displayed for large tensors)")
    end
end

-- Device management
function M.to_device(tensor, device)
    device = device or "cpu"
    return torsh.to_device(tensor, device)
end

function M.cuda_available()
    return torsh.cuda_available() or false
end

-- Random seed
function M.manual_seed(seed)
    return torsh.manual_seed(seed)
end

-- Tensor information
function M.dtype(tensor)
    return torsh.dtype(tensor) or "float32"
end

function M.device(tensor)
    return torsh.device(tensor) or "cpu"
end

-- Export module
return M