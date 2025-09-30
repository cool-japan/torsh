#!/usr/bin/env lua

-- Neural Network Example with ToRSh in Lua
-- This demonstrates building and training a simple neural network

local torsh = require('torsh')

print("ToRSh Lua Examples - Neural Network")
print("==================================")

-- Set random seed for reproducibility
torsh.manual_seed(42)

-- 1. Generate synthetic dataset
print("\n1. Generating Synthetic Dataset:")

local function generate_spiral_data(samples_per_class, classes)
    local X = {}
    local y = {}
    
    for class_num = 0, classes - 1 do
        local r = {}
        local t = {}
        
        -- Generate spiral data
        for i = 1, samples_per_class do
            local idx = (i - 1) / samples_per_class
            local radius = idx
            local theta = class_num * 4 + idx * 4 + math.random() * 0.2
            
            table.insert(r, radius)
            table.insert(t, theta)
            
            local x = radius * math.cos(theta)
            local y_coord = radius * math.sin(theta)
            
            table.insert(X, {x, y_coord})
            table.insert(y, class_num)
        end
    end
    
    return X, y
end

local X_data, y_data = generate_spiral_data(100, 3)
local X = torsh.tensor(X_data)
local y = torsh.tensor(y_data)

print(string.format("Generated dataset: X shape %s, y shape %s", 
    table.concat(X:shape(), "x"), 
    table.concat(y:shape(), "x")))

-- 2. Create one-hot encoded labels
print("\n2. Creating One-Hot Encoded Labels:")

local function create_one_hot(labels, num_classes)
    local batch_size = labels:numel()
    local one_hot = torsh.zeros(batch_size, num_classes)
    
    local labels_data = labels:data()
    local one_hot_data = one_hot:data()
    
    for i = 1, batch_size do
        local class_idx = labels_data[i]
        one_hot_data[(i-1) * num_classes + class_idx + 1] = 1.0
    end
    
    return one_hot
end

local num_classes = 3
local y_one_hot = create_one_hot(y, num_classes)
print(string.format("One-hot labels shape: %s", table.concat(y_one_hot:shape(), "x")))

-- 3. Define network architecture
print("\n3. Defining Network Architecture:")

local function init_layer(input_size, output_size)
    -- Xavier/Glorot initialization
    local scale = math.sqrt(2.0 / (input_size + output_size))
    local weight = torsh.randn(input_size, output_size) * scale
    local bias = torsh.zeros(output_size)
    return weight, bias
end

-- Network: 2 -> 128 -> 64 -> 3
local input_size = 2
local hidden1_size = 128
local hidden2_size = 64
local output_size = 3

local W1, b1 = init_layer(input_size, hidden1_size)
local W2, b2 = init_layer(hidden1_size, hidden2_size)
local W3, b3 = init_layer(hidden2_size, output_size)

print("Network architecture: 2 -> 128 -> 64 -> 3")
print(string.format("W1: %s, b1: %s", table.concat(W1:shape(), "x"), table.concat(b1:shape(), "x")))
print(string.format("W2: %s, b2: %s", table.concat(W2:shape(), "x"), table.concat(b2:shape(), "x")))
print(string.format("W3: %s, b3: %s", table.concat(W3:shape(), "x"), table.concat(b3:shape(), "x")))

-- 4. Forward pass function
local function forward(x)
    -- Layer 1: Linear + ReLU
    local z1 = torsh.matmul(x, W1) + b1
    local a1 = z1:relu()
    
    -- Layer 2: Linear + ReLU
    local z2 = torsh.matmul(a1, W2) + b2
    local a2 = z2:relu()
    
    -- Layer 3: Linear + Softmax
    local z3 = torsh.matmul(a2, W3) + b3
    local output = z3:softmax(-1)
    
    return output, {z1, a1, z2, a2, z3}
end

-- 5. Loss function
local function cross_entropy_loss(predictions, targets)
    -- Add small epsilon to prevent log(0)
    local eps = 1e-7
    local log_probs = (predictions + eps):log()
    local loss = -(targets * log_probs):sum(-1):mean()
    return loss
end

-- 6. Simple numerical gradient computation (for demonstration)
local function compute_numerical_gradient(params, loss_fn, h)
    h = h or 1e-5
    local grads = {}
    
    for i, param in ipairs(params) do
        local param_data = param:data()
        local grad_data = {}
        
        for j = 1, #param_data do
            -- Forward difference
            param_data[j] = param_data[j] + h
            local loss_plus = loss_fn()
            
            param_data[j] = param_data[j] - 2 * h
            local loss_minus = loss_fn()
            
            -- Restore original value
            param_data[j] = param_data[j] + h
            
            -- Compute gradient
            local grad = (loss_plus:data()[1] - loss_minus:data()[1]) / (2 * h)
            table.insert(grad_data, grad)
        end
        
        table.insert(grads, torsh.tensor(grad_data):reshape(param:shape()))
    end
    
    return grads
end

-- 7. Training loop
print("\n4. Training the Network:")

local learning_rate = 0.01
local epochs = 100
local print_every = 20

local params = {W1, b1, W2, b2, W3, b3}

for epoch = 1, epochs do
    -- Forward pass
    local predictions, _ = forward(X)
    local loss = cross_entropy_loss(predictions, y_one_hot)
    
    -- Compute gradients (simplified numerical gradients)
    local function loss_fn()
        local pred, _ = forward(X)
        return cross_entropy_loss(pred, y_one_hot)
    end
    
    -- For demonstration, we'll use a simplified gradient descent
    -- In practice, you'd want proper backpropagation
    if epoch % print_every == 0 then
        print(string.format("Epoch %d, Loss: %.6f", epoch, loss:data()[1]))
        
        -- Compute accuracy
        local pred_data = predictions:data()
        local y_data = y:data()
        local correct = 0
        local total = y:numel()
        
        for i = 1, total do
            local pred_class = 0
            local max_prob = pred_data[(i-1) * num_classes + 1]
            
            for j = 2, num_classes do
                if pred_data[(i-1) * num_classes + j] > max_prob then
                    max_prob = pred_data[(i-1) * num_classes + j]
                    pred_class = j - 1
                end
            end
            
            if pred_class == y_data[i] then
                correct = correct + 1
            end
        end
        
        local accuracy = correct / total * 100
        print(string.format("Accuracy: %.2f%%", accuracy))
    end
    
    -- Simple parameter update (replace with proper backprop in real implementation)
    if epoch <= 10 then  -- Only do expensive numerical gradients for first few epochs
        local grads = compute_numerical_gradient(params, loss_fn)
        
        -- Update parameters
        for i, param in ipairs(params) do
            local param_data = param:data()
            local grad_data = grads[i]:data()
            
            for j = 1, #param_data do
                param_data[j] = param_data[j] - learning_rate * grad_data[j]
            end
        end
    end
end

-- 8. Evaluation on test data
print("\n5. Final Evaluation:")

-- Generate test data
local X_test_data, y_test_data = generate_spiral_data(50, 3)
local X_test = torsh.tensor(X_test_data)
local y_test = torsh.tensor(y_test_data)

local test_predictions, _ = forward(X_test)
local test_pred_data = test_predictions:data()
local test_y_data = y_test:data()

local test_correct = 0
local test_total = y_test:numel()

for i = 1, test_total do
    local pred_class = 0
    local max_prob = test_pred_data[(i-1) * num_classes + 1]
    
    for j = 2, num_classes do
        if test_pred_data[(i-1) * num_classes + j] > max_prob then
            max_prob = test_pred_data[(i-1) * num_classes + j]
            pred_class = j - 1
        end
    end
    
    if pred_class == test_y_data[i] then
        test_correct = test_correct + 1
    end
end

local test_accuracy = test_correct / test_total * 100
print(string.format("Test Accuracy: %.2f%%", test_accuracy))

-- 9. Show some predictions
print("\n6. Sample Predictions:")

for i = 1, math.min(10, test_total) do
    local input_x = test_pred_data[(i-1) * num_classes + 1]
    local input_y = test_pred_data[(i-1) * num_classes + 2]
    local true_class = test_y_data[i]
    
    local pred_probs = {}
    for j = 1, num_classes do
        table.insert(pred_probs, test_pred_data[(i-1) * num_classes + j])
    end
    
    local pred_class = 0
    local max_prob = pred_probs[1]
    for j = 2, num_classes do
        if pred_probs[j] > max_prob then
            max_prob = pred_probs[j]
            pred_class = j - 1
        end
    end
    
    print(string.format("Input: (%.2f, %.2f), True: %d, Pred: %d, Confidence: %.3f", 
        input_x, input_y, true_class, pred_class, max_prob))
end

print("\nNeural network training completed!")
print("Note: This example uses simplified numerical gradients for demonstration.")
print("In practice, you would implement proper backpropagation for efficiency.")