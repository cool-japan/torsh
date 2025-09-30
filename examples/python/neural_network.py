#!/usr/bin/env python3
"""
Example demonstrating neural network training with ToRSh Python bindings.

This script shows how to create, train, and evaluate a simple neural network
using ToRSh in a PyTorch-compatible way.
"""

import sys
import os

# Add the python package to the path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import torsh
import torsh.nn as nn
import torsh.optim as optim
import torsh.nn.functional as F
import numpy as np

class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

def generate_data(num_samples=1000, input_size=10):
    """Generate synthetic regression data."""
    # Generate random input data
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    
    # Generate target as a simple linear combination with some noise
    weights = np.random.randn(input_size, 1).astype(np.float32)
    y = X @ weights + 0.1 * np.random.randn(num_samples, 1).astype(np.float32)
    
    return X, y

def main():
    print("ToRSh Python Bindings - Neural Network Training Example")
    print("=" * 60)
    
    # Set random seed for reproducibility (if available)
    # torch.manual_seed(42)
    # np.random.seed(42)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    input_size = 10
    num_samples = 1000
    X_np, y_np = generate_data(num_samples, input_size)
    
    print(f"Input shape: {X_np.shape}")
    print(f"Target shape: {y_np.shape}")
    
    # Convert to ToRSh tensors
    X = torsh.tensor(X_np, requires_grad=False)
    y = torsh.tensor(y_np, requires_grad=False)
    
    print(f"Input tensor: {X.shape}, dtype: {X.dtype}")
    print(f"Target tensor: {y.shape}, dtype: {y.dtype}")
    
    # Create model
    print("\n2. Creating neural network model...")
    model = SimpleNet(input_size=input_size, hidden_size=20, output_size=1)
    print(f"Model: {model}")
    
    # Print model parameters
    print("\n3. Model parameters:")
    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    print(f"Total parameters: {total_params}")
    
    for i, param in enumerate(params):
        print(f"Parameter {i}: shape {param.shape}, numel {param.numel()}")
    
    # Create optimizer
    print("\n4. Setting up optimizer...")
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print(f"Optimizer: {optimizer}")
    
    # Training loop
    print("\n5. Training the model...")
    num_epochs = 100
    batch_size = 32
    num_batches = len(X) // batch_size
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # Simple batch processing (no DataLoader for now)
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            batch_X = X[i:end_idx]
            batch_y = y[i:end_idx]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X)
            
            # Compute loss
            loss = F.mse_loss(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
    
    # Evaluation
    print("\n6. Evaluating the model...")
    model.eval()
    
    with torsh.no_grad():
        # Make predictions on the full dataset
        test_predictions = model(X)
        test_loss = F.mse_loss(test_predictions, y)
        
        print(f"Final test loss: {test_loss.item():.6f}")
        
        # Show some example predictions
        print("\n7. Example predictions:")
        print("Target vs Prediction (first 10 samples):")
        
        for i in range(min(10, len(y))):
            target_val = y[i].item()
            pred_val = test_predictions[i].item()
            print(f"Sample {i+1}: Target = {target_val:.4f}, Prediction = {pred_val:.4f}")
        
        # Compute some basic metrics
        residuals = test_predictions - y
        mae = torsh.mean(torsh.abs(residuals))
        rmse = torsh.sqrt(test_loss)
        
        print(f"\nMetrics:")
        print(f"Mean Absolute Error: {mae.item():.6f}")
        print(f"Root Mean Square Error: {rmse.item():.6f}")
    
    # Model serialization (if available)
    print("\n8. Model serialization:")
    try:
        state_dict = model.state_dict()
        print(f"Model state dict keys: {list(state_dict.keys())}")
        print("Model can be serialized (state_dict available)")
    except Exception as e:
        print(f"Model serialization not yet implemented: {e}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()