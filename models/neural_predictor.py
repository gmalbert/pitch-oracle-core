"""
Neural Network Predictor for Premier League Match Outcomes

Deep learning approach using PyTorch for complex pattern recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np


class FootballNet(nn.Module):
    """Neural network for football match prediction"""

    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super(FootballNet, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 3))  # 3 output classes
        layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_neural_model(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    """Train neural network model"""

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y_train.values)

    # Create model
    model = FootballNet(input_size=X_scaled.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    return model, scaler


def predict_neural(model, scaler, X_new):
    """Make predictions with neural network"""
    model.eval()
    with torch.no_grad():
        X_scaled = scaler.transform(X_new)
        X_tensor = torch.FloatTensor(X_scaled)
        predictions = model(X_tensor)
        return predictions.numpy()


def create_simple_neural_model():
    """Create a simpler neural network for faster training/testing"""
    # This will be created dynamically based on input size
    return None