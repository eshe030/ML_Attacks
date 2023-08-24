import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Simulated XOR Arbiter PUF
def xor_arbiter_puf(challenge):
    return challenge[0] ^ challenge[1]

# Generate artificial PUF responses with noise
def generate_artificial_puf_responses(challenge_pairs):
    responses = []
    for challenge in challenge_pairs:
        response = xor_arbiter_puf(challenge)
        if np.random.rand() < 0.1:  # 10% chance of flipping response
            response = 1 - response
        responses.append(response)
    return np.array(responses)

# Create training data
num_samples = 1000
challenge_pairs = np.random.randint(0, 2, size=(num_samples, 2))
responses = generate_artificial_puf_responses(challenge_pairs)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(challenge_pairs, responses, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Input layer: 2 input features, 16 hidden units
        self.fc2 = nn.Linear(16, 1)  # Output layer: 1 output neuron (PUF response)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize model, loss function, and optimizer
model = MLP()
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    loss.backward()
    optimizer.step()

# Evaluate model
model.eval()
with torch.no_grad():
    predicted_responses = model(X_test_tensor)
    predicted_responses = (predicted_responses >= 0.5).squeeze().numpy()
    attack_accuracy = np.mean(predicted_responses == y_test)

print("Attack Accuracy:", attack_accuracy)
