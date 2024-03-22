import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn.init as init

class ActivationFunction(nn.Module):
    def __init__(self, activation_function, **kwargs):
        super().__init__()
        self.activation_function = activation_function
        self.kwargs = kwargs

    def forward(self, x):
        return self.activation_function(x, **self.kwargs)

def xavier_init(m):
    if type(m) == nn.Linear:
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def kaiming_init(m):
    if type(m) == nn.Linear:
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)

class SimpleNN(nn.Module):
    def __init__(self, dropout_rate=0.19, n_neurons=16, activation_function=F.leaky_relu, num_hidden_layers=3,
                 **activation_kwargs):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(13, n_neurons), ActivationFunction(activation_function, **activation_kwargs)]

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(ActivationFunction(activation_function, **activation_kwargs))

        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(n_neurons, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Load the data
df = pd.read_csv('simtrain5.txt', sep='\s+')

# Split the data into features and target
X = df.iloc[:, 1:].values  # x1 to x13
y = df.iloc[:, 0].values  # y

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape y to be a column vector

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5])
    n_neurons = trial.suggest_categorical('n_neurons', [16, 32, 64])
    num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [1, 2, 3])
    activation_name = trial.suggest_categorical('activation_function', ['relu', 'leaky_relu', 'elu'])
    weight_init = trial.suggest_categorical('weight_init', ['None', 'kaiming_uniform'])
    # Map the activation function names to actual PyTorch functions
    activations = {
        'relu': F.relu,
        'leaky_relu': F.leaky_relu,
        'elu': F.elu,
    }
    activation_function = activations[activation_name]

    model = SimpleNN(dropout_rate=dropout_rate, n_neurons=n_neurons, activation_function=activation_function, num_hidden_layers=num_hidden_layers)
    if(weight_init == 'kaiming_uniform'):
        model.apply(kaiming_init)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    num_epochs = 200
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()


        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update the best validation loss if the current one is lower

    return best_val_loss    # Use optuna to optimize this


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best trial: {study.best_trial.value}")
print(f"Best parameters: {study.best_params}")
