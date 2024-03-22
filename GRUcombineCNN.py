import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split

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


# Define a simple neural network for regression
class CombinedGRUCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=13, hidden_size=64, batch_first=True)  # Adjust for input size
        self.conv1d = nn.Conv1d(in_channels=13, out_channels=32,
                                kernel_size=1)  # Treat each feature as a separate channel
        self.fc1 = nn.Linear(64 + 32, 50)  # Combining GRU and CNN outputs
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x_gru, _ = self.gru(x.unsqueeze(1))  # GRU expects 3D input
        x_gru = x_gru[:, -1, :]  # Use the output of the last time step

        x_cnn = self.conv1d(x.unsqueeze(2))  # CNN expects 3D input with channels
        x_cnn = torch.relu(x_cnn)
        x_cnn = torch.flatten(x_cnn, 1)  # Flatten the CNN outputs

        x_combined = torch.cat((x_gru, x_cnn), dim=1)
        x_combined = torch.relu(self.fc1(x_combined))
        return self.fc2(x_combined)


model = CombinedGRUCNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Prepare DataLoader
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 200
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

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
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_CombinedGRUCNN_model.pth')  # Save the best model

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

print("Training complete")





#############################################################   Do The Test   #############################################################
#############################################################   Do The Test   #############################################################
#############################################################   Do The Test   #############################################################


# Assuming simtest5.txt is your test dataset with the same format
df_test = pd.read_csv('simtest5.txt', sep='\s+')
X_test = df_test.iloc[:, 1:].values
y_test = df_test.iloc[:, 0].values

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CombinedGRUCNNModel()
model.load_state_dict(torch.load('best_CombinedGRUCNN_model.pth'))
model.eval()  # Set the model to evaluation mode

predictions = []
actuals = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        predictions.extend(outputs.view(-1).tolist())
        actuals.extend(batch_labels.view(-1).tolist())

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(actuals, predictions)
r_square = r2_score(actuals, predictions)

print(f'MSE: {mse}')
print(f'R-squared: {r_square}')

# MSE: 0.00039548758578931016
# R-squared: 0.7652661631334965