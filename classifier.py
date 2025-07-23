import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

#import for image proc
import torchvision
from torchvision import transforms
from PIL import Image

print("Pytorch Output...")

#load FashionMNIST
DATA_DIR = "."
train = datasets.FashionMNIST(DATA_DIR, train=True, download=False)
test = datasets.FashionMNIST(DATA_DIR, train=False, download=False)

#variables for data
X_train = train.data.float()
y_train = train.targets
X_test = test.data.float()
y_test = test.targets

#class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#split training into validation and training
test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

#validation set
X_valid = X_train[indices]
y_valid = y_train[indices]

#remove validation set from training set
X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

#reshape data
X_train = X_train.view(-1, 28*28)
X_valid = X_valid.view(-1, 28*28)
X_test = X_test.view(-1, 28*28)

#prepare datasets and loaders
batch_size = 64

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#define Feedforward Neural Network (ANN) model
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu5 = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.fc(out)
        return out

#hyperparameters
input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.001
num_epochs = 30

#model and optimiser
model = FeedforwardNeuralNetModel(input_size, hidden_size, output_size)
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

#####################
##TRAINING LOOP
#####################

total_step = len(train_loader)

#validation accuracy early stopping parameters AND output parameters
patience = 5
train_losses = []
val_losses = []
val_accuracies = []
best_val_accuracy = 0
patience_counter = 0
best_model_state = None

#train
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.float()
        labels = labels.long()
        
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        #output for loss per epoch and step
        if (i+1) % 100 == 0 or (i+1) == total_step:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    avg_train_loss = running_train_loss / total_step
    train_losses.append(avg_train_loss)

    #validation loss and accuracy
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.float()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    #output for train and validation loss and validation accuracy
    print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    #early stopping on val accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        #save best model weights
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        #no improvement
        patience_counter += 1

    if patience_counter >= patience:
        #output if early stopping occurs
        print(f"Early stopping triggered (val accuracy) at epoch {epoch+1}")
        model.load_state_dict(best_model_state)
        break

#load best
if best_model_state is not None:
    model.load_state_dict(best_model_state)

#plot validation and training loss graph
"""
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()
"""

# Evaluate on test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.float()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

#printing accuracies and done message
    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
print("Done!\n")

ans = ""

#get input from user until exit is typed
while ans != "exit":
    ans = input("Please enter a filepath:\n")

    #process image
    try:
        #load grayscale image
        img = torchvision.io.read_image(ans, mode=torchvision.io.ImageReadMode.GRAY)
        img = img.squeeze().float()

        #check if right size, if not --> resize
        if img.shape != (28, 28):
            img = torchvision.transforms.functional.resize(img, [28, 28])

        #flatten and convert to float
        img = img.view(-1, 28 * 28).float()

        #get label through model
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            predicted_label = predicted.item()

        print(f"Classifier: {class_names[predicted_label]}\n")

    except Exception as e:
        print(f"error: {e}")

print("Exiting...")