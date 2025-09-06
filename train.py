import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BoneFractureDataset
from model import BoneCNN  # your custom CNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Datasets
train_dataset = BoneFractureDataset(
    "/Users/rigelrai/Documents/Bennett/Year 2/Python/fracture/HBFMID/Bone Fractures Detection/train",
    transform=train_transforms
)

valid_dataset = BoneFractureDataset(
    "/Users/rigelrai/Documents/Bennett/Year 2/Python/fracture/HBFMID/Bone Fractures Detection/test",
    transform=train_transforms
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(valid_loader)}")

# Model
model = BoneCNN(num_classes=2).to(device)

# Class weights
class_counts = [0, 0]  # [fracture, normal]
for _, label in train_dataset:
    class_counts[label] += 1
total = sum(class_counts)
weights = [total/c for c in class_counts]
weights = torch.tensor(weights).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total * 100

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total * 100

    print(f"Epoch [{epoch+1}/{num_epochs}] Completed: "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Valid Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%\n")
