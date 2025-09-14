import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import BoneFractureDataset
from model import BoneCNN  
from loss import FocalLoss 

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

# Class weights (for imbalance handling)
class_counts = [0, 0]  # [fracture, normal]
for _, label in train_dataset:
    class_counts[label] += 1
total = sum(class_counts)
weights = [total / c for c in class_counts]
weights = torch.tensor(weights).to(device)

# Training Function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5):
    best_val_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
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

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)
        val_acc = 100 * val_correct / val_total

        print(f"[{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best model saved with Val Acc: {best_val_acc:.2f}%")

    return best_val_acc, train_losses, val_losses

# Run Training - CrossEntropyLoss
print("\n================ Training with CrossEntropyLoss ================\n")
model_ce = BoneCNN(num_classes=2).to(device)
criterion_ce = nn.CrossEntropyLoss(weight=weights)
optimizer_ce = optim.Adam(model_ce.parameters(), lr=0.001)

best_val_acc_ce, train_losses_ce, val_losses_ce = train_model(model_ce, train_loader, valid_loader, criterion_ce, optimizer_ce)

# Run Training - FocalLoss
print("\n================ Training with FocalLoss ================\n")
model_focal = BoneCNN(num_classes=2).to(device)
criterion_focal = FocalLoss(alpha=1, gamma=2)
optimizer_focal = optim.Adam(model_focal.parameters(), lr=0.001)

best_val_acc_focal, train_losses_focal, val_losses_focal = train_model(model_focal, train_loader, valid_loader, criterion_focal, optimizer_focal)

# Final Comparison
print("\n================ Final Results ================\n")
print(f"Best Validation Accuracy (CrossEntropyLoss): {best_val_acc_ce:.2f}%")
print(f"Best Validation Accuracy (FocalLoss):        {best_val_acc_focal:.2f}%")

# Plot Loss Curves
plt.plot(train_losses_ce, label="Train CE Loss")
plt.plot(val_losses_ce, label="Val CE Loss")
plt.plot(train_losses_focal, label="Train Focal Loss")
plt.plot(val_losses_focal, label="Val Focal Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence of Loss Functions")
plt.legend()
plt.show()
