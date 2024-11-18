import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Custom Dataset and Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root=r'C:\Project\Dataset\train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data = datasets.ImageFolder(root=r'C:\Project\Dataset\test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_data.classes)

model = models.resnet50(pretrained=True)  

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples, correct_samples = 0, 0, 0
    all_preds, all_labels = [], []
    start_time = time.time()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_samples += (preds == labels).sum().item()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Metrics Calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    fps = total_samples / (time.time() - start_time)  # Frames per second

    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, precision, recall, f1, fps

# Evaluation Loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_samples, correct_samples = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_samples += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics Calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, precision, recall, f1

# Full Training Process
epochs = 10
start_time = time.time()
for epoch in range(epochs):
    train_loss, train_acc, train_prec, train_rec, train_f1, fps = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}, FPS: {fps:.2f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

total_runtime = time.time() - start_time
print(f"Total Training Runtime: {total_runtime:.2f} seconds")

# Save model
torch.save(model.state_dict(), r'C:\Project\resnet50_cctv.pth')
