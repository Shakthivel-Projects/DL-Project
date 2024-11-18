import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Define the model
class HybridConvResBlock(nn.Module):
    # Define layers
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(HybridConvResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = self.residual(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)

class FusionResNet(nn.Module):
    # Construct the full network with Hybrid ConvRes Blocks
    def __init__(self, num_classes, dropout_rate=0.3):
        super(FusionResNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm2d(64)
        self.layer1 = HybridConvResBlock(64, 128, stride=2, dropout_rate=dropout_rate)
        self.layer2 = HybridConvResBlock(128, 256, stride=2, dropout_rate=dropout_rate)
        self.layer3 = HybridConvResBlock(256, 512, stride=2, dropout_rate=dropout_rate)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.initial_bn(self.initial_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=1)

# Custom Dataset and Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root=r'C:\Project\Dataset\train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
test_data = datasets.ImageFolder(root=r'C:\Project\Dataset\test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

# Initialize model, loss, and optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = len(train_data.classes)
model = FusionResNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
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
torch.save(model.state_dict(), r'C:\Project\fusionresnet_cctv.pth')
