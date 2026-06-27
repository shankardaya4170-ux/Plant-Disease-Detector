import torch
from torchvision import datasets , transforms
from torch.utils.data import DataLoader #its use data into small parts
import torch.nn as nn #network banane ke liye
import torch.nn.functional as F
import torch.optim as optim
# Image ready for rules
transform = transforms.Compose([ #save data in list
    transforms.Resize((224, 224)),#chage all image in same size
    transforms.ToTensor() # change the colour into number
])
train_data = datasets.ImageFolder(root='./data/train',transform=transform)
val_data = datasets.ImageFolder(root='./data/val',transform=transform)
print(len(train_data))
print(len(val_data))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
print(len(train_loader))
print(len(val_loader))


class PlantCNN(nn.Module):
    def __init__ (self):
        super(PlantCNN, self).__init__()

        self.conv1=nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)

        self.fc=nn.Linear(in_features=32*56*56,out_features=38)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x

model=PlantCNN()
# # --- 1. LOSS FUNCTON AUR OPTIMIZER ---
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # --- 2. TRAINING LOOP ---
# num_epochs = 2
# print("Training shuru ho rahi hai... Isme thoda time lagega, aaram se chalne dein.")
#
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, (images, labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward pass aur optimize
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#
#         if (i + 1) % 50 == 0:
#             print(f'Epoch: {epoch + 1}, Step: {i + 1}, Loss: {loss.item():.4f}')
#
# torch.save(model.state_dict(), 'plant_disease_model.pth')
# print("Model safely save ho gaya hai.")
#
# # EVAL
# print("Testing shuru ho rahi hai...")
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in val_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# accuracy = 100 * correct / total
# print(f"Test Data par Model ki Accuracy hai: {accuracy:.2f}%")