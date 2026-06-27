import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 1. Model ka Blueprint (Daancha) yahin add kar diya, taaki error na aaye
class PlantCNN(nn.Module):
    def __init__ (self):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=32*56*56, out_features=38)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 2. Model ko setup karna aur saved weights load karna
model = PlantCNN()
model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()  

# 3. Photo ko model ke layak banane ka rule
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 4. Streamlit Website ka Interface
st.title("🌿 Plant Disease Detector")
st.write("Apne patte ki photo upload karein aur bimari check karein!")

uploaded_file = st.file_uploader("Photo chunein (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Aapki Upload ki gayi photo", use_container_width=True)
    
    img_tensor = transform(image).unsqueeze(0)  
    
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        
    st.success(f"Prediction: Model ke hisaab se yeh Class Number [{predicted.item()}] hai!")
    st.success(f"Prediction: Model ke hisaab se yeh Class Number [{predicted.item()}] hai!")
