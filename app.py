import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from train_model import PlantCNN  # Aapki purani file se model ka dacha import kar rahe hain

# 1. Model ko setup karna aur saved weights load karna
model = PlantCNN()
# weights_only=True aur map_location lagana safe practice hai
model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()  # Model ko test mode mein daalna

# 2. Photo ko model ke layak banane ka rule (jo training mein tha)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. Streamlit Website ka Interface
st.title("🌿 Plant Disease Detector")
st.write("Apne patte ki photo upload karein aur bimari check karein!")

# File upload karne ka button
uploaded_file = st.file_uploader("Photo chunein (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Photo ko screen par dikhana
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Aapki Upload ki gayi photo", use_container_width=True)

    # Photo ko model ke andar bhejna
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224] banane ke liye unsqueeze(0) lagaya

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    st.success(f"Prediction: Model ke hisaab se yeh Class Number [{predicted.item()}] hai!")