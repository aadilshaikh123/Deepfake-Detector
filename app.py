import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import asyncio
import sys


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load the trained model (only once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = r"model\best_model_epoch_4.pth" # Replace with your actual checkpoint path

# Load state dict and remove "module." prefix if necessary
state_dict = torch.load(checkpoint_path, map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k  # remove module. prefix
    new_state_dict[name] = v

# Load the modified state dict
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

# Define the transformation pipeline (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define a function to classify an image as Fake or Real with confidence
def classify_image(image):
    # Apply the transformations
    img_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
    
    # Move the tensor to the appropriate device
    img_tensor = img_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()  # Get the probability (real class confidence)
    
    # Classify based on the output
    if prediction >= 0.5:
        confidence = prediction * 100  # Confidence in the "Real" classification
        return "Real", confidence
    else:
        confidence = (1 - prediction) * 100  # Confidence in the "Fake" classification
        return "Fake", confidence

# Streamlit UI
st.title("Deepfake Image Classifier")
st.write("Upload an image to classify it as 'Fake' or 'Real' with confidence.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Classify the image
    label, confidence = classify_image(image)
    st.write(f"The image is classified as *{label}* with a confidence of *{confidence:.2f}%*.")
