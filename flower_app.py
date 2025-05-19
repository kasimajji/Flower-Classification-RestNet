import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import mlflow
import os
import time
from io import BytesIO
import base64
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f0f8ff;
            padding: 0 !important;
            margin: 0 !important;
        }
        .stApp {
            max-width: 100%;
            padding: 0 !important;
            margin: 0 !important;
        }
        .css-1v3fvcr, .css-18e3th9, .css-1d391kg, .css-hxt7ib {
            padding: 0 !important;
            margin: 0 !important;
        }
        .block-container {
            max-width: 100%;
            padding: 0 !important;
            margin: 0 !important;
        }
        .stSidebar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .stSidebar .block-container {
            padding: 2rem 1rem !important;
        }
        .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar li {
            color: white !important;
        }
        .stSidebar h2 {
            font-size: 1.8rem;
            text-align: center;
            margin-bottom: 1.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .stSidebar h3 {
            font-size: 1.4rem;
            border-bottom: 1px solid rgba(255,255,255,0.3);
            padding-bottom: 0.5rem;
            margin-top: 1.5rem;
        }
        .stButton>button {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 7px 14px rgba(0,0,0,0.1);
        }
        .stButton>button:active {
            transform: translateY(1px);
        }
        .prediction-box {
            padding: 20px;
            border-radius: 15px;
            background-color: #ffffff;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: all 0.3s;
            border-left: 5px solid #38ef7d;
        }
        .prediction-box:hover {
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            transform: translateY(-5px);
        }
        .title {
            font-size: 3.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #FF6B6B 0%, #FFE66D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-shadow: 0px 0px 10px rgba(255,107,107,0.2);
        }
        .subtitle {
            font-size: 1.5rem;
            margin-bottom: 2rem;
            text-align: center;
            color: #6c757d;
        }
        .upload-section, .result-section {
            padding: 1rem;
            margin: 1rem;
        }
        .confidence-bar {
            height: 25px;
            border-radius: 15px;
            margin-top: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .flower-card {
            padding: 15px;
            border-radius: 15px;
            background: white;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            margin: 10px;
            text-align: center;
            transition: all 0.3s;
            overflow: hidden;
            height: 220px;
            width: 170px;
            display: inline-block;
        }
        .flower-card h4 {
            margin-top: 10px;
            margin-bottom: 10px;
            color: #333;
            font-weight: bold;
        }
        .flower-card:hover {
            transform: translateX(-10px) scale(1.05);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        .flower-img {
            border-radius: 10px;
            margin-bottom: 10px;
            height: 130px;
            width: 130px;
            object-fit: cover;
        }
        .sample-flowers-container {
            display: flex;
            flex-direction: row;
            overflow-x: auto;
            gap: 10px;
            padding: 0.5rem;
            margin: 0 auto;
            justify-content: flex-start;
            white-space: nowrap;
            scroll-snap-type: x mandatory;
        }
        .flower-card {
            flex: 0 0 auto;
            scroll-snap-align: start;
        }
        .main-content {
            padding: 0.5rem;
            min-height: 92vh;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
            margin-top: 30px;
            border-top: 1px solid #eee;
        }
        /* Fix for file uploader */
        .stFileUploader > div > div {
            width: 100% !important;
        }
        .stFileUploader > div > div > div {
            width: 100% !important;
        }
        .stFileUploader label {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        /* Chart styling */
        .st-emotion-cache-1offfay p {
            font-weight: bold !important;
            color: #333 !important;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_flower = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_obhph3sh.json")
lottie_loading = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_x62chJ.json")

# Path to the saved model
MODEL_PATH = "/Users/kasimajji/Desktop/Projects/Flower_classification_Deep learning/mlruns/200404991946074660/555ee7a0085f4f7dbe9b476272e3f949/artifacts/best_model"

# Class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Function to load the model
@st.cache_resource
def load_model():
    try:
        # Define the model architecture (ResNet50)
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_names))
        )
        
        # Try multiple possible model paths (for both local and deployment environments)
        possible_paths = [
            # Local path
            "/Users/kasimajji/Desktop/Projects/Flower_classification_Deep learning/mlruns/200404991946074660/555ee7a0085f4f7dbe9b47627e3f949/artifacts/best_model/data/model.pth",
            # Deployment paths (relative)
            "model.pth",
            "./model.pth",
            "./models/model.pth",
            "./data/model.pth"
        ]
        
        # Try each path until we find the model
        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    st.success(f"Found model at: {path}")
                    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                    model_loaded = True
                    break
                except Exception as path_error:
                    st.info(f"Found file at {path} but couldn't load it: {path_error}")
        
        # If model couldn't be loaded, just use the initialized model for demo
        if not model_loaded:
            st.warning("Could not load saved model. Using a randomly initialized model for demo purposes.")
        
        # Set the model to evaluation mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error in model loading process: {e}")
        # For deployment, return a basic model rather than None
        try:
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))
            model.eval()
            st.warning("Using fallback model for demonstration")
            return model
        except Exception as fallback_error:
            st.error(f"Even fallback model failed: {fallback_error}")
            return None

# Image preprocessing function
def preprocess_image(image):
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Function to make predictions
def predict(model, image):
    # Preprocess the image
    image_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get the predicted class and probability
    max_prob, predicted_idx = torch.max(probabilities, 0)
    predicted_class = class_names[predicted_idx.item()]
    confidence = max_prob.item()
    
    # If confidence is below threshold, classify as "unknown"
    if confidence < 0.5:  # You can adjust this threshold
        return "unknown", confidence, probabilities.cpu().numpy()
    
    return predicted_class, confidence, probabilities.cpu().numpy()

# Function to create a base64 image for download
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to create a confidence bar chart
def create_confidence_chart(probabilities):
    # Set a modern style
    plt.style.use('ggplot')
    
    # Create figure with a gradient background
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#f5f7fa')
    ax.set_facecolor('#f5f7fa')
    
    # Define vibrant colors for each class
    colors = [
        '#FF6B6B',  # Red for daisy
        '#FFD93D',  # Yellow for dandelion
        '#FF9A8B',  # Pink for rose
        '#FFDE7D',  # Gold for sunflower
        '#6A0572',  # Purple for tulip
    ]
    
    # Add a light gradient to the bars
    for i, (color, prob) in enumerate(zip(colors, probabilities)):
        # Create gradient effect
        gradient = np.linspace(0.6, 1.0, 100)
        gradient_colors = [(1-(1-int(color[1:3], 16)/255)*g, 
                           1-(1-int(color[3:5], 16)/255)*g, 
                           1-(1-int(color[5:7], 16)/255)*g, 1) for g in gradient]
        
        # Plot the bar with a more sophisticated style
        bar = ax.bar(
            i, prob, 
            color=color, 
            width=0.7,
            edgecolor='white',
            linewidth=1.5,
            alpha=0.8,
            label=class_names[i].capitalize()
        )
        
        # Add percentage labels on top of bars with improved styling
        height = prob
        ax.annotate(
            f'{height:.1%}',
            xy=(i, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha='center', 
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#333333'
        )
    
    # Customize the chart
    ax.set_ylim(0, max(probabilities) * 1.2)  # Give some headroom above the highest bar
    ax.set_ylabel('Confidence', fontsize=12, fontweight='bold', color='#333333')
    ax.set_title('Prediction Confidence by Class', fontsize=14, fontweight='bold', color='#333333', pad=20)
    
    # Set x-ticks with class names
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([name.capitalize() for name in class_names], fontsize=11)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Customize grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    # Add subtle box shadow to the figure
    fig.tight_layout(pad=2.0)
    
    # Add a legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    return fig

# Function to display sample images for each class
def display_sample_images():
    st.markdown("<h3 class='subtitle' style='margin-bottom:0.5rem;'>Sample Flowers</h3>", unsafe_allow_html=True)
    
    # Create a simple horizontal layout using Streamlit columns
    cols = st.columns(len(class_names))
    
    # Display a sample image for each class in its own column
    for i, class_name in enumerate(class_names):
        with cols[i]:
            # Get a sample image path for this class
            sample_dir = f"/Users/kasimajji/Desktop/Projects/Flower_classification_Deep learning/flowers/{class_name}"
            if os.path.exists(sample_dir):
                sample_images = os.listdir(sample_dir)
                if sample_images:
                    # Filter out non-image files
                    image_files = [f for f in sample_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if image_files:
                        sample_image_path = os.path.join(sample_dir, image_files[0])
                        try:
                            img = Image.open(sample_image_path)
                            # Resize image to ensure consistent dimensions
                            img = img.resize((130, 130), Image.LANCZOS)
                            
                            # Display the image and class name using Streamlit components
                            st.markdown(f"<p style='text-align:center; margin:0 0 5px 0; font-weight:bold;'>{class_name.capitalize()}</p>", unsafe_allow_html=True)
                            st.image(img, width=130, use_column_width=False)
                        except Exception as e:
                            st.error(f"Could not load image: {e}")


# Function to convert image to base64 for embedding in HTML
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Main app function
def main():
    # Apply the custom CSS
    local_css()
    
    # Title and introduction - with improved visibility
    st.markdown("""
    <h1 class='title' style='
        margin: 0.5rem 0;
        font-size: 2.5rem;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B 0%, #FFE66D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: black; /* fallback color */
        color: black; /* fallback for other browsers */
        display: block;
    '>
        🌸 Flower Classification App
    </h1>
""", unsafe_allow_html=True)
    
    # Sidebar content with model architecture and preprocessing information
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom:0.5rem;'>Flower Classifier</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)
        
        # Model Architecture section
        st.markdown("<h3 style='margin:0.5rem 0;'>Model Architecture</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='margin:0.3rem 0;'>
        <p>The model uses a pre-trained <b>ResNet50</b> architecture with:</p>
        <ul style='margin:0.2rem 0; padding-left:1rem;'>
            <li>Feature extraction layers (frozen)</li>
            <li>Custom classifier head with dropout</li>
            <li>5 output classes + unknown</li>
            <li>Trained with cross-entropy loss</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Preprocessing section
        st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0.5rem 0;'>Image Preprocessing</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='margin:0.3rem 0;'>
        <p>Images undergo the following preprocessing:</p>
        <ul style='margin:0.2rem 0; padding-left:1rem;'>
            <li>Resize to 224×224 pixels</li>
            <li>Convert to RGB format</li>
            <li>Normalize with ImageNet stats</li>
            <li>Data augmentation during training</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Classes section
        st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0.5rem 0;'>Classes</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='margin:0.3rem 0;'>
        <ul style='margin:0.2rem 0; padding-left:1rem;'>
            <li>Daisy</li>
            <li>Dandelion</li>
            <li>Rose</li>
            <li>Sunflower</li>
            <li>Tulip</li>
        </ul>
        <p>If confidence < 0.7, classified as "unknown"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # How to use section
        st.markdown("<hr style='margin:0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:0.5rem 0;'>How to use</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ol style='margin:0.2rem 0; padding-left:1rem;'>
            <li>Upload an image using the file uploader</li>
            <li>Click the Classify Flower button</li>
            <li>View the results and confidence scores</li>
        </ol>
        """, unsafe_allow_html=True)
    
    # Load the model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        return
    
    # Display sample images
    display_sample_images()
    
    # Upload section with improved styling - more compact
    st.markdown("<div class='upload-section' style='padding:0.8rem; margin:0.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #333; margin:0.5rem 0;'>Upload an Image</h3>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create two columns with better spacing
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display the uploaded image with a border and shadow
            image = Image.open(uploaded_file).convert('RGB')
            st.markdown("<h4 style='text-align: center; margin-bottom: 10px;'>Uploaded Image</h4>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            
        with col2:
            st.markdown("<h4 style='text-align: center; margin-bottom: 20px;'>Classification</h4>", unsafe_allow_html=True)
            # Add a prediction button with animation and improved styling
            if st.button("Classify Flower"):
                with st.spinner():
                    # Display loading animation
                    lottie_placeholder = st.empty()
                    lottie_status = st_lottie(lottie_loading, height=120, key="loading")
                    
                    # Make prediction
                    predicted_class, confidence, probabilities = predict(model, image)
                    
                    # Remove loading animation after a short delay for effect
                    time.sleep(1)
                    lottie_placeholder.empty()
                    
                    # Display prediction result with improved styling
                    if predicted_class == "unknown":
                        st.markdown(f"""
                        <div class='prediction-box' style='background: linear-gradient(135deg, #ffefba, #ffffff); border-left: 5px solid #ff9a9e;'>
                            <h3 style='color: #721c24; text-align: center;'>Unknown Flower</h3>
                            <p style='text-align: center;'>The model is not confident enough to classify this image.</p>
                            <p style='text-align: center; font-weight: bold;'>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='prediction-box' style='background: linear-gradient(135deg, #d4fc79, #96e6a1); border-left: 5px solid #38ef7d;'>
                            <h3 style='color: #155724; text-align: center;'>Prediction: {predicted_class.capitalize()}</h3>
                            <p style='text-align: center; font-weight: bold;'>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                    # Create and display confidence chart with improved styling
                    st.markdown("<h4 style='margin-top: 20px; text-align: center;'>Confidence Scores</h4>", unsafe_allow_html=True)
                    fig = create_confidence_chart(probabilities)
                    st.pyplot(fig)
                    
                    # Provide download link for the image with prediction
                    result = f"{predicted_class}_{confidence:.2f}"
                    download_link = get_image_download_link(image, f"{result}.jpg", "📥 Download Image with Prediction")
                    st.markdown(f"<div style='text-align: center; margin-top: 15px;'>{download_link}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div class='footer'>Flower Classification App © 2025</div>", unsafe_allow_html=True)
    
    # Close the main content div
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
