import torch
import torch.nn as nn
import torchvision.models as models
import os

# Define the class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def create_model():
    # Create a ResNet50 model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    
    # Modify the model with the same architecture used in your app
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(class_names))
    )
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def save_model():
    # Create the model
    model = create_model()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model state dictionary
    torch.save(model.state_dict(), "models/model.pth")
    print(f"Model saved to {os.path.abspath('models/model.pth')}")

if __name__ == "__main__":
    save_model()
