# Model Files

Place your trained model files in this directory for deployment.

## Required Files

For the application to work properly on Streamlit Cloud, you should place your trained model file here with the name `model.pth`.

## How to Export Your Model

If you've trained your model locally, you can export it using:

```python
# Save the model state dict
torch.save(model.state_dict(), "models/model.pth")
```

## Model Architecture

The application expects a ResNet50 model with the following architecture:
- Base: ResNet50
- Custom classifier head:
  - Dropout(0.5)
  - Linear(2048, 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512, 5) # 5 flower classes
