import timm
import torch
from torch import nn, optim
from mlp import SimpleMLP
from cnn import SimpleCNN
from pvt_vit import get_pvt_model, get_vit_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(model_type, in_channels, num_classes, input_height, input_width):
    """
    Create and return a model based on the specified type
    
    Args:
        model_type (str): Type of model to create
        in_channels (int): Number of input channels
        num_classes (int): Number of output classes
        input_height (int): Input image height
        input_width (int): Input image width
    
    Returns:
        torch.nn.Module: Initialized model
    """
    if model_type == 'MLP':
        mlp_input_size = input_height * input_width * in_channels
        model = SimpleMLP(input_size=mlp_input_size, num_classes=num_classes).to(device)
    elif model_type == 'CNN':
        model = SimpleCNN(in_channels, num_classes, input_height, input_width).to(device)
    elif model_type == 'VIT':
        model = get_vit_model(num_classes, in_channels)
    elif model_type == 'PVT':
        model = get_pvt_model(num_classes, in_channels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Created {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model

