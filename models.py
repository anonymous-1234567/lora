import torch
import torch.nn as nn
import torchvision.models 
from lora_utils import convert_linear_to_lora, LoRALinear

from peft import LoraConfig, get_peft_model

class LogisticRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim = 32 * 32 * 3):
        super(LogisticRegression, self).__init__()
        # Input is flattened 32x32x3 = 3072
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x, **kwargs):
        if isinstance(self.linear, LoRALinear):
            return self.linear(x, **kwargs)
        else:
            # Fallback for standard Linear (ignores the flag)
            return self.linear(x)

def get_cifar_resnet18(num_classes=10, use_lora=False, lora_rank=None):
    """
    Returns a ResNet-18 modified for CIFAR-10 image sizes.
    """
    # Load standard ResNet structure (random initialization without pretrained weights)
    model = torchvision.models.resnet18(weights=None)

    if use_lora:
        config = LoraConfig(
        r=lora_rank,                       # Rank
        lora_alpha=lora_rank,              # Alpha 
        target_modules=["conv1", "conv2", "downsample.0"], # Regex to catch all Conv2d layers
        lora_dropout=0.0,
        bias="none",
        modules_to_save=["fc"],     # Train the classifier head fully!
    )

        lora_model = get_peft_model(model, config)
    
    # Modify first conv layer to accept 32x32 images without aggressive downsampling
    # Original: kernel=7, stride=2. Modified: kernel=3, stride=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the first MaxPool (it reduces spatial dim too fast for small images)
    model.maxpool = nn.Identity()
    
    # Modify the final Fully Connected layer to output the correct number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_model(model_name, device, use_lora=False, lora_rank=4, input_dim = 32 * 32 * 3, num_classes=10):
    """
    Factory function to create models and optionally apply LoRA.
    """
    # 1. Instantiate the Base Model
    if model_name == 'logreg':
        model = LogisticRegression(num_classes=num_classes, input_dim = input_dim)
    elif model_name == 'resnet18':
        model = get_cifar_resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 2. Apply LoRA if requested
    if use_lora:
        print(f"Converting model to LoRA (Rank={lora_rank})...")
        
        if (model_name == 'logreg'):        
            convert_linear_to_lora(model, rank=lora_rank)
        else:
            model = get_cifar_resnet18(num_classes=num_classes, use_lora=True, lora_rank=lora_rank)
            
        
    return model.to(device)