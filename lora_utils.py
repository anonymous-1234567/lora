import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRATrackingContext:
    _enabled = False

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.prev = False

    def __enter__(self):
        self.prev = LoRATrackingContext._enabled
        LoRATrackingContext._enabled = self.enabled

    def __exit__(self, exc_type, exc_val, exc_tb):
        LoRATrackingContext._enabled = self.prev

    @staticmethod
    def is_enabled():
        return LoRATrackingContext._enabled





class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        """
        A Linear layer that implements Low-Rank Adaptation (LoRA).
        Effective Weight = W_frozen + (B @ A) * scaling
        """
        super(LoRALinear, self).__init__()
        
        # Standard Setup
        # We wrap a standard Linear layer but freeze it immediately
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_rank = rank
        self.lora_alpha = alpha
        #self.scaling = alpha / rank
        self.scaling = 1 #WE GET RID OF SCALING

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Storage for the gradient norm we want to capture
        self.w_grad_norm = None 

        # A is random, B is zero. 
        # This ensures the training starts exactly as the base model behavior.
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _hook_fn(self, grad):
        """
        This function runs during the BACKWARD pass.
        'grad' is dL / d(delta_W).
        """
        with torch.no_grad():
            self.w_grad_norm = grad.norm(2).item()
        # We must return gradients in a hook (returning None keeps it unchanged)
        return None

    def forward(self, x, intermediate=False):

        should_track = LoRATrackingContext.is_enabled()
        
        if should_track:
            # 1. Explicitly construct the low-rank matrix W = B @ A
            # Shape: (out, rank) @ (rank, in) -> (out, in)
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            
            # 2. Register a hook on this tensor intermediate.
            # This tells PyTorch: "When you compute dL/d(delta_w), call this function."
            if self.training and delta_w.requires_grad:
                 delta_w.register_hook(self._hook_fn)
    
            # 3. Apply the linear transformation using this constructed weight
            # lora_out = x @ delta_w.T
            lora_out = F.linear(x, delta_w)
            
            return self.base_layer(x) + lora_out
        else:
            # 1. Base (Frozen) path
            result = self.base_layer(x)
            
            # 2. LoRA (Trainable) path
            # Computation: (x @ A.T) @ B.T
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            
            return result + (lora_out * self.scaling)


def convert_linear_to_lora(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            lora_layer = LoRALinear(module.in_features, module.out_features, rank=rank)
            lora_layer.base_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_layer.base_layer.bias.data = module.bias.data.clone()
            
            setattr(model, name, lora_layer)
        else:
            convert_linear_to_lora(module, rank, zero_base)