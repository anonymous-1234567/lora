import argparse
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split

import copy

from lora_utils import LoRALinear, LoRATrackingContext

# Import our custom modules
from models import get_model

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training with LoRA')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='logreg', choices=['logreg', 'resnet18'],
                        help='Which architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'extracted_cifar10'],
                        help='Dataset')
    
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=10, choices=[2, 10])
    
    parser.add_argument('--lr', type=float, default=0.01, help='Base Learning rate or scaling factor (alpha), depending on lr scheduler')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        
    parser.add_argument('--lr_schedule', type=str, default='constant', choices=['constant', 'adaptive', 'adaptive_2', 'normalized'],
                        help='Strategy for learning rate updates')

    parser.add_argument('--val_split', type=float, default=0.1, 
                    help='Fraction of training data to use for validation (e.g., 0.1 for 10%)')

    # Utilities
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=int, default=0, help='Device number')
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def get_run_name(args):
    """Generates a unique filename based on arguments."""
    lora_str = f"_lora_{args.lora_rank}" if args.use_lora else ""
    return f"{args.dataset}_{args.model}_valsplit_{args.val_split}_epochs_{args.epochs}_bs_{args.batch_size}_num_classes_{args.num_classes}_lr_{args.lr}_wd_{args.weight_decay}_{args.lr_schedule}_seed_{args.seed}{lora_str}"


def compute_model_norm_squared(model, only_trainable=True):
    """
    Computes the aggregate norm squared of the model parameters.
    
    Args:
        model: The PyTorch model.
        only_trainable: If True, only computes norm of parameters with requires_grad=True.
    """
    total_norm_sq = 0.0
    
    for p in model.parameters():
        if only_trainable and not p.requires_grad:
            continue
            
        # Sum of squares for this parameter tensor
        param_norm = p.data.norm(2).item()
        total_norm_sq += param_norm ** 2
        
    return total_norm_sq

def compute_lipschitz_constant(data_loader, device, weight_decay=0.0):
    """Computes smoothness constant L of multinomial logistic regression loss."""
    print("Computing Lipschitz Constant L...")
    first_batch, _ = next(iter(data_loader))
    input_dim = first_batch.view(first_batch.size(0), -1).shape[1]
    
    gram_matrix = torch.zeros((input_dim, input_dim), device=device)
    total_samples = 0
    
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        inputs_flat = inputs.view(inputs.size(0), -1)
        gram_matrix += torch.matmul(inputs_flat.T, inputs_flat)
        total_samples += inputs.size(0)
        
    max_eigenvalue = torch.linalg.eigvalsh(gram_matrix)[-1].item()
    L = (max_eigenvalue / total_samples) + weight_decay
    print(f"Computed L: {L:.4f}")
    return L


def compute_full_grad_stats(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = len(data_loader.dataset)
    
    model.zero_grad()
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * batch_size
        weighted_loss = loss * (batch_size / total_samples)
        weighted_loss.backward()
        
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    
    full_grad_norm = total_norm_sq ** 0.5
    avg_loss = total_loss / total_samples
    model.zero_grad()
    return avg_loss, full_grad_norm

def get_intermediate_grad_norm(model):
    """
    Retrieves the stored gradient norms of the implicit matrix W (BA)
    from all LoRALinear layers.
    """
    norm = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # The hook populates this attribute during backward()
            if module.w_grad_norm is not None:
                norm += (module.w_grad_norm)**2
    return norm**(0.5)

def compute_accuracy(model, dataloader, device):
    """
    Computes the accuracy of the model on the provided dataloader.
    Returns the accuracy as a fraction.
    """
    model.eval() # Set model to evaluation mode (disables dropout, fixes BN stats)
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient computation for speed and memory efficiency
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions from the maximum value (logits)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    return correct / total

def compute_loss(model, dataloader, criterion, device):
    """Computes average loss on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Sum up batch loss (multiply by batch size to weight correctly)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
    return total_loss / total_samples

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Running {args.model} on {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- 1. Data Prep ---
    if args.model == 'logreg':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1)) 
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    num_classes = args.num_classes

    if args.dataset == 'extracted_cifar10':
        print('Using feature extracted dataset')
        X_train, y_train, X_test, y_test = torch.load('cifar10_resnet18_features.pt')
        input_dim = X_train.shape[1]
        
        # Create simple dataloaders for the embeddings
        train_set = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)   
    else:
        input_dim = 32 * 32 * 3


    # Calculate split sizes
    val_size = int(len(train_set) * args.val_split)
    train_size = len(train_set) - val_size

    # Split the dataset (with a fixed generator for reproducibility)
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(train_set, [train_size, val_size], generator=generator)

    print(f"Dataset split: {train_size} Train | {val_size} Val | {len(test_set)} Test")

    # Create the Loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)



    # --- 3. Model & Optimizer ---

    model = get_model(args.model, device, input_dim = input_dim, use_lora=args.use_lora, lora_rank=args.lora_rank)


    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # --- 4. Training ---
    print(f"Starting Training | Learning Rate Schedule: {args.lr_schedule}")
    

    training_history = {'step': [], 'lr': [],'batch_loss': [], 'batch_grad_norm': []}


    training_history['parameter norm squared'] = []
    training_history['intermediate grad norm'] = []
    
    global_step = 0
    total_steps = len(train_loader) * args.epochs

    #validation
    best_val_loss = 10000
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0



    for epoch in range(args.epochs):
        model.train()
        if args.model == 'resnet18':
            model.eval()

        track_intermediate_grads = (args.lr_schedule == 'adaptive')
        
        for inputs, targets in train_loader:
            with LoRATrackingContext(track_intermediate_grads):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

            #compute batch gradient norm
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            
            grad_norm = total_norm_sq ** 0.5
            

            vnormsquared = compute_model_norm_squared(model)
            training_history['parameter norm squared'].append(vnormsquared)
            
            #Update Learning Rate
            if args.lr_schedule == 'adaptive':
                
                w_grad_norm = get_intermediate_grad_norm(model)
                new_lr = args.lr/(1e-12 +  vnormsquared + w_grad_norm)
                

                
                training_history['intermediate grad norm'].append(w_grad_norm)

            if args.lr_schedule == 'adaptive_2':
                new_lr = args.lr/(1e-12 +  vnormsquared + math.sqrt(loss.item()))
            
            elif args.lr_schedule == 'normalized':
                new_lr = min(args.lr/(grad_norm)**(0.5), 1)

            else:
                new_lr = args.lr 

            new_lr = min(new_lr, 1)

            for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                
            #Take step
            optimizer.step()

            #logging 
            training_history['batch_loss'].append(loss.item())
            training_history['batch_grad_norm'].append(grad_norm)
            
            training_history['step'].append(global_step)
            training_history['lr'].append(new_lr)
            global_step += 1

        val_loss = compute_loss(model, val_loader, criterion, device)
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}] Loss: {loss.item():.4f} | Validation Loss {val_loss:.4f} | learning rate: {new_lr:.6f}")

        
        
        # --- Check for Improvement ---
        if val_loss < best_val_loss:
            print(f" -> Validation loss improved ({best_val_loss:.2f} -> {val_loss:.2f}). Saving best model.")
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_step = global_step

    print(f"\nTraining complete. Loading best model (Val Loss: {best_val_loss:.2f}) for final testing.")
    model.load_state_dict(best_model_weights)
    
    train_accuracy = compute_accuracy(model, train_loader, device)
    val_accuracy = compute_accuracy(model, val_loader, device)
    test_accuracy = compute_accuracy(model, test_loader, device)
    # --- 5. Save ---
    run_name = get_run_name(args)
    save_path = os.path.join(args.save_dir, f"{run_name}.pt")

    save_dict = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'history': training_history,
        'selected epoch': best_epoch,
        'selected step': best_step,
        'val accuracy': val_accuracy,
        'train accuracy': train_accuracy,
        'test accuracy': test_accuracy
    }

    if args.lr_schedule == 'adaptive':
        save_dict['lip_constant'] = lipschitz_L
    
    
    torch.save(save_dict, save_path)
    print(f"Saved to {save_path}")

if __name__ == '__main__':
    main()