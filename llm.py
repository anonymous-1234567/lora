import torch
import math
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader

from main import set_seed, compute_model_norm_squared
import numpy as np

set_seed(42)

# --- UPDATE #1: Explicitly sync the pad_token_id ---
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id # Explicitly set the ID



# Load model (use torch_dtype=torch.bfloat16 for modern GPUs to save VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    #device_map="auto"
    device_map={"": 1}
)


# 2. Configure and Apply LoRA
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

custom_scale = 1e-3

# 4. Override matrix A initialization
with torch.no_grad():
    for name, param in model.named_parameters():
        if "lora_A" in name and "weight" in name:
            torch.nn.init.normal_(param, mean=0.0, std=custom_scale)
            # Alternatively, for a uniform distribution:
            # nn.init.uniform_(param, a=-custom_scale, b=custom_scale)


initial_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# 3. Load and format the Alpaca dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# --- UPDATE #2: Mask the labels with -100 ---
def format_alpaca(example):
    if example.get("input", "") != "":
        text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    
    # Tokenize
    tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    
    # Mask padding tokens with -100 so the loss function ignores them entirely
    tokens["labels"] = [
        -100 if token == tokenizer.pad_token_id else token 
        for token in tokens["input_ids"]
    ]
    return tokens

tokenized_dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")

# Create DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)


import time # Add this to your imports at the top

def train_custom_sgd(model, dataloader, alpha, schedule_type="constant", steps=500):
    model.eval()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)
    history = {"loss": [], "step_size": [], "step_time": []} # Added step_time tracking
    
    vocab_size = model.config.vocab_size
    
    step_count = 0
    for batch in dataloader:
        if step_count >= steps:
            break
            
        batch = {k: v.to(model.device) for k, v in batch.items()}
        labels = batch["labels"]
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        
        # --- START TIMER ---
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Forward pass (Get raw bfloat16 logits)
        outputs = model(**model_inputs)
        logits = outputs.logits
        
        # Manual FP32 Loss Calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        shift_logits = shift_logits.view(-1, vocab_size).to(torch.float32)
        shift_labels = shift_labels.view(-1)
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)
        loss_val = loss.item()
        
        # Backward pass
        loss.backward()

        # --- THE FIX: GRADIENT CLIPPING ---
        # This forcefully scales down exploding gradients before they corrupt the weights
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
        
        # Step Size Calculation
        if schedule_type == "adaptive":
            parameter_norm_squared = compute_model_norm_squared(model)
            current_lr = min(alpha/(parameter_norm_squared + math.sqrt(loss_val) + 1e-13), 1)

        elif schedule_type == "normalized":

            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
            
            grad_norm = total_norm_sq ** 0.5
            current_lr = min(alpha / (np.sqrt(grad_norm) + 1e-13), 1)
        
        else:
            current_lr = alpha

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr


        
        optimizer.step()
        optimizer.zero_grad()

        # --- STOP TIMER ---
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        step_duration = end_time - start_time
        
        # Store metrics
        history["loss"].append(loss_val)
        history["step_size"].append(current_lr)
        history["step_time"].append(step_duration) # Store the time
        
        if step_count % 10 == 0:
            print(f"Step {step_count} | Loss: {loss_val:.4f} | LR: {current_lr:.6f} | Time: {step_duration:.3f}s")
            
        step_count += 1
        
    return history



# --- Run the experiments ---

Nsteps = 100
print("Running Constant LR Schedule...")
model.load_state_dict(initial_weights)
alpha = 0.02
history_constant = train_custom_sgd(model, dataloader, alpha=alpha, schedule_type="constant", steps=Nsteps)

torch.save(history_constant, f'llm_exp/history_constant_{alpha}.pt')



print("\nRunning Loss-Dependent LR Schedule...")
model.load_state_dict(initial_weights)
alpha=0.2
history_loss_dep = train_custom_sgd(model, dataloader, alpha=alpha, schedule_type="adaptive", steps=Nsteps)


torch.save(history_loss_dep, f'llm_exp/history_adaptive_{alpha}.pt')


print("\nRunning Normalized LR Schedule...")
model.load_state_dict(initial_weights)
alpha=0.02
history_loss_norm = train_custom_sgd(model, dataloader, alpha=alpha, schedule_type="normalized", steps=Nsteps)

torch.save(history_loss_norm, f'llm_exp/history_normalized_{alpha}.pt')
