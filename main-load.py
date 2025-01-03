import torch
import random
import numpy as np
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  # Import safetensors loader
import math
import multiprocessing

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

# 1. Initialize the tokenizer and base model
model_name = "gpt2-xl"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

# Load pre-trained GPT-2 XL model
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

# 2. Define PEFT LoRA configurations (same as during training)
peft_config_layers_0_8 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(0, 9)),
    layers_pattern="h",
    num_adapters_per_layer=1,
    layer_group=0,
    adapter_labels=["EU,Indian,ECHR,UK,CAC"],
    r_a=[64]
)

peft_config_layers_9_10 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(9, 11)),
    layers_pattern="h",
    num_adapters_per_layer=2,
    layer_group=1,
    adapter_labels=['Indian,UK,CAC','EU,ECHR'],
    r_a=[43,21]
)

peft_config_layers_11 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=[11],
    layers_pattern="h",
    num_adapters_per_layer=3,
    layer_group=2,
    adapter_labels=['Indian,UK,CAC','EU','ECHR'],
    r_a=[43,14,7]
)

peft_config_layers_12_18 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=13,
    lora_alpha=26,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(12, 19)),
    layers_pattern="h",
    num_adapters_per_layer=4,
    layer_group=3,
    adapter_labels=['Indian','UK,CAC','EU','ECHR'],
    r_a=[9,34,14,7]
)

peft_config_layers_19_47 = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=13,
    lora_alpha=26,
    target_modules=['c_attn', 'c_proj'],
    lora_dropout=0.1,
    layers_to_transform=list(range(19, 48)),
    layers_pattern="h",
    num_adapters_per_layer=5,
    layer_group=4,
    adapter_labels=['Indian','UK','CAC','EU','ECHR'],
    r_a=[9,31,3,14,7]
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config_layers_0_8, adapter_name="layers_0_8")
model.add_adapter("layers_9_10", peft_config_layers_9_10)
model.add_adapter("layers_11", peft_config_layers_11)
model.add_adapter("layers_12_18", peft_config_layers_12_18)
model.add_adapter("layers_19_47", peft_config_layers_19_47)

# Ensure all adapter parameters require gradients (if necessary)
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# 4. Load the adapter weights from the Hugging Face Hub
from huggingface_hub import hf_hub_download

repo_id = 'MHGanainy/gpt2-xl-peft-lora-progressive-adapter-layer-comp-vocab-overlap'

# Function to load adapter weights into the model's adapters
def load_adapter_weights(model, adapter_name, repo_id, subfolder=None):
    # Download the adapter weights file (safetensors format)
    filename = 'adapter_model.safetensors'
    state_dict_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        cache_dir='./cache'  # Optional
    )
    # Load the state_dict using safetensors
    adapter_state_dict = load_file(state_dict_path)

    # Adjust keys to match the model's state_dict
    adjusted_state_dict = {}
    for k, v in adapter_state_dict.items():
        new_key = k  # Keep the key as is (since we don't remove any prefix)

        # Handle 'lora_A' and 'lora_B' keys
        if 'lora_A' in new_key or 'lora_B' in new_key:
            parts = new_key.split('.')
            # Find the index of 'lora_A' or 'lora_B'
            lora_idx = parts.index('lora_A') if 'lora_A' in parts else parts.index('lora_B')
            # Insert adapter_name after 'lora_A' or 'lora_B'
            parts.insert(lora_idx + 1, adapter_name)
            new_key = '.'.join(parts)

        adjusted_state_dict[new_key] = v

    # Save a copy of model parameters before loading the adapter
    model_state_dict_before = {k: v.clone() for k, v in model.state_dict().items()}

    # Load the adjusted adapter weights into the model
    model_state_dict = model.state_dict()
    # Keep track of keys that are actually in the model's state dict
    keys_in_model = []
    for key in adjusted_state_dict.keys():
        if key in model_state_dict:
            keys_in_model.append(key)
            model_state_dict[key] = adjusted_state_dict[key]
        else:
            print(f"Key '{key}' from adjusted state dict not found in model's state dict.")

    # Now load the updated state dict into the model and capture the missing and unexpected keys
    load_result = model.load_state_dict(model_state_dict, strict=False)
    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys

    # Verify that parameters have been updated
    updated_params = []
    model_state_dict_after = model.state_dict()
    for key in keys_in_model:
        before = model_state_dict_before[key]
        after = model_state_dict_after[key]
        if not torch.equal(before, after):
            updated_params.append(key)

    # Compute boolean indicating whether all parameters have been updated
    all_params_updated = len(updated_params) == len(keys_in_model)


    if missing_keys:
        print(f"\nMissing keys when loading adapter '{adapter_name}': {missing_keys}")
    if unexpected_keys:
        print(f"\nUnexpected keys when loading adapter '{adapter_name}': {unexpected_keys}")

    if not missing_keys and not unexpected_keys and all_params_updated:
        print(f"\nAdapter '{adapter_name}' loaded successfully with all parameters updated.")
    else:
        print(f"\nAdapter '{adapter_name}' loading issues detected.")
        if not all_params_updated:
            print(f"Not all parameters were updated for adapter '{adapter_name}'.")
        print(f"Total parameters expected to update: {len(keys_in_model)}")
        print(f"Total parameters actually updated: {len(updated_params)}")

    # Return the boolean indicating whether all parameters were updated
    return all_params_updated


# Load weights for each adapter
load_adapter_weights(model, 'layers_0_8', repo_id, subfolder='layers_0_8')
load_adapter_weights(model, 'layers_9_10', repo_id, subfolder='layers_9_10')
load_adapter_weights(model, 'layers_11', repo_id, subfolder='layers_11')
load_adapter_weights(model, 'layers_12_18', repo_id, subfolder='layers_12_18')
load_adapter_weights(model, 'layers_19_47', repo_id, subfolder='layers_19_47')

# Set model to evaluation mode
model.eval()

# 5. Prepare your evaluation dataset (same as during training)
block_size = 1024

# Task types mapping
dataset = load_dataset('MHGanainy/multi_clustering', 'lex-former-8-clustered-instance-b-dataset-cluster')
dataset_name_to_task_types = {
    'santoshtyss/uk_courts_cases': [0,0,0,1,1],
    'santoshtyss/eu-court-cases': [0,1,1,2,3],
    'santoshtyss/indian_courts_cases': [0,0,0,0,0],
    'santoshtyss/ecthr_cases': [0,1,2,3,4],
    'santoshtyss/canadian_court_cases': [0,0,0,1,2]
}

# 3. Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the text
    result = tokenizer(
        examples["original_text"],
        padding='max_length',
        max_length=block_size,
        truncation=True,
    )
    input_ids = result["input_ids"]

    # Copy input_ids to labels
    labels = input_ids.copy()

    # Set the first 512 tokens of labels to -100
    labels = [[-100]*512 + ids[512:] for ids in labels]

    # Set labels to -100 where input_ids == pad_token_id
    labels = [
        [label if input_id != tokenizer.pad_token_id else -100 for input_id, label in zip(input_ids_seq, labels_seq)]
        for input_ids_seq, labels_seq in zip(input_ids, labels)
    ]
    result["labels"] = labels

    # Map the dataset_name to task_types
    dataset_names = examples['dataset_name']
    task_types_list = [
        dataset_name_to_task_types.get(name) if name in dataset_name_to_task_types else print(f"{name} not found") or None
        for name in dataset_names
    ]
    result['task_types'] = task_types_list

    return result

# Apply the tokenize_function to the dataset
def prepare_dataset(dataset_split, split="train"):
    total_cores = multiprocessing.cpu_count()
    num_cpu_cores = min(64, total_cores)
    print(f"Using {num_cpu_cores} CPU cores for '{split}' dataset processing.")

    lm_dataset = dataset_split.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_split.column_names,
        desc=f"Tokenizing {split} dataset",
        num_proc=num_cpu_cores,
        # Ensure deterministic shuffling
        load_from_cache_file=True,
        writer_batch_size=1000,
    )

    lm_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'task_types'])
    return lm_dataset

print("Preprocessing validation data...")
# cluster_id = 0 # echr
# cluster_id = 1 # eu
# cluster_id = 2 # uk
# cluster_id = 3 # indian
cluster_id = 4 # canadian
eval_dataset = prepare_dataset(dataset["validation"].filter(lambda x: x['cluster_id'] == cluster_id), "validation")

# 6. Initialize Trainer for evaluation
batch_size = 8

training_args = TrainingArguments(
    output_dir="./evaluation_output",
    per_device_eval_batch_size=batch_size,
    dataloader_num_workers=8,
    fp16=False,  # Set to True if your hardware supports it
    bf16=True,   # Set to True if your hardware supports it
    do_train=False,
    do_eval=True,
    seed=seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 7. Perform evaluation
print("Starting evaluation...")
metrics = trainer.evaluate()
perplexity = math.exp(metrics["eval_loss"])
metrics["perplexity"] = perplexity

print("Evaluation metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")
