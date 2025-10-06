import pickle
import torch 
import numpy as np

# Set random seeds for reproducibility
#torch.manual_seed(42)
#np.random.seed(42)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed_all(42)
    # Ensure deterministic behavior on CUDA
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("affinity_input_sample.pkl", "rb") as f:
    sample_dict = pickle.load(f)

print(sample_dict["s_inputs_affinity"].shape)
print(sample_dict["z_affinity"].shape)
print(sample_dict["coords_affinity"].shape)
print(sample_dict["feats"].keys())

with open("affinity_module1.pkl", "rb") as f:
    module_dict = pickle.load(f)    


from boltz.model.modules.affinity_protein import ProteinProteinAffinityModule

# Create the module
affinity_module1 = ProteinProteinAffinityModule(
    module_dict["token_s"],
    module_dict["token_z"],
    module_dict["protein_ligand_mode"],
    **module_dict["affinity_model_args1"]
).to(device)

# Load checkpoint and extract affinity_module1 weights
checkpoint = torch.load("boltz2_aff.ckpt", map_location=device, weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)

# Extract only the affinity_module1 parameters from checkpoint
affinity_state_dict = {}
prefix = "affinity_module1."
for key, value in state_dict.items():
    if key.startswith(prefix):
        # Remove the prefix to match the module's parameter names
        new_key = key[len(prefix):]
        affinity_state_dict[new_key] = value

# Load the weights into the module
if affinity_state_dict:
    missing_keys, unexpected_keys = affinity_module1.load_state_dict(affinity_state_dict, strict=False)
    print(f"Loaded {len(affinity_state_dict)} parameters from checkpoint")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
else:
    print("Warning: No affinity_module1 weights found in checkpoint")

print(len(affinity_state_dict.keys()))

print(affinity_module1)

# Set model to evaluation mode for deterministic inference
affinity_module1.eval()

# Disable gradient computation for inference
with torch.no_grad():
    dict_out_affinity1 = affinity_module1(
        s_inputs=sample_dict["s_inputs_affinity"].to(device),
        z=sample_dict["z_affinity"].to(device),
        x_pred=sample_dict["coords_affinity"].to(device),
        feats=sample_dict['feats'],
        multiplicity=1,
        use_kernels=sample_dict["use_kernels"]
    )
    
    dict_out_affinity1["affinity_probability_binary"] = (
                    torch.nn.functional.sigmoid(
                        dict_out_affinity1["affinity_logits_binary"]
                    )
                )
print(dict_out_affinity1)

# ============================================================================
# Training Loop for Finetuning
# ============================================================================

print("\n" + "="*80)
print("Starting Finetuning...")
print("="*80 + "\n")

# Set model to training mode
affinity_module1.train()

# Training hyperparameters
batch_size = 4
num_epochs = 10
learning_rate = 1e-4
target_affinity_value = 7.5  # Example target pKd value
target_affinity_binary = 1.0  # Example: binding (1) or not binding (0)

# Setup optimizer
optimizer = torch.optim.Adam(affinity_module1.parameters(), lr=learning_rate)

# Loss functions
mse_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

# Create batch by replicating the sample
def create_batch(sample_dict, batch_size):
    """Replicate sample to create a batch."""
    batch = {}
    batch["s_inputs_affinity"] = sample_dict["s_inputs_affinity"].repeat(batch_size, 1, 1).to(device)
    batch["z_affinity"] = sample_dict["z_affinity"].repeat(batch_size, 1, 1, 1).to(device)
    batch["coords_affinity"] = sample_dict["coords_affinity"].repeat(batch_size, 1, 1, 1).to(device)
    
    # Handle feats dictionary - need to replicate relevant tensors
    batch["feats"] = {}
    for key, value in sample_dict['feats'].items():
        if isinstance(value, torch.Tensor):
            # Replicate tensor features
            batch["feats"][key] = value.repeat(batch_size, *[1]*(value.ndim-1))
        else:
            # Keep non-tensor features as-is
            batch["feats"][key] = value
    
    batch["use_kernels"] = sample_dict["use_kernels"]
    return batch

# Create targets
target_values = torch.full((batch_size, 1), target_affinity_value, device=device)
target_binary = torch.full((batch_size, 1), target_affinity_binary, device=device)

# Training loop
for epoch in range(num_epochs):
    # Create batch
    batch = create_batch(sample_dict, batch_size)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    output = affinity_module1(
        s_inputs=batch["s_inputs_affinity"],
        z=batch["z_affinity"],
        x_pred=batch["coords_affinity"],
        feats=batch['feats'],
        multiplicity=1,
        use_kernels=batch["use_kernels"]
    )
    
    # Calculate losses
    loss_value = mse_loss(output["affinity_pred_value"], target_values)
    loss_binary = bce_loss(output["affinity_logits_binary"], target_binary)
    
    # Combined loss
    total_loss = loss_value + loss_binary
    
    # Backward pass
    total_loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 2 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Total Loss: {total_loss.item():.4f} | "
              f"Value Loss: {loss_value.item():.4f} | "
              f"Binary Loss: {loss_binary.item():.4f}")

print("\n" + "="*80)
print("Finetuning Complete!")
print("="*80 + "\n")

# Evaluate after training
affinity_module1.eval()
with torch.no_grad():
    final_output = affinity_module1(
        s_inputs=sample_dict["s_inputs_affinity"].to(device),
        z=sample_dict["z_affinity"].to(device),
        x_pred=sample_dict["coords_affinity"].to(device),
        feats=sample_dict['feats'],
        multiplicity=1,
        use_kernels=sample_dict["use_kernels"]
    )
    
    final_output["affinity_probability_binary"] = torch.nn.functional.sigmoid(
        final_output["affinity_logits_binary"]
    )
    
    print("Final predictions after finetuning:")
    print(f"  Predicted affinity value: {final_output['affinity_pred_value'].item():.4f}")
    print(f"  Predicted binary probability: {final_output['affinity_probability_binary'].item():.4f}")
    print(f"  Target affinity value: {target_affinity_value:.4f}")
    print(f"  Target binary: {target_affinity_binary:.4f}")

# ============================================================================
# Update Checkpoint with Finetuned Parameters
# ============================================================================

print("\n" + "="*80)
print("Updating checkpoint with finetuned parameters...")
print("="*80 + "\n")

# Get the finetuned state dict from the module
finetuned_state_dict = affinity_module1.state_dict()

# Add the "affinity_module1." prefix back to match checkpoint structure
updated_affinity_params = {}
prefix = "affinity_module1."
for key, value in finetuned_state_dict.items():
    updated_affinity_params[prefix + key] = value

# Update the checkpoint state_dict with finetuned parameters
for key, value in updated_affinity_params.items():
    if key in state_dict:
        state_dict[key] = value
    else:
        print(f"Warning: Key {key} not found in original checkpoint")

# Update the checkpoint dictionary
checkpoint["state_dict"] = state_dict

# Save the updated checkpoint
output_checkpoint_path = "boltz2_aff_finetuned.ckpt"
torch.save(checkpoint, output_checkpoint_path)

print(f"✓ Updated checkpoint saved to: {output_checkpoint_path}")
print(f"  Total parameters in checkpoint: {len(state_dict)}")
print(f"  Finetuned affinity_module1 parameters: {len(updated_affinity_params)}")
print("\n" + "="*80)
print("Checkpoint update complete!")
print("="*80)