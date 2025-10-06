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