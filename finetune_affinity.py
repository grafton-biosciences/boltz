#!/usr/bin/env python3
"""
Finetuning Script for Boltz Affinity Module using Protein-Protein Affinity Dataset
Adapted for the new affinity_train_dataset format with configurable binding threshold.

Key Features:
- Configurable binding threshold for class balancing
- Mixed precision training with automatic scaling
- Gradient accumulation for larger effective batch sizes
- Advanced learning rate scheduling
- Comprehensive logging and visualization
- 90/10 train/test split
"""

import argparse
import json
import logging
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix,
    mean_absolute_error, precision_recall_curve, r2_score, roc_curve
)
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

import sys
sys.path.insert(0, '/pscratch/sd/v/vladygin/side_projects/ML_coding_series/Boltz-tests/new_boltz/boltz/src')
from boltz.model.modules.affinity_protein import ProteinProteinAffinityModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetune_affinity.log')
    ]
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def convert_kd_to_pkd_micromolar(kd_molar: float) -> float:
    """
    Convert Kd from Molar to pKd using micromolar scale.
    pKd = -log10(Kd in μM)
    """
    kd_micromolar = kd_molar * 1e6  # Convert M to μM
    pkd = -np.log10(kd_micromolar)
    return pkd


class AffinityDataset(Dataset):
    """
    Dataset for protein-protein affinity data from affinity_train_dataset.
    
    Features:
    - LAZY LOADING: Only loads metadata at init, loads tensors on-demand
    - Auto-balance threshold: uses median Kd for ~50/50 class split
    - 3-way train/val/test splitting
    - Memory efficient for large datasets
    """

    def __init__(
        self,
        msa_dir: str,
        yaml_dir: str,
        split: str = 'train',  # 'train', 'val', or 'test'
        train_ratio: float = 0.855,  # 85.5% train (of total)
        val_ratio: float = 0.045,    # 4.5% val (of total)
        # test_ratio = 1 - train_ratio - val_ratio = 10%
        binding_threshold: float = 1e-6,  # Default: 1 μM
        auto_balance_threshold: bool = True,  # Auto-select threshold for 50/50 balance
        max_samples: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True
    ):
        self.msa_dir = Path(msa_dir)
        self.yaml_dir = Path(yaml_dir)
        self.split = split
        self.binding_threshold = binding_threshold
        self.auto_balance_threshold = auto_balance_threshold
        self.verbose = verbose
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Only load metadata (not the heavy tensor data)
        self.sample_info = []  # Light metadata only
        self.failed_samples = []
        self._scan_samples(max_samples)

        # Split data into train/val/test
        self._split_data(seed)

        # Calculate statistics from metadata
        self._calculate_statistics()

    def _scan_samples(self, max_samples: Optional[int]):
        """Scan samples and load only metadata (not tensor data)."""
        sample_dirs = sorted([d for d in self.msa_dir.iterdir() if d.is_dir()])
        
        if max_samples is not None:
            sample_dirs = sample_dirs[:max_samples]

        logger.info(f"Scanning {len(sample_dirs)} samples (lazy loading)...")

        # First pass: collect all metadata
        for sample_dir in tqdm(sample_dirs, desc="Scanning samples", disable=not self.verbose):
            sample_name = sample_dir.name

            # Paths
            affinity_input_path = sample_dir / "affinity_input_sample.pkl"
            affinity_module_path = sample_dir / "affinity_module1.pkl"
            metadata_path = self.yaml_dir / sample_name / "manifest.json"

            if not all([affinity_input_path.exists(),
                       affinity_module_path.exists(),
                       metadata_path.exists()]):
                self.failed_samples.append((sample_name, "Missing files"))
                continue

            try:
                # Only load the small JSON metadata, not the tensor data
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                kd_molar = metadata["kd_value"]
                if kd_molar is None or kd_molar <= 0:
                    self.failed_samples.append((sample_name, "Invalid Kd value"))
                    continue

                kd_micromolar = kd_molar * 1e6
                pkd_micromolar = convert_kd_to_pkd_micromolar(kd_molar)

                # Binding strength categories (fixed thresholds)
                if kd_molar < 1e-9:
                    binding_strength = "strong"
                elif kd_molar < 1e-6:
                    binding_strength = "moderate"
                elif kd_molar < 1e-3:
                    binding_strength = "weak"
                else:
                    binding_strength = "non_binder"

                # Store only paths and metadata (no tensors!)
                # is_binder will be set after auto-threshold calculation
                sample_info = {
                    "name": sample_name,
                    "pdb_id": metadata.get("pdb_id", "unknown"),
                    "pdb_id_original": metadata.get("pdb_id_original", "unknown"),
                    "affinity_input_path": str(affinity_input_path),
                    "affinity_module_path": str(affinity_module_path),
                    "kd_molar": kd_molar,
                    "kd_micromolar": kd_micromolar,
                    "pkd_micromolar": pkd_micromolar,
                    "is_binder": None,  # Will be set below
                    "binding_strength": binding_strength,
                    "mutations": metadata.get("mutations", None)
                }
                self.sample_info.append(sample_info)

            except Exception as e:
                self.failed_samples.append((sample_name, str(e)))
                if self.verbose:
                    logger.warning(f"Failed to scan {sample_name}: {e}")

        logger.info(f"Successfully scanned {len(self.sample_info)} samples")
        if self.failed_samples:
            logger.warning(f"Failed to scan {len(self.failed_samples)} samples")

        # Auto-calculate threshold for 50/50 balance if enabled
        if self.auto_balance_threshold and len(self.sample_info) > 0:
            all_kd_values = [s["kd_molar"] for s in self.sample_info]
            median_kd = np.median(all_kd_values)
            self.binding_threshold = median_kd
            logger.info(f"Auto-balanced threshold: {median_kd:.2e} M (median Kd)")

        # Apply threshold to determine is_binder for each sample
        for sample in self.sample_info:
            sample["is_binder"] = float(sample["kd_molar"] < self.binding_threshold)

    def _split_data(self, seed: int):
        """Split data into train/val/test sets with stratification."""
        np.random.seed(seed)

        binders = [s for s in self.sample_info if s["is_binder"] == 1.0]
        non_binders = [s for s in self.sample_info if s["is_binder"] == 0.0]

        np.random.shuffle(binders)
        np.random.shuffle(non_binders)

        # Calculate split indices for each class
        # train_ratio + val_ratio + test_ratio = 1.0
        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        
        # Binders split
        n_train_binders = int(len(binders) * self.train_ratio)
        n_val_binders = int(len(binders) * self.val_ratio)
        # Rest goes to test
        
        # Non-binders split
        n_train_non_binders = int(len(non_binders) * self.train_ratio)
        n_val_non_binders = int(len(non_binders) * self.val_ratio)

        if self.split == 'train':
            self.sample_info = (binders[:n_train_binders] + 
                               non_binders[:n_train_non_binders])
        elif self.split == 'val':
            self.sample_info = (binders[n_train_binders:n_train_binders + n_val_binders] + 
                               non_binders[n_train_non_binders:n_train_non_binders + n_val_non_binders])
        else:  # test
            self.sample_info = (binders[n_train_binders + n_val_binders:] + 
                               non_binders[n_train_non_binders + n_val_non_binders:])

        np.random.shuffle(self.sample_info)
        logger.info(f"{self.split} set: {len(self.sample_info)} samples")

    def _calculate_statistics(self):
        """Calculate dataset statistics from metadata."""
        if not self.sample_info:
            self.stats = {}
            return

        pkd_values = [s["pkd_micromolar"] for s in self.sample_info]
        n_binders = sum(s["is_binder"] for s in self.sample_info)
        n_non_binders = len(self.sample_info) - n_binders

        self.stats = {
            "n_samples": len(self.sample_info),
            "pkd_mean": np.mean(pkd_values),
            "pkd_std": np.std(pkd_values),
            "pkd_min": np.min(pkd_values),
            "pkd_max": np.max(pkd_values),
            "pkd_median": np.median(pkd_values),
            "n_binders": int(n_binders),
            "n_non_binders": int(n_non_binders),
            "binder_ratio": n_binders / len(self.sample_info) if len(self.sample_info) > 0 else 0,
            "n_strong": sum(1 for s in self.sample_info if s["binding_strength"] == "strong"),
            "n_moderate": sum(1 for s in self.sample_info if s["binding_strength"] == "moderate"),
            "n_weak": sum(1 for s in self.sample_info if s["binding_strength"] == "weak"),
            "n_non_binder_cat": sum(1 for s in self.sample_info if s["binding_strength"] == "non_binder")
        }

        if self.verbose:
            logger.info(f"Dataset statistics for {self.split}:")
            logger.info(f"  pKd (μM scale): {self.stats['pkd_mean']:.2f} ± {self.stats['pkd_std']:.2f}")
            logger.info(f"  pKd range: [{self.stats['pkd_min']:.2f}, {self.stats['pkd_max']:.2f}]")
            logger.info(f"  Binders/Non-binders: {self.stats['n_binders']}/{self.stats['n_non_binders']} "
                       f"({self.stats['binder_ratio']:.1%} binders)")
            logger.info(f"  By strength - Strong: {self.stats['n_strong']}, Moderate: {self.stats['n_moderate']}, "
                       f"Weak: {self.stats['n_weak']}, Non-binder: {self.stats['n_non_binder_cat']}")

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        """Load sample data on-demand (lazy loading)."""
        info = self.sample_info[idx]
        
        # Load tensor data on-demand
        try:
            with open(info["affinity_input_path"], "rb") as f:
                affinity_data = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {info['name']}: {e}")
            # Return None - will be filtered by collate_fn
            return None

        return {
            # Input features
            "s_inputs_affinity": affinity_data["s_inputs_affinity"],
            "z_affinity": affinity_data["z_affinity"],
            "coords_affinity": affinity_data["coords_affinity"],
            "feats": affinity_data.get("feats", {}),
            "use_kernels": affinity_data.get("use_kernels", False),

            # Targets
            "target_pkd": torch.tensor(info["pkd_micromolar"], dtype=torch.float32),
            "target_binary": torch.tensor(info["is_binder"], dtype=torch.float32),
            "target_kd_micromolar": torch.tensor(info["kd_micromolar"], dtype=torch.float32),

            # Metadata
            "sample_name": info["name"],
            "pdb_id": info["pdb_id"],
            "binding_strength": info["binding_strength"]
        }


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for handling variable-sized tensors with proper batching."""
    if not batch:
        return None

    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    collated = {}
    batch_size = len(batch)

    # First pass: determine max dimensions across batch
    # s_inputs_affinity: [batch=1, seq, features] -> need max seq
    # z_affinity: [batch=1, seq, seq, features] -> need max seq  
    # coords_affinity: [batch=1, model=1, atoms, 3] -> need max atoms
    
    seq_lens = []
    atom_counts = []
    
    for b in batch:
        s = b["s_inputs_affinity"]
        if s.dim() == 3 and s.shape[0] == 1:
            seq_lens.append(s.shape[1])
        else:
            seq_lens.append(s.shape[0])
        
        c = b["coords_affinity"]
        if c.dim() == 4:  # [batch=1, model=1, atoms, 3]
            atom_counts.append(c.shape[2])
        elif c.dim() == 3:  # [model=1, atoms, 3]
            atom_counts.append(c.shape[1])
        else:
            atom_counts.append(c.shape[0])
    
    max_seq_len = max(seq_lens)
    max_atoms = max(atom_counts)

    # Handle s_inputs_affinity with padding
    s_inputs_list = []
    for b in batch:
        s = b["s_inputs_affinity"]
        # Remove batch dim if present: [1, seq, feat] -> [seq, feat]
        if s.dim() == 3 and s.shape[0] == 1:
            s = s.squeeze(0)
        # Pad sequence dimension
        if s.shape[0] < max_seq_len:
            pad_size = max_seq_len - s.shape[0]
            s = torch.nn.functional.pad(s, (0, 0, 0, pad_size), value=0)
        s_inputs_list.append(s)
    collated["s_inputs_affinity"] = torch.stack(s_inputs_list)  # [batch, seq, feat]

    # Handle z_affinity with padding
    z_list = []
    for b in batch:
        z = b["z_affinity"]
        # Remove batch dim if present: [1, seq, seq, feat] -> [seq, seq, feat]
        if z.dim() == 4 and z.shape[0] == 1:
            z = z.squeeze(0)
        # Pad both sequence dimensions
        if z.shape[0] < max_seq_len:
            pad_size = max_seq_len - z.shape[0]
            z = torch.nn.functional.pad(z, (0, 0, 0, pad_size, 0, pad_size), value=0)
        z_list.append(z)
    collated["z_affinity"] = torch.stack(z_list)  # [batch, seq, seq, feat]

    # Handle coords_affinity with padding
    coords_list = []
    for b in batch:
        c = b["coords_affinity"]
        # Expected: [1, 1, atoms, 3] -> squeeze to [atoms, 3], pad, then expand
        if c.dim() == 4:  # [batch=1, model=1, atoms, 3]
            c = c.squeeze(0).squeeze(0)  # [atoms, 3]
        elif c.dim() == 3:  # [model=1, atoms, 3]
            c = c.squeeze(0)  # [atoms, 3]
        # Pad atoms dimension
        if c.shape[0] < max_atoms:
            pad_size = max_atoms - c.shape[0]
            c = torch.nn.functional.pad(c, (0, 0, 0, pad_size), value=0)
        # Add model dimension back: [atoms, 3] -> [1, atoms, 3]
        c = c.unsqueeze(0)
        coords_list.append(c)
    collated["coords_affinity"] = torch.stack(coords_list)  # [batch, model=1, atoms, 3]

    # Handle scalar targets
    for field in ["target_pkd", "target_binary", "target_kd_micromolar"]:
        if field in batch[0]:
            collated[field] = torch.stack([b[field] for b in batch])

    # Handle feats dictionary - need to pad and stack tensors properly
    if "feats" in batch[0] and isinstance(batch[0]["feats"], dict):
        collated_feats = {}
        
        # Critical tensors that map between tokens and atoms
        # token_to_rep_atom: [batch=1, tokens, atoms] -> one-hot mapping
        # atom_to_token: [batch=1, atoms, tokens] -> one-hot mapping
        
        for key in ["token_to_rep_atom", "r_set_to_rep_atom", "token_to_center_atom"]:
            if key in batch[0]["feats"]:
                tensors = []
                for b in batch:
                    t = b["feats"][key]
                    # Remove batch dim: [1, tokens, atoms] -> [tokens, atoms]
                    if t.dim() == 3 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    # Pad: [tokens, atoms] -> [max_seq, max_atoms]
                    tokens_pad = max_seq_len - t.shape[0]
                    atoms_pad = max_atoms - t.shape[1]
                    t = torch.nn.functional.pad(t, (0, atoms_pad, 0, tokens_pad), value=0)
                    tensors.append(t)
                collated_feats[key] = torch.stack(tensors)  # [batch, max_seq, max_atoms]
        
        if "atom_to_token" in batch[0]["feats"]:
            tensors = []
            for b in batch:
                t = b["feats"]["atom_to_token"]
                # Remove batch dim: [1, atoms, tokens] -> [atoms, tokens]
                if t.dim() == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)
                # Pad: [atoms, tokens] -> [max_atoms, max_seq]
                atoms_pad = max_atoms - t.shape[0]
                tokens_pad = max_seq_len - t.shape[1]
                t = torch.nn.functional.pad(t, (0, tokens_pad, 0, atoms_pad), value=0)
                tensors.append(t)
            collated_feats["atom_to_token"] = torch.stack(tensors)  # [batch, max_atoms, max_seq]
        
        # Handle 1D mask tensors (token-level)
        for key in ["token_pad_mask", "mol_type", "affinity_token_mask"]:
            if key in batch[0]["feats"]:
                tensors = []
                for b in batch:
                    t = b["feats"][key]
                    # Remove batch dim: [1, tokens] -> [tokens]
                    if t.dim() == 2 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    # Pad to max_seq_len
                    if t.shape[0] < max_seq_len:
                        pad_size = max_seq_len - t.shape[0]
                        # For masks, pad with 0 (False for pad positions)
                        t = torch.nn.functional.pad(t, (0, pad_size), value=0)
                    tensors.append(t)
                collated_feats[key] = torch.stack(tensors)  # [batch, max_seq]
        
        # Copy remaining feats (non-tensor or scalar values)
        for key in batch[0]["feats"]:
            if key not in collated_feats:
                val = batch[0]["feats"][key]
                if isinstance(val, torch.Tensor):
                    # For other tensors, try to batch them if possible
                    if val.dim() >= 1 and val.shape[0] == 1:
                        # Repeat for batch size
                        collated_feats[key] = val.expand(batch_size, *val.shape[1:])
                    else:
                        collated_feats[key] = val
                else:
                    collated_feats[key] = val
        
        collated["feats"] = collated_feats
    else:
        collated["feats"] = batch[0].get("feats", {})

    collated["use_kernels"] = batch[0]["use_kernels"]
    collated["sample_names"] = [b["sample_name"] for b in batch]
    collated["pdb_ids"] = [b["pdb_id"] for b in batch]
    collated["binding_strengths"] = [b["binding_strength"] for b in batch]

    return collated


class AffinityFinetuner:
    """Finetuning class for affinity prediction."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = config.get("use_amp", True) and torch.cuda.is_available()

        logger.info(f"Using device: {self.device}")
        logger.info(f"Mixed precision training: {self.use_amp}")

        # Initialize wandb
        self.use_wandb = config.get("use_wandb", False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb()

        # Load model
        self.model = self._load_model()

        # Setup training
        self._setup_training()

        # Tracking
        self.train_history = []
        self.val_history = []
        self.best_metrics = {
            "val_loss": float("inf"),
            "val_mae": float("inf"),
            "val_r2": -float("inf"),
            "epoch": -1
        }
        self.patience_counter = 0

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        run_name = f"boltz_affinity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=self.config.get("wandb_project", "boltz-affinity"),
            name=run_name,
            config=self.config,
            tags=["affinity", "boltz", "protein-protein"]
        )
        logger.info(f"Initialized W&B run: {run_name}")

    def _load_model(self) -> ProteinProteinAffinityModule:
        """Load the affinity module."""
        # Get module config from first sample
        module_num = self.config.get("module", 1)
        sample_dir = list(Path(self.config["msa_dir"]).iterdir())[0]
        module_config_path = sample_dir / f"affinity_module{module_num}.pkl"

        logger.info(f"Loading module config from {module_config_path}")

        # Load pickle file with module config
        with open(module_config_path, "rb") as f:
            module_dict = pickle.load(f)

        # Get the correct args key based on module number
        args_key = f"affinity_model_args{module_num}"

        # Create module
        model = ProteinProteinAffinityModule(
            module_dict["token_s"],
            module_dict["token_z"],
            module_dict["protein_ligand_mode"],
            **module_dict[args_key]
        ).to(self.device)

        # Load pretrained weights
        checkpoint_path = Path(self.config.get("checkpoint_path"))
        if checkpoint_path.exists():
            logger.info(f"Loading pretrained weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Extract affinity module parameters
            affinity_state_dict = {}
            prefix = "affinity_module1."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    affinity_state_dict[new_key] = value

            if affinity_state_dict:
                model.load_state_dict(affinity_state_dict, strict=False)
                logger.info(f"Loaded {len(affinity_state_dict)} pretrained parameters")

            self.original_checkpoint = checkpoint
            self.original_state_dict = state_dict
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            self.original_checkpoint = None
            self.original_state_dict = {}

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model

    def _setup_training(self):
        """Setup optimizer, scheduler, losses."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-5)
        )

        scheduler_type = self.config.get("scheduler_type", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get("cosine_T0", 10),
                T_mult=self.config.get("cosine_Tmult", 2),
                eta_min=self.config.get("min_lr", 1e-7)
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5,
                patience=self.config.get("scheduler_patience", 5),
                min_lr=self.config.get("min_lr", 1e-7)
            )
        else:
            self.scheduler = None

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        if self.use_amp:
            self.scaler = GradScaler()

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = []
        epoch_pkd_losses = []
        epoch_binary_losses = []

        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")

        for batch_idx, batch in enumerate(progress_bar):
            if batch is None or isinstance(batch.get("s_inputs_affinity"), list):
                continue

            # Move to device
            s_inputs = batch["s_inputs_affinity"].to(self.device)
            z = batch["z_affinity"].to(self.device)
            coords = batch["coords_affinity"].to(self.device)
            target_pkd = batch["target_pkd"].to(self.device)
            target_binary = batch["target_binary"].to(self.device)
            
            # Move feats tensors to device
            feats = {}
            for key, val in batch["feats"].items():
                if isinstance(val, torch.Tensor):
                    feats[key] = val.to(self.device)
                else:
                    feats[key] = val

            with autocast(device_type='cuda', enabled=self.use_amp):
                output = self.model(
                    s_inputs=s_inputs,
                    z=z,
                    x_pred=coords,
                    feats=feats,
                    multiplicity=1,
                    use_kernels=batch["use_kernels"]
                )

                pred_pkd = output["affinity_pred_value"].squeeze()
                pred_binary = output["affinity_logits_binary"].squeeze()

                if pred_pkd.dim() == 0:
                    pred_pkd = pred_pkd.unsqueeze(0)
                if pred_binary.dim() == 0:
                    pred_binary = pred_binary.unsqueeze(0)

                loss_pkd = self.mse_loss(pred_pkd, target_pkd)
                loss_binary = self.bce_loss(pred_binary, target_binary)

                total_loss = (
                    self.config.get("pkd_loss_weight", 1.0) * loss_pkd +
                    self.config.get("binary_loss_weight", 0.5) * loss_binary
                )
                total_loss = total_loss / accumulation_steps

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("gradient_clip", 1.0)
                )
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_losses.append(total_loss.item() * accumulation_steps)
            epoch_pkd_losses.append(loss_pkd.item())
            epoch_binary_losses.append(loss_binary.item())

            progress_bar.set_postfix({
                "loss": f"{total_loss.item() * accumulation_steps:.4f}",
                "pkd": f"{loss_pkd.item():.4f}",
                "bin": f"{loss_binary.item():.4f}"
            })

        return {
            "loss": np.mean(epoch_losses),
            "pkd_loss": np.mean(epoch_pkd_losses),
            "binary_loss": np.mean(epoch_binary_losses)
        }

    def validate(self, dataloader: DataLoader, epoch: int, split_name: str = "val") -> Dict:
        """Validate the model on a given split."""
        self.model.eval()

        all_losses = []
        all_pkd_preds = []
        all_pkd_targets = []
        all_binary_preds = []
        all_binary_targets = []
        all_binding_strengths = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                if batch is None or isinstance(batch.get("s_inputs_affinity"), list):
                    continue

                s_inputs = batch["s_inputs_affinity"].to(self.device)
                z = batch["z_affinity"].to(self.device)
                coords = batch["coords_affinity"].to(self.device)
                target_pkd = batch["target_pkd"].to(self.device)
                target_binary = batch["target_binary"].to(self.device)
                
                # Move feats tensors to device
                feats = {}
                for key, val in batch["feats"].items():
                    if isinstance(val, torch.Tensor):
                        feats[key] = val.to(self.device)
                    else:
                        feats[key] = val

                with autocast(device_type='cuda', enabled=self.use_amp):
                    output = self.model(
                        s_inputs=s_inputs,
                        z=z,
                        x_pred=coords,
                        feats=feats,
                        multiplicity=1,
                        use_kernels=batch["use_kernels"]
                    )

                    pred_pkd = output["affinity_pred_value"].squeeze()
                    pred_binary = output["affinity_logits_binary"].squeeze()

                    if pred_pkd.dim() == 0:
                        pred_pkd = pred_pkd.unsqueeze(0)
                    if pred_binary.dim() == 0:
                        pred_binary = pred_binary.unsqueeze(0)

                    loss_pkd = self.mse_loss(pred_pkd, target_pkd)
                    loss_binary = self.bce_loss(pred_binary, target_binary)
                    total_loss = (
                        self.config.get("pkd_loss_weight", 1.0) * loss_pkd +
                        self.config.get("binary_loss_weight", 0.5) * loss_binary
                    )

                all_losses.append(total_loss.item())
                all_pkd_preds.extend(pred_pkd.cpu().numpy())
                all_pkd_targets.extend(target_pkd.cpu().numpy())
                all_binary_preds.extend(torch.sigmoid(pred_binary).cpu().numpy())
                all_binary_targets.extend(target_binary.cpu().numpy())
                all_binding_strengths.extend(batch["binding_strengths"])

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_pkd_preds, all_pkd_targets,
            all_binary_preds, all_binary_targets,
            all_binding_strengths
        )
        metrics["loss"] = np.mean(all_losses) if all_losses else float('inf')

        return metrics

    def _calculate_metrics(
        self, pkd_preds, pkd_targets, binary_preds, binary_targets, binding_strengths
    ) -> Dict:
        """Calculate comprehensive metrics."""
        pkd_preds = np.array(pkd_preds, dtype=np.float32)
        pkd_targets = np.array(pkd_targets, dtype=np.float32)
        binary_preds = np.array(binary_preds, dtype=np.float32)
        binary_targets = np.array(binary_targets, dtype=np.float32)

        if len(pkd_preds) == 0:
            return {"pkd_mae": 0, "pkd_rmse": 0, "pkd_r2": 0, "binary_accuracy": 0}
        
        # Handle NaN values
        valid_mask = ~(np.isnan(pkd_preds) | np.isnan(pkd_targets))
        if not valid_mask.any():
            logger.warning("All predictions are NaN!")
            return {"pkd_mae": float('inf'), "pkd_rmse": float('inf'), "pkd_r2": -1, "binary_accuracy": 0}
        
        pkd_preds = pkd_preds[valid_mask]
        pkd_targets = pkd_targets[valid_mask]
        binary_preds = binary_preds[valid_mask]
        binary_targets = binary_targets[valid_mask]
        binding_strengths = [s for i, s in enumerate(binding_strengths) if valid_mask[i]]

        # Regression metrics
        mae = mean_absolute_error(pkd_targets, pkd_preds)
        rmse = np.sqrt(np.mean((pkd_preds - pkd_targets) ** 2))
        r2 = r2_score(pkd_targets, pkd_preds) if len(pkd_preds) > 1 else 0

        # Correlation
        if len(pkd_preds) > 1:
            pearson_r, _ = stats.pearsonr(pkd_targets, pkd_preds)
            spearman_r, _ = stats.spearmanr(pkd_targets, pkd_preds)
        else:
            pearson_r = spearman_r = 0

        # Binary classification
        binary_preds_class = (binary_preds > 0.5).astype(float)
        accuracy = np.mean(binary_preds_class == binary_targets)

        cm = confusion_matrix(binary_targets, binary_preds_class)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        else:
            sensitivity = specificity = precision = f1 = 0

        # AUC
        if len(np.unique(binary_targets)) > 1:
            fpr, tpr, _ = roc_curve(binary_targets, binary_preds)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.5

        return {
            "pkd_mae": mae,
            "pkd_rmse": rmse,
            "pkd_r2": r2,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "binary_accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

    def _create_plots(
        self, pkd_preds, pkd_targets, binary_preds, binary_targets, binding_strengths, epoch, split_name: str = "val"
    ):
        """Create plots for a given split (train, val, or test)."""
        pkd_preds = np.array(pkd_preds, dtype=np.float32)
        pkd_targets = np.array(pkd_targets, dtype=np.float32)
        binary_preds = np.array(binary_preds, dtype=np.float32)
        binary_targets = np.array(binary_targets, dtype=np.float32)

        module_num = self.config.get("module", 1)
        module_tag = f"m{module_num}"

        if len(pkd_preds) == 0:
            logger.warning(f"No predictions available for {split_name} plots at epoch {epoch}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{split_name.upper()} Set - Module {module_num} - Epoch {epoch}', fontsize=14, fontweight='bold')

        # 1. pKd Correlation
        ax = axes[0, 0]
        ax.scatter(pkd_targets, pkd_preds, alpha=0.5, s=20)
        ax.plot([pkd_targets.min(), pkd_targets.max()],
                [pkd_targets.min(), pkd_targets.max()], 'k--', alpha=0.5)
        ax.set_xlabel('True pKd (μM scale)')
        ax.set_ylabel('Predicted pKd')
        ax.set_title(f'pKd Predictions')
        mae = mean_absolute_error(pkd_targets, pkd_preds)
        ax.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))

        # 2. Residual Plot
        ax = axes[0, 1]
        residuals = pkd_preds - pkd_targets
        ax.scatter(pkd_targets, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('True pKd')
        ax.set_ylabel('Residual')
        ax.set_title('Residual Plot')

        # 3. ROC Curve
        ax = axes[1, 0]
        if len(np.unique(binary_targets)) > 1:
            fpr, tpr, _ = roc_curve(binary_targets, binary_preds)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Only one class present', ha='center', va='center')
            ax.set_title('ROC Curve (N/A)')

        # 4. Confusion Matrix
        ax = axes[1, 1]
        cm = confusion_matrix(binary_targets, (binary_preds > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-binder', 'Binder'],
                    yticklabels=['Non-binder', 'Binder'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()
        plt.savefig(f'{split_name}_{module_tag}_epoch_{epoch}.png', dpi=150)
        plt.close()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Main training loop with train/val/test splits."""
        logger.info("Starting training...")

        for epoch in range(self.config["num_epochs"]):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)

            # Validation on val set (used for early stopping)
            val_metrics = self.validate(val_loader, epoch, split_name="val")
            self.val_history.append(val_metrics)

            # Also evaluate on test set
            test_metrics = self.validate(test_loader, epoch, split_name="test")

            # Generate plots at specified intervals
            should_plot = (epoch + 1) % self.config.get("plot_every", 5) == 0 or epoch == 0
            if should_plot:
                # Generate plots for all splits
                self._generate_split_plots(train_loader, epoch, "train")
                self._generate_split_plots(val_loader, epoch, "val")
                self._generate_split_plots(test_loader, epoch, "test")

            # Scheduler step (based on validation loss)
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Log
            logger.info(f"\nEpoch [{epoch+1}/{self.config['num_epochs']}] ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['pkd_mae']:.3f}")
            logger.info(f"  Test Loss: {test_metrics['loss']:.4f}, MAE: {test_metrics['pkd_mae']:.3f}, "
                       f"R²: {test_metrics['pkd_r2']:.3f}, Acc: {test_metrics['binary_accuracy']:.1%}")

            # Check best (using validation MAE for model selection)
            if val_metrics["pkd_mae"] < self.best_metrics["val_mae"]:
                self.best_metrics.update({
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["pkd_mae"],
                    "val_r2": val_metrics["pkd_r2"],
                    "test_mae": test_metrics["pkd_mae"],
                    "test_r2": test_metrics["pkd_r2"],
                    "epoch": epoch + 1
                })
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  New best model! Val MAE: {val_metrics['pkd_mae']:.4f}, Test MAE: {test_metrics['pkd_mae']:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.get("patience", 20):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"\nBest model at epoch {self.best_metrics['epoch']} with Val MAE: {self.best_metrics['val_mae']:.4f}, "
                   f"Test MAE: {self.best_metrics.get('test_mae', 'N/A')}")

    def _generate_split_plots(self, loader: DataLoader, epoch: int, split_name: str):
        """Generate plots for a specific data split."""
        pkd_preds, pkd_targets, binary_preds, binary_targets, binding_strengths = [], [], [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if batch is None or isinstance(batch.get("s_inputs_affinity"), list):
                    continue
                    
                # Move to device
                s_inputs = batch["s_inputs_affinity"].to(self.device)
                z = batch["z_affinity"].to(self.device)
                coords = batch["coords_affinity"].to(self.device)
                
                # Move feats tensors to device
                feats = {}
                for key, val in batch["feats"].items():
                    if isinstance(val, torch.Tensor):
                        feats[key] = val.to(self.device)
                    else:
                        feats[key] = val
                
                output = self.model(
                    s_inputs=s_inputs,
                    z=z,
                    x_pred=coords,
                    feats=feats,
                    multiplicity=1,
                    use_kernels=batch["use_kernels"]
                )
                
                pred_pkd = output["affinity_pred_value"].squeeze()
                pred_binary = output["affinity_logits_binary"].squeeze()
                
                if pred_pkd.dim() == 0:
                    pred_pkd = pred_pkd.unsqueeze(0)
                if pred_binary.dim() == 0:
                    pred_binary = pred_binary.unsqueeze(0)
                
                pkd_preds.extend(pred_pkd.cpu().numpy().flatten())
                binary_preds.extend(torch.sigmoid(pred_binary).cpu().numpy().flatten())
                pkd_targets.extend(batch["target_pkd"].numpy().flatten())
                binary_targets.extend(batch["target_binary"].numpy().flatten())
                binding_strengths.extend(batch["binding_strengths"])
        
        if len(pkd_preds) > 0:
            self._create_plots(pkd_preds, pkd_targets, binary_preds, binary_targets, 
                              binding_strengths, epoch, split_name)

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint."""
        module_num = self.config.get("module", 1)
        module_tag = f"m{module_num}"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "best_metrics": self.best_metrics
        }

        path = f"checkpoint_{module_tag}_best.pt" if is_best else f"checkpoint_{module_tag}_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best and self.original_checkpoint is not None:
            self._save_boltz_checkpoint()

    def _save_boltz_checkpoint(self):
        """Save in Boltz format."""
        module_num = self.config.get("module", 1)
        module_tag = f"m{module_num}"

        finetuned_state = self.model.state_dict()
        prefix = f"affinity_module{module_num}."
        for key, value in finetuned_state.items():
            self.original_state_dict[prefix + key] = value

        self.original_checkpoint["state_dict"] = self.original_state_dict
        output_path = f"boltz2_aff_{module_tag}_finetuned.ckpt"
        torch.save(self.original_checkpoint, output_path)
        logger.info(f"Saved Boltz checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Finetune Boltz affinity module")

    # Module selection
    parser.add_argument("--module", type=int, default=1, choices=[1, 2],
        help="Which affinity module to train (1 or 2)")

    # Data paths
    parser.add_argument("--msa_dir", type=str,
        default="/pscratch/sd/v/vladygin/side_projects/ML_coding_series/Boltz-tests/new_boltz/boltz/affinity_train_dataset/msa_folder_train")
    parser.add_argument("--yaml_dir", type=str,
        default="/pscratch/sd/v/vladygin/side_projects/ML_coding_series/Boltz-tests/new_boltz/boltz/affinity_train_dataset/yaml_folder_train")
    parser.add_argument("--checkpoint_path", type=str,
        default="/pscratch/sd/v/vladygin/side_projects/ML_coding_series/Boltz-tests/.boltz/boltz2_aff.ckpt")

    # Training params
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Loss weights
    parser.add_argument("--pkd_loss_weight", type=float, default=1.0)
    parser.add_argument("--binary_loss_weight", type=float, default=0.5)

    # Binding threshold for class balancing
    parser.add_argument("--binding_threshold", type=float, default=1e-6,
        help="Kd threshold in M for binary classification (default: 1e-6 = 1 μM)")
    parser.add_argument("--auto_balance_threshold", action="store_true", default=True,
        help="Auto-select threshold using median Kd for ~50/50 class balance")
    parser.add_argument("--no_auto_balance", dest="auto_balance_threshold", action="store_false",
        help="Disable auto-balance and use fixed binding_threshold")

    # Data split (train 85.5%, val 4.5%, test 10%)
    parser.add_argument("--train_ratio", type=float, default=0.855)
    parser.add_argument("--val_ratio", type=float, default=0.045)
    parser.add_argument("--max_samples", type=int, default=None)

    # Scheduler
    parser.add_argument("--scheduler_type", type=str, default="cosine",
        choices=["cosine", "plateau", "none"])
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_Tmult", type=int, default=2)

    # Other
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="boltz-affinity")

    args = parser.parse_args()
    set_seed(args.seed)
    config = vars(args)

    logger.info("="*60)
    logger.info(f"Boltz Affinity Finetuning - Module {args.module}")
    logger.info("="*60)
    logger.info(f"Training affinity_module{args.module}")
    if args.auto_balance_threshold:
        logger.info("Auto-balance threshold: ENABLED (will use median Kd)")
    else:
        logger.info(f"Binding threshold: {args.binding_threshold} M ({args.binding_threshold*1e6:.1f} μM)")

    # Create datasets (train/val/test)
    logger.info("\nCreating datasets...")

    dataset_kwargs = dict(
        msa_dir=args.msa_dir,
        yaml_dir=args.yaml_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        binding_threshold=args.binding_threshold,
        auto_balance_threshold=args.auto_balance_threshold,
        max_samples=args.max_samples,
        seed=args.seed
    )

    train_dataset = AffinityDataset(split="train", **dataset_kwargs)
    val_dataset = AffinityDataset(split="val", **dataset_kwargs)
    test_dataset = AffinityDataset(split="test", **dataset_kwargs)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Update config
    config.update({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_stats": train_dataset.stats,
        "val_stats": val_dataset.stats,
        "test_stats": test_dataset.stats,
        "binding_threshold_used": train_dataset.binding_threshold  # actual threshold used
    })

    # Create trainer
    trainer = AffinityFinetuner(config)

    # Train with all 3 loaders
    trainer.train(train_loader, val_loader, test_loader)

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

