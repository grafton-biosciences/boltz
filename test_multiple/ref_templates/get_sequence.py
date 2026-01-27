# pip install biopython
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
from pathlib import Path


def extract_sequences(pdb_path: str):
    path = Path(pdb_path)
    if path.suffix.lower() == ".cif" or path.suffix.lower().endswith(".mmcif"):
        structure = MMCIFParser(QUIET=True).get_structure("struct", pdb_path)
    else:
        structure = PDBParser(QUIET=True).get_structure("struct", pdb_path)

    ppb = PPBuilder()
    seqs = {}  # chain_id -> sequence
    for model in structure:
        for chain in model:
            peptides = ppb.build_peptides(chain)
            if not peptides:
                continue
            # If there are breaks, join with 'X' to mark gaps (or just ''.join(..) if you prefer)
            seq = "X".join(str(peptide.get_sequence()) for peptide in peptides)
            seqs[chain.id] = seq
        break  # typically one model; remove if you want all
    return seqs


# Example usage
seqs = extract_sequences("target.pdb")
for chain_id, seq in seqs.items():
    print(f">{chain_id}\n{seq}")
