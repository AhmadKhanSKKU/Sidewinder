# my-attacker/src/advGenerate/test_mamba.py

# Import Libraries

import torch
from mamba_pytorch import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

# Create dummy network flow: Batch Size 32, Seq Length 512, Embedding Dim 128
dummy_input = torch.randn(32, 512, 128).to(device)

# Initialize Mamba Block
mamba_model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to(device)

try:
    output = mamba_model(dummy_input)
    print(f"Success! Output shape: {output.shape}")
    print("Mamba pure PyTorch implementation is fully functional.")
except Exception as e:
    print(f"Error during Mamba forward pass: {e}")
