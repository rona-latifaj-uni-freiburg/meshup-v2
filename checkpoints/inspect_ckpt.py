import torch

ckpt = torch.load("stage2_model.pt", map_location="cpu")

print("Top-level keys in checkpoint:")
print(ckpt.keys())

# If it's wrapped in 'model' or 'state_dict', inspect further
state_dict = ckpt.get('model') or ckpt.get('state_dict') or ckpt

print("\nParameter names and shapes:")
for k, v in state_dict.items():
    print(f"{k:60} {tuple(v.shape)}")
