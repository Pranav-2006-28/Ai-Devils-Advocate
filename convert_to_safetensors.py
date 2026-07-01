import torch
import os
from safetensors.torch import save_file

folder = r"c:\Users\prana\Desktop\Projects\MYPROJECT\MYPROJECT ADVISE\ai-devils-advocate\cuad-main\train_models\inlegalbert-cuad"
bin_file = os.path.join(folder, "pytorch_model.bin")
safe_file = os.path.join(folder, "model.safetensors")

if os.path.exists(bin_file):
    print("Loading bin file...")
    state_dict = torch.load(bin_file, weights_only=False)
    print("Saving safetensors...")
    save_file(state_dict, safe_file)
    print("Deleting bin file...")
    os.remove(bin_file)
    print("Done!")
else:
    print("bin file not found.")
