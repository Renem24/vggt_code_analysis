import torch
import os
from glob import glob
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)



input_obj = "forest"



# Load and preprocess example images (replace with your own image paths)
image_names = []

image_names = glob(os.path.join(f"/home/lym/lym/Github/vggt_code_analysis/examples/kitchen/images", "00.png"))
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

# print(model)
# dict_keys(['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images'])

depth = predictions['world_points']
depth = depth.squeeze(0).squeeze(-1)

print(f"Depth shape: {depth.shape}")

print(f"Depth: {depth}")

fig, axes = plt.subplots(2, 4, figsize=(20,10))
axes = axes.flatten()

for i in range(depth.shape[0]):
    base_filename = os.path.basename(image_names[i])
    filename_no_ext = os.path.splitext(base_filename)[0]
    
    depth_map = depth[i].cpu().numpy() if depth.device.type == 'cuda' else depth[i].numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_map, cmap='plasma')
    # plt.colorbar(label='Depth')
    plt.axis('off')
    plt.title(f'Depth Map - {filename_no_ext}')

    vis_file_path = os.path.join(f"/home/lym/lym/Github/vggt_code_analysis/output_depthmaps", f"{filename_no_ext}_depth.png")
    plt.savefig(vis_file_path, bbox_inches='tight', dpi=150)
    plt.close()