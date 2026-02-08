#!/bin/bash

# Download Omni6DPose 3D meshes
# Run this from the meshup_v2 directory

echo "==================================================================="
echo "Downloading Omni6DPose PAM Dataset (3D Meshes)"
echo "==================================================================="

# Install required Python packages if needed
pip install requests tqdm

# Run the Python download script
python download_omni6d_meshes.py

echo ""
echo "==================================================================="
echo "Download complete! Meshes are in: data/Omni6DPose/PAM/"
echo "==================================================================="
