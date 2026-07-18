# Mesh Creator Setup (SAM3D + SAM2/SAM3 masks)

This workspace is configured to run single-image 3D reconstruction with
[facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects),
using image/mask pairs as your supervisor described.

## 1) Folder layout

Use this structure at the repository root:

- `image/0000.jpg` (RGB image)
- `mask/0000.jpg` (binary object mask: foreground > 0)
- Outputs are created automatically in:
  - `sam3D/meta/0000.yaml`
  - `sam3D/splat/0000.ply`
  - `sam3D/mesh/0000.ply`

## 2) Clone SAM3D code (already done once in this workspace)

```bash
git clone https://github.com/facebookresearch/sam-3d-objects.git
```

If it is already cloned, skip this step.

## 3) Create environment + install SAM3D

Run:

```bash
bash scripts/setup_sam3d_env.sh
```

Notes:
- You need approved access to SAM3D checkpoints on Hugging Face.
- The script uses a blender-free dependency profile by default (`SKIP_BLENDER=1`) to avoid common `bpy` wheel errors.
- In this default mode it also skips `nvidia-pyindex` (not needed because package indexes are set directly).
- The installer uses `pip --no-cache-dir` to reduce disk quota pressure during installs.
- The installer skips `flash_attn` by default (`INSTALL_FLASH_ATTN=0`) to avoid common build failures.
- The installer skips `gsplat` by default (`INSTALL_GSPLAT=0`) to avoid heavy CUDA build failures on login/headless nodes.
- If checkpoints are already present, you can skip download:

```bash
SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh
```

If you hit disk quota errors, run this cleanup and retry:

```bash
pip cache purge
conda clean -a -y
SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh
```

If you explicitly want upstream full dependency installation (including Blender tooling), run:

```bash
SKIP_BLENDER=0 bash scripts/setup_sam3d_env.sh
```

If you are on a compatible GPU/toolchain and want `flash_attn`, run:

```bash
INSTALL_FLASH_ATTN=1 SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh
```

If you are on a GPU compute node and want to build gsplat:

```bash
INSTALL_GSPLAT=1 SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh
```

## 4) (Optional but recommended) Normalize masks

This makes masks binary and resizes masks to match image resolution:

```bash
conda activate sam3d-objects
python scripts/prepare_masks.py --base_path .
```

Dry-run mode:

```bash
python scripts/prepare_masks.py --base_path . --dry_run
```

## 5) Run reconstruction

Single image:

```bash
conda activate sam3d-objects
python process_image.py --base_path . --image_name 0000.jpg --sam3d_repo ./sam-3d-objects
```

Batch mode (all files with matching names in `image/` and `mask/`):

```bash
python process_image.py --base_path . --all --sam3d_repo ./sam-3d-objects
```

## 6) Orientation alignment with your rendering pipeline

Your script already supports export-time mesh rotation:

```bash
python process_image.py \
  --base_path . \
  --image_name 0000.jpg \
  --sam3d_repo ./sam-3d-objects \
  --rotate_x_deg -90
```

Use `--rotate_x_deg`, `--rotate_y_deg`, and `--rotate_z_deg` until:
- front direction matches MeshUp front direction,
- diffusion image-conditioning and DINO conditioning are in the same frame.

The used rotation is saved in `sam3D/meta/<name>.yaml` under `export_rotation_deg_xyz`.

## 7) About SAM3 vs SAM2 for mask generation

Both are fine for this pipeline as long as final mask files are binary and filename-matched with images.

Recommended practical workflow:
- Generate masks with SAM3 (or SAM2) in its own repo/tooling.
- Export one mask per image.
- Copy masks into this workspace under `mask/` with the same filename as `image/`.
- Run `scripts/prepare_masks.py` once to guarantee compatibility.

## 8) Submit as SLURM jobs

Create log directory once:

```bash
mkdir -p logs
```

Generate mask with SAM2 (for one image):

```bash
sbatch --partition=<usable_gpu_partition> --export=ALL,BASE_PATH=/work/dlclarge1/latifajr-mesh_creator_for_meshup,IMAGE_NAME=bugatti-centodieci.jpg scripts/slurm_sam2_mask.sh
```

Run SAM3D reconstruction for one image:

```bash
sbatch --partition=<usable_gpu_partition> --export=ALL,BASE_PATH=/work/dlclarge1/latifajr-mesh_creator_for_meshup,IMAGE_NAME=bugatti-centodieci.jpg scripts/slurm_sam3d_reconstruct.sh
```

Run SAM3D reconstruction for all matched image/mask pairs:

```bash
sbatch --partition=<usable_gpu_partition> --export=ALL,BASE_PATH=/work/dlclarge1/latifajr-mesh_creator_for_meshup,RUN_ALL=1 scripts/slurm_sam3d_reconstruct.sh
```

Tune optional job parameters with exported vars:
- `MODEL_ID` for SAM2 checkpoint choice
- `SEED` for SAM3D inference
- `ROTATE_X_DEG`, `ROTATE_Y_DEG`, `ROTATE_Z_DEG` for export alignment
- `SAM3D_REPO` if your SAM3D clone is in a non-default location

## 9) Quick troubleshooting

- Error: cannot locate sam-3d-objects repo
  - Pass `--sam3d_repo /absolute/path/to/sam-3d-objects`
- Error: config not found
  - Ensure `sam-3d-objects/checkpoints/hf/pipeline.yaml` exists
- Empty/bad meshes
  - Verify mask quality and object coverage
  - Try different seed: `--seed 123`
  - Check orientation and adjust rotation flags

- Error while installing PyTorch3D: `ModuleNotFoundError: No module named 'torch'`
  - This happens when PyTorch3D build isolation cannot see torch.
  - The setup script now preinstalls torch and installs PyTorch3D with `--no-build-isolation`.
  - Rerun: `SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh`

- Error while installing gsplat: `ModuleNotFoundError: No module named 'torch'`
  - This is the same build-isolation issue.
  - The setup script now installs gsplat with `--no-build-isolation`.
  - Rerun: `SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh`

- Error while installing gsplat: `OSError: CUDA_HOME environment variable is not set`
  - gsplat requires a CUDA compiler toolchain (`nvcc`) and CUDA paths.
  - The setup script now installs `cuda-toolkit=12.1` into the conda env if needed and exports `CUDA_HOME`.
  - Rerun: `SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh`

- Error while building gsplat: `IndexError: list index out of range` from `torch.utils.cpp_extension`
  - This can happen on headless/login nodes where no GPU is visible during build.
  - The setup script now sets a default `TORCH_CUDA_ARCH_LIST` so extension build can proceed.
  - You can override it if needed, for example: `TORCH_CUDA_ARCH_LIST="8.0" SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh`
