#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result


def queued_job_count(user: str) -> int:
    result = run(["squeue", "-h", "-u", user, "-o", "%i"], check=False)
    if result.returncode != 0:
        return 999
    return len([line for line in result.stdout.splitlines() if line.strip()])


def submit(cmd: list[str], log, poll_seconds: int) -> str:
    while True:
        result = run(cmd, check=False)
        if result.returncode == 0:
            return result.stdout.strip().split(";")[0]
        if "QOSMax" in result.stderr or "job submit limit" in result.stderr:
            log.write(f"[WAIT] submit limit hit for: {' '.join(cmd)}\n")
            log.flush()
            time.sleep(poll_seconds)
            continue
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")


def load_submitted(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    submitted = set()
    for line in state_path.read_text().splitlines():
        if not line or line.startswith("image\t"):
            continue
        submitted.add(line.split("\t", 1)[0])
    return submitted


def append_state(state_path: Path, row: tuple[str, str, str]) -> None:
    write_header = not state_path.exists()
    with state_path.open("a", encoding="utf-8") as fh:
        if write_header:
            fh.write("image\tmask_job\tmesh_job\n")
        fh.write("\t".join(row) + "\n")
        fh.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit dense image SAM2/SAM3D jobs under small Slurm submit limits.")
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", default="dev_gpu_h100")
    parser.add_argument("--mask-time", default="00:20:00")
    parser.add_argument("--mesh-time", default="00:30:00")
    parser.add_argument("--max-submitted", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--user", default=None)
    args = parser.parse_args()

    base = args.base_path.resolve()
    manifest = args.manifest.resolve()
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = logs / "dense_images_job_state.tsv"
    progress_path = logs / "dense_images_submitter.log"

    user = args.user or run(["bash", "-lc", "printf %s \"$USER\""]).stdout.strip()
    mask_script = base / "scripts" / "slurm_sam2_mask_adaptive.sh"
    mesh_script = base / "scripts" / "slurm_sam3d_reconstruct_5k_adaptive.sh"
    images = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    submitted = load_submitted(state_path)

    with progress_path.open("a", encoding="utf-8") as log:
        log.write(f"[START] images={len(images)} max_submitted={args.max_submitted}\n")
        log.flush()

        for image_name in images:
            mesh_out = base / "sam3D" / "mesh" / f"{Path(image_name).stem}.ply"
            if image_name in submitted:
                log.write(f"[SKIP] already submitted {image_name}\n")
                log.flush()
                continue
            if mesh_out.exists():
                append_state(state_path, (image_name, "EXISTS", "EXISTS"))
                submitted.add(image_name)
                log.write(f"[SKIP] mesh exists {image_name}\n")
                log.flush()
                continue

            mask_exists = (base / "mask" / image_name).exists()
            needed_slots = 1 if mask_exists else 2
            while queued_job_count(user) > args.max_submitted - needed_slots:
                log.write(f"[WAIT] queue full before {image_name}\n")
                log.flush()
                time.sleep(args.poll_seconds)

            if mask_exists:
                mask_job = "EXISTS"
            else:
                mask_job = submit([
                    "sbatch",
                    "--parsable",
                    f"--chdir={base}",
                    f"--partition={args.partition}",
                    f"--time={args.mask_time}",
                    f"--export=ALL,BASE_PATH={base},IMAGE_NAME={image_name}",
                    str(mask_script),
                ], log, args.poll_seconds)

            dependency = [] if mask_job == "EXISTS" else [f"--dependency=afterok:{mask_job}"]
            mesh_job = submit([
                "sbatch",
                "--parsable",
                f"--chdir={base}",
                f"--partition={args.partition}",
                f"--time={args.mesh_time}",
                *dependency,
                f"--export=ALL,BASE_PATH={base},IMAGE_NAME={image_name}",
                str(mesh_script),
            ], log, args.poll_seconds)

            append_state(state_path, (image_name, mask_job, mesh_job))
            submitted.add(image_name)
            log.write(f"[SUBMIT] {image_name} mask={mask_job} mesh={mesh_job}\n")
            log.flush()

        log.write("[DONE] all images submitted or skipped\n")
        log.flush()


if __name__ == "__main__":
    main()
