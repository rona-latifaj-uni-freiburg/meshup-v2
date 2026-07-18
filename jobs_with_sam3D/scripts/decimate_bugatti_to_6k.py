import os
import pymeshlab

SRC = "jobs_with_sam3D/meshes/upright_wheels_down/bugatti-centodieci_upright_wheels_down.ply"
OUT = "jobs_with_sam3D/meshes/decimated/bugatti-centodieci_upright_wheels_down_6kverts.ply"
TARGET_VERTS = 6000


def decimate_to_faces(target_faces: int):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(SRC)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        qualitythr=0.5,
        planarquadric=True,
    )
    m = ms.current_mesh()
    return ms, m.vertex_number(), m.face_number()


def main():
    lo, hi = 7000, 30000
    best = None

    for _ in range(12):
        mid = (lo + hi) // 2
        ms, v, f = decimate_to_faces(mid)
        err = abs(v - TARGET_VERTS)
        if best is None or err < best[0]:
            best = (err, v, f, mid, ms)
        if v > TARGET_VERTS:
            hi = mid - 1
        elif v < TARGET_VERTS:
            lo = mid + 1
        else:
            best = (0, v, f, mid, ms)
            break

    best_err, bv, bf, bmid, bms = best

    for tf in [max(1000, bmid - 1500), max(1000, bmid - 750), bmid + 750, bmid + 1500]:
        ms, v, f = decimate_to_faces(tf)
        err = abs(v - TARGET_VERTS)
        if err < best_err:
            best_err, bv, bf, bmid, bms = err, v, f, tf, ms

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    bms.save_current_mesh(OUT)

    print(f"source: {SRC}")
    print(f"saved: {OUT}")
    print(f"chosen_target_faces: {bmid}")
    print(f"result_vertices: {bv}")
    print(f"result_faces: {bf}")
    print(f"vertex_error: {best_err}")


if __name__ == "__main__":
    main()
