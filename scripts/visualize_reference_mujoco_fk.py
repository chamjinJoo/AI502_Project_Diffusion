"""Visualize 65D reference chunks with MuJoCo forward kinematics.

This is a lightweight diagnostic renderer: it strips mesh geoms from a G1 MJCF,
uses MuJoCo only for FK, and draws the resulting body positions as a skeleton.
It is meant for checking whether history/target/prediction are aligned in the
same root-relative frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternions shaped [..., 4] in MuJoCo/wxyz order."""
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, 1e-8, None)


def _load_fk_model(xml_path: Path) -> mujoco.MjModel:
    """Load an MJCF after removing mesh-only visual geometry.

    Some bundled STL files are ASCII and can fail MuJoCo's mesh decoder. Meshes
    are not needed for FK, so removing them keeps the kinematic tree intact.
    """
    root = ET.parse(xml_path).getroot()
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag in {"mesh", "texture", "material"}:
                asset.remove(child)

    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.attrib.get("type") == "mesh":
                parent.remove(child)

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _chunk_to_qpos(chunk: np.ndarray, base_height: float) -> np.ndarray:
    """Convert a [T, 65] chunk to MuJoCo qpos [T, 36]."""
    if chunk.ndim != 2 or chunk.shape[1] != 65:
        raise ValueError(f"chunk must have shape [T, 65], got {chunk.shape}")
    qpos = np.zeros((chunk.shape[0], 36), dtype=np.float64)
    qpos[:, :3] = chunk[:, 62:65]
    qpos[:, 2] += base_height
    qpos[:, 3:7] = _normalize_quat(chunk[:, 58:62])
    qpos[:, 7:36] = chunk[:, :29]
    return qpos


def _fk_body_positions(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Run MuJoCo FK and return body positions [T, nbody, 3]."""
    data = mujoco.MjData(model)
    xpos = np.zeros((qpos.shape[0], model.nbody, 3), dtype=np.float32)
    for i, state in enumerate(qpos):
        data.qpos[:] = state[: model.nq]
        mujoco.mj_forward(model, data)
        xpos[i] = data.xpos
    return xpos


def _set_equal_axes(ax, points: np.ndarray) -> None:
    """Set a stable 3D view box for all frames."""
    xyz_min = points.min(axis=(0, 1))
    xyz_max = points.max(axis=(0, 1))
    center = (xyz_min + xyz_max) * 0.5
    radius = float(np.max(xyz_max - xyz_min) * 0.55)
    radius = max(radius, 0.8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _draw_skeleton(ax, points: np.ndarray, edges: list[tuple[int, int]], color: str, label: str, alpha: float = 1.0) -> None:
    """Draw one FK skeleton from body positions [nbody, 3]."""
    first = True
    for parent, child in edges:
        xs = [points[parent, 0], points[child, 0]]
        ys = [points[parent, 1], points[child, 1]]
        zs = [points[parent, 2], points[child, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=2.0, alpha=alpha, label=label if first else None)
        first = False
    ax.scatter(points[1:, 0], points[1:, 1], points[1:, 2], color=color, s=8, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", type=Path, required=True, help="History chunk shaped [H, 65]")
    parser.add_argument("--target", type=Path, required=True, help="Target future chunk shaped [K, 65]")
    parser.add_argument("--pred", type=Path, required=True, help="Predicted future chunk shaped [K, 65]")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("AI502TermProject-main/gear_sonic/data/robots/g1/g1_29dof_old.xml"),
        help="G1 MJCF used for FK; mesh geoms are stripped at load time.",
    )
    parser.add_argument("--output", type=Path, default=Path("samples/future_reference_mujoco_fk.gif"))
    parser.add_argument("--meta_output", type=Path, default=None)
    parser.add_argument("--base_height", type=float, default=0.8, help="Constant z offset for root-relative chunks")
    parser.add_argument("--fps", type=int, default=8, help="GIF playback FPS")
    args = parser.parse_args()

    cond = np.load(args.cond).astype(np.float32)  # [H, 65]
    target = np.load(args.target).astype(np.float32)  # [K, 65]
    pred = np.load(args.pred).astype(np.float32)  # [K, 65]
    if target.shape != pred.shape:
        raise ValueError(f"target and pred shapes must match, got {target.shape} and {pred.shape}")

    model = _load_fk_model(args.xml)
    if model.nq != 36:
        raise ValueError(f"expected 36 qpos values for G1 29DOF, got model.nq={model.nq}")

    cond_xpos = _fk_body_positions(model, _chunk_to_qpos(cond, args.base_height))  # [H, B, 3]
    target_xpos = _fk_body_positions(model, _chunk_to_qpos(target, args.base_height))  # [K, B, 3]
    pred_xpos = _fk_body_positions(model, _chunk_to_qpos(pred, args.base_height))  # [K, B, 3]

    edges = []
    for body_id in range(1, model.nbody):
        parent = int(model.body_parentid[body_id])
        if parent > 0:
            edges.append((parent, body_id))

    all_points = np.concatenate([cond_xpos, target_xpos, pred_xpos], axis=0)
    total_frames = cond.shape[0] + target.shape[0]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int) -> None:
        ax.clear()
        _set_equal_axes(ax, all_points[:, 1:])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-65)

        if frame_idx < cond.shape[0]:
            _draw_skeleton(ax, cond_xpos[frame_idx], edges, "black", "history")
            ax.set_title(f"MuJoCo FK reference check - history frame {frame_idx + 1}/{cond.shape[0]}")
        else:
            future_idx = frame_idx - cond.shape[0]
            _draw_skeleton(ax, target_xpos[future_idx], edges, "black", "target")
            _draw_skeleton(ax, pred_xpos[future_idx], edges, "red", "prediction", alpha=0.82)
            ax.set_title(f"MuJoCo FK reference check - future frame {future_idx + 1}/{target.shape[0]}")
        ax.legend(loc="upper right")

    anim = plt.matplotlib.animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    meta_path = args.meta_output or args.output.with_suffix(".json")
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    meta_path.write_text(
        json.dumps(
            {
                "type": "mujoco_fk_skeleton",
                "xml": str(args.xml),
                "output": str(args.output),
                "cond": str(args.cond),
                "target": str(args.target),
                "pred": str(args.pred),
                "base_height": args.base_height,
                "body_names": body_names,
                "note": "Meshes are stripped; body positions come from MuJoCo FK. Root-relative chunks receive a constant z offset only for display.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {args.output}")
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()
