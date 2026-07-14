import argparse
import json
import os

import h5py
import numpy as np


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _require_keys(group, keys, demo_name):
    missing = [key for key in keys if key not in group]
    if missing:
        raise KeyError(f"{demo_name} is missing required keys: {missing}")


def convert_demo(src_demo, dst_demo):
    required = (
        "o",
        "a",
        "l515_pc_xyzs_l515",
        "l515_pc_rgbs",
        "pyft_xyzs_l515",
        "pyft_quats_l515",
        "pyft_fs",
        "pyft_ts",
    )
    _require_keys(src_demo, required, src_demo.name)

    obs_indices = np.asarray(src_demo["o"][:, 0], dtype=np.int64)
    action_indices = np.asarray(src_demo["a"][:], dtype=np.int64)
    num_samples = obs_indices.shape[0]

    pc_xyz = src_demo["l515_pc_xyzs_l515"][obs_indices].astype(np.float32)
    pc_rgb = src_demo["l515_pc_rgbs"][obs_indices].astype(np.float32)
    pointcloud = np.concatenate([pc_xyz, pc_rgb], axis=-1)

    robot0_eef_pos = src_demo["pyft_xyzs_l515"][obs_indices].astype(np.float32)
    robot0_eef_quat = src_demo["pyft_quats_l515"][obs_indices].astype(np.float32)

    action_indices = action_indices[:, 0]
    pyft_xyzs = src_demo["pyft_xyzs_l515"][:].astype(np.float32)
    pyft_quats = src_demo["pyft_quats_l515"][:].astype(np.float32)
    pyft_forces = src_demo["pyft_fs"][:].astype(np.float32)
    pyft_torques = src_demo["pyft_ts"][:].astype(np.float32)
    action_pos = pyft_xyzs[action_indices]
    action_quat = pyft_quats[action_indices]
    action_force = pyft_forces[action_indices]
    action_torque = pyft_torques[action_indices]
    action_with_wrench = np.concatenate(
        [action_pos, action_quat, action_force, action_torque],
        axis=-1,
    )

    obs_group = dst_demo.create_group("obs")
    obs_group.create_dataset("pointcloud", data=pointcloud, compression="gzip", compression_opts=1)
    obs_group.create_dataset("robot0_eef_pos", data=robot0_eef_pos, compression="gzip", compression_opts=1)
    obs_group.create_dataset("robot0_eef_quat", data=robot0_eef_quat, compression="gzip", compression_opts=1)
    dst_demo.create_dataset(
        "action_with_wrench",
        data=action_with_wrench,
        compression="gzip",
        compression_opts=1,
    )

    dst_demo.attrs["num_samples"] = num_samples
    _copy_attrs(src_demo, dst_demo)


def convert_file(src_path, dst_path):
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        raise ValueError("source and destination must be different files")
    if os.path.exists(dst_path):
        raise FileExistsError(f"destination already exists: {dst_path}")

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        if "data" not in src:
            raise KeyError("source file is missing root 'data' group")

        src_data = src["data"]
        dst_data = dst.create_group("data")
        _copy_attrs(src_data, dst_data)

        demos = sorted(src_data.keys())
        total_samples = 0
        name_map = {}
        for idx, src_name in enumerate(demos):
            dst_name = f"demo_{idx}"
            name_map[dst_name] = src_name
            dst_demo = dst_data.create_group(dst_name)
            convert_demo(src_data[src_name], dst_demo)
            total_samples += int(dst_demo.attrs["num_samples"])

        dst_data.attrs["num_samples"] = total_samples
        dst_data.attrs["source_file"] = os.path.basename(src_path)
        dst_data.attrs["demo_name_map"] = json.dumps(name_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    convert_file(args.src, args.dst)


if __name__ == "__main__":
    main()
