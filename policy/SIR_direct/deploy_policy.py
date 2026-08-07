# RMBench deploy adapter for sir_baseline (temporal_scene_graphs), Pattern A (in-process).
#
# Additive variant of policy/SIR (Pattern B, client/server) -- does not modify or replace it.
# Purpose: isolate whether the client/server socket split itself (JSON+base64-numpy
# serialization, the execute_horizon/update_obs bookkeeping across two processes, etc.) is
# introducing any behavior difference versus running the exact same RMBenchModelAdapter directly
# in the same process as the sim, matching how RMBench's Pattern A examples (pi05, DP) work.
#
# Verified before building this (see conversation record): for the image-only configuration,
# sir_baseline's actual runtime import chain never touches torch_geometric (the one real,
# confirmed dependency the RMBench conda env lacks), a checkpoint saved under sir_env's torch
# (2.9.1) loads fine under RMBench env's torch (2.7.1), and torchvision/einops/hydra/omegaconf are
# all present in the RMBench env. So Pattern A is expected to work for the image-only checkpoint;
# this variant exists to confirm that empirically, not just in isolated import checks. It reuses
# graph mode too, but that path still needs torch_geometric available in whichever env you run
# eval.sh in (i.e. it's not expected to help the graph-only checkpoint unless the RMBench env gets
# torch_geometric installed).
#
# Because there's no socket boundary, get_model() runs in the SAME process as eval()/reset_model()
# -- this makes the heavy sir_baseline/torch imports happen in RMBench's own eval_policy.py
# process, unlike policy/SIR where they're confined to the separate server process by design.
import os
import sys

import numpy as np

RMBENCH_CAMERAS = ("head_camera", "left_camera", "right_camera")


def _iter_named_cameras(cameras_obj):
    out = {}
    if getattr(cameras_obj, "collect_wrist_camera", False):
        out["left_camera"] = cameras_obj.left_camera
        out["right_camera"] = cameras_obj.right_camera
    for cam, name in zip(getattr(cameras_obj, "static_camera_list", []), getattr(cameras_obj, "static_camera_name", [])):
        out[name] = cam
    return out


def _raw_actor_segmentation(camera) -> np.ndarray:
    seg = np.asarray(camera.get_picture("Segmentation"))
    return seg[..., 1].astype(np.uint32)


def _raw_depth_mm(camera) -> np.ndarray:
    position = np.asarray(camera.get_picture("Position"))
    return (-position[..., 2] * 1000.0).clip(0, 65535).astype(np.uint16)


def _compute_bboxes(instance_map: np.ndarray) -> np.ndarray:
    rows = []
    for instance_id in np.unique(instance_map):
        if instance_id <= 0:
            continue
        ys, xs = np.where(instance_map == instance_id)
        if ys.size == 0:
            continue
        rows.append([int(instance_id), int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return np.asarray(rows, dtype=np.int32) if rows else np.zeros((0, 5), dtype=np.int32)


def _build_instance_id_to_name(task_env) -> dict:
    out = {}
    scene = getattr(task_env, "scene", None)
    if scene is None:
        return out
    for actor in scene.get_all_actors():
        out[int(actor.per_scene_id)] = actor.name or ""
    return out


def encode_obs(observation):
    images = {}
    for cam in RMBENCH_CAMERAS:
        cam_obs = observation.get("observation", {}).get(cam)
        if cam_obs is not None and "rgb" in cam_obs:
            images[cam] = np.asarray(cam_obs["rgb"])
    vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    return images, vector


def _grab_graph_payload(TASK_ENV, id_to_name_cache: dict):
    cameras = _iter_named_cameras(TASK_ENV.cameras)
    graphs = {}
    for cam_name, camera in cameras.items():
        if cam_name not in RMBENCH_CAMERAS:
            continue
        seg = _raw_actor_segmentation(camera)
        graphs[cam_name] = {
            "bboxes_xyxy": _compute_bboxes(seg).tolist(),
            "depth_mm": _raw_depth_mm(camera).tolist(),
            "intrinsic": camera.get_intrinsic_matrix().tolist(),
            "extrinsic": camera.get_extrinsic_matrix().tolist(),
            "instance_id_to_name": id_to_name_cache,
        }
    return graphs


def _as_bool(value, default=True):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes")


def get_model(usr_args):  # Runs in-process here (Pattern A) -- unlike policy/SIR, this is NOT
    # confined to a separate server process. sir_repo_path still needs adding to sys.path since
    # sir_baseline isn't installed as a package in the RMBench env.
    sir_repo_path = usr_args["sir_repo_path"]
    if sir_repo_path not in sys.path:
        sys.path.insert(0, sir_repo_path)

    from sir_baseline.rmbench_adapter import RMBenchModelAdapter

    checkpoint_epoch = usr_args.get("checkpoint_epoch")
    checkpoint_tag = f"epoch_{int(checkpoint_epoch)}" if checkpoint_epoch else "final"

    return RMBenchModelAdapter(
        checkpoint=usr_args["checkpoint"],
        device=usr_args.get("device"),
        execute_horizon=int(usr_args.get("execute_horizon", 64)),
        config_overrides=usr_args.get("config_overrides", []),
        use_ema=_as_bool(usr_args.get("use_ema"), default=True),
        checkpoint_tag=checkpoint_tag,
    )


def eval(TASK_ENV, model, observation):
    """Same logic as policy/SIR/deploy_policy.py's eval(), but every model.call(func_name=X,
    obs=Y) becomes a direct model.X(Y) -- model is a real RMBenchModelAdapter instance here, not
    a ModelClient socket stub, so no RPC dispatch is needed."""
    # Same convention as policy/SIR: eval() gets no usr_args here either (Pattern A's harness
    # calls eval_func(TASK_ENV, model, observation) with exactly those 3 args), so this is read
    # from the env var eval.sh sets from deploy_policy.yml's use_graph field.
    use_graph = os.environ.get("SIR_USE_GRAPH", "") == "1"

    if TASK_ENV.take_action_cnt == 0:
        instruction = TASK_ENV.get_instruction()
        model.set_instruction({"instruction": instruction})
        TASK_ENV._sir_id_to_name = _build_instance_id_to_name(TASK_ENV) if use_graph else {}

    images, vector = encode_obs(observation)
    payload = {"vector": vector, "images": images}
    if use_graph:
        payload["graphs"] = _grab_graph_payload(TASK_ENV, TASK_ENV._sir_id_to_name)

    actions = model.get_action(payload)
    for i, action in enumerate(actions):
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        if i == len(actions) - 1:
            break
        observation = TASK_ENV.get_obs()
        images, vector = encode_obs(observation)
        update_payload = {"vector": vector, "images": images}
        if use_graph:
            update_payload["graphs"] = _grab_graph_payload(TASK_ENV, TASK_ENV._sir_id_to_name)
        model.update_obs(update_payload)


def reset_model(model):
    model.reset_model()
