# RMBench deploy adapter for sir_baseline (temporal_scene_graphs), Pattern B (client/server).
#
# This module is imported on BOTH sides of the socket split:
#   - Client (this RMBench conda env, script/eval_policy_client.py): imports the module (running
#     any top-level code) and calls eval()/... every sim step. get_model() is fetched but never
#     called here (see script/eval_policy_client.py:243/332) -- ModelClient is used instead.
#   - Server (sir_baseline's own env, script/policy_model_server.py): imports the module and
#     calls get_model(usr_args) once to build the real model object.
# So top-level imports MUST stay light (numpy/sys/os/json only) -- torch/sir_baseline only ever
# get imported inside get_model()'s body, which never executes client-side. See
# rmbench_onboarding.md Part 2.2 and RMBench/script/eval_policy_client.py for the confirmed
# contract this relies on.
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
    """Best-effort: SAPIEN scene actor enumeration -- see
    scripts/precompute_segdepth_rmbench.py's _build_instance_metadata docstring assumption 3 in
    the sir_baseline repo for the same technique/caveats."""
    out = {}
    scene = getattr(task_env, "scene", None)
    if scene is None:
        return out
    for actor in scene.get_all_actors():
        out[int(actor.id)] = actor.name or ""
    return out


def encode_obs(observation):  # Post-process the RMBench observation dict from TASK_ENV.get_obs()
    images = {}
    for cam in RMBENCH_CAMERAS:
        cam_obs = observation.get("observation", {}).get(cam)
        if cam_obs is not None and "rgb" in cam_obs:
            images[cam] = np.asarray(cam_obs["rgb"])
    vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    return images, vector


def _grab_graph_payload(TASK_ENV, id_to_name_cache: dict):
    """On-the-fly per-camera segmentation/depth grab, mirroring
    scripts/precompute_segdepth_rmbench.py's technique but eval-time/live. Only called when
    usr_args['use_graph'] is set in deploy_policy.yml (Tier 2, best-effort)."""
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
    """CLI --overrides only eval()s numeric-looking strings (see
    RMBench/script/policy_model_server.py:parse_args_and_config), so 'true'/'false' arrive as
    plain strings, not bools -- bool('false') is True in Python, so this needs explicit parsing."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes")


def get_model(usr_args):  # Only ever called server-side -- heavy imports live here.
    # usr_args['sir_repo_path'] points at the temporal_scene_graphs checkout, e.g.
    # /mnt/projects/Temporal_Scene_Graphs/sir_rmbench/temporal_scene_graphs
    sir_repo_path = usr_args["sir_repo_path"]
    if sir_repo_path not in sys.path:
        sys.path.insert(0, sir_repo_path)

    from sir_baseline.rmbench_adapter import RMBenchModelAdapter

    checkpoint_epoch = usr_args.get("checkpoint_epoch")
    checkpoint_tag = f"epoch_{int(checkpoint_epoch)}" if checkpoint_epoch else "final"

    return RMBenchModelAdapter(
        checkpoint=usr_args["checkpoint"],
        device=usr_args.get("device"),
        execute_horizon=int(usr_args.get("execute_horizon", 1)),
        config_overrides=usr_args.get("config_overrides", []),
        use_ema=_as_bool(usr_args.get("use_ema"), default=True),
        checkpoint_tag=checkpoint_tag,
    )


def eval(TASK_ENV, model, observation):
    """Runs client-side (RMBench conda env) every sim step -- see module docstring.

    use_graph is read from the SIR_USE_GRAPH env var (set by eval.sh from deploy_policy.yml's
    use_graph field) rather than usr_args, because eval() never receives usr_args directly here
    -- only get_model() does, and get_model() never runs client-side (see module docstring).
    This flag is a client-side compute optimization only: the server already no-ops the graph
    path when the checkpoint's manager.graph_modalities is empty, regardless of this flag.
    """
    use_graph = os.environ.get("SIR_USE_GRAPH", "") == "1"

    if TASK_ENV.take_action_cnt == 0:
        instruction = TASK_ENV.get_instruction()
        model.call(func_name="set_instruction", obs={"instruction": instruction})
        # Cache the live scene's actor id->name mapping once per episode (cheap, reused every
        # step) rather than every eval() call.
        TASK_ENV._sir_id_to_name = _build_instance_id_to_name(TASK_ENV) if use_graph else {}

    images, vector = encode_obs(observation)
    payload = {"vector": vector, "images": images}
    if use_graph:
        payload["graphs"] = _grab_graph_payload(TASK_ENV, TASK_ENV._sir_id_to_name)

    # get_action already caps its return to usr_args['execute_horizon'] actions (default 1) --
    # see RMBenchModelAdapter.get_action. We execute exactly that many steps here and then
    # return; the outer harness (script/eval_policy_client.py) calls eval() again next sim step
    # with a freshly-fetched observation, whose get_action() call is what pushes that frame into
    # the server's history and runs inference on it. For every action *before* the last one in
    # this chunk we push the intermediate frame via update_obs (history only, no inference) so a
    # multi-slot chunk (execute_horizon > 1) doesn't desync the obs-window from real elapsed sim
    # time -- but we must NOT push the post-chunk frame here too, or the next eval() call's
    # get_action would double-count it (see RMBenchModelAdapter.update_obs vs get_action).
    actions = model.call(func_name="get_action", obs=payload)
    for i, action in enumerate(actions):
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        if i == len(actions) - 1:
            break  # last slot in the chunk -- let the next eval() call's get_action handle it
        observation = TASK_ENV.get_obs()
        images, vector = encode_obs(observation)
        update_payload = {"vector": vector, "images": images}
        if use_graph:
            update_payload["graphs"] = _grab_graph_payload(TASK_ENV, TASK_ENV._sir_id_to_name)
        model.call(func_name="update_obs", obs=update_payload)


def reset_model(model):
    """Dead on the client/server path -- RMBench/script/eval_policy_client.py calls
    model.call(func_name='reset_model') directly (dispatches to
    RMBenchModelAdapter.reset_model server-side), bypassing this function entirely. Kept only for
    interface completeness / Pattern-A compatibility."""
    pass
