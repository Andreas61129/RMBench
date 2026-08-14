"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _patch_cam_shapes(cfg, real_shape):
    # YAML anchors (&image_shape / *image_shape) are resolved by the YAML parser at load time,
    # so head_cam/left_cam/right_cam/front_cam each get an independent copy of the placeholder
    # [3, -1, -1] -- patching only head_cam's shape (the historical single-camera behavior)
    # leaves any other active camera key stuck at -1, -1, which crashes obs_encoder.output_shape()
    # (torch.zeros can't allocate a negative dimension). Patch every rgb camera key that's
    # actually present in this config's shape_meta.obs instead.
    for cam_key in ("head_cam", "front_cam", "left_cam", "right_cam"):
        if cam_key in cfg.task.shape_meta.obs:
            cfg.task.shape_meta.obs[cam_key].shape = list(real_shape)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    real_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.image_shape = real_shape
    _patch_cam_shapes(cfg, real_shape)
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = real_shape
    _patch_cam_shapes(cfg, real_shape)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
