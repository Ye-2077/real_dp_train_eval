# Train & Eval DP on Real Robot

## `Quick Start`
**1. Prepare config file**
- In `diffusion_policy/config/task` add task config file, refer to `real_ur5_image.yaml`
- Modified `diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml`, change the task to your own task name

``` yaml
defaults:
  - _self_
  - task: real_ur5_image # change to your own task name
```

- add dataset path in task_config.yaml
```yaml
dataset_path: data/demo_clean_wok
```

**ATTENTION**
- Must include obs `Gripper_Pose` and set it in config file, if you want to use directly, like this

``` yaml
shape_meta: &shape_meta
  obs:
    camera_0:
      shape: ${task.image_shape}
      type: rgb
    ...
    Gripper_Pose:
      shape: [1]
      type: low_dim
  action: 
    shape: [7]
```

**2. eval**
```bash
$ eval_real_robot_UR5.py --input=Models/epoch=0050-train_loss=0.013.ckpt --output=record -ri=172.16.11.101
```

**3. train**
``` bash
$ python train.py --config-name=train_diffusion_unet_real_UR5_image_workspace
```

- continue from checkpoint
  - Modified `train.py`, add output_dir = ${Path_to_your_checkpont}
  ```Python
  workspace: BaseWorkspace = cls(cfg)
  # for example: workspace: BaseWorkspace = cls(
  #   cfg, 
  #   output_dir="/home/cyn/document/diffusion_policy/data/outputs/2025.01.20/15.02.15_train_diffusion_unet_image_real_ur5_image")
  ```
  - Modified workspace.yaml, for example `diffusion_policy/config/train_diffusion_unet_real_UR5_image_workspace.yaml`
    - change run_dir of `multi_run`, dir of  `hydra.run` and `hydra.sweep` to exactly `${Path_to_your_checkpont}`, for example 
    ```
    /home/cyn/document/diffusion_policy/data/outputs/2025.01.20/15.02.15_train_diffusion_unet_image_real_ur5_image
    ```