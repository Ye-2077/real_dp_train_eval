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
```
$ eval_real_robot_UR5.py --input=Models/epoch=0050-train_loss=0.013.ckpt --output=record -ri=172.16.11.101
```
