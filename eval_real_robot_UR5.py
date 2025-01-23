"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
# from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.real_env_for_eval import RealEnv_for_eval
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform


OmegaConf.register_new_resolver("eval", eval, replace=True)

# python eval_real_robot_UR5.py --input=Models/epoch=0050-train_loss=0.013.ckpt --output=record -ri=172.16.11.101


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, default="172.16.11.101", help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")



def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv_for_eval(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=False,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)

            # Should be the same as demo
            # realsense exposure
            env.realsense.set_exposure(exposure=None, gain=None)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=None)

            print("Waiting for realsense")
            time.sleep(5.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                # assert action.shape[-1] == 2
                del result

            print('Ready!')
            while True:
                # # ========= human control loop ==========
                # print("Human in control!")
                # state = env.get_robot_state()
                # target_pose = state['TargetTCPPose']
                # t_start = time.monotonic()
                # iter_idx = 0
                # while True:
                #     # calculate timing
                #     t_cycle_end = t_start + (iter_idx + 1) * dt
                #     t_sample = t_cycle_end - command_latency
                #     t_command_target = t_cycle_end + dt

                #     # pump obs
                #     obs = env.get_obs()

                #     # visualize
                #     episode_id = env.replay_buffer.n_episodes
                #     vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                #     match_episode_id = episode_id
                #     if match_episode is not None:
                #         match_episode_id = match_episode
                #     if match_episode_id in episode_first_frame_map:
                #         match_img = episode_first_frame_map[match_episode_id]
                #         ih, iw, _ = match_img.shape
                #         oh, ow, _ = vis_img.shape
                #         tf = get_image_transform(
                #             input_res=(iw, ih), 
                #             output_res=(ow, oh), 
                #             bgr_to_rgb=False)
                #         match_img = tf(match_img).astype(np.float32) / 255
                #         vis_img = np.minimum(vis_img, match_img)

                #     text = f'Episode: {episode_id}'
                #     cv2.putText(
                #         vis_img,
                #         text,
                #         (10,20),
                #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #         fontScale=0.5,
                #         thickness=1,
                #         color=(255,255,255)
                #     )
                #     cv2.imshow('default', vis_img[...,::-1])
                #     key_stroke = cv2.pollKey()
                #     if key_stroke == ord('q'):
                #         # Exit program
                #         env.end_episode()
                #         exit(0)
                #     elif key_stroke == ord('c'):
                #         # Exit human control loop
                #         # hand control over to the policy
                #         break

                #     precise_wait(t_sample)
                #     # get teleop command
                #     sm_state = sm.get_motion_state_transformed()
                #     # print(sm_state)
                #     dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                #     drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
  
                #     if not sm.is_button_pressed(0):
                #         # translation mode
                #         drot_xyz[:] = 0
                #     else:
                #         dpos[:] = 0
                #     if not sm.is_button_pressed(1):
                #         # 2D translation mode
                #         dpos[2] = 0    

                #     drot = st.Rotation.from_euler('xyz', drot_xyz)
                #     target_pose[:3] += dpos
                #     target_pose[3:] = (drot * st.Rotation.from_rotvec(
                #         target_pose[3:])).as_rotvec()
                #     # clip target pose
                #     target_pose[:2] = np.clip(target_pose[:2], [0.25, -0.45], [0.77, 0.40])

                #     # execute teleop command
                #     env.exec_actions(
                #         actions=[target_pose], 
                #         timestamps=[t_command_target-time.monotonic()+time.time()])
                #     precise_wait(t_cycle_end)
                #     iter_idx += 1


                # ========= human control loop ==========
                print("goto init control and wait!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_command_target = t_cycle_end + 4
                    # pump obs
                    obs = env.get_obs()
                    
                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]

                    if 'camera_1' in obs:
                        vis_img2 = obs[f'camera_1'][-1,:,:,::-1].copy()
                        vis_img2 = cv2.resize(vis_img2, (0, 0), fx = 0.1, fy = 0.1)
                        W1, H1 = vis_img.shape[1::-1]  # 大图尺寸
                        W2, H2 = vis_img2.shape[1::-1]  # 小图尺寸
                        vis_img[H1-H2 : H1, W1-W2:W1] = vis_img2

                    # point_cloud = obs[f'camera_{vis_camera_idx}_depth']
                    # print(point_cloud.any())

                    text = 'Episode: {}, Time: {:.1f}'.format(
                        episode_id, time.monotonic() - t_start
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    # key_stroke = cv2.pollKey()
                    key = cv2.waitKey(1)
                    # if key == ord('k'):
                    #     depth_file_path = "/home/demphi/touch_ws/src/diffusion_policy/data/depth/3.png"
                    #     # cv2.imwrite(depth_file_path, frame)

                    if key == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key == ord('c'):
                        # Exit human control loop
                        # hand control over to the policy
                        img_save_root = "/home/rob/songzherui/touch_ws/src/diffusion_policy/data/eval_img_save/episode"
                        if 'camera_0' in obs:
                            cv2.imwrite(img_save_root+str(episode_id)+"_init_out_view.png", (obs['camera_0'][-1,:,:,::-1].copy()*255).astype(np.uint8) )
                        if 'camera_1' in obs:
                            
                            cv2.imwrite(img_save_root+str(episode_id)+"_init_hand_view.png", (obs['camera_1'][-1,:,:,::-1].copy()*255).astype(np.uint8))
                        break
                    elif key==ord('i'):
                        env.exec_actions(
                            # actions=[[0.4146965, -0.114739, 0.13797, -1.5657, 1.55159, -0.996,5.1]], 
                            actions=[[0.4976, -0.1089744, 0.467, -1.238, 1.2668, -1.19559,5.1]],    ######## tcp_pose + gripper
                            timestamps=[t_command_target-time.monotonic()+time.time()])
                    elif key==ord('r'):
                        env.exec_actions(
                            # actions=[[0.4146965, -0.114739, 0.13797, -1.5657, 1.55159, -0.996,5.1]], 
                            actions=[[0.42, -0.233, 0.1954, -1.332, 1.78, -1.2078,5.1]],    ######## tcp_pose + gripper
                            # 0.408, -0.207287, 0.228, -1.25, 1.66, -1.14
                            timestamps=[t_command_target-time.monotonic()+time.time()])     
                    elif key==ord('l'):
                        env.exec_actions(
                            # actions=[[0.4146965, -0.114739, 0.13797, -1.5657, 1.55159, -0.996,5.1]], 
                            actions=[[0.45958, 0.183, 0.180, -1.696, 1.325362, -0.794,5.1]],    ######## tcp_pose + gripper
                            #0.45838, 0.1925, 0.224, -1.46, 1.5798, -0.89
                            timestamps=[t_command_target-time.monotonic()+time.time()])     
                    elif key==ord('m'):
                        env.exec_actions(
                            # actions=[[0.4146965, -0.114739, 0.13797, -1.5657, 1.55159, -0.996,5.1]], 
                            # actions=[[0.3659, -0.00547575, 0.118, -1.8, 1.736942, -0.62,5.1]],    ######## tcp_pose + gripper
                            actions=[[0.43724650002413534, -0.009136663972630174, 0.16258519323065612, -1.5374798969098493, 1.6130417575527476, -0.972797139225697,5.1]],    ######## tcp_pose + gripper
                            #0.45838, 0.1925, 0.224, -1.46, 1.5798, -0.89
                            timestamps=[t_command_target-time.monotonic()+time.time()])    
                    elif key==ord('p'):
                        env.exec_actions(
                            # actions=[[0.242, -0.22, 0.393, -1.35498, 1.9, -1.07,209]],    ######## tcp_pose + gripper
                            actions=[[0.317667, -0.258424, 0.358917, -1.28645, 1.704, -1.13939,209]],    ######## tcp_pose + gripper close
                            timestamps=[t_command_target-time.monotonic()+time.time()])
                    elif key==ord('o'):
                        env.exec_actions(
                            # actions=[[0.242, -0.22, 0.393, -1.35498, 1.9, -1.07,209]],    ######## tcp_pose + gripper
                            actions=[[0.317667, -0.258424, 0.358917, -1.28645, 1.704, -1.13939,5.1]],    ######## tcp_pose + gripper open
                            timestamps=[t_command_target-time.monotonic()+time.time()])
                    elif key == ord('d'):
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                    elif key == ord('t'):
                        get_touch_gripper_signal = False
                        use_touch = True
                        geted_touch_gripper_data = obs['Gripper_Pose'][1][0]
                    elif key == ord('e'):
                        use_touch = False

                    precise_wait(t_cycle_end)
                    iter_idx += 1








                
                # ========== policy control loop ==============
                
                
                
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        # if delta_action:
                        #     assert len(action) == 1
                        #     if perv_target_pose is None:
                        #         perv_target_pose = obs['robot_eef_pose'][-1]
                        #     this_target_pose = perv_target_pose.copy()
                        #     this_target_pose[[0,1]] += action[-1]
                        #     perv_target_pose = this_target_pose
                        #     this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        # else:
                        #     this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                        #     this_target_poses[:] = target_pose
                        #     this_target_poses[:,[0,1]] = action
                        this_target_poses = action
                        assert this_target_poses.shape[1] == 7

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # clip actions
                        this_target_poses[:,:2] = np.clip(
                            this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        curr_pose = obs['robot_eef_pose'][-1]
                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        if dist < 0.03:
                            # in termination area
                            curr_timestamp = obs['timestamp'][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp
                                if term_area_time > 0.5:
                                    terminate = True
                                    print('Terminated by the policy!')
                        else:
                            # out of the area
                            term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
