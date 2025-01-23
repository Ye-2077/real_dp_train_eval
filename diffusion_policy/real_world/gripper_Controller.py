import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from diffusion_policy.real_world.robotiq85_my_control.robotiq_gripper import RobotiqGripper




class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class gripper_Controller(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_ip, 
            port=63352,
            frequency=15,
            # frequency=125,
            move_max_speed=250.0,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            verbose=False
            ):
        super().__init__(name="gripper_Controller")
    
        self.robot_ip = robot_ip
        self.port = port
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose


        gripper = RobotiqGripper()
        gripper.connect(robot_ip,port)
        # gripper.activate()
        self.gripper = gripper

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            # 'gripper_position': 0.0,
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[gripper_Controller] Controller process spawned at ")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        # assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)
        # print("input_queue_put",pos)
        # print("input_queue_put_time",time.monotonic() - time.time() + target_time)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # start connection
        try:
            
            time.sleep(1)
            # get initial
            curr_pos = self.gripper.get_current_position()
            # curr_pos = 100.0
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos,0,0,0,0,0]]
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                # print('controller', target_pos, target_vel)

                # print("target_pos:",target_pos)
                # print("target_pos_time:",t_target)
                # self.gripper.move(target_pos,speed = target_vel)
                self.gripper.move(target_pos,speed=255,force=255)

                # get_position = self.gripper.get_current_position()

                # time.sleep(1e-3)

                # get state from robot
                state = {
                    # 'gripper_position': get_position,
                    'gripper_timestamp': time.time() 
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:  
                        target_pos = command['target_pos'] 
                        # print(target_pos)
                        target_time = command['target_time']
                        # print(target_time)
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        
                        # print("getget",pose_interp(time.monotonic()+0.6)[0])
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                    
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
            self.ready_event.clear()
            if self.verbose:
                print(f"[gripper_Controller] Disconnected from gripper")