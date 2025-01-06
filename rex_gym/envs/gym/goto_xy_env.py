"""This file implements the gym environment of rex alternating legs.

"""
import time
import math
import random
import pybullet

from gym import spaces
import numpy as np
from .. import rex_gym_env
from ...model import rex_constants
from rex_gym.model.gait_planner import GaitPlanner
from ...model.kinematics import Kinematics

NUM_LEGS = 4


class RexGoToXYEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex.

  It simulates the locomotion of a rex, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the rex walks in 2000 steps and penalizes the energy
  expenditure or how near rex is to the target position.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}
    load_ui = True
    is_terminating = False

    def __init__(self,
                 debug=False,
                 urdf_version=None,
                 control_time_step=0.005,
                 action_repeat=5,
                 control_latency=0,
                 pd_latency=0,
                 on_rack=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 render=False,
                 num_steps_to_log=2000,
                 env_randomizer=None,
                 log_path=None,
                 target_position=None,
                 backwards=None,
                 signal_type="ik",
                 terrain_type="plane",
                 terrain_id=None,
                 mark='base'):
        """Initialize the rex alternating legs gym environment.

    Args:
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. Refer to
        rex_gym_env for more details.
      control_time_step: The time step between two successive control signals.
      action_repeat: The number of simulation steps that an action is repeated.
      control_latency: The latency between get_observation() and the actual
        observation. See minituar.py for more details.
      pd_latency: The latency used to get motor angles/velocities used to
        compute PD controllers. See rex.py for more details.
      on_rack: Whether to place the rex on rack. This is only used to debug
        the walk gait. In this mode, the rex's base is hung midair so
        that its walk gait is clearer to visualize.
      motor_kp: The P gain of the motor.
      motor_kd: The D gain of the motor.
      remove_default_joint_damping: Whether to remove the default joint damping.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode. If the
        number of steps is over num_steps_to_log, the environment will still
        be running, but only first num_steps_to_log will be recorded in logging.
      env_randomizer: An instance (or a list) of EnvRanzomier(s) that can
        randomize the environment during when env.reset() is called and add
        perturbations when env.step() is called.
      log_path: The path to write out logs. For the details of logging, refer to
        rex_logging.proto.
    """
        super(RexGoToXYEnv,
              self).__init__(urdf_version=urdf_version,
                             accurate_motor_model_enabled=True,
                             motor_overheat_protection=True,
                             hard_reset=False,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=False,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat,
                             target_position=target_position,
                             signal_type=signal_type,
                             backwards=backwards,
                             debug=debug,
                             terrain_id=terrain_id,
                             terrain_type=terrain_type,
                             mark=mark)
        # (eventually) allow different feedback ranges/action spaces for different signals
        action_max = 1
        action_dim = 1
        action_high = np.array([action_max] * action_dim)
        action_low = np.array([-action_max] * action_dim)
        self.action_space = spaces.Box(action_low, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = -20
        self._signal_type = signal_type
        self._gait_planner = GaitPlanner("walk")
        self._kinematics = Kinematics()
        self.goal_reached = False
        self._stay_still = False
        self.is_terminating = False
        self._target_position = (-1.2, 0.5)
        self._target_length = math.sqrt(self._target_position[0] ** 2 + self._target_position[1] ** 2)
        self.prev_x = 0.0
        self.first_reward = True
        self.backwards = False
        self.last_pos = (0.0, 0.0)
        self._out_of_traj = False
        self.orientation_stay_still = False
        self.initialized = True

    def reset(self):
        self.orientation_goal_reached = False
        self.orientation_is_terminating = False
        self.orientation_stay_still = False
        self.prev_x = 0.0
        self.last_pos = (0.0, 0.0)
        self.first_reward = True
        self._out_of_traj = False
        self.init_pose = rex_constants.INIT_POSES["stand"]
        if self._signal_type == 'ol':
            self.init_pose = rex_constants.INIT_POSES["stand_ol"]
        super(RexGoToXYEnv, self).reset(initial_motor_angles=self.init_pose, reset_duration=0.5)
        self.goal_reached = False
        self.is_terminating = False
        self._stay_still = False
        self.backwards = False
        step = 0.6
        period = 0.65
        base_x = self._base_x
        if self.backwards:
            step = -.3
            period = .5
            base_x = .0
        if self._is_render and self._signal_type == 'ik':
            if self.load_ui:
                self.setup_ui(base_x, step, period)
                self.load_ui = False
        
        if self.initialized:
            # get target orientation from the target position
            self._target_orient = math.atan2(self._target_position[1], self._target_position[0])
            self._init_orient = 0
            self.clockwise = self._orientation_solve_direction()
            if self._is_debug:
                print(f"Start Orientation: {self._init_orient}, Target Orientation: {self._target_orient}")
                print("Turning right") if self.clockwise else print("Turning left")
            q = self.pybullet_client.getQuaternionFromEuler([0, 0, self._init_orient])
            self.pybullet_client.resetBasePositionAndOrientation(self.rex.quadruped, self.rex.init_position, q)
            
            self.orientation_is_terminating = False
            self.orientation_goal_reached = False
            while not self.orientation_goal_reached:
                self._handle_orientation()

            print("Orientation reached!")
            self.orientation_reached_time = self.rex.GetTimeSinceReset()
            self._last_frame_time = time.time()

        return self._get_observation()

    def setup_ui(self, base_x, step, period):
        self.base_x_ui = self._pybullet_client.addUserDebugParameter("base_x",
                                                                     self._ranges["base_x"][0],
                                                                     self._ranges["base_x"][1],
                                                                     base_x)
        self.base_y_ui = self._pybullet_client.addUserDebugParameter("base_y",
                                                                     self._ranges["base_y"][0],
                                                                     self._ranges["base_y"][1],
                                                                     self._ranges["base_y"][2])
        self.base_z_ui = self._pybullet_client.addUserDebugParameter("base_z",
                                                                     self._ranges["base_z"][0],
                                                                     self._ranges["base_z"][1],
                                                                     self._ranges["base_z"][2])
        self.roll_ui = self._pybullet_client.addUserDebugParameter("roll",
                                                                   self._ranges["roll"][0],
                                                                   self._ranges["roll"][1],
                                                                   self._ranges["roll"][2])
        self.pitch_ui = self._pybullet_client.addUserDebugParameter("pitch",
                                                                    self._ranges["pitch"][0],
                                                                    self._ranges["pitch"][1],
                                                                    self._ranges["pitch"][2])
        self.yaw_ui = self._pybullet_client.addUserDebugParameter("yaw",
                                                                  self._ranges["yaw"][0],
                                                                  self._ranges["yaw"][1],
                                                                  self._ranges["yaw"][2])
        self.step_length_ui = self._pybullet_client.addUserDebugParameter("step_length", -0.7, 0.7, step)
        self.step_rotation_ui = self._pybullet_client.addUserDebugParameter("step_rotation", -1.5, 1.5, 0.)
        self.step_angle_ui = self._pybullet_client.addUserDebugParameter("step_angle", -180., 180., 0.)
        self.step_period_ui = self._pybullet_client.addUserDebugParameter("step_period", 0.2, 0.9, period)

    def _read_inputs(self, base_pos_coeff, gait_stage_coeff):
        position = np.array(
            [
                self._pybullet_client.readUserDebugParameter(self.base_x_ui),
                self._pybullet_client.readUserDebugParameter(self.base_y_ui) * base_pos_coeff,
                self._pybullet_client.readUserDebugParameter(self.base_z_ui) * base_pos_coeff
            ]
        )
        orientation = np.array(
            [
                self._pybullet_client.readUserDebugParameter(self.roll_ui) * base_pos_coeff,
                self._pybullet_client.readUserDebugParameter(self.pitch_ui) * base_pos_coeff,
                self._pybullet_client.readUserDebugParameter(self.yaw_ui) * base_pos_coeff
            ]
        )
        step_length = self._pybullet_client.readUserDebugParameter(self.step_length_ui) * gait_stage_coeff
        step_rotation = self._pybullet_client.readUserDebugParameter(self.step_rotation_ui)
        step_angle = self._pybullet_client.readUserDebugParameter(self.step_angle_ui)
        step_period = self._pybullet_client.readUserDebugParameter(self.step_period_ui)
        return position, orientation, step_length, step_rotation, step_angle, step_period

    def _is_within_bounds_of_target_pos(self, lower_bounds, upper_bounds):
        last_x, last_y = self.last_pos
        
        return (
            lower_bounds[0] <= last_x <= upper_bounds[0]
            ) and (
            lower_bounds[1] <= last_y <= upper_bounds[1]
        )
    
    def _missed_target(self, bounds):
        last_x, last_y = self.last_pos
        return (
            last_x > bounds[0]
        )  or ( 
            last_y > bounds[1]
        )
    
    def _reward(self):
        lower_bounds = (0.8 * self._target_position[0], 0.8 * self._target_position[1])
        upper_bounds = (1.2 * self._target_position[0], 1.2 * self._target_position[1])
        if self.goal_reached:
            if self.first_reward:
                self.first_reward = False
                if self._is_within_bounds_of_target_pos(lower_bounds, upper_bounds):
                    # the agent is close to the target
                    reward = 1
                    print("GOAL REACHED!", reward, self.last_pos)
                else:
                    reward = -1
                    print("STOPPED OUT OF TRAJECTORY!", reward, self.last_pos)
            else:
                reward = 0
        elif self._missed_target(upper_bounds) and not self._out_of_traj:
                reward = -1
                self._out_of_traj = True
                self.is_terminating = True # out of trajectory
                print("WENT OUT OF TRAJECTORY!", reward, self.last_pos)
        else:
            reward = 0

        return reward

    @staticmethod
    def _evaluate_gait_stage_coeff(current_t, action, end_t=0.0):
        # ramp function
        p = 0.8 + action[0]
        if end_t <= current_t <= p + end_t:
            return current_t
        else:
            return 1.0

    def _termination(self):
        return self.is_terminating or self.is_fallen() or self._out_of_traj
    
    def _handle_action(self, action):
        self.last_pos = self.rex.GetBasePosition()
        self.last_pos = (-self.last_pos[0], -self.last_pos[1])

        if self._stay_still:
            # stop -- from before
            self.rex.Step(self.init_pose)
        elif action < -0.5: # the agent should stop
            # stop
            self.goal_reached = True
            self.is_terminating = True
            self._stay_still = True
            self.rex.Step(self.init_pose)
        else:
            current_position = self.rex.GetBasePosition()
            current_length = math.sqrt(current_position[0] ** 2 + current_position[1] ** 2)
            current_norm_distance = round(current_length / self._target_length, 1) if self._target_length else 0.0
            norm_distance = current_norm_distance
        
            while norm_distance < current_norm_distance + 0.1:
                t = self.rex.GetTimeSinceReset() - self.orientation_reached_time
                action_ = self.walking_step(t, 1.0/8, 0.05, 0.1, self.init_pose)
                action_ = super(RexGoToXYEnv, self)._transform_action_to_motor_command(action_)
                self.rex.Step(action_)  
                current_position_ = self.rex.GetBasePosition()
                current_length_ = math.sqrt(current_position_[0] ** 2 + current_position_[1] ** 2)
                norm_distance = round(current_length_ / self._target_length, 1) if self._target_length else 0.0
            
            self.last_pos_x = - self.rex.GetBasePosition()[0]

    def walking_step(self, t, period, l_a, f_a, initial_pose):
        start_coeff = self._evaluate_gait_stage_coeff(t, [0.0])
        l_a *= start_coeff
        f_a *= start_coeff
        l_extension = l_a * math.cos(2 * math.pi / period * t)
        f_extension = f_a * math.cos(2 * math.pi / period * t)
        l_swing = -l_extension
        swing = -f_extension
        pose = np.array([0.0, l_extension , f_extension,
                        0.0, l_swing, swing,
                        0.0, l_swing, swing,
                        0.0, l_extension, f_extension])
        signal = initial_pose + pose
        return signal
    
    def _handle_orientation(self):
        """Step forward the simulation, given the action.

        Args:
          action: A list of desired motor angles for eight motors.

        Returns:
          observations: The angles, velocities and torques of all motors.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """
        self._last_base_position = self.rex.GetBasePosition()
        self._last_base_orientation = self.rex.GetBaseOrientation()
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.rex.GetBasePosition()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

        action = self._orientation_transform_action_to_motor_command()
        self.rex.Step(action)
    
    def _orientation_transform_action_to_motor_command(self):
        if self.orientation_stay_still:
            return self.init_pose
        t = self.rex.GetTimeSinceReset()
        self._orientation_check_target_position(t)
        action = self.orientation_step(t)
        action = super(RexGoToXYEnv, self)._transform_action_to_motor_command(action)
        return action

    def _orientation_solve_direction(self):
        diff = abs(self._init_orient - self._target_orient)
        clockwise = False
        if self._init_orient < self._target_orient:
            if diff > 3.14:
                clockwise = True
        else:
            if diff < 3.14:
                clockwise = True
        return clockwise
    
    @staticmethod
    def _orientation_evaluate_base_stage_coeff(current_t, end_t=0.0, width=0.001):
        # sigmoid function
        beta = p = width
        if p - beta + end_t <= current_t <= p - (beta / 2) + end_t:
            return (2 / beta ** 2) * (current_t - p + beta) ** 2
        elif p - (beta/2) + end_t <= current_t <= p + end_t:
            return 1 - (2 / beta ** 2) * (current_t - p) ** 2
        else:
            return 1

    @staticmethod
    def _orientation_evaluate_gait_stage_coeff(current_t, end_t=0.0):
        # ramp function
        p = .8
        if end_t <= current_t <= p + end_t:
            return current_t
        else:
            return 1.0
        
    def _orientation_check_target_position(self, t):
        current_z = self.pybullet_client.getEulerFromQuaternion(self.rex.GetBaseOrientation())[2]
        if current_z < 0:
            current_z += 6.28
        if abs(self._target_orient - current_z) <= 0.01:
            self.orientation_goal_reached = True
            if not self.orientation_is_terminating:
                self.orientation_end_time = t
                self.orientation_is_terminating = True

    def orientation_step(self, t):
        base_pos_coeff = self._orientation_evaluate_base_stage_coeff(t, width=1.5)
        gait_stage_coeff = self._orientation_evaluate_gait_stage_coeff(t)
        step_dir_value = -0.5 * gait_stage_coeff
        if self.clockwise:
            step_dir_value *= -1
        position = np.array([0.009,
                                self._base_y * base_pos_coeff,
                                self._base_z * base_pos_coeff])
        orientation = np.array([self._base_roll * base_pos_coeff,
                                self._base_pitch * base_pos_coeff,
                                self._base_yaw * base_pos_coeff])
        step_length = (self.step_length if self.step_length is not None else 0.02)
        step_rotation = (self.step_rotation if self.step_rotation is not None else step_dir_value) #+ action[0]
        step_angle = self.step_angle if self.step_angle is not None else 0.0
        step_period = (self.step_period if self.step_period is not None else 0.75) #+ action[1]
        if self.orientation_goal_reached:
            self.orientation_stay_still = True
        frames = self._gait_planner.loop(step_length, step_angle, step_rotation, step_period, 1.0)
        fr_angles, fl_angles, rr_angles, rl_angles, _ = self._kinematics.solve(orientation, position, frames)
        signal = [
            fl_angles[0], fl_angles[1], fl_angles[2],
            fr_angles[0], fr_angles[1], fr_angles[2],
            rl_angles[0], rl_angles[1], rl_angles[2],
            rr_angles[0], rr_angles[1], rr_angles[2]
        ]
        return signal
    
    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
          action: A list of desired motor angles for eight motors.

        Returns:
          observations: The angles, velocities and torques of all motors.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """
        self._last_base_position = self.rex.GetBasePosition()
        self._last_base_orientation = self.rex.GetBaseOrientation()
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
                print("Sleeping")
            base_pos = self.rex.GetBasePosition()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)

        self._handle_action(action)
        reward = self._reward()
        done = self._termination()
        self._env_step_counter += 1
        if done:
            # print("Episode done!", self._env_step_counter)
            self.rex.Terminate()
        # print(self._env_step_counter, self._last_base_position)
        # print("Observation:", self._get_observation(), "Reward:", reward, "Action", action)
        return np.array(self._get_observation()), reward, done, {'action': action}


    def is_fallen(self):
        """Decide whether the rex has fallen.

    If the up directions between the base and the world is large (the dot
    product is smaller than 0.85), the rex is considered fallen.

    Returns:
      Boolean value that indicates whether the rex has fallen.
    """
        orientation = self.rex.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85

    def _get_true_observation(self):
        """Get the true observations of this environment.

    It includes the roll, pitch and their corresponding rates, and the 
    distance in x-axis from the target position.

    Returns:
      The observation list.
    """
        observation = []
        current_position = self.rex.GetBasePosition()
        current_position = (-current_position[0], -current_position[1])
        current_length = math.sqrt(current_position[0] ** 2 + current_position[1] ** 2)
        if self.initialized:
            # norm_distance = round(current_length / self._target_length, 1) if self._target_length else 0.0
            norm_distance = round(0.5 * (current_position[0] / self._target_position[0] + (current_position[1] / self._target_position[1])), 1)
        else:
            norm_distance = 0.0
        observation.extend([norm_distance])
        self._true_observation = np.array(observation)
        return self._true_observation


    def _get_observation(self):
        observation = []
        current_position = self.rex.GetBasePosition()
        current_position = (-current_position[0], -current_position[1])
        current_length = math.sqrt(current_position[0] ** 2 + current_position[1] ** 2)
        if self.initialized:
            # norm_distance = round(current_length / self._target_length, 1) if self._target_length else 0.0
            norm_distance = round(0.5 * (current_position[0] / self._target_position[0] + (current_position[1] / self._target_position[1])), 1)
        else:
            norm_distance = 0.0
        observation.extend([norm_distance])
        self._observation = np.array(observation)
        print("Observation:", self._observation)
        return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        upper_bound = np.zeros(self._get_observation_dimension())
        upper_bound[0] = 3.0
        return upper_bound

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound
