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
from ...model.gait_planner import GaitPlanner
from ...model.kinematics import Kinematics

NUM_LEGS = 4


class RexGoToEnv(rex_gym_env.RexGymEnv):
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
        super(RexGoToEnv,
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
        self._target_position = 1.0
        self.prev_x = 0.0
        self.first_reward = True
        self.backwards = False
        self.last_pos_x = 0.0
        self._out_of_traj = False

    def reset(self):
        self.prev_x = 0.0
        self.last_pos_x = 0.0
        self.first_reward = True
        self._out_of_traj = False
        self.init_pose = rex_constants.INIT_POSES["stand"]
        if self._signal_type == 'ol':
            self.init_pose = rex_constants.INIT_POSES["stand_ol"]
        super(RexGoToEnv, self).reset(initial_motor_angles=self.init_pose, reset_duration=0.5)
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
        self._target_position = 1.0
        if self._is_render and self._signal_type == 'ik':
            if self.load_ui:
                self.setup_ui(base_x, step, period)
                self.load_ui = False
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

    # def _reward(self):
    #     current_base_position = self.rex.GetBasePosition()
    #     current_x = -current_base_position[0]
    #     if self.goal_reached:
    #         if self.first_reward:
    #             self.first_reward = False
    #             if 0.75 * self._target_position <= current_x <= 1.25*self._target_position:
    #                 # the agent is close to the target
    #                 reward = 1
    #             # elif 0.5 * self._target_position <= current_x <= 1.5*self._target_position:
    #             #     reward = 0.5
    #             # elif 0.25 * self._target_position <= current_x <= 1.75*self._target_position:
    #             #     reward = -0.5
    #             else:
    #                 reward = -1
    #         else:
    #             reward = 0
    #     else:
    #         if current_x < self._target_position:
    #             reward = 0.1
    #         elif current_x < 1.25 * self._target_position:
    #             reward = 0
    #         # if current_x < 1.25 * self._target_position:
    #         #     # intermediate reward
    #         #     reward = (current_x - self.prev_x) / self._target_position
    #         # elif current_x < 1.75 * self._target_position:
    #         #     reward = -(current_x - self.prev_x) / self._target_position
    #         else:
    #             reward = -1
    #             self.is_terminating = True # out of trajectory

    #     self.prev_x = current_x
        
    #     # if reward != 0:
    #     #     print(f"Reward: {reward}")
    #     return reward


    def _reward(self):
        lower_bound = 0.85 * self._target_position
        upper_bound = 1.15 * self._target_position
        if self.goal_reached:
            if self.first_reward:
                self.first_reward = False
                if lower_bound <= self.last_pos_x and self.last_pos_x <= upper_bound:
                    # the agent is close to the target
                    reward = 1
                    print("GOAL REACHED!", reward, self.last_pos_x)
                else:
                    reward = -1
                    print("STOPPED OUT OF TRAJECTORY!", reward, self.last_pos_x)
            else:
                reward = 0
        elif self.last_pos_x > upper_bound and not self._out_of_traj:
                reward = -1
                self._out_of_traj = True
                self.is_terminating = True # out of trajectory
                print("WENT OUT OF TRAJECTORY!", reward, self.last_pos_x)
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
        # print("Action:", action)
        self.last_pos_x = - self.rex.GetBasePosition()[0]
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
            current_pos = - round(self.rex.GetBasePosition()[0] / self._target_position, 1)
            pos = current_pos
            while pos < current_pos + 0.1:
                t = self.rex.GetTimeSinceReset()
                action_ = self.walking_step(t, 1.0/8, 0.05, 0.1, self.init_pose)
                action_ = super(RexGoToEnv, self)._transform_action_to_motor_command(action_)
                self.rex.Step(action_)  
                pos = - round(self.rex.GetBasePosition()[0] / self._target_position, 1)
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

    # def _signal(self, t, action):
    #     period = 1.0 / 8
    #     l_a = 0.1
    #     f_a = l_a * 2
    #     initial_pose = self.init_pose
    #     if action > 1 or action < -1:
    #         print("Invalid action value. The action value should be in the range [-1, 1].")
        
    #     if action < -1: #-0.8 # the agent should stop
    #         self.goal_reached = True
    #         self.is_terminating = True
    #         self._stay_still = True
    #         return initial_pose
    #     else:
    #         current_pos = - round(self.rex.GetBasePosition()[0] / self._target_position, 1)
    #         pos = current_pos
    #         print(current_pos, "resolving:")
    #         while pos < current_pos + 0.1:
    #             signal = self.walking_step(t, period, l_a, f_a, initial_pose)
    #             pos = - round(self.rex.GetBasePosition()[0] / self._target_position, 1)
    #             t = self.rex.GetTimeSinceReset()

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
        norm_x_distance = - round(current_position[0] / self._target_position, 1) if self._target_position else 0.0
        observation.extend([norm_x_distance])
        self._true_observation = np.array(observation)
        return self._true_observation


    def _get_observation(self):
        observation = []
        current_position = self.rex.GetBasePosition()
        norm_x_distance = - round(current_position[0] / self._target_position, 1) if self._target_position else 0.0
        observation.extend([norm_x_distance])
        self._observation = np.array(observation)
        # print(self._observation)
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
