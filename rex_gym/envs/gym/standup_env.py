"""This file implements the gym environment of rex alternating legs.

"""
import math
import time
import random

from gym import spaces
import numpy as np
from .. import rex_gym_env
from ...model import rex_constants
from ...model.rex import Rex

NUM_LEGS = 4
NUM_MOTORS = 3 * NUM_LEGS


class RexStandupEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex.

  It simulates the locomotion of a rex, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the rex walks in 1000 steps and penalizes the energy
  expenditure.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}

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
                 remove_default_joint_damping=False,
                 render=False,
                 num_steps_to_log=1000,
                 env_randomizer=None,
                 log_path=None,
                 signal_type="ol",
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
        self.init = False
        super(RexStandupEnv,
              self).__init__(urdf_version=urdf_version,
                             accurate_motor_model_enabled=True,
                             motor_overheat_protection=True,
                             hard_reset=False,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=remove_default_joint_damping,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat,
                             signal_type=signal_type,
                             debug=debug,
                             terrain_id=terrain_id,
                             terrain_type=terrain_type,
                             mark=mark)

        action_dim = 1
        action_low = np.array([-0.02] * action_dim)
        action_high = np.array([0.0] * action_dim)
        self.action_space = spaces.Box(action_low, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 30
        self._cam_pitch = -30
        if self._on_rack:
            self._cam_pitch = 0
        self.prev_pose = rex_constants.INIT_POSES['new_rest_position']
        self.m_angle = random.random() + 2
        self.init_angle = self.m_angle
        self.prev_pose = [
            self.prev_pose[0], self.prev_pose[1], self.m_angle,
            self.prev_pose[3], self.prev_pose[4], self.m_angle,
            self.prev_pose[6], self.prev_pose[7], self.m_angle,
            self.prev_pose[9], self.prev_pose[10], self.m_angle
        ]
        self.prev_z = self.rex.GetBasePosition()[2]
        self.sum_reward = 0
        self.init = True
        self.terminate = False

    def reset(self):
        self.terminate = False
        if self.init:
          self.prev_pose = rex_constants.INIT_POSES['new_rest_position']
          self.m_angle = random.random() + 2
          self.init_angle = self.m_angle
          self.prev_pose = [
              self.prev_pose[0], self.prev_pose[1], self.m_angle,
              self.prev_pose[3], self.prev_pose[4], self.m_angle,
              self.prev_pose[6], self.prev_pose[7], self.m_angle,
              self.prev_pose[9], self.prev_pose[10], self.m_angle
          ]
          self.sum_reward = 0
          super(RexStandupEnv, self).reset(initial_motor_angles=self.prev_pose,
                                    reset_duration=0.5)
          self.prev_z = self.rex.GetBasePosition()[2]
        else:
          super(RexStandupEnv, self).reset(initial_motor_angles=rex_constants.INIT_POSES['new_rest_position'],
                                         reset_duration=0.5)
        return self._get_observation()

    def _signal(self, action):
      self.m_angle = self.m_angle + action
      self.prev_pose = [
        self.prev_pose[0], self.prev_pose[1], self.m_angle,
        self.prev_pose[3], self.prev_pose[4], self.m_angle,
        self.prev_pose[6], self.prev_pose[7], self.m_angle,
        self.prev_pose[9], self.prev_pose[10], self.m_angle
      ]
      # print("m_angle: ", round(self.prev_pose[], 2), "action: ", round(action, 2))
      return self.prev_pose

    @staticmethod
    def _convert_from_leg_model(leg_pose):
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            motor_pose[3 * i] = leg_pose[3 * i]
            motor_pose[3 * i + 1] = leg_pose[3 * i + 1]
            motor_pose[3 * i + 2] = leg_pose[3 * i + 2]
          
        return motor_pose

    def _transform_action_to_motor_command(self, action):
        action = self._signal(action[0])
        action = self._convert_from_leg_model(action)
        action = super(RexStandupEnv, self)._transform_action_to_motor_command(action)
        return action

    def _termination(self):
        return self.is_fallen() or self.terminate

    def is_fallen(self):
        """Decide whether the rex has fallen.

    If the up directions between the base and the world is large (the dot
    product is smaller than 0.85), the rex is considered fallen.

    Returns:
      Boolean value that indicates whether the rex has fallen.
    """
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        return math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.5 or self.m_angle < -1.5 or self.m_angle > 3.2

    def _reward(self):
        # Reward is based on the movement in the z direction of the base.
        current_z = self.rex.GetBasePosition()[2]
        if 1.9 <= self.m_angle <= 2.1:
          reward = 1
          self.terminate = True
        elif self.m_angle < 1.9 or self.m_angle > 1.05 * self.init_angle:
          reward = -1
          self.terminate = True
        else:
          reward = current_z - self.prev_z
        self.prev_z = current_z
        # print(reward)
        self.sum_reward += reward
        return reward

    def _get_true_observation(self):
        """Get the true observations of this environment.

    It includes the roll, the pitch, the roll dot and the pitch dot of the base.
    Also, eight motor angles are added into the
    observation.

    Returns:
      The observation list, which is a numpy array of floating-point values.
    """
        if self.init:
          observation = [self.m_angle]
        else:
          observation = [0]        
        # observation.extend(self.rex.GetMotorAngles().tolist())
        self._true_observation = np.array(observation)
        return self._true_observation

    def _get_observation(self):
        # roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
        # roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
        # observation = [roll, pitch, roll_rate, pitch_rate]
        if self.init:
          observation = [self.m_angle]
        else:
          observation = [0]
        # observation.extend(self.rex.GetMotorAngles().tolist())
        self._observation = np.array(observation)
        return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        upper_bound = np.ones(self._get_observation_dimension()) * 2 * math.pi
        return upper_bound

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound


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

        action = self._transform_action_to_motor_command(action)
        self.rex.Step(action)
        reward = self._reward()
        # print("Reward: ", reward)
        done = self._termination()
        # @TODO fix logging
        # if self._log_path is not None:
        #     rex_logging.update_episode_proto(self._episode_proto, self.rex, action,
        #                                      self._env_step_counter)
        self._env_step_counter += 1
        # print("Step: ", self._env_step_counter, self.sum_reward)
        if done:
            self.rex.Terminate()
            print("Episode done!", self._env_step_counter, self.sum_reward)
            
        return np.array(self._get_observation()), reward, done, {'action': action}
