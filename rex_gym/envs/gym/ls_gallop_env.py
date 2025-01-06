"""This file implements the gym environment of rex trotting environment.

"""
import math
import random

from gym import spaces
import numpy as np
import pybullet
from .. import rex_gym_env
from ...model import rex_constants
from ...model.gait_planner import GaitPlanner
from ...model.kinematics import Kinematics
from rex_gym.model import mark_constants

NUM_LEGS = 4
NUM_MOTORS = 3 * NUM_LEGS

class RexLSGallopEnv(rex_gym_env.RexGymEnv):
    """The gym environment for the rex.

  In this env, Rex performs a trotting style locomotion specified by
  extension_amplitude, swing_amplitude, and step_frequency. Each diagonal pair
  of legs will move according to the reference trajectory:
      extension = extsion_amplitude * cos(2 * pi * step_frequency * t + phi)
      swing = swing_amplitude * sin(2 * pi * step_frequency * t + phi)
  And the two diagonal leg pairs have a phase (phi) difference of pi. The
  reference signal may be modified by the feedback actions from a balance
  controller (e.g. a neural network).

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
                 signal_type="ol",
                 terrain_type="plane",
                 terrain_id=None,
                 mark='base'):
        """Initialize the rex trotting gym environment.

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
        super(RexLSGallopEnv,
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
                             mark=mark,
                             drift_weight=2,
                             energy_weight=0,
                             shake_weight=0.01,
                             distance_weight=1,)
                            
        action_dim = 8
        action_high = np.array([0.3] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = -20
        self._signal_type = signal_type
        self._gait_planner = GaitPlanner("trotting")
        self._kinematics = Kinematics()
        self.goal_reached = False
        self._stay_still = False
        self.is_terminating = False
        self._backwards = False
        self._target_position = None
        self.prev_x = 0.0
        self.prev_y = 0.0


    def reset(self):
        self.init_pose = rex_constants.INIT_POSES["stand"]
        if self._signal_type == 'ol':
            self.init_pose = rex_constants.INIT_POSES["stand_ol"]
        super(RexLSGallopEnv, self).reset(initial_motor_angles=self.init_pose, reset_duration=0.5)
        self.goal_reached = False
        self.is_terminating = False
        self._stay_still = False
        if self._backwards is None:
            self.backwards = random.choice([True, False])
        else:
            self.backwards = self._backwards
        step = 0.6
        period = 0.65
        base_x = self._base_x
        if self.backwards:
            step = -.3
            period = .5
            base_x = .0

        self._target_position = None
        # if not self._target_position or self._random_pos_target:
        #     bound = -3 if self.backwards else 3
        #     self._target_position = random.uniform(bound//2, bound)
        #     self._random_pos_target = True
        if self._is_render and self._signal_type == 'ik':
            if self.load_ui:
                self.setup_ui(base_x, step, period)
                self.load_ui = False
        # if self._is_debug:
        #     print(f"Target Position x={self._target_position}, Random assignment: {self._random_pos_target}, Backwards: {self.backwards}")
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


    def _reward(self):
        current_base_position = self.rex.GetBasePosition()
        # observation = self._get_observation()
        # forward gait
        current_x = -current_base_position[0]
        forward_reward = current_x - self.prev_x
        # Cap the forward reward if a cap is set.
        forward_reward = min(forward_reward, self._forward_reward_cap)
        # Penalty for sideways translation.
        # drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
        drift_reward = -abs(current_base_position[1] - self.prev_y)

        self.prev_x = current_x
        self.prev_y = current_base_position[1]

        # Penalty for sideways rotation of the body.
        orientation = self.rex.GetBaseOrientation()
        rot_matrix = pybullet.getMatrixFromQuaternion(orientation)
        local_up_vec = rot_matrix[6:]
        shake_reward = -abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
        # shake_reward = -abs(observation[4])
        energy_reward = 0.0
        # energy_reward = -np.abs(
        #     np.dot(self.rex.GetMotorTorques(),
        #            self.rex.GetMotorVelocities())) * self._time_step
        objectives = [forward_reward, energy_reward, drift_reward, shake_reward]
        if shake_reward < 0.1:
            shake_reward /= 2
        weighted_objectives = [o * w for o, w in zip(objectives, self._objective_weights)]
        reward = sum(weighted_objectives)
        # print(objectives, self._objective_weights, reward)
        self._objectives.append(objectives)
        return reward
    
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

    def _check_target_position(self, t):
        if self._target_position:
            current_x = abs(self.rex.GetBasePosition()[0])
            # give 0.15 stop space
            if current_x >= abs(self._target_position) - 0.15:
                self.goal_reached = True
                if not self.is_terminating:
                    self.end_time = t
                    self.is_terminating = True

    @staticmethod
    def _evaluate_base_stage_coeff(current_t, end_t=0.0, width=0.001):
        # sigmoid function
        beta = p = width
        if p - beta + end_t <= current_t <= p - (beta / 2) + end_t:
            return (2 / beta ** 2) * (current_t - p + beta) ** 2
        elif p - (beta/2) + end_t <= current_t <= p + end_t:
            return 1 - (2 / beta ** 2) * (current_t - p) ** 2
        else:
            return 1

    @staticmethod
    def _evaluate_gait_stage_coeff(current_t, action, end_t=0.0):
        # ramp function
        p = 0.8 + action[0]
        if end_t <= current_t <= p + end_t:
            return current_t
        else:
            return 1.0

    @staticmethod
    def _evaluate_brakes_stage_coeff(current_t, action, end_t=0.0, end_value=0.0):
        # ramp function
        p = 0.8 + action[1]
        if end_t <= current_t <= p + end_t:
            return 1 - (current_t - end_t)
        else:
            return end_value

    def _signal(self, t, action):
        """Generates the gallop gait for the robot.

        Args:
        t: Current time in simulation.

        Returns:
        A numpy array of the reference leg positions.
        """
        gallop_signal = np.array([
            0,  action[1], action[5], # front left angles
            0, action[0], action[4], # front right angles
            0,  action[3], action[7], # rear left angles
            0,  action[2], action[6], # rear right angles
        ])
        return gallop_signal


    def _convert_from_leg_model(self, leg_pose):
        """Converts leg space action into motor commands.

        Args:
        leg_pose: A numpy array. leg_pose[0:NUM_LEGS] are leg swing angles
            and leg_pose[NUM_LEGS:2*NUM_LEGS] contains leg extensions.

        Returns:
        A numpy array of the corresponding motor angles for the given leg pose.
        """
        motor_pose = np.zeros(NUM_MOTORS)
        for i in range(NUM_LEGS):
            current_idx = 3 * i
            motor_pose[current_idx] = 0
            motor_pose[current_idx + 1] = leg_pose[current_idx + 2] + leg_pose[current_idx] # e + s
            motor_pose[current_idx + 2] = leg_pose[current_idx + 2] - leg_pose[current_idx] # e - s
        return motor_pose

    def _transform_action_to_motor_command(self, action):
        """Generates the motor commands for the given action.

        Swing/extension offsets and the reference leg trajectory will be added on
        top of the inputs before the conversion.

        Args:
        action: A numpy array contains the leg swings and extensions that will be
            added to the reference trotting trajectory. action[0:NUM_LEGS] are leg
            swing angles, and action[NUM_LEGS:2*NUM_LEGS] contains leg extensions.

        Returns:
        A numpy array of the desired motor angles for the given leg space action.
        """
        # Add the reference trajectory (i.e. the trotting signal).
        action = self._signal(self.rex.GetTimeSinceReset(), action)

        return self._convert_from_leg_model(action) + self.init_pose


    def is_fallen(self):
        """Decides whether Rex is in a fallen state.

    If the roll or the pitch of the base is greater than 0.3 radians, the
    rex is considered fallen.

    Returns:
      Boolean value that indicates whether Rex has fallen.
    """
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        return math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.5

    
    def _get_true_observation(self):
        """Get the true observations of this environment.

    It includes the roll, the pitch, the roll dot and the pitch dot of the base.
    Also, eight motor angles are added into the
    observation.

    Returns:
      The observation list, which is a numpy array of floating-point values.
    """
        roll, pitch, _ = self.rex.GetTrueBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetTrueBaseRollPitchYawRate()
        observation = [roll, pitch, roll_rate, pitch_rate]
        observation.extend(self.rex.GetMotorAngles().tolist())
        self._true_observation = np.array(observation)
        return self._true_observation

    def _get_observation(self):
        roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
        roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
        observation = [roll, pitch, roll_rate, pitch_rate]
        observation.extend(self.rex.GetMotorAngles().tolist())
        self._observation = np.array(observation)
        return self._observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See _get_true_observation() for the
      details of each element of an observation.
    """
        upper_bound_roll = 2 * math.pi
        upper_bound_pitch = 2 * math.pi
        upper_bound_roll_dot = 2 * math.pi / self._time_step
        upper_bound_pitch_dot = 2 * math.pi / self._time_step
        upper_bound_motor_angle = 2 * math.pi
        upper_bound = [
            upper_bound_roll, upper_bound_pitch, upper_bound_roll_dot, upper_bound_pitch_dot
        ]

        upper_bound.extend([upper_bound_motor_angle] * mark_constants.MARK_DETAILS['motors_num'][self.mark])
        return np.array(upper_bound)

    def _get_observation_lower_bound(self):
        lower_bound = -self._get_observation_upper_bound()
        return lower_bound
