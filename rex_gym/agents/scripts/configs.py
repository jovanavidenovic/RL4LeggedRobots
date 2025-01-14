# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example configurations using the PPO algorithm."""

from ..ppo.algorithm import PPOAlgorithm

from ..scripts import networks


def default():
    """Default configuration for PPO."""
    # General
    algorithm = PPOAlgorithm
    num_agents = 25
    eval_episodes = 25
    use_gpu = False
    # Network
    network = networks.ForwardGaussianPolicy
    weight_summaries = dict(all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
    policy_layers = 200, 100
    value_layers = 200, 100
    init_mean_factor = 0.05
    init_logstd = -1
    # Optimization
    update_every = 25
    policy_optimizer = 'AdamOptimizer'
    value_optimizer = 'AdamOptimizer'
    update_epochs_policy = 50
    update_epochs_value = 50
    policy_lr = 1e-4
    value_lr = 3e-4
    # Losses
    discount = 0.985
    kl_target = 1e-2
    kl_cutoff_factor = 2
    kl_cutoff_coef = 1000
    kl_init_penalty = 1
    return locals()


def gallop_ik():
    """Configuration for Rex gallop task based on inverse kinematics controller."""
    locals().update(default())
    # Environment
    env = 'RexGalloping-v0'
    max_length = 2000
    steps = 1e6  # 1M
    return locals()


def gallop_ol():
    """Configuration for Rex gallop task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexGalloping-v0'
    max_length = 1000
    steps = 1e7  # 10M
    return locals()

def new_gallop_ol():
    """Configuration for Rex trotting task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexNewGalloping-v0'
    max_length = 250
    steps = 1e7 # 10M
    return locals()

def ls_gallop_ol():
    """Configuration for Rex trotting task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexLSGalloping-v0'
    max_length = 250
    steps = 1e7 # 10M
    return locals()

def new_gallop_linear_ol():
    """Configuration for Rex trotting task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexNewGallopingLinear-v0'
    network = networks.LinearGaussianPolicy
    max_length = 250
    steps = 1e7 # 10M
    return locals()

def new_gallop_rnn_ol():
    """Configuration for Rex trotting task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexNewGallopingRNN-v0'
    network = networks.RecurrentGaussianPolicy
    max_length = 250
    discount = 1.0
    steps = 1e7 # 10M
    # discount = 1
    return locals()

def trotting_ol():
    """Configuration for Rex trotting task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexTrotting-v0'
    max_length = 500
    steps = 2 * 1e7 # 10M
    discount = 1
    kl_target = 5 * 1e-3
    return locals()

def trotting_rnn_ol():
    """Configuration for Rex trotting task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexTrottingRNN-v0'
    network = networks.RecurrentGaussianPolicy
    max_length = 500
    steps = 2 * 1e7 # 10M
    discount = 1
    kl_target = 5 * 1e-3
    return locals()

def walk_ol():
    """Configuration for Rex walk task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexWalk-v0'
    max_length = 2000
    steps = 2e6  # 2M
    return locals()


def goto_ol():
    """Configuration for Rex walk task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexGoTo-v0'
    update_every = 100
    update_epochs_policy = 200
    update_epochs_value = 200
    max_length = 25
    steps = 2e6  # 2M
    discount = 1.0
    network = networks.GoToLinearGaussianPolicy
    weight_summaries = dict(all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
    policy_layers = 1, 
    value_layers = 1,
    return locals()

def gotoxy_ol():
    """Configuration for Rex walk task based on open loop controller."""
    locals().update(default())
    # Environment
    env = 'RexGoToXY-v0'
    update_every = 100
    update_epochs_policy = 200
    update_epochs_value = 200
    max_length = 25
    steps = 2e6  # 2M
    discount = 1.0
    network = networks.GoToLinearGaussianPolicy
    weight_summaries = dict(all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
    policy_layers = 1, 
    value_layers = 1,
    return locals()

def walk_ik():
    """Configuration for Rex walk task based on inverse kinematics controller."""
    locals().update(default())
    # Environment
    env = 'RexWalk-v0'
    max_length = 2000
    steps = 1e6  # 1M
    return locals()


def turn_ol():
    """Configuration for Rex turn task."""
    locals().update(default())
    # Environment
    env = 'RexTurn-v0'
    max_length = 1000
    steps = 1e6  # 1M
    return locals()


def turn_ik():
    """Configuration for Rex turn task."""
    locals().update(default())
    # Environment
    env = 'RexTurn-v0'
    max_length = 1000
    steps = 1e6  # 1M
    return locals()


def standup_ol():
    """Configuration for Rex stand up task."""
    locals().update(default())
    # Environment
    env = 'RexStandup-v0'
    max_length = 500
    steps = 1e6  # 1M
    discount = 1.0
    network = networks.GoToLinearGaussianPolicy
    return locals()


def go():
    """Configuration for Rex go-to task."""
    locals().update(default())
    # Environment
    env = 'RexGo-v0'
    max_length = 1000
    steps = 5e6  # 5M
    return locals()


def poses_ik():
    """Configuration for Rex reach-a-pose task."""
    locals().update(default())
    # Environment
    env = 'RexPoses-v0'
    max_length = 1000
    steps = 1e6  # 1M
    return locals()
