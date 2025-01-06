import gym
from gym.envs.registration import registry


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
    return btenvs


register(
    id='RexGalloping-v0',
    entry_point='rex_gym.envs.gym.gallop_env:RexReactiveEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexLSGalloping-v0',
    entry_point='rex_gym.envs.gym.ls_gallop_env:RexLSGallopEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexNewGalloping-v0',
    entry_point='rex_gym.envs.gym.new_gallop_env:RexGallopEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexNewGallopingLinear-v0',
    entry_point='rex_gym.envs.gym.new_gallop_env:RexGallopEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexNewGallopingRNN-v0',
    entry_point='rex_gym.envs.gym.new_gallop_env:RexGallopEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexTrotting-v0',
    entry_point='rex_gym.envs.gym.trotting_env:RexTrottingEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexTrottingRNN-v0',
    entry_point='rex_gym.envs.gym.trotting_env:RexTrottingEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexWalk-v0',
    entry_point='rex_gym.envs.gym.walk_env:RexWalkEnv',
    max_episode_steps=2500,
    reward_threshold=5.0,
)

register(
    id='RexTurn-v0',
    entry_point='rex_gym.envs.gym.turn_env:RexTurnEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexStandup-v0',
    entry_point='rex_gym.envs.gym.standup_env:RexStandupEnv',
    max_episode_steps=400,
    reward_threshold=5.0,
)

register(
    id='RexGo-v0',
    entry_point='rex_gym.envs.gym.go_env:RexGoEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='RexGoTo-v0',
    entry_point='rex_gym.envs.gym.goto_env:RexGoToEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)


register(
    id='RexGoToXY-v0',
    entry_point='rex_gym.envs.gym.goto_env:RexGoToXYEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)


register(
    id='RexPoses-v0',
    entry_point='rex_gym.envs.gym.poses_env:RexPosesEnv',
    max_episode_steps=400,
    reward_threshold=5.0,
)
