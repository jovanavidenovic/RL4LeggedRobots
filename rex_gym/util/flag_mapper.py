ENV_ID_TO_POLICY = {
    'gallop_ol': ('rex_gym/policies/gallop/ol', 'model.ckpt-4000000'),
    'goto_ol': ('rex_gym/policies_final/goto/ol', 'model.ckpt-93750'),
    'goto_xy_ol': ('rex_gym/policies_final/goto_xy/ol', 'model.ckpt-93750'),
    'new_gallop_ol': ('rex_gym/policies_final/new_gallop/ol', 'model.ckpt-13375000'),
    'new_gallop_linear_ol': ('rex_gym/policies_final/new_gallop_linear/ol', 'model.ckpt-13375000'),
    'new_gallop_rnn_ol': ('rex_gym/policies_final/new_gallop_rnn/ol', 'model.ckpt-20000000'),
    'trotting_ol': ('rex_gym/policies_final/trotting/ol', 'model.ckpt-20000000'),
    'gallop_ik': ('rex_gym/policies/gallop/ik', 'model.ckpt-2000000'),
    'walk_ik': ('rex_gym/policies/walk/ik', 'model.ckpt-2000000'),
    'walk_ol': ('rex_gym/policies/walk/ol', 'model.ckpt-4000000'),
    'standup_ol': ('rex_gym/policies/standup/ol', 'model.ckpt-2000000'),
    'turn_ik': ('rex_gym/policies/turn/ik', 'model.ckpt-2000000'),
    'turn_ol': ('rex_gym/policies/turn/ol', 'model.ckpt-2000000'),
    'poses_ik': ('rex_gym/policies/poses', 'model.ckpt-2000000'),
}

ENV_ID_TO_ENV_NAMES = {
    'gallop': 'RexReactiveEnv',
    'walk': 'RexWalkEnv',
    'turn': 'RexTurnEnv',
    'standup': 'RexStandupEnv',
    'go': 'RexGoEnv',
    'poses': 'RexPosesEnv',
    'trotting': 'RexTrottingEnv',
    'trotting_rnn': 'RexTrottingEnv',
    'ls_gallop': 'RexLSGallopEnv',
    'new_gallop': 'RexGallopEnv',
    'new_gallop_linear': 'RexGallopEnv',
    'new_gallop_rnn': 'RexGallopEnv',
    "goto": "RexGoToEnv",
    "goto_xy": "RexGoToXYEnv",
}

DEFAULT_SIGNAL = {
    'gallop': 'ik',
    'walk': 'ik',
    'turn': 'ol',
    'standup': 'ol',
    'go': 'ik',
    'poses': 'ik'
}

TERRAIN_TYPE = {
    'mounts': 'png',
    'maze': 'png',
    'hills': 'csv',
    'random': 'random',
    'plane': 'plane'
}
