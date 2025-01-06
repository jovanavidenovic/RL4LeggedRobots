import numpy as np

ARM_POSES = {
    'rest': np.array([
        -1.6, -1.6, 0.,
        0., 1.6, 0.
    ])
}

INIT_POSES = {
    'stand': np.array([
        0., -0.88643435, 1.30197369,
        0., -0.88643435, 1.30197369,
        0., -0.88643435, 1.30197369,
        0., -0.88643435, 1.30197369
    ]),
    'stand_ol': np.array([
        0.15192765, -0.90412283, 1.48156545,
        -0.15192765, -0.90412283, 1.48156545,
        0.15192765, -0.90412283, 1.48156545,
        -0.15192765, -0.90412283, 1.48156545
    ]),
    'gallop': np.array([
        0.15192765, -0.90412283, 1.48156545,
        -0.15192765, -0.90412283, 1.48156545,
        0.15192765, -0.90412283, 1.48156545,
        -0.15192765, -0.90412283, 1.48156545
    ]),
    'stand_low': np.array([
        0.1, -0.82, 1.35,
        -0.1, -0.82, 1.35,
        0.1, -0.87, 1.35,
        -0.1, -0.87, 1.35
    ]),
    'stand_high': np.array([
        0, -0.658319, 1.0472,
        0, -0.658319, 1.0472,
        0, -0.658319, 1.0472,
        0, -0.658319, 1.0472
    ]),
    'rest_position': np.array([
        -0.4, -1.5, 6,
        0.4, -1.5, 6,
        -0.4, -1.5, 6,
        0.4, -1.5, 6
    ]),
    'new_rest_position': np.array([
        -0.4, -1.5, 3,
        0.4, -1.5,  3,
        -0.4, -1.5, 3,
        0.4, -1.5,  3
    ])
}
