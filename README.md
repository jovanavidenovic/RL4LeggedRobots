<div align="center">

# Reinforcement Learning for <br> Locomotion in Quadruped Robots

Jovana Videnović and Haris Kupinić

Faculty of Engineering, University of Porto

</div>

## Abstract
This project explores the application of reinforcement learning for developing agile and adaptive locomotion in quadruped robots. By employing deep reinforcement learning (DRL) techniques, including Proximal Policy Optimization (PPO), and PyBullet simulation framework we trained policies to execute high-speed gaits like galloping and trotting, as well as navigate to specified targets. A hybrid two-stage training approach enabled fine-tuning of policies for challenging terrains, overall achieving up to 1.9x speed improvement over baseline inverse-kinematics based implementations. The results demonstrate the potential of DRL to generalize robust locomotion strategies across varying environments, surpassing traditional heuristic-based methods in speed and adaptability.

## Installation

To set up the repository locally, follow these steps:

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/jovanavidenovic/RL4LeggedRobots.git
    cd RL4LeggedRobots
    ```
2. Create a new conda environment and activate it:
   ```bash
    conda create -n rl4locomotion python=3.7.16
    conda activate rl4locomotion
    ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install .
   ```

## Running and evaluation

This repository support two locomotion policies: trotting and galloping, and navigation to a specific target. To run the training and evaluation of the policies, follow the instructions below.

### Training
To start a new training, run the following command with the desired environment name:
```bash
rex-gym train --env ENV_NAME --log-dir LOG_DIR_PATH
```
ENV_NAME can be one of the following: `trotting`, `new_gallop`, `gotoxy`.

### Evaluation
To perform the evaluation of the trained policies, run the following command with the desired environment name:
```bash
rex-gym policy --env ENV_NAME
```
ENV_NAME can be one of the following: `trotting`, `new_gallop`, `gotoxy`.
