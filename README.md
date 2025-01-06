<div align="center">

# Reinforcement Learning for Locomotion in Quadruped Robots

[Jovana Videnović] and [Haris Kupinić]

Faculty of Engineering, University of Porto

</div>

## Abstract

## Installation

To set up the repository locally, follow these steps:

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/jovanavidenovic/DAM4SAM.git
    cd DAM4SAM
    ```
2. Create a new conda environment and activate it:
   ```bash
    conda create -n rl4locomotion python=3.7.16
    conda activate rl4locomotion
    ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
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