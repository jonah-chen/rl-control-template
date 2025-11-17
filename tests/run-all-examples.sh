#!/bin/sh

# exit script on error
set -e

# Janky way to run all examples. Only looking for the most basic errors.

## Atari Example
python src/main.py exp_path=experiments/atari_example/Atari/DQN.json idxs=0
python src/main.py experiment=atari_example/Atari/EQRC idxs=0

## Continuing Example
python src/main.py experiment=continuing_example/Forager/EQRC idxs=0

## Optuna MountainCar
python src/main.py exp_path=experiments/optuna_example/MountainCar/DQN.json idxs=0
python src/main.py exp_path=experiments/optuna_example/MountainCar/EQRC.json idxs=0
python experiments/optuna_example/learning_curve.py

## Replay MountainCar
python src/main.py exp_path=experiments/replay_example/MountainCar/DQN.json idxs=0
python src/main.py exp_path=experiments/replay_example/MountainCar/EQRC.json idxs=0
python experiments/replay_example/learning_curve.py


## Basic Examples

## Acrobot
python src/main.py exp_path=experiments/example/Acrobot/DQN.json idxs=0
python src/main.py experiment=example/Acrobot/EQRC idxs=0
python src/main.py exp_path=experiments/example/Acrobot/ESARSA.json idxs=0
python src/main.py exp_path=experiments/example/Acrobot/SoftmaxAC.json idxs=0

## Breakout
python src/main.py exp_path=experiments/example/Breakout/DQN.json idxs=0
python src/main.py experiment=example/Breakout/EQRC idxs=0
python src/main.py exp_path=experiments/example/Breakout/PrioritizedDQN.json idxs=0

## Cartpole
python src/main.py exp_path=experiments/example/Cartpole/DQN.json idxs=0
python src/main.py experiment=example/Cartpole/EQRC idxs=0
python src/main.py exp_path=experiments/example/Cartpole/ESARSA.json idxs=0
python src/main.py exp_path=experiments/example/Cartpole/SoftmaxAC.json idxs=0

## MountainCar
python src/main.py exp_path=experiments/example/MountainCar/DQN.json idxs=0
python src/main.py experiment=example/MountainCar/EQRC idxs=0
python src/main.py exp_path=experiments/example/MountainCar/ESARSA.json idxs=0
python src/main.py exp_path=experiments/example/MountainCar/SoftmaxAC.json idxs=0

## Learning Curve
python experiments/example/learning_curve.py