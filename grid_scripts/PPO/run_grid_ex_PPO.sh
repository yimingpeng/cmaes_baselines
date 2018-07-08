#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/PPO/

# setting the grid env
need sgegrid
qsub -t 1-5:1 PPO_HalfCheetah.sh
qsub -t 1-5:1 PPO_Hopper.sh
qsub -t 1-5:1 PPO_InvertedDoublePendulum.sh 
qsub -t 1-5:1 PPO_InvertedPendulum.sh 
qsub -t 1-5:1 PPO_InvertedPendulumSwingup.sh 
qsub -t 1-5:1 PPO_Reacher.sh 
qsub -t 1-5:1 PPO_Walker2D.sh 
qsub -t 1-5:1 PPO_BipedalWalker.sh 
qsub -t 1-5:1 PPO_BipedalWalkerHardcore.sh 
qsub -t 1-5:1 PPO_LunarLanderContinuous.sh 





