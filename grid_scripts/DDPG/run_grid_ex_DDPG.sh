#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/

# clone the repository
git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
cd ./cmaes_baselines/grid_scripts/DDPG/

# setting the grid env
need sgegrid
qsub -t 1-5:1 DDPG_HalfCheetah.sh
qsub -t 1-5:1 DDPG_Hopper.sh
qsub -t 1-5:1 DDPG_InvertedDoublePendulum.sh 
qsub -t 1-5:1 DDPG_InvertedPendulum.sh 
qsub -t 1-5:1 DDPG_InvertedPendulumSwingup.sh 
qsub -t 1-5:1 DDPG_Reacher.sh 
qsub -t 1-5:1 DDPG_Walker2D.sh 
qsub -t 1-5:1 DDPG_BipedalWalker.sh 
qsub -t 1-5:1 DDPG_BipedalWalkerHardcore.sh 
qsub -t 1-5:1 DDPG_LunarLanderContinuous.sh 





