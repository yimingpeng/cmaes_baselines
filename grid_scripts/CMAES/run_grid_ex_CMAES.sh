#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/CMAES/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/CMAES/

# setting the grid env
need sgegrid
qsub -t 1-5:1 CMAES_HalfCheetah.sh
qsub -t 1-5:1 CMAES_Hopper.sh
qsub -t 1-5:1 CMAES_InvertedDoublePendulum.sh 
qsub -t 1-5:1 CMAES_InvertedPendulum.sh 
qsub -t 1-5:1 CMAES_InvertedPendulumSwingup.sh 
qsub -t 1-5:1 CMAES_Reacher.sh 
qsub -t 1-5:1 CMAES_Walker2D.sh 
qsub -t 1-5:1 CMAES_BipedalWalker.sh 
qsub -t 1-5:1 CMAES_BipedalWalkerHardcore.sh 
qsub -t 1-5:1 CMAES_LunarLanderContinuous.sh 





