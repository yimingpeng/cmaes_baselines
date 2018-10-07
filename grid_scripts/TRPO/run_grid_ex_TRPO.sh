#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/TRPO/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/TRPO/

# setting the grid env
need sgegrid
qsub -t 1-5:1 TRPO_HalfCheetah.sh
qsub -t 1-5:1 TRPO_Hopper.sh
qsub -t 1-5:1 TRPO_InvertedDoublePendulum.sh
qsub -t 1-5:1 TRPO_InvertedPendulum.sh
qsub -t 1-5:1 TRPO_InvertedPendulumSwingup.sh
qsub -t 1-5:1 TRPO_Reacher.sh
qsub -t 1-5:1 TRPO_Walker2D.sh
qsub -t 1-5:1 TRPO_BipedalWalker.sh
qsub -t 1-5:1 TRPO_BipedalWalkerHardcore.sh
qsub -t 1-5:1 TRPO_LunarLanderContinuous.sh





