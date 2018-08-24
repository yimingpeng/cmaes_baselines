#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/CMAES_Layer_Entire/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/CMAES_Layer_Entire/

# setting the grid env
need sgegrid
qsub -t 1-10:1 CMAES_Layer_Entire_HalfCheetah.sh
qsub -t 1-10:1 CMAES_Layer_Entire_Hopper.sh
qsub -t 1-10:1 CMAES_Layer_Entire_InvertedDoublePendulum.sh
qsub -t 1-10:1 CMAES_Layer_Entire_InvertedPendulum.sh
qsub -t 1-10:1 CMAES_Layer_Entire_InvertedPendulumSwingup.sh
qsub -t 1-10:1 CMAES_Layer_Entire_Reacher.sh
qsub -t 1-10:1 CMAES_Layer_Entire_Walker2D.sh
qsub -t 1-10:1 CMAES_Layer_Entire_BipedalWalker.sh
qsub -t 1-10:1 CMAES_Layer_Entire_BipedalWalkerHardcore.sh
qsub -t 1-10:1 CMAES_Layer_Entire_LunarLanderContinuous.sh





