#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/CMAES_Layer_uniform/

# setting the grid env
need sgegrid
qsub -t 1-5:1 CMAES_Layer_uniform_HalfCheetah.sh
qsub -t 1-5:1 CMAES_Layer_uniform_Hopper.sh
qsub -t 1-5:1 CMAES_Layer_uniform_InvertedDoublePendulum.sh 
qsub -t 1-5:1 CMAES_Layer_uniform_InvertedPendulum.sh 
qsub -t 1-5:1 CMAES_Layer_uniform_InvertedPendulumSwingup.sh 
qsub -t 1-5:1 CMAES_Layer_uniform_Reacher.sh 
qsub -t 1-5:1 CMAES_Layer_uniform_Walker2D.sh 
qsub -t 1-5:1 CMAES_Layer_uniform_BipedalWalker.sh 
qsub -t 1-5:1 CMAES_Layer_uniform_BipedalWalkerHardcore.sh 
qsub -t 1-5:1 CMAES_Layer_uniform_LunarLanderContinuous.sh 





