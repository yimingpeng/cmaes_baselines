#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/uber_ga_original/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/uber_ga_original/

# setting the grid env
need sgegrid
qsub -t 1-5:1 uber_ga_original_HalfCheetah.sh
qsub -t 1-5:1 uber_ga_original_Hopper.sh
qsub -t 1-5:1 uber_ga_original_InvertedDoublePendulum.sh
qsub -t 1-5:1 uber_ga_original_InvertedPendulum.sh
qsub -t 1-5:1 uber_ga_original_InvertedPendulumSwingup.sh
qsub -t 1-5:1 uber_ga_original_Reacher.sh
qsub -t 1-5:1 uber_ga_original_Walker2D.sh
qsub -t 1-5:1 uber_ga_original_BipedalWalker.sh
qsub -t 1-5:1 uber_ga_original_BipedalWalkerHardcore.sh
qsub -t 1-5:1 uber_ga_original_LunarLanderContinuous.sh





