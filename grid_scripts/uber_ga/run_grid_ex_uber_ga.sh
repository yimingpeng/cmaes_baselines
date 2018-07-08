#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/

# clone the repository
git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
cd ./cmaes_baselines/grid_scripts/uber_ga/

# setting the grid env
need sgegrid
qsub -t 1-5:1 uber_ga_HalfCheetah.sh
qsub -t 1-5:1 uber_ga_Hopper.sh
qsub -t 1-5:1 uber_ga_InvertedDoublePendulum.sh 
qsub -t 1-5:1 uber_ga_InvertedPendulum.sh 
qsub -t 1-5:1 uber_ga_InvertedPendulumSwingup.sh 
qsub -t 1-5:1 uber_ga_Reacher.sh 
qsub -t 1-5:1 uber_ga_Walker2D.sh 
qsub -t 1-5:1 uber_ga_BipedalWalker.sh 
qsub -t 1-5:1 uber_ga_BipedalWalkerHardcore.sh 
qsub -t 1-5:1 uber_ga_LunarLanderContinuous.sh 





