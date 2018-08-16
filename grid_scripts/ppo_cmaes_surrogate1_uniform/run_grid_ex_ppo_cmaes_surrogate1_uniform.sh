#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/achen/cmaes_baselines/grid_scripts/ppo_cmaes_surrogate1_uniform/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/ppo_cmaes_surrogate1_uniform/

# setting the grid env
need sgegrid
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_HalfCheetah.sh
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_Hopper.sh
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_InvertedDoublePendulum.sh 
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_InvertedPendulum.sh 
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_InvertedPendulumSwingup.sh 
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_Reacher.sh 
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_Walker2D.sh 
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_BipedalWalker.sh 
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_BipedalWalkerHardcore.sh 
qsub -t 1-5:1 ppo_cmaes_surrogate1_uniform_LunarLanderContinuous.sh 





