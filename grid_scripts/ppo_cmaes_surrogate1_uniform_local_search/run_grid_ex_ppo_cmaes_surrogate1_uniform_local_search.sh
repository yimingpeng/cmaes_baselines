#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/ppo_cmaes_surrogate1_uniform_local_search/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/ppo_cmaes_surrogate1_uniform_local_search/

# setting the grid env
need sgegrid
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_HalfCheetah.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_Hopper.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_InvertedDoublePendulum.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_InvertedPendulum.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_InvertedPendulumSwingup.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_Reacher.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_Walker2D.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_BipedalWalker.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_BipedalWalkerHardcore.sh
qsub -t 1-10:1 ppo_cmaes_surrogate1_uniform_local_search_LunarLanderContinuous.sh





