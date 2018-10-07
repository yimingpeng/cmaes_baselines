#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/openai_es/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/openai_es/

# setting the grid env
need sgegrid
qsub -t 1-5:1 openai_es_HalfCheetah.sh
qsub -t 1-5:1 openai_es_Hopper.sh
qsub -t 1-5:1 openai_es_InvertedDoublePendulum.sh
qsub -t 1-5:1 openai_es_InvertedPendulum.sh
qsub -t 1-5:1 openai_es_InvertedPendulumSwingup.sh
qsub -t 1-5:1 openai_es_Reacher.sh
qsub -t 1-5:1 openai_es_Walker2D.sh
qsub -t 1-5:1 openai_es_BipedalWalker.sh
qsub -t 1-5:1 openai_es_BipedalWalkerHardcore.sh
qsub -t 1-5:1 openai_es_LunarLanderContinuous.sh





