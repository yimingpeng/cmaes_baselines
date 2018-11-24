#! /bin/bash

# the repository
cd /vol/grid-solar/sgeusers/yimingpeng/cmaes_baselines/grid_scripts/trpo_mpi/

# clone the repository
# git clone https://yimingpeng:Aa19820713@github.com/yimingpeng/cmaes_baselines &
# cd ./cmaes_baselines/grid_scripts/trpo_mpi/

# setting the grid env
need sgegrid
qsub -t 1-5:1 trpo_mpi_HalfCheetah.sh
qsub -t 1-5:1 trpo_mpi_Hopper.sh
qsub -t 1-5:1 trpo_mpi_InvertedDoublePendulum.sh
qsub -t 1-5:1 trpo_mpi_InvertedPendulum.sh
qsub -t 1-5:1 trpo_mpi_InvertedPendulumSwingup.sh
qsub -t 1-5:1 trpo_mpi_Reacher.sh
qsub -t 1-5:1 trpo_mpi_Walker2D.sh
qsub -t 1-5:1 trpo_mpi_BipedalWalker.sh
qsub -t 1-5:1 trpo_mpi_BipedalWalkerHardcore.sh
qsub -t 1-5:1 trpo_mpi_LunarLanderContinuous.sh





