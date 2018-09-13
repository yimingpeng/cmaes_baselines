#!/bin/bash
#SBATCH --job-name=PPO_Hopper      # job name (shows up in the queue)
#SBATCH --account=nesi00272     # Project Account
#SBATCH --time=200:00:00         # Walltime (HH:MM:SS)
#SBATCH -D /nesi/project/nesi00272/cmaes_baselines/baselines/ppo/
#SBATCH --mem-per-cpu=4096      # memory/cpu (in MB)
#SBATCH --ntasks=1  # number of tasks (e.g. MPI)
#SBATCH --cpus-per-task=1  # number of cores per task (e.g. OpenMP)
#SBATCH --partition=long        # specify a partition
#SBATCH --hint=nomultithread    # don't use hyperthreading
#SBATCH --array=1-10     # Array definition
#SBATCH --error=%A_%a.err
#SBATCH --output=%A_%a.out

export PATH=/home/yiming.peng/miniconda3/bin/:$PATH
source activate cmaes_baselines
srun python run_pybullet.py --env HopperBulletEnv-v0 --seed $SLURM_ARRAY_TASK_ID
