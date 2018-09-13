#!/usr/bin/env python

"""Description:
"""
__author__ = "Yiming Peng"
__copyright__ = "Copyright 2018, baselines"
__credits__ = ["Yiming Peng"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Yiming Peng"
__email__ = "yiming.peng@ecs.vuw.ac.nz"
__status__ = "Prototype"

# Scripts for generating GCP startup scripts
import os


f = open("../../hpc_scripts/template.sl")
algorithms = ["PPO", "CMAES", "CMAES_Layer_Entire",
              "CMAES_Layer_uniform", "DDPG",
              "ACKTR", "openai_es", "uber_ga",
              "TRPO", "ppo_cmaes_surrogate1_uniform", "ppo_cmaes_surrogate1_uniform_local_search"]
bullet_problems = ["HalfCheetah", "Hopper", "InvertedDoublePendulum",
                   "InvertedPendulum", "InvertedPendulumSwingup",
                   "Walker2D"]
gym_problems = ["LunarLanderContinuous", "BipedalWalker", "BipedalWalkerHardcore"]
seeds = range(5)
# Generate for Bullet problems
for algorithm in algorithms:
    for problem in bullet_problems:
        directory = "../../hpc_scripts/" + str(algorithm)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f1 = open(directory + "/" + algorithm + "_" +
                  problem + ".sl", 'w')
        for line in f:
            # if 'source activate cmaes_baselines' in line and algorithm == "DDPG":
            #         line = line.replace("cmaes_baselines", "ddpg_baselines")
            if '--job-name=ddpg_Walker2D' in line:
                line = line.replace("--job-name=ddpg_Walker2D", "--job-name="+algorithm+"_"+problem)
            if "/nesi/project/nesi00272/cmaes_baselines/baselines/ddpg/" in line:
                line = line.replace("ddpg",algorithm.lower())
            if "Walker2DBulletEnv-v0" in line:
                if algorithm == "DDPG":
                    line = "srun python main.py --env-id " + problem + "BulletEnv-v0" + " --seed $SLURM_ARRAY_TASK_ID\n"
                else:
                    line = "srun python run_pybullet.py --env " + problem + "BulletEnv-v0" + " --seed $SLURM_ARRAY_TASK_ID\n"
            f1.write(line)
        f1.close()
        f.seek(0)
    # f3 = open(directory + "/run_grid_ex_" + algorithm + ".sh", 'w')
    # for line in f2:
    #     if "ACKTR" in line:
    #         line = line.replace("ACKTR", algorithm)
    #     f3.write(line)
    # f3.close()
    # f2.seek(0)

# Generate for gym control problems
for algorithm in algorithms:
    for problem in gym_problems:
        directory = "../../hpc_scripts/" + str(algorithm)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f1 = open(directory + "/" + algorithm + "_" +
                  problem + ".sl", 'w')
        for line in f:
            if '--job-name=ddpg_Walker2D' in line:
                line = line.replace("--job-name=ddpg_Walker2D", "--job-name="+algorithm+"_"+problem)
            if "/nesi/project/nesi00272/cmaes_baselines/baselines/ddpg/" in line:
                line = line.replace("ddpg",algorithm.lower())
            if "Walker2DBulletEnv-v0" in line:
                if algorithm == "DDPG":
                    line = "python main.py --env-id " + problem + "-v2" + " --seed $SLURM_ARRAY_TASK_ID\n"
                else:
                    line = "python run_simple_ctrl.py --env " + problem + "-v2" + " --seed $SLURM_ARRAY_TASK_ID\n"
            f1.write(line)
        f1.close()
        f.seek(0)

    # f3 = open(directory + "/run_grid_ex_" + algorithm + ".sh", 'w')
    # for line in f2:
    #     if "ACKTR" in line:
    #         line = line.replace("ACKTR", algorithm)
    #     f3.write(line)
    # f3.close()
    # f2.seek(0)
f.close()
