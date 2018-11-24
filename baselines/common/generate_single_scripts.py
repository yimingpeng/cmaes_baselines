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

person = 'yimingpeng'
# person = 'achen'

if person == 'achen':
    f = open("../../single_scripts/aaron_template.sh")
    f2 = open("../../single_scripts/run_aaron_grid_ex_template.sh")
else:
    f = open("../../single_scripts/template.sh")
    f2 = open("../../single_scripts/run_grid_ex_template.sh")
algorithms = ["PPO", "CMAES",
              "CMAES_Layer_uniform", "DDPG",
              "ACKTR", "openai_es", "uber_ga_original",
              "TRPO", "ppo_cmaes_surrogate1_uniform", "ppo_cmaes_surrogate1_uniform_local_search"]
bullet_problems = ["HalfCheetah", "Hopper", "InvertedDoublePendulum",
                   "InvertedPendulum", "InvertedPendulumSwingup", "Reacher",
                   "Walker2D"]
gym_problems = ["LunarLanderContinuous", "BipedalWalker", "BipedalWalkerHardcore"]
seeds = range(5)
# Generate for Bullet problems
for algorithm in algorithms:
    for problem in bullet_problems:
        directory = "../../single_scripts/" + str(algorithm)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f1 = open(directory + "/" + algorithm + "_" +
                  problem + ".sh", 'w')
        for line in f:
            # if 'source activate cmaes_baselines' in line and algorithm == "DDPG":
            #         line = line.replace("cmaes_baselines", "ddpg_baselines")
            if 'pyName="run_pybullet.py"' in line:
                if algorithm == "DDPG":
                    line = line.replace("run_pybullet.py", "main.py")
            if "$experimentName/ppo/" in line:
                line = "cd ../../$experimentName/" + algorithm.lower() + "/\n"
            if "BipedalWalker-v2" in line:
                if algorithm == "DDPG":
                    line = "\t( python $pyName --env-id " + problem + "BulletEnv-v0" + " --seed $i  &> "+ problem +"_\"$i\".out)\n"
                else:
                    line = "\t( python $pyName --env " + problem + "BulletEnv-v0" + " --seed $i  &> "+ problem +"_\"$i\".out)\n"
            f1.write(line)
        f1.close()
        f.seek(0)
    f3 = open(directory + "/run_grid_ex_" + algorithm + ".sh", 'w')
    for line in f2:
        if "ACKTR" in line:
            line = line.replace("ACKTR", algorithm)
        f3.write(line)
    f3.close()
    f2.seek(0)

# Generate for gym control problems
for algorithm in algorithms:
    for problem in gym_problems:
        directory = "../../single_scripts/" + str(algorithm)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f1 = open(directory + "/" + algorithm + "_" +
                  problem + ".sh", 'w')
        for line in f:
            if 'pyName="run_pybullet.py"' in line:
                if algorithm == "DDPG":
                    line = line.replace("run_pybullet.py", "main.py")
                else:
                    line = 'pyName="run_simple_ctrl.py"\n'
            if "$experimentName/ppo/" in line:
                line = "cd ../../$experimentName/" + algorithm.lower() + "/\n"
            if "BipedalWalker-v2" in line:
                if algorithm == "DDPG":
                    line = "\t( python $pyName --env-id " + problem + "-v0" + " --seed $i &> "+ problem +"_\"$i\".out)\n"
                else:
                    if problem == "LunarLanderContinuous" or problem == "BipedalWalker":
                        line = "\t( python $pyName --env " + problem + "-v2" + " --seed $i &> "+ problem +"_\"$i\".out)\n"
                    else:
                        line = "\t( python $pyName --env " + problem + "-v0" + " --seed $i &> "+ problem +"_\"$i\".out)\n"
            f1.write(line)
        f1.close()
        f.seek(0)

    f3 = open(directory + "/run_grid_ex_" + algorithm + ".sh", 'w')
    for line in f2:
        if "ACKTR" in line:
            line = line.replace("ACKTR", algorithm)
        f3.write(line)
    f3.close()
    f2.seek(0)
f.close()



