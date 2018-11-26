#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/ppo_cmaes_surrogate1_uniform_original/

for i in {0..5}
do
	( python $pyName --env ReacherBulletEnv-v0 --seed $i  &> Reacher_"$i".out)
     echo "Complete the process $i"
done