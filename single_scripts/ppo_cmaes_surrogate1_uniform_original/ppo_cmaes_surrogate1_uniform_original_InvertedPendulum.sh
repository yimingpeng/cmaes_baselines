#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/ppo_cmaes_surrogate1_uniform_original/

for i in {0..5}
do
	( python $pyName --env InvertedPendulumBulletEnv-v0 --seed $i  &> InvertedPendulum_"$i".out)
     echo "Complete the process $i"
done