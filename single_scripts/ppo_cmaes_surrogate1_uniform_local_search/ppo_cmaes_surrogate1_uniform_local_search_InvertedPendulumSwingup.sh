#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/ppo_cmaes_surrogate1_uniform_local_search/

for i in {0..5}
do
	( python $pyName --env InvertedPendulumSwingupBulletEnv-v0 --seed $i  &> InvertedPendulumSwingup_"$i".out)
     echo "Complete the process $i"
done