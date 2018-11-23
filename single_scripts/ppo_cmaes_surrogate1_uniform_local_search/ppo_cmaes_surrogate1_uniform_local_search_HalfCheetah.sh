#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/ppo_cmaes_surrogate1_uniform_local_search/

for i in {0..5}
do
	( python $pyName --env HalfCheetahBulletEnv-v0 --seed $i  &> HalfCheetah_"$i".out)
     echo "Complete the process $i"
done