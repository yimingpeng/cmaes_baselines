#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/cmaes_layer_uniform/

for i in {0..5}
do
	( python $pyName --env HopperBulletEnv-v0 --seed $i  &> Hopper_"$i".out)
     echo "Complete the process $i"
done