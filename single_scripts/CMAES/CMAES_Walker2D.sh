#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/cmaes/

for i in {0..5}
do
	( python $pyName --env Walker2DBulletEnv-v0 --seed $i  &> Walker2D_"$i".out)
     echo "Complete the process $i"
done