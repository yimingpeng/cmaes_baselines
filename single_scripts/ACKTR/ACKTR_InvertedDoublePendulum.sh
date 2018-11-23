#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/acktr/

for i in {0..5}
do
	( python $pyName --env InvertedDoublePendulumBulletEnv-v0 --seed $i  &> InvertedDoublePendulum_"$i".out)
     echo "Complete the process $i"
done