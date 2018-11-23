#!/bin/bash

experimentName="baselines"

pyName="main.py"

cd ../../$experimentName/ddpg/

for i in {0..5}
do
	( python $pyName --env-id InvertedPendulumSwingupBulletEnv-v0 --seed $i  &> InvertedPendulumSwingup_"$i".out)
     echo "Complete the process $i"
done