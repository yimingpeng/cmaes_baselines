#!/bin/bash

experimentName="baselines"

pyName="main.py"

cd ../../$experimentName/ddpg/

for i in {0..5}
do
	( python $pyName --env-id ReacherBulletEnv-v0 --seed $i  &> Reacher_"$i".out)
     echo "Complete the process $i"
done