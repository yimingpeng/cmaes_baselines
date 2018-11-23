#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd ../../$experimentName/acktr/

for i in {0..5}
do
	( python $pyName --env HalfCheetahBulletEnv-v0 --seed $i  &> HalfCheetah_"$i".out)
     echo "Complete the process $i"
done