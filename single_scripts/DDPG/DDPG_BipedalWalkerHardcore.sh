#!/bin/bash

experimentName="baselines"

pyName="main.py"

cd ../../$experimentName/ddpg/

for i in {0..5}
do
	( python $pyName --env-id BipedalWalkerHardcore-v0 --seed $i &> BipedalWalkerHardcore_"$i".out)
     echo "Complete the process $i"
done