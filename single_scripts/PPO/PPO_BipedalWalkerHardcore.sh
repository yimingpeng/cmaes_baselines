#!/bin/bash

experimentName="baselines"

pyName="run_simple_ctrl.py"

cd ../../$experimentName/ppo/

for i in {0..5}
do
	( python $pyName --env BipedalWalkerHardcore-v2 --seed $i &> BipedalWalkerHardcore_"$i".out)
     echo "Complete the process $i"
done