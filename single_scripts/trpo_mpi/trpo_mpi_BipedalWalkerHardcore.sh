#!/bin/bash

experimentName="baselines"

pyName="run_simple_ctrl.py"

cd ../../$experimentName/trpo_mpi/

for i in {0..5}
do
	( python $pyName --env BipedalWalkerHardcore-v0 --seed $i &> BipedalWalkerHardcore_"$i".out)
     echo "Complete the process $i"
done