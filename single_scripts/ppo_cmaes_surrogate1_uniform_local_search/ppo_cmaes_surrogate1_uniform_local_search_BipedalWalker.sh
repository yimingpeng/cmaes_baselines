#!/bin/bash

experimentName="baselines"

pyName="run_simple_ctrl.py"

cd ../../$experimentName/ppo_cmaes_surrogate1_uniform_local_search/

for i in {0..5}
do
	( python $pyName --env BipedalWalker-v2 --seed $i &> BipedalWalker_"$i".out)
     echo "Complete the process $i"
done