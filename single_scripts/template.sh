#!/bin/bash

experimentName="baselines"

pyName="run_pybullet.py"

cd $experimentName/ppo/

for i in {0..5}
do
     (python $pyName --env BipedalWalker-v2 --seed "$i" &)
     echo "Complete the process $i"
done