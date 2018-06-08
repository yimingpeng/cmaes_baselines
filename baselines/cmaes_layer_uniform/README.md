# CMAES only with layer training, where only partial paras will be selected with some proportion for training
Created by Yiming

- Updates: [7/6/2018]
    1. CMAES is used to optimize a different topology (same in policy) from the one of PPO
    2. CMAES is used to train one layer at one time, to address its efficiency problem
    3. Not all parameters in one layer will be trained at one time, a proportion of parameteres uniformly seleted from all layer params
       will be trained
    4. Results is reported at every 10,000 steps, the best solution from last generation will be re-evaluated for 2048 steps.
       The reason of reporting in this way is, because during the tracking points, one generation's evaluation is highly likely
        not completed, we don't have the current best.