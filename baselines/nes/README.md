# Natural Evolutionary Strategy only
Created by Yiming

- Updates: [7/6/2018]
    1. CMAES is used to optimize a different topology from the one of PPO
    2. CMAES is used to train all layers at the same time, thus its efficiency is a concern
    3. Results is reported at every 10,000 steps, the best solution from last generation will be re-evaluated for 2048 steps.
       The reason of reporting in this way is, because during the tracking points, one generation's evaluation is highly likely
        not completed, we don't have the current best.