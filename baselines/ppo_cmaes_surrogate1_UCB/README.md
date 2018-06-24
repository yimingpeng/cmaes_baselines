# PPOSGD_CMAES Surrogate 1
# CMAES used to train the policy, the objective function is using the PPO surrogate objective
# Add UCB for explorasively select between real evaluation and surrogate models
Created by Yimingl

1. New Implementation with new fitness function
2. CMAES train the entire policy layer
3. PPO is used only to train the value function for a given number of times.