# PPOSGD_CMAES Surrogate 1 Local Search
## 1. CMAES used to train the policy, the objective function is using the PPO surrogate objective,
## 2. Add Gradient descent as local search in optimizing the surrogate objective function

Created by Yiming

1. New Implementation with new fitness function
2. CMAES train the entire policy layer
3. PPO is used only to train the value function for a given number of times.
4. PPO gradient is also used to train policy after cmaes training, works as a local search.