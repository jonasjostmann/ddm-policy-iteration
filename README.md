# Dynamic Decision Making in Energy Systems

#### A solution approach for a simplified Energy System using Policy Iteration  

Making optimal decisions in the energy sector is an important research topic today, due to resource scarcity and rapidly changing prices. When it comes to optimizing such decisions, limits are often set by uncertainty. In this paper, the implementation of a solution approach for making optimal decisions in the context of dynamic decision making is presented. Therefore, the policy iteration algorithm from the field of dynamic programming in the context of an energy storage decision problem is presented and implemented. Within the evaluation of this solution approach hurdles like the exponentially growing state space are considered and further results are presented.

As part of the seminar "Dynamic decision making in energy systems and transportation" a simplified version of this optimization problem is covered. 
The corresponding seminar thesis aims to show how such a dynamic decision problem can be solved by using an Algorithm called Policy Iteration.
This Python Code represents the actual representation of the Policy Iteration Algorithm for the above stated Optimization Problem.

### Simplifications:

1. Only Energy Storage and Market are covered (Energy is bought from market and later on sold to the market) 
1. Energy Storage Size and Price Levels are discrete and have a minimum and a maximum value.

### User Input:

In the ``config.py`` of the package ``policyiter`` the discretization of the Energy and Price Levels can be adjusted, as well as 
the maximum time steps which are considered. 

### Output:

The main method of the ``main.py`` script returns the optimal policy, which were determined by the Policy Iteration Algorithm.
Additionally some interesting parameters like computation time, the optimal V of the Initial State and the amount of Predecision and Postdecision States are returned.  