# Dynamic Decision Making in Energy Systems

#### A solution approach for a simplified Energy System using Policy Iteration  

Since, Energy is an equally expensive and important good nowadays, it is crucial to optimize its utilization.
Because energy prices are changing volatile, in some cases an energy storage is a good solution to overcome this issue and save money. 
The idea behind this approach is seems to be very simple: Energy is bought from the market when the prices are low and is stored in
the energy storage unit. When the energy price reaches a certain point the energy supply from the market is substituted with the 
own storage unit.
Although, this approach seems easy to implement in the first place, it is very hard in practice. 
One reason for this is the amount of exogenous factors which affect the energy price and the demand.

As part of the Seminar "Dynamic Decision Making in Energy Systems and Transportation" a simplified version of this optimization problem is covered. 
This seminar thesis aims to show how such a dynamic decision problem can be solved by using an Algorithm called Policy Iteration.

The corresponding Python Code can be found in this Repository.

### Simplifications:

1. Only Energy Storage and Market are covered (Energy is bought from market and later on sold to the market) 
1. Energy Storage Size and Price Levels are discrete and have a minimum and a maximum value.

### User Input:

In the ``config.py`` of the package ``policyiter`` the discretization of the Energy and Price Levels can be adjusted, as well as 
the maximum time steps which are considered. 

### Output:

The main method returns the optimal policy, which were determined by the Policy Iteration Algorithm.
Additionally some interesting parameters like computation time, the optimal V of the Initial State and the amount of Predecision and Postdecision States are returned.  