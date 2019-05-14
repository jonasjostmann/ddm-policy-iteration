import pandas as pd
import numpy as np
import random

# Initialize empty policy
# Policy will be build as an array, which indices are the id of pre decision states and reference to the id of the next
# post-decision state to imply the decision which were made

# TODO: generate Probabilities automatically with seed to create reproducable results (Durchführbar mit ziehen ohne
# Todo: ohne zurücklegen aus einem Zustandsraum, den man vorher definiert hat
# Definition of the Probabilities from one Price Level to another
prob_matrix = [[0.3, 0.7], [0.7, 0.3]]

# Definition of possible Price levels
price_min = 1
price_max = 2
price_step_size = 1
price_levels = np.arange(price_min, price_max+1, price_step_size)

# Definition of possible Energy levels
energy_min = 0
energy_max = 2
energy_step_size = 1
energy_levels = np.arange(energy_min, energy_max+1, energy_step_size)

max_time = 1
time_horizon = np.arange(1, max_time+1)

pre_decision_states = []
post_decision_states = []

# initialize start state
# Definition of states: [Price, Energy-Level]
pre_decision_states.append({"v": None, "state": [1,0], "trans_mat": None})

# initialize empty policy
policy = []

'''
Create tree
Method to create the tree
'''
# TODO: in order to be able to differ in discretization step size change all references to which need normal numbers
#TODO: Add Probabilities to post decision states
def create_tree():

    last_post = []
    last_pre_states = [0]

    # Counter for Pre and Post Decision states
    n_pre_states = 0
    n_post_states = 0

    for t in time_horizon:

        last_pre_states_temp = []

        for pre_state in last_pre_states:
            pre_decision_states[pre_state]["trans_mat"] = pd.DataFrame(columns = [energy_levels], index=[pre_decision_states[pre_state]["state"][0]])
            pre_decision_states[pre_state]["trans_mat"] = pre_decision_states[pre_state]["trans_mat"].astype('object')

            # Insert None Values to build structure of DataFrame
            for e_l in energy_levels:
                # TODO: Hier ist eine Abhängigkeit zum Wert des Energy States -> Umwandeln in Loc und heraussuchen des Price wertes vorher
                pre_decision_states[pre_state]["trans_mat"].iloc[0, e_l] = [dict(v= None, post_state= None)]

            for post_state in pre_decision_states[pre_state]["trans_mat"].columns:

                # Caused by Multiindex issue position 0 must be selected
                post_state = post_state[0]
                post_decision_states.append({"v": None, "state": [pre_decision_states[pre_state]["state"][0], post_state], "trans_mat": None})

                # Save Post decision state in pre-decision state
                pre_decision_states[pre_state]["trans_mat"].iloc[0, post_state][0]["post_state"] = n_post_states

                # Initialize Dataframe for post_decision_state
                post_decision_states[n_post_states]["trans_mat"] = pd.DataFrame(columns = [post_state], index=[price_levels])

                for row in post_decision_states[n_post_states]["trans_mat"].index:

                    # Caused by Multiindex issue position 0 must be selected
                    row = row[0]
                    post_price = row
                    post_energy_level = post_decision_states[n_post_states]["trans_mat"].columns[0] # Energy Level equals col name

                    #TODO: Preisübergänge durchgehen mit Wahrscheinlichkeiten und Neue Predecision Zustände definieren
                    # row is equal to the price (because row index is set to the price)
                    pre_decision_states.append({"v": None, "state": [post_price, post_energy_level], "trans_mat": None})
                    # After appending an additional pre state the counter must be increased by one
                    n_pre_states = n_pre_states + 1

                    # Set pre state in post decision trans matrix
                    post_decision_states[n_post_states]["trans_mat"].loc[row, post_energy_level] = [dict(v=None, pre_state=n_pre_states)]

                    # Append new pre-decision state to list
                    last_pre_states_temp.append(n_pre_states)

                # Increase index of post_decision_states
                n_post_states =  n_post_states +1

        last_pre_states = last_pre_states_temp


'''
Create random Policy
Method to create a random Policy
'''
def create_random_policy():

    # Number of pre decision states to iterate over
    n_pre_states = len(pre_decision_states)

    # initialize policy
    policy = [None]*n_pre_states

    counter=0

    for pre_state in range(0, n_pre_states):

        # Last States have an empty trans_mat because the time horizon is reached and no decision can be made
        if(pre_decision_states[pre_state]["trans_mat"] is None):
            counter=counter+1
        else:
            random_state = random.choice(pre_decision_states[pre_state]["trans_mat"].columns.levels[0])

            policy[pre_state] = pre_decision_states[pre_state]["trans_mat"].iloc[0, random_state][0]["post_state"]

    print(policy)
    print(counter)

create_tree()

create_random_policy()

