import pandas as pd
import numpy as np
import random
from policyiter import evaluation, improvement, treebuilder


'''
Create random Policy
Method to create a random Policy
'''
def create_random_policy(pre_decision_states, post_decision_states):
    # Number of pre decision states to iterate over
    n_pre_states = len(pre_decision_states)

    # initialize policy
    policy = [None] * n_pre_states

    counter = 0

    for pre_state in range(0, n_pre_states):

        # Last States have an empty trans_mat because the time horizon is reached and no decision can be made
        if (pre_decision_states[pre_state]["trans_mat"] is None):
            counter = counter + 1
        else:
            random_state = random.choice(pre_decision_states[pre_state]["trans_mat"].columns.levels[0])
            price_index = pre_decision_states[pre_state]["trans_mat"].index[0]

            policy[pre_state] = pre_decision_states[pre_state]["trans_mat"].loc[price_index, random_state][0][0]["post_state"]

    return policy



def main():

    # Definition of possible Price levels
    price_min = 1
    price_max = 2
    price_step_size = 1
    price_levels = np.arange(price_min, price_max + price_step_size, price_step_size)

    # Definition of possible Energy levels
    energy_min = 1
    energy_max = 2
    energy_step_size = 1
    energy_levels = np.arange(energy_min, energy_max + energy_step_size, energy_step_size)

    # TODO: generate Probabilities automatically with seed to create reproducable results (Durchführbar mit ziehen ohne
    # TODO: ohne zurücklegen aus einem Zustandsraum, den man vorher definiert hat
    # Definition of the Probabilities from one Price Level to another
    PROB_MATRIX = pd.DataFrame([[0.3, 0.7], [0.7, 0.3]], columns=price_levels, index=price_levels)

    max_time = 3
    time_horizon = np.arange(1, max_time + 1)

    pre_decision_states = []
    post_decision_states = []

    # initialize start state
    # Definition of states: [Price, Energy-Level]
    pre_decision_states.append({"v": None, "state": [1, 0], "trans_mat": None})

    # initialize empty policy
    policy = []

    # Create the tree
    pre_decision_states, post_decision_states = treebuilder.create_tree(time_horizon,
                                                                        energy_levels,
                                                                        price_levels,
                                                                        pre_decision_states,
                                                                        post_decision_states)

    # Create a random policy
    policy = create_random_policy(pre_decision_states, post_decision_states)

    counter = 0

    while (True):
        print(counter)
        counter = counter + 1

        # Evaluate_policy
        pre_decision_states, post_decision_states = evaluation.evaluate_policy(policy,
                                                                               pre_decision_states,
                                                                               post_decision_states,
                                                                               PROB_MATRIX)

        # Policy improvement
        policy_new = improvement.improve_policy(policy,
                                                pre_decision_states,
                                                post_decision_states,
                                                PROB_MATRIX).copy()

        print(policy)
        print(policy_new)

        if (policy_new == policy):
            break

        policy = policy_new


if __name__ == "__main__":
    main()
