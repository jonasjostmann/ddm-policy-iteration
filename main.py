import pandas as pd
import numpy as np
import random
import time
from policyiter import evaluation, improvement, treebuilder, config
import copy


def create_random_policy(pre_decision_states_copy, post_decision_states):
    """
    Method to create a random Policy
    :param pre_decision_states_copy: Pre-Decision states represented by an array
    :param post_decision_states: Post-Decision states represented by an array
    :return: policy represented by an array (indecis: Pre-Decision states, values: Post-Decision states)
    """
    # Number of pre decision states to iterate over
    n_pre_states = len(pre_decision_states_copy)

    # initialize policy

    policy_copy = [None] * n_pre_states

    counter = 0

    for pre_state in range(0, n_pre_states):

        # Last States have an empty trans_mat because the time horizon is reached and no decision can be made
        if (pre_decision_states_copy[pre_state]["trans_mat"] is not None):
            random.seed(config.SEED)
            random_state = random.choice(pre_decision_states_copy[pre_state]["trans_mat"].columns.levels[0])
            price_index = pre_decision_states_copy[pre_state]["trans_mat"].index[0]

            policy_copy[pre_state] = pre_decision_states_copy[pre_state]["trans_mat"].loc[price_index][random_state].iloc[0][0][
                "post_state"]

    return policy_copy


def main():

    # Definition of possible Price levels
    PRICE_MIN = config.PRICE_MIN
    PRICE_MAX = config.PRICE_MAX
    PRICE_STEP_SIZE = config.PRICE_STEP_SIZE
    price_levels = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP_SIZE, PRICE_STEP_SIZE)

    # Definition of possible Energy levels
    ENERGY_MIN = config.ENERGY_MIN
    ENERGY_MAX = config.ENERGY_MAX
    ENERGY_STEP_SIZE = config.ENERGY_STEP_SIZE
    energy_levels = np.arange(ENERGY_MIN, ENERGY_MAX + ENERGY_STEP_SIZE, ENERGY_STEP_SIZE)

    # Efficiency Coefficient
    EFF_COEFF = config.EFF_COEFF

    # MAX PULL and PUSH
    MAX_PULL = config.MAX_PULL
    MAX_PUSH = config.MAX_PUSH

    # Definition of the Probabilities from one Price Level to another
    TRANS_PROB = config.TRANS_PROB
    # Check if dimensions are specified correct
    if len(TRANS_PROB) != len(price_levels):
        raise ValueError('You have to specify a transition probability for each Price Level!')
    PROB_MATRIX = pd.DataFrame(TRANS_PROB, columns=price_levels, index=price_levels)

    MAX_TIME = config.MAX_TIME
    time_horizon = np.arange(1, MAX_TIME + 1)

    pre_decision_states = []
    post_decision_states = []

    # initialize start state
    # Definition of states: [Price, Energy-Level]
    INITIAL_STATE = config.INITIAL_STATE
    pre_decision_states.append({"v": None, "state": INITIAL_STATE, "trans_mat": None})

    # initialize empty policy
    policy = []

    # Create the tree
    start_time_tree = time.process_time()
    new_tree = copy.deepcopy(treebuilder.create_tree(time_horizon,
                                                                        energy_levels,
                                                                        price_levels,
                                                                        copy.deepcopy(pre_decision_states),
                                                                        copy.deepcopy(post_decision_states),
                                                                        MAX_PULL,
                                                                        MAX_PUSH))

    pre_decision_states = copy.deepcopy(new_tree["pre"])
    post_decision_states = copy.deepcopy(new_tree["post"])

    stop_time_tree = time.process_time()

    # Create a random policy
    policy = copy.deepcopy(create_random_policy(copy.deepcopy(pre_decision_states), copy.deepcopy(post_decision_states)))

    new_eval = copy.deepcopy(evaluation.evaluate_policy(copy.deepcopy(policy),
                                                                           copy.deepcopy(pre_decision_states),
                                                                           copy.deepcopy(post_decision_states),
                                                                           PROB_MATRIX.copy(),
                                                                           EFF_COEFF))

    # Value for the intitial state of the random Policy
    pre_first = copy.deepcopy(new_eval["pre"])

    counter = 0

    start_time_policy_iteration = time.process_time()

    while (True):

        counter = counter + 1
        print(f"Iteration: #{counter}")

        # Evaluate_policy
        new_eval = copy.deepcopy(evaluation.evaluate_policy(copy.deepcopy(policy),
                                                                               copy.deepcopy(pre_decision_states),
                                                                               copy.deepcopy(post_decision_states),
                                                                               PROB_MATRIX.copy(),
                                                                               EFF_COEFF))

        pre_decision_states = copy.deepcopy(new_eval["pre"])
        post_decision_states = copy.deepcopy(new_eval["post"])

        # Policy improvement
        policy_new = copy.deepcopy(improvement.improve_policy(copy.deepcopy(policy),
                                                copy.deepcopy(pre_decision_states),
                                                copy.deepcopy(post_decision_states),
                                                PROB_MATRIX,
                                                EFF_COEFF))

        if (policy_new == policy):

            stop_time_policy_iteration = time.process_time()
            time_policy_iteration = stop_time_policy_iteration - start_time_policy_iteration
            time_tree = stop_time_tree - start_time_tree
            print(f"\nPOLICY CONVERGED!\n"
                  "\n"
                  f"# Pre-Decision-States:                                  {len(pre_decision_states)}"
                  "\n"
                  f"# Post-Decision-States:                                 {len(post_decision_states)}"
                  "\n"
                  f"Different Price Levels:                                 {len(price_levels)}"
                  "\n"
                  f"Different Energy Levels:                                {len(energy_levels)}"
                  "\n"
                  f"Time for building the tree (Seconds):                   {time_tree}"
                  "\n"
                  f"Time for Policy Iteration (Seconds):                    {time_policy_iteration}"
                  "\n"
                  f"# Iterations:                                           {counter}"
                  "\n"
                  f"Chosen Initial State:                                   Price: {INITIAL_STATE[0]}, Energy-Level: {INITIAL_STATE[1]}"
                  "\n"
                  f"Expected reward for Initial State (OPTIMAL POLICY):     {pre_decision_states[0]['v']}"
                  "\n"
                  f"Expected reward for initial state (RANDOM POLICY):      {pre_first[0]['v']}")
            break


        policy = copy.deepcopy(policy_new)

if __name__ == "__main__":
    main()
