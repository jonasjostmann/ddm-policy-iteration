import pandas as pd
import numpy as np
import random
import time
from policyiter import evaluation, improvement, treebuilder, config
import copy
import csv

SEED = 42

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
    PRICE_MIN = 1
    PRICE_MAX = 2
    PRICE_STEP_SIZE = 1
    price_levels = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP_SIZE, PRICE_STEP_SIZE)

    # Definition of possible Energy levels
    ENERGY_MIN = 0
    ENERGY_MAX = 7
    ENERGY_STEP_SIZE = 1
    energy_levels = np.arange(ENERGY_MIN, ENERGY_MAX + ENERGY_STEP_SIZE, ENERGY_STEP_SIZE)

    # Efficiency Coefficient
    EFF_COEFF = 1

    # MAX PULL and PUSH
    MAX_PULL = 10
    MAX_PUSH = 10

    # Definition of the Probabilities from one Price Level to another
    TRANS_PROB = [[0.5, 0.5], [0.5, 0.5]]
    # Check if dimensions are specified correct
    if len(TRANS_PROB) != len(price_levels):
        raise ValueError('You have to specify a transition probability for each Price Level!')
    PROB_MATRIX = pd.DataFrame(TRANS_PROB, columns=price_levels, index=price_levels)

    MAX_TIME = 3
    time_horizon = np.arange(1, MAX_TIME + 1)

    pre_decision_states = []
    post_decision_states = []

    # initialize start state
    # Definition of states: [Price, Energy-Level]
    INITIAL_STATE = [1, 0]
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

    stop_time_tree = time.process_time()

    pre_decision_states = copy.deepcopy(new_tree["pre"])
    post_decision_states = copy.deepcopy(new_tree["post"])



    # Create a random policy
    policy = copy.deepcopy(create_random_policy(copy.deepcopy(pre_decision_states), copy.deepcopy(post_decision_states)))

    start_time_rnd_policy = time.process_time()
    new_eval = copy.deepcopy(evaluation.evaluate_policy(copy.deepcopy(policy),
                                                                           copy.deepcopy(pre_decision_states),
                                                                           copy.deepcopy(post_decision_states),
                                                                           PROB_MATRIX.copy(),
                                                                           EFF_COEFF))

    stop_time_policy_iteration = time.process_time()

    # Initialization of evaluation Results
    initial_v = []
    iterations = []
    eval_time = []
    impr_time = []

    # Value of the initial Random Policy
    rnd_policy_v = new_eval["pre"][0]["v"]
    initial_v.append(rnd_policy_v)

    counter = 0
    iterations.append(counter)

    time_rnd_policy = stop_time_policy_iteration - start_time_rnd_policy
    eval_time.append(time_rnd_policy)

    impr_time.append(0)

    start_time_complete = time.process_time()
    while (True):

        counter = counter + 1
        iterations.append(counter)
        print(f"Iteration: #{counter}")

        start_time_eval = time.process_time()
        # Evaluate_policy
        new_eval = copy.deepcopy(evaluation.evaluate_policy(copy.deepcopy(policy),
                                                                               copy.deepcopy(pre_decision_states),
                                                                               copy.deepcopy(post_decision_states),
                                                                               PROB_MATRIX.copy(),
                                                                               EFF_COEFF))

        stop_time_eval = time.process_time()
        iter_eval_time = stop_time_eval - start_time_eval
        eval_time.append(iter_eval_time)

        pre_decision_states = copy.deepcopy(new_eval["pre"])
        post_decision_states = copy.deepcopy(new_eval["post"])

        new_init_v = pre_decision_states[0]["v"].copy()
        initial_v.append(new_init_v)

        start_time_impr = time.process_time()
        # Policy improvement
        policy_new = copy.deepcopy(improvement.improve_policy(copy.deepcopy(policy),
                                                copy.deepcopy(pre_decision_states),
                                                copy.deepcopy(post_decision_states),
                                                PROB_MATRIX,
                                                EFF_COEFF))

        stop_time_impr = time.process_time()
        iter_impr_time = stop_time_impr - start_time_impr
        impr_time.append(iter_impr_time)

        print(policy)
        print(policy_new)

        if (policy_new == policy):
            stop_time_complete = time.process_time()
            time_complete = stop_time_complete - start_time_complete

            fields = [initial_v, iterations, eval_time, impr_time]
            csv_file_name = csv_file_name = f"improvementExperimentRange_1-{str(MAX_TIME)}_maxTime_{str(MAX_TIME)}_" \
                f"maxPull_{str(MAX_PULL)}_maxPush_{str(MAX_PUSH)}"
            with open(f'results/{csv_file_name}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            break


        policy = copy.deepcopy(policy_new)

if __name__ == "__main__":
    main()
