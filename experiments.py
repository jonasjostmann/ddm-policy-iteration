import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from policyiter import evaluation, improvement, treebuilder, config


def create_random_policy(pre_decision_states, post_decision_states):
    """
    Method to create a random Policy
    :param pre_decision_states: Pre-Decision states represented by an array
    :param post_decision_states: Post-Decision states represented by an array
    :return: policy represented by an array (indecis: Pre-Decision states, values: Post-Decision states)
    """
    # Number of pre decision states to iterate over
    n_pre_states = len(pre_decision_states)

    # initialize policy
    policy = [None] * n_pre_states

    counter = 0

    for pre_state in range(0, n_pre_states):

        # Last States have an empty trans_mat because the time horizon is reached and no decision can be made
        if (pre_decision_states[pre_state]["trans_mat"] is not None):
            random.seed(config.SEED)
            random_state = random.choice(pre_decision_states[pre_state]["trans_mat"].columns.levels[0])
            price_index = pre_decision_states[pre_state]["trans_mat"].index[0]

            policy[pre_state] = pre_decision_states[pre_state]["trans_mat"].loc[price_index][random_state].iloc[0][0][
                "post_state"]

    return policy


def search_optimal_policy(time_horizon, energy_levels, price_levels, pre_decision_states, post_decision_states,
                          MAX_PULL, MAX_PUSH, PROB_MATRIX, EFF_COEFF, INITIAL_STATE):
    # initialize empty policy
    policy = []

    # Create the tree
    start_time_tree = time.process_time()
    pre_decision_states, post_decision_states = treebuilder.create_tree(time_horizon,
                                                                        energy_levels,
                                                                        price_levels,
                                                                        pre_decision_states,
                                                                        post_decision_states,
                                                                        MAX_PULL,
                                                                        MAX_PUSH)
    stop_time_tree = time.process_time()

    # Create a random policy
    policy = create_random_policy(pre_decision_states, post_decision_states)

    counter = 0

    start_time_policy_iteration = time.process_time()
    while (True):
        counter = counter + 1
        print(f"Iteration: #{counter}")

        # Evaluate_policy
        pre_decision_states, post_decision_states = evaluation.evaluate_policy(policy.copy(),
                                                                               pre_decision_states.copy(),
                                                                               post_decision_states.copy(),
                                                                               PROB_MATRIX.copy(),
                                                                               EFF_COEFF)

        # Policy improvement
        policy_new = improvement.improve_policy(policy,
                                                pre_decision_states.copy(),
                                                post_decision_states.copy(),
                                                PROB_MATRIX,
                                                EFF_COEFF).copy()

        print(policy)
        print(policy_new)

        if (policy_new == policy):
            pre_decision_states, post_decision_states = evaluation.evaluate_policy(policy.copy(),
                                                                                   pre_decision_states.copy(),
                                                                                   post_decision_states.copy(),
                                                                                   PROB_MATRIX.copy(),
                                                                                   EFF_COEFF)
            stop_time_policy_iteration = time.process_time()
            time_policy_iteration = stop_time_policy_iteration - start_time_policy_iteration
            time_tree = stop_time_tree - start_time_tree

            # returns 1. #pre_decision_states, 2. #post_decision_states, 3. #price_levels, 4. #energy_levels,
            # 5. Time time_tree
            return(pre_decision_states[0]['v'])

        policy = policy_new


def main():
    # Definition of possible Price levels
    PRICE_MIN = 1
    PRICE_MAX = 2
    PRICE_STEP_SIZE = 1
    price_levels = np.arange(PRICE_MIN, PRICE_MAX + PRICE_STEP_SIZE, PRICE_STEP_SIZE)

    # Efficiency Coefficient
    EFF_COEFF = 1

    # MAX PULL and PUSH
    MAX_PULL = 2
    MAX_PUSH = 2

    # Definition of the Probabilities from one Price Level to another
    k = len(price_levels)
    np.random.seed(23)
    result = np.identity(k) + np.random.uniform(low=0., high=.25, size=(k, k))
    result /= result.sum(axis=1, keepdims=1)
    #TRANS_PROB = result
    TRANS_PROB = [[0.5, 0.5], [0.5,0.5]]
    PROB_MATRIX = pd.DataFrame(TRANS_PROB, columns=price_levels, index=price_levels)

    MAX_TIME = 4
    time_horizon = np.arange(1, MAX_TIME + 1)

    pre_decision_states = []
    post_decision_states = []

    # initialize start state
    # Definition of states: [Price, Energy-Level]
    INITIAL_STATE = [1,0]
    pre_decision_states.append({"v": None, "state": INITIAL_STATE, "trans_mat": None})

    n = 7
    experiment_range = np.arange(1,n,1)
    cost_per_unit = -0.01

    profit = []
    time_list = []

    csv_file_name = f"profitExperimentRange_1-{str(n)}_maxTime_{str(MAX_TIME)}_maxPull_{str(MAX_PULL)}_maxPush_{str(MAX_PUSH)}"

    for i in tqdm(experiment_range):

        # Definition of possible Energy levels
        ENERGY_MIN = 0
        ENERGY_MAX = i
        ENERGY_STEP_SIZE = 1
        energy_levels = np.arange(ENERGY_MIN, ENERGY_MAX + ENERGY_STEP_SIZE, ENERGY_STEP_SIZE)

        start_time_iter = time.process_time()

        v = search_optimal_policy(time_horizon,
                                  energy_levels,
                                  price_levels,
                                  pre_decision_states,
                                  post_decision_states,
                                  MAX_PULL,
                                  MAX_PUSH,
                                  PROB_MATRIX,
                                  EFF_COEFF,
                                  INITIAL_STATE)

        stop_time_iter = time.process_time()

        time_iter = stop_time_iter - start_time_iter

        profit.append(i*cost_per_unit+v)
        time_list.append(time_iter)

        fields = [i, v, profit, time_list]

        with open(f'results/{csv_file_name}', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)


    file_name = "RemoteMarnginalStorageSize"+ str(MAX_TIME) +"_timeSteps_" + str(max(experiment_range)) + "_eLevels_" + str(ENERGY_STEP_SIZE) + "_stepSize_" + str(MAX_PULL) + "_maxPull_" + str(MAX_PUSH) + "_maxPush"
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(experiment_range, profit)
    plt.xlabel("Size of Energy Storage")
    plt.ylabel("Profit")
    plt.title("Marginal Storage Size")
    plt.savefig('results/'+ file_name +'.png')

    plt.show()


if __name__ == "__main__":
    main()
