import pandas as pd
import numpy as np
from tqdm import tqdm
import copy


def calc_contribution(pre_state, post_state, pre_decision_states_copy, post_decision_states_copy, eff_coeff):
    """
    Calculate Contribution - Method to calculate the returned contribution by making a decision a.
    :param pre_state: index of considered pre-decision state
    :param post_state: index of the post state reached stated by the policy
    :param pre_decision_states_copy: a copy of the current pre-decision states
    :param post_decision_states_copy: a copy of the current post-decision states
    :param eff_coeff: efficiency coefficient
    :return: skalar value of direct contribution
    """

    contribution = 0

    # If pre_state is smaller then post_state energy level then energy is bought
    if(pre_decision_states_copy[pre_state]["state"][1]<post_decision_states_copy[post_state]["state"][1]):

        contribution = (pre_decision_states_copy[pre_state]["state"][1] - post_decision_states_copy[post_state]["state"][1]) \
                       * pre_decision_states_copy[pre_state]["state"][0] * (1 / eff_coeff)

    # pre_state is greater then post_state energy level then energy is sold
    elif (pre_decision_states_copy[pre_state]["state"][1] > post_decision_states_copy[post_state]["state"][1]):

        contribution = (pre_decision_states_copy[pre_state]["state"][1] - post_decision_states_copy[post_state]["state"][1]) \
                       * pre_decision_states_copy[pre_state]["state"][0] * eff_coeff


    return contribution


def evaluate_policy(policy, pre_decision_states_copy, post_decision_states_copy, prob_matrix, eff_coeff):
    """
    Policy Evaluation - Method to evaluate a given Policy
    :param policy: current policy which has to be evaluated
    :param pre_decision_states_copy: a copy of the current pre-decision states
    :param post_decision_states_copy: a copy of the current post-decision states
    :param prob_matrix: transition probabilities
    :param eff_coeff: efficiency coefficient
    :return: returns pre- and post-decision states with updated v-values
    """
    print("\n Policy is evaluated...")
    n_pre_states = len(pre_decision_states_copy)

    for pre_dec in tqdm(range(0, n_pre_states)):

        # Initialize List with pre_decision states which are considered to calculate V-Value for Predecision state of
        # current iteration of outer loop
        pre_dec_v_list = [pre_dec]

        # Initialize matrix for linear equation system
        v = np.zeros((len(pre_decision_states_copy),
                      len(pre_decision_states_copy) + 1))

        if pre_decision_states_copy[pre_dec]["trans_mat"] is None:
            pre_decision_states_copy[pre_dec]["v"] = 0
        else:
            # Iterate over all pre_decision states in the list which are considered to calculate V-Value for Predecision
            # state of current iteration of outer loop
            for pre_dec_v in pre_dec_v_list:

                # If trans_mat is none No decision can be made, thus V-Value is 0
                if pre_decision_states_copy[pre_dec_v]["trans_mat"] is None:
                    contribution = 0
                else:
                    contribution = copy.deepcopy(calc_contribution(pre_dec_v, policy[pre_dec_v], pre_decision_states_copy,
                                                     post_decision_states_copy, eff_coeff))

                # Set to -1 in order to calculate the corresponding value in linear equation system
                # TRANSFORMATION STEPS IN ORDER TO CALC: Vi = contri(i) + wj * Vj + ... + wn * Vn
                v[pre_dec_v, pre_dec_v] = -1
                v[pre_dec_v, n_pre_states] = -1 * contribution

                # Iterate over all possible random states and calculate expected contribution EC = p1*v1 + ... + pn*vn
                for row in post_decision_states_copy[policy[pre_dec]]["trans_mat"].index:

                    # Due to the indexing issue of columns
                    row = row[0]

                    # Get necessary indices and values for further calculation in order to save space
                    pre_v_cols = post_decision_states_copy[policy[pre_dec_v]]["trans_mat"].columns[0]
                    # New pre_state which is reached with the probability prob_matrix[pre_dec_price][pre_v_price]
                    pre_v_state = post_decision_states_copy[policy[pre_dec_v]]["trans_mat"].loc[row, pre_v_cols][0]['pre_state']
                    # Expected V-Value of this pre_decision state (Contribution of decision (Policy) + Expected Contributions)
                    pre_v = pre_decision_states_copy[pre_v_state]['v']
                    pre_v_state_price = pre_decision_states_copy[pre_v_state]["state"][0]
                    pre_dec_price = pre_decision_states_copy[pre_dec_v]["state"][0]

                    if pre_decision_states_copy[pre_v_state]["trans_mat"] is None:
                        v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                    pre_v_state_price]
                        v[pre_v_state, n_pre_states] = 0
                        v[pre_v_state, pre_v_state] = 1

                    # If no v-value is stored the v-values must be calculated
                    else:
                        v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                    pre_v_state_price]

                        # Append pre_decision_states to the list for which the v-value must be calculated
                        if(pre_v_state not in pre_dec_v_list):
                            pre_dec_v_list.append(pre_v_state)

            # Avoid singular matrix by set diagonal to one where no values are provided
            for i in range(0, len(v[:, 0])):
                if (not np.any(v[i, :])):
                    v[i,i] = 1

            # SOLVE LINEAR EQUATION SYSTEM
            a = v[:, range(0, n_pre_states)]
            b = v[:, n_pre_states]

            # Solver for linear equations of the numpy package is used
            x = np.linalg.solve(a, b)

            pre_decision_states_copy[pre_dec]["v"] = copy.deepcopy(x[pre_dec])

    print("Evaluation completed successfully!")

    return dict(pre=copy.deepcopy(pre_decision_states_copy), post=copy.deepcopy(post_decision_states_copy))
