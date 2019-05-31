import pandas as pd
import numpy as np

# TODO: Assumption that contribution of last states in time horizon is zero
'''
Calculate Contribution
Method to calculate the returned contribution by making a decision a.
'''
def calc_contribution(pre_state, post_state, pre_decision_states, post_decision_states, eff_coeff):
    # Calculate contribution (Old Energy Level - New Energy Level) * Price per Energy

    contribution = 0

    # If pre_state is smaller then post_state energy level then energy is bought
    if(pre_decision_states[pre_state]["state"][1]<post_decision_states[post_state]["state"][1]):

        contribution = (pre_decision_states[pre_state]["state"][1] - post_decision_states[post_state]["state"][1]) \
                       * pre_decision_states[pre_state]["state"][0] * (1/eff_coeff)

    # pre_state is greater then post_state energy level then energy is sold
    elif (pre_decision_states[pre_state]["state"][1]>post_decision_states[post_state]["state"][1]):

        contribution = (pre_decision_states[pre_state]["state"][1] - post_decision_states[post_state]["state"][1]) \
                       * pre_decision_states[pre_state]["state"][0] * eff_coeff


    return contribution



'''
Policy Evaluation
Method to evaluate a given Policy
'''
def evaluate_policy(policy, pre_decision_states, post_decision_states, prob_matrix, eff_coeff):

    print("\n Policy is evaluated...")

    n_pre_states = len(pre_decision_states)

    for pre_dec in range(0, n_pre_states):

        # Initialize List with pre_decision states which are considered to calculate V-Value for Predecision state of
        # current iteration of outer loop
        pre_dec_v_list = [pre_dec]

        # Initialize matrix for linear equation system
        v = np.zeros((len(pre_decision_states),
                      len(pre_decision_states) + 1))

        if pre_decision_states[pre_dec]["trans_mat"] is None:
            pre_decision_states[pre_dec]["v"] = 0
        else:
            # Iterate over all pre_decision states in the list which are considered to calculate V-Value for Predecision
            # state of current iteration of outer loop
            for pre_dec_v in pre_dec_v_list:
                # Check if List is not None in order to prevent errors
                #if pre_dec_v_list is not None:
                #    pre_dec_v_list.remove(pre_dec_v)

                # If trans_mat is none No decision can be made, thus V-Value is 0
                if pre_decision_states[pre_dec_v]["trans_mat"] is None:
                    contribution = 0
                else:
                    contribution = calc_contribution(pre_dec_v, policy[pre_dec_v], pre_decision_states,
                                                     post_decision_states, eff_coeff)

                # Set to -1 in order to calculate the corresponding value in linear equation system
                # TRANSFORMATION STEPS IN ORDER TO CALC: Vi = contri(i) + wj * Vj + ... + wn * Vn
                v[pre_dec_v, pre_dec_v] = -1
                v[pre_dec_v, n_pre_states] = -1 * contribution

                # Iterate over all possible random states and calculate expected contribution EC = p1*v1 + ... + pn*vn
                for row in post_decision_states[policy[pre_dec]]["trans_mat"].index:

                    # Due to the indexing issue of columns
                    row = row[0]

                    # Get necessary indices and values for further calculation in order to save space
                    pre_v_cols = post_decision_states[policy[pre_dec_v]]["trans_mat"].columns[0]
                    # New pre_state which is reached with the probability prob_matrix[pre_dec_price][pre_v_price]
                    pre_v_state = post_decision_states[policy[pre_dec_v]]["trans_mat"].loc[row, pre_v_cols][0]['pre_state']
                    # Expected V-Value of this pre_decision state (Contribution of decision (Policy) + Expected Contributions)
                    pre_v = pre_decision_states[pre_v_state]['v']
                    pre_v_state_price = pre_decision_states[pre_v_state]["state"][0]
                    pre_dec_price = pre_decision_states[pre_dec_v]["state"][0]

                    # If pre_v is not None the v value was determined in a previous iteration and can be used in this calculation
                    if (pre_v is not None):

                        v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                    pre_v_state_price]

                        # Index n_pre_states leads to last column of matrix (represents the result of this line of the
                        # equation
                        v[pre_v_state, n_pre_states] = pre_decision_states[pre_v_state]["v"]
                        v[pre_v_state, pre_v_state] = 1

                    # In last pre_decision states the trans_mat is None because no decision can be made v-value is zero
                    elif pre_decision_states[pre_v_state]["trans_mat"] is None:
                        v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                    pre_v_state_price]
                        v[pre_v_state, n_pre_states] = 0
                        v[pre_v_state, pre_v_state] = 1

                    # If no v-value is stored the v-values must be calculated
                    else:
                        v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                    pre_v_state_price]

                        # Check if List is None to prevent errors
                        #if pre_dec_v_list is None:
                        #    pre_dec_v_list = []
                        # Append pre_decision_states to the list for which the v-value must be calculated
                        if(pre_v_state not in pre_dec_v_list):
                            pre_dec_v_list.append(pre_v_state)

            # TODO: Herausfiltern alle Zeilen, welche NUR ZERO Werte haben, um calc zu verschnellern
            # TODO: Teil mit lines entfernen
            lines = []

            for i in range(0, len(v[:, 0])):
                if (not np.any(v[i, :])):
                    v[i,i] = 1

            # SOLVE LINEAR EQUATION SYSTEM
            a = v[:, range(0, n_pre_states)]
            b = v[:, n_pre_states]

            # lstq solver is used, because normal solver cannot handle singular matrices
            x = np.linalg.solve(a, b)

            pre_decision_states[pre_dec]["v"] = x[pre_dec]

            # TODO: Calculate expected V-Value of post decision states for trans Matrix

    print("Evaluation completed successfully!")

    return pre_decision_states, post_decision_states
