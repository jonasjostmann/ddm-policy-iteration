import pandas as pd
import numpy as np

# TODO: Assumption that contribution of last states in time horizon is zero
'''
Calculate Contribution
Method to calculate the returned contribution by making a decision a.
'''
def calc_contribution(pre_state, post_state, pre_decision_states, post_decision_states):
    # Calculate contribution (Old Energy Level - New Energy Level) * Price per Energy
    contribution = (pre_decision_states[pre_state]["state"][1] - post_decision_states[post_state]["state"][1]) \
                   * pre_decision_states[pre_state]["state"][0]

    return contribution



'''
Policy Evaluation
Method to evaluate a given Policy
'''
def evaluate_policy(policy, pre_decision_states, post_decision_states, prob_matrix):
    n_pre_states = len(pre_decision_states)

    for pre_dec in range(0, n_pre_states):

        pre_dec_v_list = [pre_dec]

        # Initialize matrix for linear euitation system
        v = np.zeros((len(pre_decision_states),
                      len(pre_decision_states) + 1))

        for pre_dec_v in pre_dec_v_list[:]:

            # Check if List is not None in order to prevent errors
            if pre_dec_v_list is not None:
                pre_dec_v_list.remove(pre_dec_v)

            if pre_decision_states[pre_dec_v]["trans_mat"] is None:
                pre_decision_states[pre_dec_v]["v"] = 0
                break

            contribution = calc_contribution(pre_dec, policy[pre_dec], pre_decision_states, post_decision_states)

            v[pre_dec_v, pre_dec_v] = -1

            v[pre_dec_v, n_pre_states] = -1 * contribution

            for row in post_decision_states[policy[pre_dec]]["trans_mat"].index:

                # Due to the indexing issue of columns (see above)
                row = row[0]

                pre_v_cols = post_decision_states[policy[pre_dec]]["trans_mat"].columns[0]

                pre_v_state = post_decision_states[policy[pre_dec]]["trans_mat"].loc[row, pre_v_cols][0]['pre_state']
                pre_v = pre_decision_states[pre_v_state]['v']

                pre_v_state_price = pre_decision_states[pre_v_state]["state"][0]

                pre_dec_price = pre_decision_states[pre_dec]["state"][0]

                if (pre_v is not None):

                    v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                pre_v_state_price]

                    v[pre_v_state, n_pre_states] = pre_decision_states[pre_v_state]["v"]
                    v[pre_v_state, pre_v_state] = 1

                elif pre_decision_states[pre_v_state]["trans_mat"] is None:
                    v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                pre_v_state_price]
                    v[pre_v_state, n_pre_states] = 0
                    v[pre_v_state, pre_v_state] = 1

                else:
                    v[pre_dec_v, pre_v_state] = prob_matrix.loc[pre_dec_price,
                                                                pre_v_state_price]

                    # Check if List is None to prevent errors
                    if pre_dec_v_list is None:
                        pre_dec_v_list = []
                    pre_dec_v_list.append(pre_v_state)

        # TODO: Herausfiltern alle Zeilen, welche NUR ZERO Werte haben, um calc zu verschnellern
        # TODO: Teil mit lines entfernen
        lines = []

        for i in range(0, len(v[:, 0])):
            if (np.all(np.all(v[:, i] == 0, axis=0))):
                lines.append(i)

        a = v[:, range(0, n_pre_states)]

        b = v[:, n_pre_states]

        # lstq solver is used because normal solver canot handle singular matrices
        x = np.linalg.lstsq(a, b, rcond=-1)[0]

        pre_decision_states[pre_dec]["v"] = x[pre_dec]

    return pre_decision_states, post_decision_states
