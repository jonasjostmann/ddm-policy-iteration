import numpy as np
from policyiter import evaluation

'''
Policy Improvement
Method to improve current Policy
'''
# TODO: Call improve policy maybe recursive? To improve until no change in polcy
def improve_policy(policy, pre_decision_states, post_decision_states, prob_matrix):
    print("\nImprove current Policy...")

    # Todo: Auf copys achten: hier wurde z.B. auf das gleiche Objekt referenziert
    policy_temp = policy.copy()

    n_pre_states = len(pre_decision_states)

    for pre_dec in range(0, n_pre_states):

        pre_dec_state = pre_decision_states[pre_dec]

        # Improve policy only for pre decision states where a decision can be made
        if pre_dec_state["trans_mat"] is not None:

            # Initialize action_values array which holds the v-value
            # for each of the possible decisions in the pre_state
            action_values = [None] * len(pre_dec_state["trans_mat"].iloc[0, :])

            # Counter for the decision index (represented by the post_state at this location in trans_matrix)
            for a in range(0, len(pre_dec_state["trans_mat"].iloc[0, :])):

                post_state = pre_dec_state["trans_mat"].iloc[0, a][0]["post_state"]

                action_values[a] = evaluation.calc_contribution(pre_dec,
                                                                post_state,
                                                                pre_decision_states,
                                                                post_decision_states)

                for row in post_decision_states[post_state]["trans_mat"].index:
                    row = row[0]

                    col_name = post_decision_states[post_state]["trans_mat"].columns[0]

                    new_pre_state_id = post_decision_states[post_state]["trans_mat"].loc[row, col_name][0]['pre_state']
                    new_pre_state = pre_decision_states[new_pre_state_id]

                    post_price = post_decision_states[post_state]["state"][0]
                    pre_price = new_pre_state["state"][0]

                    action_values[a] += prob_matrix.loc[post_price, pre_price] * new_pre_state['v']


            best_a = np.argmax(action_values)

            policy_temp[pre_dec] = pre_decision_states[pre_dec]["trans_mat"].iloc[0, best_a][0]["post_state"]

    # New generated Policy
    policy = policy_temp.copy()

    print("Policy was successfully improved!\n")

    return policy
