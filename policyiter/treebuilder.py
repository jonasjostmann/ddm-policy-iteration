import pandas as pd


'''
Create tree
Method to create the tree
'''
# TODO: in order to be able to differ in discretization step size change all references to which need normal numbers
# TODO: Add Probabilities to post decision states
def create_tree(time_horizon, energy_levels, price_levels, pre_decision_states, post_decision_states):
    last_post = []
    last_pre_states = [0]

    # Counter for Pre and Post Decision states
    n_pre_states = 0
    n_post_states = 0

    for t in time_horizon:

        last_pre_states_temp = []

        for pre_state in last_pre_states:
            pre_decision_states[pre_state]["trans_mat"] = pd.DataFrame(columns=[energy_levels],
                                                                       index=[
                                                                           pre_decision_states[pre_state]["state"][0]])
            pre_decision_states[pre_state]["trans_mat"] = pre_decision_states[pre_state]["trans_mat"].astype('object')

            # Insert None Values to build structure of DataFrame
            for e_l in range(0, len(energy_levels)):
                # TODO: Hier ist eine Abhängigkeit zum Wert des Energy States
                #  -> Umwandeln in Loc und heraussuchen des Price wertes vorher
                pre_decision_states[pre_state]["trans_mat"].iloc[0, e_l] = [dict(v=None, post_state=None)]

            for post_state in pre_decision_states[pre_state]["trans_mat"].columns:

                # Caused by Multiindex issue position 0 must be selected
                post_state = post_state[0]
                post_decision_states.append({"v": None,
                                             "state": [pre_decision_states[pre_state]["state"][0], post_state],
                                             "trans_mat": None})

                price_index = pre_decision_states[pre_state]["trans_mat"].index[0]

                # Save Post decision state in pre-decision state
                pre_decision_states[pre_state]["trans_mat"].loc[price_index, post_state][0][0]["post_state"] = n_post_states

                # Initialize Dataframe for post_decision_state
                post_decision_states[n_post_states]["trans_mat"] = pd.DataFrame(columns=[post_state],
                                                                                index=[price_levels])

                for row in post_decision_states[n_post_states]["trans_mat"].index:
                    # Caused by Multiindex issue position 0 must be selected
                    row = row[0]
                    post_price = row
                    # Energy Level equals col name
                    post_energy_level = post_decision_states[n_post_states]["trans_mat"].columns[0]

                    # TODO: Preisübergänge durchgehen mit Wahrscheinlichkeiten und Neue Predecision Zustände definieren
                    # row is equal to the price (because row index is set to the price)
                    pre_decision_states.append({"v": None, "state": [post_price, post_energy_level], "trans_mat": None})
                    # After appending an additional pre state the counter must be increased by one
                    n_pre_states = n_pre_states + 1

                    # Set pre state in post decision trans matrix
                    post_decision_states[n_post_states]["trans_mat"].loc[row, post_energy_level] = [dict(v=None,
                                                                                                         pre_state=n_pre_states)]

                    # Append new pre-decision state to list
                    last_pre_states_temp.append(n_pre_states)

                # Increase index of post_decision_states
                n_post_states = n_post_states + 1

        last_pre_states = last_pre_states_temp.copy()

    return pre_decision_states, post_decision_states