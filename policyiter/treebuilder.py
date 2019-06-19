import pandas as pd
import numpy as np
import copy


def create_tree(time_horizon, energy_levels, price_levels, pre_decision_states_copy, post_decision_states_copy, max_pull, max_push,
                verbosity=1):
    """
    Create tree - Method to create the decision tree
    :param time_horizon: time horizon for which the tree has to be be created
    :param energy_levels: Energy levels which are allowed
    :param price_levels: Price levels which are allowed
    :param pre_decision_states_copy: current pre-decision states
    :param post_decision_states_copy: current post-decision states
    :param max_pull: maximal pullable energy at a point in time
    :param max_push: maximal pushable energy at a point in time
    :param verbosity: verbosity
    :return: decision tree
    """
    if(verbosity>=1):
        print("\nBuilding the tree...")

    last_post = []
    last_pre_states = [0]

    # Counter for Pre and Post Decision states
    n_pre_states = 0
    n_post_states = 0

    for t in time_horizon:

        last_pre_states_temp = []

        for pre_state in last_pre_states:

            # Reduce Possible Energy Levels could be reached by decisions, w.r.t constraints MAX_PULL, MAX_PUSH
            def check_state_transition(x):
                diff = pre_decision_states_copy[pre_state]["state"][1] - x

                if((diff > 0 and diff <= max_pull) or (diff < 0 and abs(diff)<= max_push) or diff==0):
                    return x
                return None

            possible_energy_levels = np.array(list(map(check_state_transition, energy_levels)))
            possible_energy_levels = possible_energy_levels[possible_energy_levels!=np.array(None)]

            pre_decision_states_copy[pre_state]["trans_mat"] = pd.DataFrame(columns=[possible_energy_levels],
                                                                            index=[
                                                                           pre_decision_states_copy[pre_state]["state"][0]])
            pre_decision_states_copy[pre_state]["trans_mat"] = pre_decision_states_copy[pre_state]["trans_mat"].astype('object')

            # Insert None Values to build structure of DataFrame
            # Iterate over all energy levels represented by their index in this for loop
            for e_l in range(0, len(possible_energy_levels)):
                pre_decision_states_copy[pre_state]["trans_mat"].iloc[0, e_l] = [dict(v=None, post_state=None)]

            for post_state in pre_decision_states_copy[pre_state]["trans_mat"].columns:

                # Caused by Multiindex issue position 0 must be selected
                post_state = post_state[0]
                post_decision_states_copy.append({"v": None,
                                             "state": [pre_decision_states_copy[pre_state]["state"][0], post_state],
                                             "trans_mat": None})

                # Save Post decision state in pre-decision state
                # To be able to use loc price value of row must be read
                price_index = pre_decision_states_copy[pre_state]["trans_mat"].index[0]
                pre_decision_states_copy[pre_state]["trans_mat"].loc[price_index][post_state].iloc[0][0]["post_state"] = n_post_states

                # Initialize Dataframe for post_decision_state
                post_decision_states_copy[n_post_states]["trans_mat"] = pd.DataFrame(columns=[post_state],
                                                                                     index=[price_levels])

                for row in post_decision_states_copy[n_post_states]["trans_mat"].index:
                    # Caused by Multiindex issue position 0 must be selected
                    row = row[0]
                    post_price = row
                    # Energy Level equals col name
                    post_energy_level = post_decision_states_copy[n_post_states]["trans_mat"].columns[0]

                    # row is equal to the price (because row index is set to the price)
                    pre_decision_states_copy.append({"v": None, "state": [post_price, post_energy_level], "trans_mat": None})
                    # After appending an additional pre state the counter must be increased by one
                    n_pre_states = n_pre_states + 1

                    # Set pre state in post decision trans matrix
                    post_decision_states_copy[n_post_states]["trans_mat"].loc[row, post_energy_level] = [dict(v=None,
                                                                                                              pre_state=n_pre_states)]

                    # Append new pre-decision state to list
                    last_pre_states_temp.append(n_pre_states)

                # Increase index of post_decision_states
                n_post_states = n_post_states + 1

        last_pre_states = last_pre_states_temp.copy()

    if(verbosity>=1):
        print("Tree was sucessfully built!\n")

    return dict(pre=copy.deepcopy(pre_decision_states_copy), post=copy.deepcopy(post_decision_states_copy))