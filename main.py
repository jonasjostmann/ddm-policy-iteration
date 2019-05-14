import pandas as pd
import numpy as np
import random
from scipy.sparse.linalg import lsqr

# Initialize empty policy
# Policy will be build as an array, which indices are the id of pre decision states and reference to the id of the next
# post-decision state to imply the decision which were made

# Definition of possible Price levels
price_min = 1
price_max = 2
price_step_size = 1
price_levels = np.arange(price_min, price_max+1, price_step_size)

# Definition of possible Energy levels
energy_min = 0
energy_max = 2
energy_step_size = 1
energy_levels = np.arange(energy_min, energy_max+1, energy_step_size)

# TODO: generate Probabilities automatically with seed to create reproducable results (Durchführbar mit ziehen ohne
# Todo: ohne zurücklegen aus einem Zustandsraum, den man vorher definiert hat
# Definition of the Probabilities from one Price Level to another
prob_matrix = pd.DataFrame([[0.3, 0.7], [0.7, 0.3]], columns=price_levels, index=price_levels)

max_time = 1
time_horizon = np.arange(1, max_time+1)

pre_decision_states = []
post_decision_states = []

# initialize start state
# Definition of states: [Price, Energy-Level]
pre_decision_states.append({"v": None, "state": [1,0], "trans_mat": None})

# initialize empty policy
policy = []

'''
Create tree
Method to create the tree
'''
# TODO: in order to be able to differ in discretization step size change all references to which need normal numbers
#TODO: Add Probabilities to post decision states
def create_tree():

    last_post = []
    last_pre_states = [0]

    # Counter for Pre and Post Decision states
    n_pre_states = 0
    n_post_states = 0

    for t in time_horizon:

        last_pre_states_temp = []

        for pre_state in last_pre_states:
            pre_decision_states[pre_state]["trans_mat"] = pd.DataFrame(columns = [energy_levels],
                                                                       index=[pre_decision_states[pre_state]["state"][0]])
            pre_decision_states[pre_state]["trans_mat"] = pre_decision_states[pre_state]["trans_mat"].astype('object')

            # Insert None Values to build structure of DataFrame
            for e_l in energy_levels:
                # TODO: Hier ist eine Abhängigkeit zum Wert des Energy States
                #  -> Umwandeln in Loc und heraussuchen des Price wertes vorher
                pre_decision_states[pre_state]["trans_mat"].iloc[0, e_l] = [dict(v= None, post_state= None)]

            for post_state in pre_decision_states[pre_state]["trans_mat"].columns:

                # Caused by Multiindex issue position 0 must be selected
                post_state = post_state[0]
                post_decision_states.append({"v": None,
                                             "state": [pre_decision_states[pre_state]["state"][0], post_state],
                                             "trans_mat": None})

                # Save Post decision state in pre-decision state
                pre_decision_states[pre_state]["trans_mat"].iloc[0, post_state][0]["post_state"] = n_post_states

                # Initialize Dataframe for post_decision_state
                post_decision_states[n_post_states]["trans_mat"] = pd.DataFrame(columns = [post_state],
                                                                                index=[price_levels])

                for row in post_decision_states[n_post_states]["trans_mat"].index:

                    # Caused by Multiindex issue position 0 must be selected
                    row = row[0]
                    post_price = row
                    # Energy Level equals col name
                    post_energy_level = post_decision_states[n_post_states]["trans_mat"].columns[0]

                    #TODO: Preisübergänge durchgehen mit Wahrscheinlichkeiten und Neue Predecision Zustände definieren
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
                n_post_states =  n_post_states +1

        last_pre_states = last_pre_states_temp


'''
Create random Policy
Method to create a random Policy
'''
def create_random_policy():

    # Number of pre decision states to iterate over
    n_pre_states = len(pre_decision_states)

    # initialize policy
    policy = [None]*n_pre_states

    counter=0

    for pre_state in range(0, n_pre_states):

        # Last States have an empty trans_mat because the time horizon is reached and no decision can be made
        if(pre_decision_states[pre_state]["trans_mat"] is None):
            counter=counter+1
        else:
            random_state = random.choice(pre_decision_states[pre_state]["trans_mat"].columns.levels[0])

            policy[pre_state] = pre_decision_states[pre_state]["trans_mat"].iloc[0, random_state][0]["post_state"]

    return policy

    print(policy)
    print(counter)


'''
Calculate Contribution
Method to calculate the returned contribution by making a decision a.
'''
def calc_contribution(pre_state, post_state):

    # Calculate contribution (Old Energy Level - New Energy Level) * Price per Energy
    contribution = (pre_decision_states[pre_state]["state"][1]-post_decision_states[post_state]["state"][1]) \
                   * pre_decision_states[pre_state]["state"][0]

    return contribution

# TODO: Assumption that contribution of last states in time horizon is zero
'''
Policy Evaluation
Method to evaluate a given Policy
'''
def evaluate_policy(policy):

    n_pre_states = len(pre_decision_states)

    for pre_dec in range(0, n_pre_states):

        pre_dec_v_list = [pre_dec]

        # Initialize matrix for linear euitation system
        v = np.zeros((len(pre_decision_states),
                           len(pre_decision_states)+1))

        for pre_dec_v in pre_dec_v_list:

            if pre_decision_states[pre_dec_v]["trans_mat"] is None:
                pre_decision_states[pre_dec_v]["v"] = 0
                break

            contribution = calc_contribution(pre_dec, policy[pre_dec])

            v[pre_dec_v, pre_dec_v] = -1

            v[pre_dec_v, n_pre_states] = -1 * contribution

            print(contribution)
            print(policy)

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
                    pre_dec_v_list.add(pre_v_state)

        # TODO: Herausfiltern alle Zeilen, welche NUR ZERO Werte haben, um calc zu verschnellern

        lines=[]

        for i in range(0, len(v[:, 0])):
            if(np.all(np.all(v[:,i] == 0, axis = 0))):
                lines.append(i)



        a = v[:, range(0, n_pre_states)]

        b = v[:, n_pre_states]


        x = np.linalg.lstsq(a, b, rcond=-1)[0]

        print("Ergebnis: ") # TODO: Problem 7x 8 Matrix muss aber 7x7 matrix sein
        print(x)

        pre_decision_states[pre_dec]["v"] = x[pre_dec]


#%% Create the tree
create_tree()

#%% Create a random policy
policy = create_random_policy()

#%% Evaluate_policy

evaluate_policy(policy)


