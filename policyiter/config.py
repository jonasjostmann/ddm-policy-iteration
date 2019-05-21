# DEFINITION OF PRICE LEVELS (DISCRETIZATION CAN BE MODIFIED WITH THE STEP SIZE)
PRICE_MIN = 1
PRICE_MAX = 2
PRICE_STEP_SIZE = 1

# DEFINITION OF ENERGY LEVELS (DISCRETIZATION CAN BE MODIFIED WITH THE STEP SIZE)
ENERGY_MIN = 0
ENERGY_MAX = 1
ENERGY_STEP_SIZE = 1

# TRANSITION PROBABILITIES BETWEEN THE DIFFERENT PRICES (FOR EACH PRICE A TRANSITION PROBABILITY MUST BE SPECIFIED)
TRANS_PROB = [[0.4, 0.6], [0.6, 0.4]]

# MAXIMUM TIME STEPS (MUST BE A NATURAL NUMBER GREATER THEN 0)
MAX_TIME = 2

# Initialize state
# Definition of states: [Price, Energy-Level]
INITIAL_STATE = [1, 0]

# Set a seed for the random policy creation
SEED = 42

# EFFICENCY COEFFICIENT MUST BE BETWEEN 0 AND 1
EFF_COEFF = 0.5

