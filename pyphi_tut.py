
import pyphi 
import numpy as np
import itertools

# pyphi.config.PROGRESS_BARS = False
# pyphi.config.PARALLEL = False
# pyphi.config.SHORTCIRCUIT_SIA = False
# pyphi.config.VALIDATE_SUBSYSTEM_STATES = False

unit_labels = ['A','B','C','D','E']
n_units = len(unit_labels)

unit_activation_fx = pyphi.network_generator.ising.probability
k = 4

tm = np.random.randint(0,100,size=(5,5))
weights = tm/tm.sum(axis=0)

print(weights)

substrate = pyphi.network_generator.build_network(
    [unit_activation_fx] * n_units,
    weights,temperature = 1/k,
    node_labels = unit_labels)

