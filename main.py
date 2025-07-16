import os
import sys

# Add the library path
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(working_dir, 'FedFB'))

from DP_run_private import sim_dp_man

if __name__ == '__main__':
    sim_dp_man(
        method='fedfb',
        model='multilayer perceptron',
        dataset='adult',
        ε=1,
        num_sim=1,
        seed=24,
        num_rounds=10,
        local_epochs=2,
        learning_rate=0.001,
        alpha=1
    )

