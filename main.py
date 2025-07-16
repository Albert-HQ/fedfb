import os
import sys

# Add the library path
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(working_dir, 'FedFB'))

from DP_run_private import sim_dp_man

def parse_args():
    parser = argparse.ArgumentParser(description="Run FedFB with client-level DP")
    parser.add_argument("--epsilon", type=float, default=1.0, help="DP epsilon")
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sim_dp_man(
        method="fedfb",
        model="multilayer perceptron",
        dataset="adult",
        ε=args.epsilon,
        num_sim=1,
        seed=24,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        alpha=args.alpha,

