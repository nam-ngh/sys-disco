import argparse
from src.data_generator import DataGenerator
from config import DATA, GEN_F2

# Script to generate synthetic data for either f1: u - x^2 = 0 or f2: u' - a*u = 0
# Run from command line: python gen_data.py f1 or python gen_data.py f2
# Data generated is saved in data/

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "F", choices=["f1", "f2"], 
        help="Which equation would you like to generate synthetic data for?"
    )
    args = parser.parse_args()

    gen = DataGenerator(**DATA)

    # call the desire generator function
    if args.F == "f1":
        gen.f1()
    elif args.F == "f2":
        gen.f2(a=GEN_F2["a"], n_trajectories=GEN_F2["n_trajectories"])