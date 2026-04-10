import argparse
from src.data_generator import DataGenerator
from config import DATA

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
        gen.f2(a=1)