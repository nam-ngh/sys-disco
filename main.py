import argparse
import numpy as np
from src.detector import DetectorODE, DetectorAlg
from config import F1, F2

def get_data():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "F", choices=["f1", "f2", "custom"], 
        help="Which dataset from data/ would you like to model?"
    )
    args = parser.parse_args()

    # load the desired dataset
    if args.F == "f1":
        data = np.load('data/f1.npy')
    elif args.F == "f2":
        data = np.load('data/f2.npy')
    elif args.F == "custom":
        # load any custom dataset with name custom.npy in data/
        data = np.load('data/custom.npy')
    
    print(f"Data shape: {data.shape}")
    return data, args.F

def inspect(detector):
    print(f'\nFirst row of data:')
    print(detector._data[1:2])
    print(f'\nFirst row of polynomial features:')
    print(detector._poly[1:2])
    print(f'\nFirst row of Jacobian:')
    print(detector._J[1:2])

def main():
    # load data to model
    data, F = get_data()
    # detector instance
    if F == "f2":
        de = DetectorODE()
        de.ingest(data)
        de.solve_linear_system(max_polynomial=F2["max_polynomial"])
    elif F == "f1":
        de = DetectorAlg()
        de.ingest(data)
        delxF = 2 * de._data[:, 0]
        deluF = - np.ones((de._data.shape[0]))
        delF = np.column_stack([delxF, deluF])
        de.solve_linear_system(delF, max_polynomial=F1["max_polynomial"])

if __name__ == "__main__":
    main()