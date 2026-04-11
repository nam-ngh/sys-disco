# synthetic data generation configs, ignore if you already have data to model
DATA = {
    "npts": 1000,
    "noise": 0.0,
    "x_min": 0.0,
    "x_max": 5.0,
}
GEN_F2 = {
    "a": 1.0,
    "n_trajectories": 5,
}

# modeling configs, should be tuned for specific datasets
F1 = {
    "max_polynomial": 2,
}

F2 = {
    "max_polynomial": 2,
}