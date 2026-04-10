# synthetic data generation configs, ignore if you already have data to model
DATA = {
    "npts": 10000,
    "noise": 0.01,
    "x_min": 0.0,
    "x_max": 10.0,
}

# modeling configs, should be tuned for specific datasets
F1 = {
    "max_polynomial": 2,
}

F2 = {
    "max_polynomial": 2,
}