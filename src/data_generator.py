import os
import numpy as np

class DataGenerator:
    def __init__(self, npts=10000, noise=0.01, x_min=0.0, x_max=100.0, seed=42):
        self.npts = npts
        self.noise = noise
        self.x_min = x_min
        self.x_max = x_max
        self._points = None
        np.random.seed(seed)


    def _save(self, name):
        '''Save generated points to data/'''
        assert self._points is not None, "No points generated yet, run a generator function first"
        save_path = f'data/{name}.npy'
        np.save(save_path, self._points)
        print(f'Points saved to {save_path}')

    def f1(self):
        '''Generate synthetic points on manifold F(x, u) = x^2 - u = 0'''
        # init points array
        self._points = np.zeros((self.npts, 2)) # (x, u)

        # random x values
        self._points[:, 0] = np.random.uniform(
            self.x_min, self.x_max, self.npts
        )

        # u values satisfying F = 0 with a bit of noise
        x2 = self._points[:, 0] ** 2
        noises = np.random.uniform(
            -self.noise, self.noise, self.npts
        ) * np.abs(x2)

        self._points[:, 1] = x2 + noises
        print(f'Points generated, shape {self._points.shape}. First point: {self._points[0]}')
        self._save(name='f1')

    def f2(self, a=1):
        '''Generate synthetic points on manifold F(x, u, u') = u' - au = 0'''
        self._points = np.zeros((self.npts, 3)) # (x, u, u')

        self._points[:, 0] = np.linspace(self.x_min, self.x_max, self.npts)

        u = np.exp(a * self._points[:, 0])
        dxu = a * u  # dxu = a*e^(ax)

        noise_u = np.random.uniform(-self.noise, self.noise, self.npts) * np.abs(u)
        noise_dxu = np.random.uniform(-self.noise, self.noise, self.npts) * np.abs(dxu)

        self._points[:, 1] = u + noise_u
        self._points[:, 2] = dxu + noise_dxu

        print(f'Points generated, shape {self._points.shape}. First point: {self._points[0]}')
        self._save(name='f2')