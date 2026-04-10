import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement

class DetectorODE:
    def ingest(self, data, standardize=True):
        '''
        Calculates d^N+1_x(u) and store data instance. 
        Assumes data is shaped (M, N) and columns are ordered by 
        increasing order of derivatives d^n_x(u).
        '''
        raw = np.column_stack([data, np.gradient(data[:, -1], data[:, 0])])
        print(f'Raw data first rows:\n{raw[:2, :]}')

        if standardize:
            mean = np.zeros(raw.shape[1])
            std  = np.ones(raw.shape[1])
            mean[1:] = raw[:, 1:].mean(axis=0)
            std[1:]  = raw[:, 1:].std(axis=0) + 1e-10
            self._data = (raw - mean) / std
            self._standard_mean = mean
            self._standard_std = std
            print(f'Standardized data first rows:\n{self._data[:2, :]}')
        else:
            self._data = raw
            self._standard_mean = None
            self._standard_std = None

    # basis functions evaluation
    def _poly_eval(self, exclude_highest_der=True, max_order=None):
        if exclude_highest_der:
            base = self._data[:, :-1]
        else:
            base = self._data
        
        # define data's original number of dimensions
        nsams = base.shape[0] # M
        ndims = base.shape[1] # N
        
        if max_order is None:
            max_order = ndims

        # zero order term - is not necessary
        # cols = [np.ones(nsams)]
        # grads = [np.zeros((nsams, ndims))]
        cols = []
        grads = []

        # loop through orders and possible combinations of dimensions in each
        print('\nPolynomial combinations:')
        for o in range(1, max_order + 1):
            print(f'Order {o}')
            for idx in combinations_with_replacement(range(ndims), o):
                # store product of selected dimensions
                cols.append(np.prod(base[:, idx], axis=1))

                # calculate corresponding partials
                grad = np.zeros((nsams, ndims))
                for n in set(idx):
                    count = idx.count(n)
                    remaining = list(idx)
                    remaining.remove(n)
                    grad[:, n] = count * (np.prod(base[:, remaining], axis=1) if remaining else np.ones(nsams))
                grads.append(grad)
                print(idx)
        
        self._poly = np.column_stack(cols) # (M, P)
        self._J = np.stack(grads, axis=1) # (M, P, N)
        print(f'\nPolynomial features shape: {self._poly.shape}')
    
    def solve_linear_system(self, max_polynomial: int, verbose=False):
        self._poly_eval(max_order=max_polynomial)
        cols_1 = []
        cols_2 = []
        P = self._poly.shape[1]
        N = self._J.shape[2]
        for p in range(P):
            component = self._J[:, p, 0]
            for n in range(2, N):
                component += self._J[:, p, n-1] * self._data[:, n]
            
            if np.allclose(component, 0):
                print(f'Warning: All zero values for column corresponding to parameter {p}')
            
            cols_1.append(component.copy())
            component *= -self._data[:, N-1]
            component -= self._data[:, -1] * self._poly[:, p]
            cols_2.append(component)
        
        # full linear system matrix (M, 2P)
        A = np.column_stack(cols_1 + cols_2) 
        
        # solve for null space of A
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        i_null = np.argmin(np.where(S > 1e-100, S, np.inf))
        params = Vt[i_null]
        print(f'\nParameters ({params.shape[0]}):')
        print(params)
        if params.sum() == 1:
            print('Warning: trivial null space solution, check data and polynomial features.')
        else:
            residual = np.sum(A @ params)
            if residual < 1e-1:
                print(f'\nSYSTEM SOLVED: Orthogonality constraint residual = {residual}')
            else:
                print(f'Warning: high residual {residual} for null space solution, check data and polynomial features.')

        if verbose:
            print(f'\nSVD: U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}')
            print(f'\nSingular values: {S}')
            print(f'\nNon-trivial null space position: {i_null}')
            print(f'\nFirst rows of linear system matrix:')
            print(A[:2, :])

        self._thetas = params
        self._A = A


class DetectorAlg:
    def ingest(self, data, standardize=True):
        '''
        Ingest data and normalise if needed.
        '''
        raw = data.copy()
        print(f'Raw data first rows:\n{raw[:2, :]}')
        if standardize:
            mean = np.zeros(raw.shape[1])
            std  = np.ones(raw.shape[1])
            mean = raw.mean(axis=0)
            std  = raw.std(axis=0) + 1e-10
            self._data = (raw - mean) / std
            self._standard_mean = mean
            self._standard_std = std
            print(f'Standardized data first rows:\n{self._data[:2, :]}')
        else:
            self._data = raw
            self._standard_mean = None
            self._standard_std = None
    
    # polynomial basis functions
    def _polyeval(self, max_order=2):
        cols = []
        print('\nPolynomial combinations:')
        for d in range(1, max_order + 1):
            print(f'Order {d}')
            for idx in combinations_with_replacement(range(self._data.shape[1]), d):
                cols.append(np.prod(self._data[:, idx], axis=1))
                print(f'{idx}')
        return np.column_stack(cols) # (N, P)

    def solve_linear_system(self, delF, max_polynomial=2, verbose=False):
        A = (delF[:, :, None] * self._polyeval(max_order=max_polynomial)[:, None, :]).reshape(self._data.shape[0], -1)
        print(f'Linear system shaped {A.shape}')
        # solve for null space of A
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        i_null = np.argmin(np.where(S > 1e-100, S, np.inf))
        params = Vt[i_null]
        print(f'\nParameters ({params.shape[0]}):')
        print(params)
        if params.sum() == 1:
            print('Warning: trivial null space solution, check data and polynomial features.')
        else:
            residual = np.sum(A @ params)
            if residual < 1e-1:
                print(f'\nSYSTEM SOLVED: Orthogonality constraint residual = {residual}')
            else:
                print(f'Warning: high residual {residual} for null space solution, check data and polynomial features.')

        if verbose:
            print(f'\nSVD: U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}')
            print(f'\nSingular values: {S}')
            print(f'\nNon-trivial null space position: {i_null}')
            print(f'\nFirst rows of linear system matrix:')
            print(A[:2, :])

        self._thetas = params
        self._A = A