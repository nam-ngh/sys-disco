import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement

class Detector(ABC):
    @abstractmethod
    def ingest(self, data, standardize=True):
        '''Data ingestion'''
        pass

    @abstractmethod
    def poly_eval(self):
        pass
    
    @abstractmethod
    def build_linear_system(self, *args, **kwargs):
        '''Sets self._A'''
        pass

    @abstractmethod
    def integrate(self):
        '''Integrate data with learned parameters'''
        pass

    def solve_linear_system(self, kernel_thres=0.1, verbose=False):
        U, S, Vt = np.linalg.svd(self._A, full_matrices=False)
        null_kernel_idx = np.where((S > 1e-30) & (S < kernel_thres))[0]
        if len(null_kernel_idx) == 0:
            raise ValueError(
                'No non-trivial null space found, try checking data quality or raising kernel_thres (not recommended)'
            )
        else:
            for idx in (null_kernel_idx):
                params = Vt[idx]
                theta_x = params[params.shape[0]//2:]
                vx = self._poly @ theta_x
                if not np.allclose(vx, 0):
                    print(f'\nNon-trivial null space position: {idx}/{len(S)}')
                    print(f'\nParameters ({params.shape[0]}), norm {np.linalg.norm(params)}:')
                    print(params)
                    self._params = params
                    break
            else:
                params = Vt[idx]
                print('Warning: all recovered symmetries have zero vx, using last null space vector')
                print(f'\nParameters ({params.shape[0]}), norm {np.linalg.norm(params)}:')
                print(params)
                self._params = params

        if params.sum() == 1:
            print('Warning: trivial null space solution, check data and polynomial features.')
        else:
            residual = np.sum(self._A @ params)
            if residual < 1e-1:
                print(f'\nSYSTEM SOLVED: Orthogonality constraint residual = {residual}')
            else:
                print(f'Warning: high residual {residual} for null space solution, check data and polynomial features.')

        if verbose:
            print(f'\nSVD: U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}')
            print(f'\nSingular values:\n {S}')
            print(f'\nFirst rows of linear system matrix:')
            print(self._A[:2, :])

    def plot_integral_curves(self, data, eps=0.1, steps=10):
        assert data.shape[1] <= 3, f"Data must have at most 3 dimensions, got {data.shape[1]}"
        
        curves = [data]
        for _ in range(steps):
            curves.append(self.integrate(curves[-1], eps=eps))
        
        if data.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(10, 7))
            for i in range(len(curves) - 1):
                ax.quiver(
                    curves[i][:, 0], curves[i][:, 1],
                    curves[i+1][:, 0] - curves[i][:, 0],
                    curves[i+1][:, 1] - curves[i][:, 1],
                    angles='xy', scale_units='xy', scale=1,
                    color=plt.cm.cool(i / steps), alpha=0.7
                )
            ax.scatter(data[:, 0], data[:, 1], s=5, c='navy', zorder=3, label='original')
            ax.set_xlabel('x'); ax.set_ylabel('u')

        elif data.shape[1] == 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            for i in range(len(curves) - 1):
                ax.quiver(
                    curves[i][:, 0], curves[i][:, 1], curves[i][:, 2],
                    curves[i+1][:, 0] - curves[i][:, 0],
                    curves[i+1][:, 1] - curves[i][:, 1],
                    curves[i+1][:, 2] - curves[i][:, 2],
                    color=plt.cm.cool(i / steps), alpha=0.7
                )
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, c='navy', zorder=3, label='original')
            ax.set_xlabel('x'); ax.set_ylabel('u'); ax.set_zlabel("u'")

        ax.set_title('Symmetry vector field')
        ax.legend()
        plt.tight_layout()
        plt.show()

#############################################
class DetectorODE(Detector):
    def ingest(self, data, npts, standardize=True):
        '''
        Process trajectory data for ODEs, shaped (M, N).
        M is total number of points across all trajectories.
        N is number of dimensions (x, u, dxu, d2xu, ...).
        Different trajectories should be stacked vertically.
        Columns are ordered by increasing order of derivatives dnxu.
        '''
        n_trajectories = data.shape[0] // npts
        grads = np.concatenate([
            np.gradient(data[i*npts:(i+1)*npts, -1], data[i*npts:(i+1)*npts, 0])
            for i in range(n_trajectories)
        ])
        raw = np.column_stack([data, grads])
        self._raw = raw
        print(f'Raw data first rows:\n{raw[:2, :]}')

        if standardize:
            mean = np.zeros(raw.shape[1])
            std  = np.ones(raw.shape[1])
            mean[1:] = raw[:, 1:].mean(axis=0)
            std[1:]  = raw[:, 1:].std(axis=0) + 1e-10
            self._processed = (raw - mean) / std
            self._standard_mean = mean
            self._standard_std = std
            print(f'Standardized data first rows:\n{self._processed[:2, :]}')
        else:
            self._processed = raw
            self._standard_mean = None
            self._standard_std = None

    # basis functions evaluation
    def poly_eval(self, data, max_order=None, is_print=True):
        '''Assumes data shape (M, N)'''
        base = data
        
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
        if is_print:
            print('\nPolynomial combinations:')
        for o in range(1, max_order + 1):
            if is_print:
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
                if is_print:
                    print(idx)
        
        poly = np.column_stack(cols) # (M, P)
        J = np.stack(grads, axis=1) # (M, P, N)
        if is_print:
            print(f'\nPolynomial features shape: {poly.shape}')
            print(f'\nJacobian shape: {J.shape}')
        return poly, J
    
    def build_linear_system(self, max_polynomial: int):
        '''Builds linear system once with original data'''
        poly, J = self.poly_eval(self._processed[:, :-1], max_order=max_polynomial)
        cols_1 = []
        cols_2 = []
        P = poly.shape[1]
        N = J.shape[2]
        for p in range(P):
            component = J[:, p, 0]
            for n in range(2, N):
                component += J[:, p, n-1] * self._processed[:, n]
            
            if np.allclose(component, 0):
                print(f'Warning: All zero values for column corresponding to parameter {p}')
            
            cols_1.append(component.copy()) # P components corresponding to dn-1xu thetas
            component *= -self._processed[:, -2]
            component -= self._processed[:, -1] * poly[:, p]
            cols_2.append(component) # P components corresponding to x thetas
        
        # full linear system matrix (M, 2P)
        A = np.column_stack(cols_1 + cols_2) 
        self._A = A
        self._poly = poly
        self._poly_order = max_polynomial
        print(f'Linear system shaped {A.shape}')
    
    def integrate(self, data, eps: float=0.001):
        '''data must be of shaped (M, N)'''
        # standardise input the same way as training data
        data = np.column_stack([data, np.gradient(data[:, -1], data[:, 0])])
        if self._standard_mean is not None and self._standard_std is not None:
            data_std = (data - self._standard_mean) / self._standard_std
        else:
            data_std = data

        theta_x = self._params[self._params.shape[0]//2:]
        theta_dN_1xu = self._params[:self._params.shape[0]//2]
        poly, _ = self.poly_eval(data_std[:, :-1], max_order=self._poly_order, is_print=False)
        vx = poly @ theta_x
        vdN_1xu = poly @ theta_dN_1xu

        if np.allclose(vx, 0):
            vx = vdN_1xu / data_std[:, -2]
            print('Warning: zero vx values, using vdN_1xu/dNxu instead for integration.')
        
        vs = [vx,]
        for n in range(data.shape[1] - 2):
            vn = data_std[:, n+2] * vx
            vs.append(vn)
        vs = np.column_stack(vs)  # (M, N)
        
        # apply flow in standardised space, convert back to original
        new_data_std = data_std[:, :-1] + eps * vs
        new_data = new_data_std * self._standard_std[:-1] + self._standard_mean[:-1]
        return new_data
    
#############################################
class DetectorAlg(Detector):
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
            self._processed = (raw - mean) / std
            self._standard_mean = mean
            self._standard_std = std
            print(f'Standardized data first rows:\n{self._processed[:2, :]}')
        else:
            self._processed = raw
            self._standard_mean = None
            self._standard_std = None
    
    # polynomial basis functions
    def poly_eval(self, data, max_order=2, is_print=True):
        cols = []
        if is_print:
            print('\nPolynomial combinations:')
        for d in range(1, max_order + 1):
            if is_print:
                print(f'Order {d}')
            for idx in combinations_with_replacement(range(data.shape[1]), d):
                cols.append(np.prod(data[:, idx], axis=1))
                if is_print:
                    print(f'{idx}')
        return np.column_stack(cols) # (N, P)
    
    def build_linear_system(self, delF, max_polynomial=2):
        poly = self.poly_eval(data=self._processed, max_order=max_polynomial)
        A = (delF[:, :, None] * poly[:, None, :]).reshape(self._processed.shape[0], -1)
        print(f'Linear system shaped {A.shape}')
        self._A = A
        self._poly = poly
        self._poly_order = max_polynomial

    def integrate(self, data, eps: float=0.001):
        if self._standard_mean is not None and self._standard_std is not None:
            data_std = (data - self._standard_mean) / self._standard_std
        else:
            data_std = data

        vs = []
        polys = self.poly_eval(data_std, max_order=self._poly_order, is_print=False)
        for j in range(data_std.shape[1]):
            vj = polys @ self._params[j*polys.shape[1]: (j+1)*polys.shape[1]]
            vs.append(vj)

        vs = np.column_stack(vs)
        new_data_std = data_std + eps*vs
        new_data = new_data_std * self._standard_std + self._standard_mean
        return new_data