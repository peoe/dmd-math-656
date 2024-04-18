import numpy as np

from logging import getLogger, INFO
from scipy.linalg import svd


class DMD:
    def __init__(self, max_rank=None):
        self.logger = getLogger("DMD")
        self.logger.setLevel(INFO)
        
        self.max_rank = max_rank
        self._X = []
        self._Y = []
        self._len = 0

    def update(self, x, y):
        self._X.append(x)
        self._Y.append(y)
        self._len += 1

    def _tsvd(self):
        _U, _S, _Vh = svd(self.X, full_matrices=False)
        if self.max_rank is not None:
            _U = _U[:, :self.max_rank]
            _S = _S[:self.max_rank]
            _Vh = _Vh[:self.max_rank, :]

        return _U, (1. / _S) * _Vh.conj().T

    @property
    def X(self):
        return np.array(self._X).T

    @property
    def Y(self):
        return np.array(self._Y).T
    
    @property
    def A(self):
        _U, V_S_inv = self._tsvd()
        return (_U.conj().T @ self.Y) @ V_S_inv
    
    @property
    def modes(self):
        _U, V_S_inv = self._tsvd()
        _A = (_U.conj().T @ self.Y) @ V_S_inv
        evals, evecs = np.linalg.eig(_A)
        return _U @ evecs, evals
    
    @property
    def residuals(self):
        _U, V_S_inv = self._tsvd()
        _A = (_U.conj().T @ self.Y) @ V_S_inv
        evals, evecs = np.linalg.eig(_A)
        residuals = np.linalg.norm(self.Y @ (V_S_inv @ evecs) - (_U @ evecs) * evals, axis=0)
        return residuals, _U @ evecs, evals
    
    def __call__(self, x):
        _U, V_S_inv = self._tsvd()
        _A = self.Y @ V_S_inv

        return _A @ (_U.conj().T @ x)
    
    def __len__(self):
        return self._len
    

class RDMD(DMD):
    def __init__(self, random_rank, oversampling, num_reorth=3):
        super().__init__(max_rank=random_rank)
        self.logger = getLogger("RDMD")
        self.logger.setLevel(INFO)

        self.total_rank = random_rank + oversampling
        self.random_rank = random_rank
        self.oversampling = oversampling
        self.num_reorth = num_reorth
        self._Q = None
        self._Omega = None
        self.compressed = False

    def update(self, x, y):
        super().update(x, y)
        if self._Omega is None:
            self._Omega = np.zeros((x.shape[0], self.total_rank))
        self._Omega += np.outer(x, np.random.randn(self.total_rank))

    def _compress(self):
        self._Omega += np.outer(self._Y[-1], np.random.randn(self.total_rank))
        self._Q = np.zeros(self._Omega.shape)
        v = self._Omega[:, 0]
        nv = np.linalg.norm(v)
        self._Q[:, 0] = v / nv
        for i in range(1, self._Omega.shape[1]):
            v = self._Omega[:, i]
            v_o = np.zeros((i, 1))
            for _ in range(self.num_reorth):
                dv = (self._Q[:, :i].conj().T @ v).reshape(-1, 1)
                v_o += dv
                v -= np.squeeze(self._Q[:, :i] @ dv)
            nv = np.linalg.norm(v)
            self._Q[:, i] = v / nv
        _Qh = self._Q.conj().T
        self._X = [ _Qh @ x for x in self._X ]
        self._Y = [ _Qh @ y for y in self._Y ]
        self.compressed = True

    def _tsvd(self):
        if not self.compressed:
            self._compress()
        return super()._tsvd()
    
    @property
    def modes(self):
        if not self.compressed:
            self._compress()
        modes, evals = super().modes
        return self._Q @ modes, evals
    
    @property
    def residuals(self):
        if not self.compressed:
            self._compress()

        _U, V_S_inv = self._tsvd()
        _A = (_U.conj().T @ self.Y) @ V_S_inv

        evals, evecs = np.linalg.eig(_A)
        residuals = np.linalg.norm(self.Y @ (V_S_inv @ evecs) - (_U @ evecs) * evals, axis=0) # no need to project back, because self._Q has orthonormal columns
        return residuals, self._Q @ (_U @ evecs), evals
    
    def __call__(self, x):
        _U, V_S_inv = self._tsvd()
        _A = self.Y @ V_S_inv

        return self._Q @ (_A @ _U.conj().T @ (self._Q.conj().T @ x))
    