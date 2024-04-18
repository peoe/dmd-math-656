import numpy as np

from .dmd import DMD
from logging import getLogger, INFO
from scipy.linalg import qr_delete, svd, solve
    

class StreamingDMD(DMD): #MARK: Streaming
    def __init__(self, max_hist=None, max_rank=None, adaptive=None, reset_ratio=None, reset_tol=None, num_reorth=3):
        super().__init__(max_rank=max_rank)
        self.logger = getLogger("StreamingDMD")
        self.logger.setLevel(INFO)

        self.adaptive = False
        if adaptive is not None:
            assert reset_ratio is not None and reset_tol is not None
            assert reset_ratio > 1
            self.adaptive = True
            self.reset_ratio = reset_ratio
            self.reset_tol = reset_tol
            self._last_prediction_error = np.inf
        self.max_hist = max_hist
        self.num_reorth = num_reorth
        self._Q = None
        self._R = None

    def update(self, x, y):
        super().update(x, y)

        if self.max_hist is not None:
            if self._len > self.max_hist:
                self._len -= 1
                self._X = self._X[1:]
                self._Y = self._Y[1:]

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        if len(self) == 1:
            self._Q, self._R = np.linalg.qr(np.hstack([self.X, y]))
        else:
            # If history is adaptive truncate to given length
            if self.adaptive:
                ny = np.linalg.norm(y)
                pred_error = np.linalg.norm(y - self.__call__(x)) / ny

                if pred_error > self.reset_tol * self._last_prediction_error:
                    self.logger.info(f"RESET    |    Memory has {self._R.shape[1]} snapshots. Prediction error {pred_error:6.3e} is larger than {self.reset_tol * self._last_prediction_error:6.3e} ({self.reset_tol:6.3e} * {self._last_prediction_error:6.3e})!")
                    self._Q, self._R = qr_delete(self._Q, self._R, 0, p=int(self._R.shape[1] / self.reset_ratio), which="col")

                self._last_prediction_error = np.linalg.norm(y - self.__call__(x)) / ny

            # Update QR decomposition by appending y
            # Reorthonormalize
            y_o = np.zeros((self._Q.shape[1], 1))
            y_diff = y.copy()
            for _ in range(self.num_reorth):
                py = np.einsum("ji,jk->ik", self._Q.conj(), y_diff)
                y_o += py
                y_diff -= self._Q @ py
            nqy = np.linalg.norm(y_diff)
            qy = y_diff / nqy
            self._Q = np.hstack([self._Q, qy])
            self._R = np.hstack([
                np.vstack([self._R, np.zeros((1, self._R.shape[1]))]),
                np.vstack([y_o, nqy])
            ])

        # Truncate history to a fixed length
        if self.max_hist is not None:
            if self._R.shape[1] > self.max_hist:
                self._Q, self._R = qr_delete(self._Q, self._R, 0, which="col")

    def reconstruct(self, modes, amps, times=[]):
        Qmodes, Rmodes = np.linalg.qr(modes)
        if not times:
            times = range(len(self))
        G = Qmodes.conj().T @ self.X
        vander = np.array([
            np.power(amp, times) for amp in amps
        ])
        alpha = solve(
            (Rmodes.conj().T @ Rmodes) * (vander.conj() @ vander.T),
            (vander.conj() * (Rmodes.conj().T @ G)) @ np.ones(G.shape[1]),
            assume_a="pos"
        )
        return modes @ np.diag(alpha) @ vander

    def _tsvd(self):
        _Rx = self._R[:, :-1]

        _U, _S, _Vh = svd(_Rx, full_matrices=False, lapack_driver="gesvd")
        if self.max_rank is not None:
            _U = _U[:, :self.max_rank]
            _S = _S[:self.max_rank]
            _Vh = _Vh[:self.max_rank, :]

        return _U, (1. / _S) * _Vh.conj().T
    
    @property
    def A(self):
        _U, V_S_inv = self._tsvd()
        return _U.conj().T @ (self._R[:, 1:] @ V_S_inv)
    
    @property
    def modes(self):
        _U, V_S_inv = self._tsvd()
        _A = (_U.conj().T @ self._R[:, 1:]) @ V_S_inv
        evals, evecs = np.linalg.eig(_A)
        return self._Q @ (_U @ evecs), evals
    
    @property
    def residuals(self):
        _U, V_S_inv = self._tsvd()
        _Ry = self._R[:, 1:]
        _A = (_U.conj().T @ _Ry) @ V_S_inv
        evals, evecs = np.linalg.eig(_A)
        residuals = np.linalg.norm(_Ry @ (V_S_inv @ evecs) - (_U @ evecs) * evals, axis=0)
        P = np.argsort(residuals)
        residuals = residuals[P]
        evals = evals[P]
        evecs = evecs[:, P]
        return residuals, self._Q @ (_U @ evecs), evals
    
    def __call__(self, x):
        _U, V_S_inv = self._tsvd()
        _A = self._R[:, 1:] @ V_S_inv @ _U.conj().T

        return self._Q @ (_A @ (self._Q.conj().T @ x))
