import numpy as np
import pandas as pd


class LeastSquaresRegression:
    def __init__(self, x: pd.Series, y: pd.Series):
        self.x = x
        self.y = y
        self.n = self.x.shape[0]

        self._x_bar, self._y_bar = self._calc_means()

        self._var = self._calc_variance()
        self._cov = self._calc_covariance()

        self._beta1 = self._calc_beta1()
        self._beta0 = self._calc_beta0()

    def _calc_means(self) -> tuple[np.float64, np.float64]:
        return (np.mean(self.x), np.mean(self.y))

    def _calc_variance(self) -> np.float64:
        """
        Returns the variance of a series
        """
        total: np.float64 = np.float64(0)

        for i in range(0, self.n):
            total += np.float64((self.x[i] - self._x_bar) ** 2)

        return total / (self.n - 1)

    def _calc_covariance(self) -> np.float64:
        """
        Return cov(a,b) given two series of equal length
        """

        total: np.float64 = np.float64(0)

        n = self.x.shape[0]

        for i in range(0, n):
            total += np.float64((self.x[i] - self._x_bar) * (self.y[i] - self._y_bar))

        return np.float64(total / (n - 1))

    def _calc_beta1(self) -> np.float64:
        return self._cov / self._var

    def _calc_beta0(self) -> np.float64:
        return self._y_bar - (self._beta1 * self._x_bar)

    def get_output(self):
        return (self._beta1, self._beta0)
