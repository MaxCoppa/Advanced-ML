import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List

class TimeSeriesGenerator(ABC):
    def __init__(self, T: int, seed: int = None):
        self.T = T
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Returns:
            x: np.ndarray of shape (T, d)
        """
        pass

class ARGenerator(TimeSeriesGenerator):
    def __init__(self, T, phi, sigma=1.0, seed=None):
        super().__init__(T, seed)
        self.phi = np.array(phi)
        self.p = len(phi)
        self.sigma = sigma

    def generate(self):
        x = np.zeros(self.T)
        eps = self.rng.normal(scale=self.sigma, size=self.T)

        for t in range(self.p, self.T):
            x[t] = np.dot(self.phi, x[t-self.p:t][::-1]) + eps[t]

        return x[:, None]

class NonlinearARGenerator(TimeSeriesGenerator):
    def __init__(self, T, p=5, sigma=0.1, seed=None):
        super().__init__(T, seed)
        self.p = p
        self.sigma = sigma

    def generate(self):
        x = np.zeros(self.T)
        eps = self.rng.normal(scale=self.sigma, size=self.T)

        for t in range(self.p, self.T):
            x[t] = np.tanh(x[t-self.p:t].sum()) + eps[t]

        return x[:, None]

class LongMemoryGenerator(TimeSeriesGenerator):
    def __init__(self, T, L=50, sigma=0.1, seed=None):
        super().__init__(T, seed)
        self.L = L
        self.sigma = sigma

    def generate(self):
        x = np.zeros(self.T)
        eps = self.rng.normal(scale=self.sigma, size=self.T)

        for t in range(1, self.T):
            start = max(0, t - self.L)
            x[t] = np.sin(x[start:t].sum()) + eps[t]

        return x[:, None]

class SwitchingARGenerator(TimeSeriesGenerator):
    def __init__(self, T, phis, transition_matrix, sigma=0.1, seed=None):
        super().__init__(T, seed)
        self.phis = phis
        self.P = transition_matrix
        self.sigma = sigma
        self.num_states = len(phis)

    def generate(self):
        x = np.zeros(self.T)
        z = np.zeros(self.T, dtype=int)
        eps = self.rng.normal(scale=self.sigma, size=self.T)

        for t in range(1, self.T):
            z[t] = self.rng.choice(self.num_states, p=self.P[z[t-1]])
            x[t] = self.phis[z[t]] * x[t-1] + eps[t]

        return x[:, None]

class LatentFactorGenerator(TimeSeriesGenerator):
    def __init__(self, T, d=5, sigma=0.1, seed=None):
        super().__init__(T, seed)
        self.d = d
        self.A = self.rng.normal(size=(d, 1))
        self.sigma = sigma

    def generate(self):
        f = np.zeros(self.T)
        eps_f = self.rng.normal(scale=0.5, size=self.T)
        eps_x = self.rng.normal(scale=self.sigma, size=(self.T, self.d))

        for t in range(1, self.T):
            f[t] = 0.8 * f[t-1] + eps_f[t]

        x = f[:, None] @ self.A.T + eps_x
        return x

def generate_datasets(
    generators: Dict[str, TimeSeriesGenerator],
    n_datasets: int
) -> Dict[str, List[np.ndarray]]:

    datasets = {name: [] for name in generators}

    for name, gen in generators.items():
        for _ in range(n_datasets):
            datasets[name].append(gen.generate())

    return datasets
