import numpy as np
from scipy.stats import norm


class AsianOption:
    def __init__(self, S, K, r, T, σ, n, m, dt, t=0.0):
        self.__S = S
        self.__K = K
        self.__r = r
        self.__Δ = T - t
        self.__σ = σ
        self.__dt = dt
        self.__n = n
        self.__m = m

        self.__S_path = []

        σhat_sqΔ = self.__Δ * (σ ** 2) * (n + 1) * (2 * n + 1) / (6 * n ** 2)
        self.__μhatΔ = self.__Δ * (r - (σ ** 2) / 2) * (n + 1) / (2 * n) + σhat_sqΔ / 2

        self.__d1hat = (np.log(S / K) + self.__μhatΔ + σhat_sqΔ / 2) / np.sqrt(σhat_sqΔ)
        self.__d2hat = self.__d1hat - np.sqrt(σhat_sqΔ)

    def __gen_paths(self):
        drift = np.exp((self.__r - (self.__σ ** 2) / 2) * self.__dt)
        gf = drift * np.exp(self.__σ * np.sqrt(self.__dt) * np.random.normal(size=(self.__m, self.__n)))

        for i in range(self.__m):
            self.__S_path.append([self.__S * gf[i][0]])
            for j in range(1, self.__n):
                self.__S_path[i].append(self.__S_path[i][j - 1] * gf[i][j])

    def __gen_geo_payoff(self):
        geo_mean = np.exp(np.sum(np.log(self.__S_path), axis=1) / self.__n)
        self.__geo_payoff = np.exp(-self.__r * self.__Δ) * np.max(geo_mean - self.__K, 0)

    def __gen_arith_payoff(self):
        arith_mean = np.mean(self.__S_path, axis=1)
        self.__arith_payoff = np.exp(-self.__r * self.__Δ) * np.max(arith_mean - self.__K, 0)

    def __θ(self):
        return np.cov(self.__arith_payoff, self.__geo_payoff) / np.var(self.__geo_payoff)

    def get_arith_payoff(self):
        self.__gen_paths()
        self.__gen_geo_payoff()
        self.__gen_arith_payoff()

        # TODO: Check this
        Z = self.__arith_payoff + self.__θ() * (self.geo_call() - self.__geo_payoff)
        Zmean = np.mean(Z)
        Zstd = np.std(Z)
        confcv = [Zmean - 1.96 * Zstd / np.sqrt(self.__m), Zmean + 1.96 * Zstd / np.sqrt(self.__m)]

    def geo_call(self):
        return np.exp(-self.__r * self.__Δ) * (self.__S * np.exp(self.__μhatΔ) * norm.cdf(self.__d1hat) -
                                               self.__K * norm.cdf(self.__d2hat))

    def geo_put(self):
        return np.exp(-self.__r * self.__Δ) * (self.__K * norm.cdf(-self.__d2hat) -
                                               self.__S * np.exp(self.__μhatΔ) * norm.cdf(-self.__d1hat))
