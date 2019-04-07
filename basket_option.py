import numpy as np
from scipy.stats import norm


class BasketOption:
    def __init__(self, S1, S2, K, r, T, σ1, σ2, ρ, m=int(1e4), t=0.0):
        self.__S1, self.__S2 = S1, S2
        self.__K, self.__r = K, r
        self.__σ1, self.__σ2 = σ1, σ2
        self.__ρ, self.__Δ = ρ, T - t

        self.__B0 = self.__geo_mean([self.__S1, self.__S2], axis=0)
        σB_sqΔ = self.__Δ * (self.__σ1 ** 2 + 2 * self.__ρ * self.__σ1 * self.__σ2 + self.__σ2 ** 2) / 4
        self.__μBΔ = self.__Δ * (self.__r - (self.__σ1 ** 2 + self.__σ2 ** 2) / 4) + σB_sqΔ / 2

        self.__d1hat = (np.log(self.__B0 / self.__K) + self.__μBΔ + σB_sqΔ / 2) / np.sqrt(σB_sqΔ)
        self.__d2hat = self.__d1hat - np.sqrt(σB_sqΔ)

    @staticmethod
    def __geo_mean(x, axis=1):
        return np.exp(np.sum(np.log(x), axis=axis) / len(x))

    def geo_call_cf(self):
        return np.exp(-self.__r * self.__Δ) * (self.__B0 * np.exp(self.__μBΔ) * norm.cdf(self.__d1hat) -
                                               self.__K * norm.cdf(self.__d2hat))

    def geo_put_cf(self):
        return np.exp(-self.__r * self.__Δ) * (self.__K * norm.cdf(-self.__d2hat) -
                                               self.__B0 * np.exp(self.__μBΔ) * norm.cdf(-self.__d1hat))


# test, closed-form formula
bo = BasketOption(S1=100, S2=100, K=100, r=0.05, T=3, σ1=0.3, σ2=0.3, ρ=0.5)
print(bo.geo_call_cf())
