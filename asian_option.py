from math import sqrt, log, exp, pi
from scipy.stats import norm


class AsianOption:
    def __init__(self, S, K, r, T, σ, n, t=0.0):
        self.__S = S
        self.__K = K
        self.__r = r
        self.__Δ = T - t

        σhat_sqΔ = self.__Δ * (σ ** 2) * (n + 1) * (2 * n + 1) / (6 * n ** 2)
        self.__μhatΔ = self.__Δ * (r - (σ ** 2) / 2) * (n + 1) / (2 * n) + σhat_sqΔ / 2

        self.__d1hat = (log(S / K) + self.__μhatΔ + σhat_sqΔ / 2) / sqrt(σhat_sqΔ)
        self.__d2hat = self.__d1hat - sqrt(σhat_sqΔ)

    def geo_call(self):
        return exp(-self.__r * self.__Δ) * (self.__S * exp(self.__μhatΔ) * norm.cdf(self.__d1hat) -
                                            self.__K * norm.cdf(self.__d2hat))

    def geo_put(self):
        return exp(-self.__r * self.__Δ) * (self.__K * norm.cdf(-self.__d2hat) -
                                            self.__S * exp(self.__μhatΔ) * norm.cdf(-self.__d1hat))
