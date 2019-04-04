import numpy as np
from scipy.stats import norm


class BasketOption:
    def __init__(self, S1, S2, K, r, T, σ1, σ2, ρ, m=int(1e4), t=0.0):
        self.__S1, self.__S2 = S1, S2
        self.__K, self.__r = K, r
        self.__σ1, self.__σ2 = σ1, σ2
        self.__ρ, self.__Δ = ρ, T - t
