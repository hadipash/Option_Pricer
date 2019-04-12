import numpy as np
from scipy.stats import norm

np.random.seed(51)


class AsianOption:
    def __init__(self, S, K, r, T, σ, n, option_type, m=int(1e5), t=0.0):
        self.__S, self.__K, self.__r = S, K, r
        self.__σ, self.__Δ = σ, T - t
        self.__n, self.__m, self.__dt = n, m, T / n
        self.__option_type = option_type.lower()

        self.__S_path = []

    def __gen_paths(self):
        if not len(self.__S_path):
            drift = np.exp((self.__r - (self.__σ ** 2) / 2) * self.__dt)
            growth_factor = drift * np.exp(self.__σ * np.sqrt(self.__dt) * np.random.randn(self.__m, self.__n))
            self.__S_path = self.__S * np.cumprod(growth_factor, 1)

    def __gen_geo_payoff(self):
        self.__gen_paths()
        geo_mean = np.exp(np.sum(np.log(self.__S_path), axis=1) / self.__n)
        if self.__option_type == 'call':
            self.__geo_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(geo_mean - self.__K, 0)
        elif self.__option_type == 'put':
            self.__geo_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(self.__K - geo_mean, 0)

    def __gen_arith_payoff(self):
        self.__gen_paths()
        arith_mean = np.mean(self.__S_path, axis=1)
        if self.__option_type == 'call':
            self.__arith_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(arith_mean - self.__K, 0)
        elif self.__option_type == 'put':
            self.__arith_payoff = np.exp(-self.__r * self.__Δ) * np.maximum(self.__K - arith_mean, 0)

    def __θ(self):
        covXY = np.mean(self.__arith_payoff * self.__geo_payoff) - \
                np.mean(self.__arith_payoff) * np.mean(self.__geo_payoff)
        return covXY / (np.var(self.__geo_payoff)+1e-20)

    def control_variate(self):
        self.__gen_geo_payoff()
        self.__gen_arith_payoff()

        Z = self.__arith_payoff + self.__θ() * (self.closed_form() - self.__geo_payoff)
        Zmean = np.mean(Z)
        Zstd = np.std(Z)
        # confidence interval
        confcv = [Zmean - 1.96 * Zstd / np.sqrt(self.__m), Zmean + 1.96 * Zstd / np.sqrt(self.__m)]

        return confcv

    def closed_form(self):
        σhat_sqΔ = self.__Δ * (self.__σ ** 2) * (self.__n + 1) * (2 * self.__n + 1) / (6 * self.__n ** 2)
        μhatΔ = self.__Δ * (self.__r - (self.__σ ** 2) / 2) * (self.__n + 1) / (2 * self.__n) + σhat_sqΔ / 2

        d1hat = (np.log(self.__S / self.__K) + μhatΔ + σhat_sqΔ / 2) / np.sqrt(σhat_sqΔ)
        d2hat = d1hat - np.sqrt(σhat_sqΔ)

        if self.__option_type == 'call':
            return np.exp(-self.__r * self.__Δ) * (self.__S * np.exp(μhatΔ) * norm.cdf(d1hat) -
                                                   self.__K * norm.cdf(d2hat))
        elif self.__option_type == 'put':
            return np.exp(-self.__r * self.__Δ) * (self.__K * norm.cdf(-d2hat) -
                                                   self.__S * np.exp(μhatΔ) * norm.cdf(-d1hat))

    def geo_std_MC(self):
        self.__gen_geo_payoff()
        return np.mean(self.__geo_payoff)

    def arith_std_MC(self):
        self.__gen_arith_payoff()
        return np.mean(self.__arith_payoff)


def run_test(num, option):
    print('\nCase {:d}:'.format(num))
    print('Arithmetic standard MC\t{:f}'.format(option.arith_std_MC()))
    print('Arithmetic MC with Control Variate\t' + str(option.control_variate()))
    print('Geometric closed-form formula\t{:f}'.format(option.closed_form()))
    print('Geometric standard MC\t{:f}'.format(option.geo_std_MC()))


if __name__ == '__main__':
    print('Put Options')
    run_test(1, AsianOption(S=100, K=100, r=0.05, T=3, σ=0.3, n=50, option_type='Put'))
    run_test(2, AsianOption(S=100, K=100, r=0.05, T=3, σ=0.3, n=100, option_type='Put'))
    run_test(3, AsianOption(S=100, K=100, r=0.05, T=3, σ=0.4, n=50, option_type='Put'))

    print('\nCall Options:')
    run_test(1, AsianOption(S=100, K=100, r=0.05, T=3, σ=0.3, n=50, option_type='Call'))
    run_test(2, AsianOption(S=100, K=100, r=0.05, T=3, σ=0.3, n=100, option_type='Call'))
    run_test(3, AsianOption(S=100, K=100, r=0.05, T=3, σ=0.4, n=50, option_type='Call'))
