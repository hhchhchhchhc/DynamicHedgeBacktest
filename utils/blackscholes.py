import math

import scipy


class black_scholes:
    @staticmethod
    def d1(S, K, V, T):
        return (math.log(S / float(K)) + (V ** 2 / 2) * T) / (V * math.sqrt(T))

    @staticmethod
    def d2(S, K, V, T):
        return black_scholes.d1(S, K, V, T) - (V * math.sqrt(T))

    @staticmethod
    def pv(S, K, V, T, cp):
        if cp == 'C':
            return S * scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T)) - K * scipy.stats.norm.cdf(
                black_scholes.d2(S, K, V, T))
        elif cp == 'P':
            return K * scipy.stats.norm.cdf(-black_scholes.d2(S, K, V, T)) - S * scipy.stats.norm.cdf(
                -black_scholes.d1(S, K, V, T))
        else:
            return black_scholes.pv(S, K, V, T, 'P') + black_scholes.pv(S, K, V, T, 'C')

    @staticmethod
    def delta(S, K, V, T, cp):
        '''for a 1% move'''
        delta = scipy.stats.norm.cdf(black_scholes.d1(S, K, V, T))
        if cp == 'C':
            delta = delta
        elif cp == 'P':
            delta = (delta - 1)
        elif cp =='S':
            delta = (2 * delta - 1)

        return delta * S * 0.01

    @staticmethod
    def gamma(S, K, V, T, cp):
        '''for a 1% move'''
        gamma = scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T)) / (S * V * math.sqrt(T))
        return gamma * S * 0.01 * S * 0.01 * (1 if cp != 'S' else 2)

    @staticmethod
    def vega(S, K, V, T, cp):
        '''for a 10% move'''
        vega = (S * math.sqrt(T) * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T)))
        return vega * V * 0.1 * (1 if cp != 'S' else 2)

    @staticmethod
    def theta(S, K, V, T, cp):
        '''for 1h'''
        theta = -((S * V * scipy.stats.norm.pdf(black_scholes.d1(S, K, V, T))) / (2 * math.sqrt(T)))
        return theta / 24/365.25 * (1 if cp != 'S' else 2)

    @staticmethod
    # Define the Black-Scholes implied volatility function
    def bs_iv(price, F, K, T, cp):
        # Set an initial guess for the volatility
        sigma = 0.5
        # Set a tolerance level for the error
        tol = 1e-6
        # Set a maximum number of iterations
        max_iter = 100
        # Initialize a counter for the iterations
        i = 0
        # Start the loop
        while True:
            # Calculate the option price and delta using the current volatility
            f = black_scholes.pv(F, K, sigma, T, cp) - price
            df = black_scholes.vega(F, K, sigma, T, cp)
                # Update the volatility using the Newton-Raphson formula
            sigma = sigma - f / (df if abs(df) > 1e-18 else (1e-18 if df>0 else -1e-18))
                # Check if the absolute error is below the tolerance level
            if abs(f) < tol:
                # Return the volatility
                return sigma
            # Increment the counter
            i += 1
            # Check if the maximum number of iterations is reached
            if i >= max_iter:
                # Return None
                return None
