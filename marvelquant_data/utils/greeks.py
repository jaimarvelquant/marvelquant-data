"""
Option Greeks calculation utilities using the Black-Scholes model.

Provides OptionPricing class for implied volatility and Greeks (Delta, Gamma,
Theta, Vega, Rho) calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass
class OptionPricing:
    """
    Black-Scholes option pricing and Greeks calculator.

    Args:
        spot: Spot (underlying) price
        strike: Strike price
        rate: Annualized risk-free rate (e.g., 0.05 for 5%)
        time_years: Time to expiry in years
    """

    spot: float
    strike: float
    rate: float
    time_years: float

    IV_MIN: float = 1e-4
    IV_MAX: float = 5.0  # 500% cap to prevent runaway solver

    def __post_init__(self) -> None:
        self.spot = float(max(self.spot, 1e-8))
        self.strike = float(max(self.strike, 1e-8))
        self.rate = float(self.rate)
        self.time_years = float(max(self.time_years, 1e-8))

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _sigma_sqrt_t(self, sigma: float) -> float:
        return sigma * math.sqrt(self.time_years)

    def d1(self, sigma: float) -> float:
        sigma_sqrt_t = self._sigma_sqrt_t(sigma)
        if sigma_sqrt_t <= 0:
            return math.inf if self.spot >= self.strike else -math.inf

        return (
            math.log(self.spot / self.strike)
            + (self.rate + 0.5 * sigma * sigma) * self.time_years
        ) / sigma_sqrt_t

    def d2(self, sigma: float) -> float:
        return self.d1(sigma) - self._sigma_sqrt_t(sigma)

    # -------------------------------------------------------------------------
    # Pricing
    # -------------------------------------------------------------------------
    def call_price(self, sigma: float) -> float:
        if sigma <= 0:
            return max(0.0, self.spot - self.strike * math.exp(-self.rate * self.time_years))

        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        return self.spot * norm.cdf(d1) - self.strike * math.exp(-self.rate * self.time_years) * norm.cdf(d2)

    def put_price(self, sigma: float) -> float:
        if sigma <= 0:
            return max(0.0, self.strike * math.exp(-self.rate * self.time_years) - self.spot)

        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        return self.strike * math.exp(-self.rate * self.time_years) * norm.cdf(-d2) - self.spot * norm.cdf(-d1)

    # -------------------------------------------------------------------------
    # IV Solver
    # -------------------------------------------------------------------------
    def implied_vol(self, price: float, option_type: str) -> float:
        """Solve for implied volatility using Brent's method."""

        option_type = option_type.upper()
        price = float(max(price, 0.0))

        if option_type not in {"CE", "PE", "CALL", "PUT"}:
            raise ValueError(f"Invalid option type: {option_type}")

        intrinsic = (
            max(0.0, self.spot - self.strike * math.exp(-self.rate * self.time_years))
            if option_type in {"CE", "CALL"}
            else max(0.0, self.strike * math.exp(-self.rate * self.time_years) - self.spot)
        )

        if price <= intrinsic + 1e-4:
            return self.IV_MIN

        pricing_func = self.call_price if option_type in {"CE", "CALL"} else self.put_price

        def objective(vol: float) -> float:
            return pricing_func(vol) - price

        try:
            low_val = objective(self.IV_MIN)
            high_val = objective(self.IV_MAX)

            if low_val * high_val > 0:
                # Cannot bracket root, fall back to closer bound
                return self.IV_MIN if abs(low_val) < abs(high_val) else self.IV_MAX

            return max(self.IV_MIN, brentq(objective, self.IV_MIN, self.IV_MAX, xtol=1e-4))
        except Exception:
            return self.IV_MIN

    # -------------------------------------------------------------------------
    # Greeks
    # -------------------------------------------------------------------------
    def delta(self, sigma: float, option_type: str) -> float:
        option_type = option_type.upper()
        d1 = self.d1(sigma)
        if option_type in {"CE", "CALL"}:
            return norm.cdf(d1)
        return norm.cdf(d1) - 1.0

    def gamma(self, sigma: float) -> float:
        sigma_sqrt_t = self._sigma_sqrt_t(sigma)
        if sigma_sqrt_t <= 0:
            return 0.0
        return norm.pdf(self.d1(sigma)) / (self.spot * sigma_sqrt_t)

    def theta(self, sigma: float, option_type: str) -> float:
        option_type = option_type.upper()
        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        term1 = -(self.spot * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(self.time_years))

        if option_type in {"CE", "CALL"}:
            term2 = -self.rate * self.strike * math.exp(-self.rate * self.time_years) * norm.cdf(d2)
        else:
            term2 = self.rate * self.strike * math.exp(-self.rate * self.time_years) * norm.cdf(-d2)
        return term1 + term2

    def vega(self, sigma: float) -> float:
        return self.spot * math.sqrt(self.time_years) * norm.pdf(self.d1(sigma)) / 100.0

    def rho(self, sigma: float, option_type: str) -> float:
        option_type = option_type.upper()
        d2 = self.d2(sigma)
        if option_type in {"CE", "CALL"}:
            return self.strike * self.time_years * math.exp(-self.rate * self.time_years) * norm.cdf(d2) / 100.0
        return -self.strike * self.time_years * math.exp(-self.rate * self.time_years) * norm.cdf(-d2) / 100.0
"""
Option Greeks Calculation Utilities using Black-Scholes Model.

This module provides the OptionPricing class for calculating implied volatility
and option Greeks (Delta, Gamma, Theta, Vega, Rho) using the Black-Scholes analytical formula.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class OptionPricing:
    """
    Black-Scholes Option Pricing and Greeks Calculator.
    """

    def __init__(self, S: float, K: float, r: float, T: float):
        """
        Initialize OptionPricing calculator.

        Args:
            S: Spot price of the underlying asset
            K: Strike price of the option
            r: Risk-free interest rate (annualized, e.g., 0.05 for 5%)
            T: Time to expiration in years
        """
        self.S = float(S)
        self.K = float(K)
        self.r = float(r)
        self.T = float(T)

        # Avoid division by zero or log of zero/negative
        self.T = max(self.T, 1e-5)  # Minimum 1e-5 years (~5 minutes)
        self.S = max(self.S, 1e-5)

        self.IV_LOWER_BOUND = 0.0001
        self.IV_UPPER_BOUND = 5.0  # 500% volatility cap for sanity

    def BS_d1(self, sigma: float) -> float:
        """Calculate d1 term of Black-Scholes."""
        if sigma < self.IV_LOWER_BOUND:
            return np.inf if self.S > self.K else -np.inf

        sigma_sqrt_t = sigma * np.sqrt(self.T)
        if sigma_sqrt_t == 0:
             return np.inf if self.S > self.K else -np.inf

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / sigma_sqrt_t
        return d1

    def BS_d2(self, sigma: float) -> float:
        """Calculate d2 term of Black-Scholes."""
        if sigma < self.IV_LOWER_BOUND:
             return np.inf if self.S > self.K else -np.inf

        return self.BS_d1(sigma) - sigma * np.sqrt(self.T)

    def BS_CallPricing(self, sigma: float) -> float:
        """Calculate theoretical Call price."""
        if sigma <= 0:
            return max(0.0, self.S - self.K * np.exp(-self.r * self.T))

        d1 = self.BS_d1(sigma)
        d2 = self.BS_d2(sigma)
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def BS_PutPricing(self, sigma: float) -> float:
        """Calculate theoretical Put price."""
        if sigma <= 0:
            return max(0.0, self.K * np.exp(-self.r * self.T) - self.S)

        d1 = self.BS_d1(sigma)
        d2 = self.BS_d2(sigma)
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def ImplVolWithBrent(self, option_price: float, option_type: str) -> float:
        """
        Calculate Implied Volatility using Brent's method.

        Args:
            option_price: Market price of the option
            option_type: 'CE' (Call) or 'PE' (Put)

        Returns:
            Implied Volatility (decimal)
        """
        pricing_func = self.BS_CallPricing if option_type == 'CE' else self.BS_PutPricing

        # Intrinsic value check
        intrinsic = 0.0
        if option_type == 'CE':
            intrinsic = max(0.0, self.S - self.K * np.exp(-self.r * self.T))
        else:
            intrinsic = max(0.0, self.K * np.exp(-self.r * self.T) - self.S)

        if option_price <= intrinsic + 0.001:
             return self.IV_LOWER_BOUND

        try:
            def objective(sigma):
                return pricing_func(sigma) - option_price

            # Check bounds first
            y_min = objective(self.IV_LOWER_BOUND)
            y_max = objective(self.IV_UPPER_BOUND)

            if y_min * y_max > 0:
                # Root not bracketed
                if abs(y_min) < abs(y_max):
                    return self.IV_LOWER_BOUND
                else:
                    return self.IV_UPPER_BOUND

            iv = brentq(objective, self.IV_LOWER_BOUND, self.IV_UPPER_BOUND, xtol=1e-4)
            return max(iv, self.IV_LOWER_BOUND)

        except Exception:
            return self.IV_LOWER_BOUND

    def Delta(self, sigma: float, option_type: str) -> float:
        """Calculate Delta."""
        d1 = self.BS_d1(sigma)
        if option_type == 'CE':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    def Gamma(self, sigma: float) -> float:
        """Calculate Gamma (same for Call and Put)."""
        if sigma <= self.IV_LOWER_BOUND or self.S <= 0 or self.T <= 0:
            return 0.0
        d1 = self.BS_d1(sigma)
        return norm.pdf(d1) / (self.S * sigma * np.sqrt(self.T))

    def Theta(self, sigma: float, option_type: str) -> float:
        """Calculate Theta (annualized)."""
        if sigma <= self.IV_LOWER_BOUND:
             return 0.0

        d1 = self.BS_d1(sigma)
        d2 = self.BS_d2(sigma)

        term1 = -(self.S * norm.pdf(d1) * sigma) / (2 * np.sqrt(self.T))

        if option_type == 'CE':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return term1 + term2
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return term1 + term2

    def Vega(self, sigma: float) -> float:
        """Calculate Vega (same for Call and Put)."""
        if sigma <= self.IV_LOWER_BOUND:
             return 0.0
        d1 = self.BS_d1(sigma)
        # Vega is typically reported as change per 1% vol change, so divide by 100
        return self.S * np.sqrt(self.T) * norm.pdf(d1) / 100.0

    def Rho(self, sigma: float, option_type: str) -> float:
        """Calculate Rho."""
        if sigma <= self.IV_LOWER_BOUND:
             return 0.0

        d2 = self.BS_d2(sigma)
        if option_type == 'CE':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100.0
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100.0

