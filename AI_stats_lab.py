"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    Compute analytical probabilities and verify using Monte Carlo simulation
    """

    # Analytical values
    analytic_gt5 = math.exp(-5)
    analytic_lt5 = 1 - math.exp(-5)
    analytic_interval = math.exp(-3) - math.exp(-7)

    # Monte Carlo simulation
    samples = np.random.exponential(scale=1, size=100000)
    simulated_gt5 = np.mean(samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Validate candidate PDF and plot
    """

    # Define function
    def f(x):
        return 2 * x * np.exp(-x**2)

    # Compute integral
    integral_value, _ = quad(f, 0, np.inf)

    # Check if valid PDF
    is_valid_pdf = (integral_value > 0) and (abs(integral_value - 1) < 1e-3)

    # Plot PDF on [0,3]
    x = np.linspace(0, 3, 500)
    y = f(x)

    plt.plot(x, y)
    plt.title("PDF f(x) = 2x e^{-x^2}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    return integral_value, is_valid_pdf


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    Analytical and simulated probabilities for Exp(1)
    """

    # Analytical values
    analytic_gt5 = math.exp(-5)
    analytic_interval = math.exp(-1) - math.exp(-3)

    # Monte Carlo simulation
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    Analytical and simulated probabilities for N(10,2^2)
    """

    # Analytical probabilities
    analytic_le12 = norm.cdf(12, loc=10, scale=2)
    analytic_interval = norm.cdf(12, loc=10, scale=2) - norm.cdf(8, loc=10, scale=2)

    # Monte Carlo simulation
    samples = np.random.normal(loc=10, scale=2, size=100000)

    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval
