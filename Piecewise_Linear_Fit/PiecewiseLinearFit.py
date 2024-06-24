"""
Find optimal number of break points for a piecewise linear fit.
Use the Bayesian Information Criterion (BIC) to find the optimal number of breaks.
Fit the model with the optimal number of breaks using ElasticNetCV for regularization.
"""

__date__ = "2024-06-24"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
import pwlf


def plot_piecewise_linear_fit(
    x: np.array,
    y: np.array,
    y_hat: np.array,
    val_col: str,
    optimal_breaks: int,
    outlier_mask: np.array,
):
    """
    Plots the piecewise linear fit of the given data set

    Args:
        x: np.array - x values
        y: np.array - y values
        y_hat: np.array - predicted y values
        val_col: str - column name for the value
        optimal_breaks: int - optimal number of breaks
        outlier_mask: np.array - mask for the outliers

    Returns: 
        fig: matplotlib figure object of the piecewise linear fit  
    """
    print(
        "Plotting piecewise linear fit with optimal number of breaks: ", optimal_breaks
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, c="r")
    ax.plot(x, y_hat)

    if np.any(outlier_mask):
        ax.scatter(x[outlier_mask], y[outlier_mask], c="b")

    ax.set_ylabel(val_col)
    ax.set_xlabel("Index")
    ax.set_title("Piecewise Linear Fit of " + val_col)
    ax.legend(["Data", "Piecewise Linear Fit", "Outliers"])
    return fig


def find_optimal_pwlf(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    complexity_penalty: int = 20,
    max_breaks: int = 7,
    outlier_threshold: int = 3,
    plot_results: bool = True,
):
    """
    This function finds the optimal number of breaks for a piecewise linear fit using the Bayesian Information Criterion (BIC)
    and then plots the piecewise linear fit of the given data set

    Args:
        df: DataFrame
        date_col: column name for the date
        val_col: column name for the value
        complexity_penalty: complexity penalty for the BIC
        max_breaks: maximum number of breaks to consider
        outlier_threshold: threshold for identifying outliers

    Returns:
        optimal_breaks: optimal number of breaks
        my_pwlf: PiecewiseLinFit model
        outliers: DataFrame containing the outliers
    """
    # Convert the date column to datetime if it's not already
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, val_col])

    # Convert the datetime to Unix timestamp (number of seconds since 1970-01-01 00:00:00 UTC)
    x = (df[date_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    y = df[val_col]

    # BIC calculation
    bic_values = []
    n = len(y)
    for i in range(2, max_breaks + 1):
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        res = my_pwlf.fit(i)
        # calculate the residual sum of squares
        yhat = my_pwlf.predict(x)
        rss = np.sum((y - yhat) ** 2)
        # calculate the BIC
        bic = n * np.log(rss / n) + complexity_penalty * i * np.log(n)
        bic_values.append(bic)

    optimal_breaks = np.argmin(bic_values) + 2  # +2 because the range starts from 2

    # Fit the model with the optimal number of breaks using ElasticNetCV for regularization
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(optimal_breaks)
    A = my_pwlf.assemble_regression_matrix(breaks, x)
    en_model = ElasticNetCV(
        cv=5,
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1],
        fit_intercept=False,
        max_iter=1000,
        n_jobs=-1,
    )
    en_model.fit(A, y)

    # Calculate the residuals
    y_hat = my_pwlf.predict(x)
    residuals = y - y_hat

    # Identify outliers as points that are more than 'outlier_threshold' standard deviations from the mean residual
    outlier_mask = np.abs(residuals - np.mean(residuals)) > outlier_threshold * np.std(
        residuals
    )
    outliers = df[outlier_mask]

    # Plot the piecewise linear fit
    fig = None
    if plot_results:
        fig = plot_piecewise_linear_fit(
            x, y, y_hat, val_col, optimal_breaks, outlier_mask
        )

    return optimal_breaks, my_pwlf, outliers, fig
