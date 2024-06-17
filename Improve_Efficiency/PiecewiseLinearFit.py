"""
Find optimal number of break points for a piecewise linear fit.
Use the Bayesian Information Criterion (BIC) to find the optimal number of breaks.
Fit the model with the optimal number of breaks using ElasticNetCV for regularization.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
import pwlf

def find_and_plot_optimal_pwlf(
        df, 
        date_col, 
        val_col, 
        complexity_penalty=2,  
        max_breaks=5, 
        outlier_threshold=2
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
    # drop na
    df = df.dropna(subset=[date_col, val_col])

    # Convert the datetime to Unix timestamp (number of seconds since 1970-01-01 00:00:00 UTC)
    x = (df[date_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    y = df[val_col]

    bic_values = []
    n = len(y)
    for i in range(2, max_breaks+1):
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        res = my_pwlf.fit(i)
        # calculate the residual sum of squares
        yhat = my_pwlf.predict(x)
        rss = np.sum((y - yhat)**2)
        # calculate the BIC
        bic = n * np.log(rss/n) + complexity_penalty * i * np.log(n)
        bic_values.append(bic)

    optimal_breaks = np.argmin(bic_values) + 2  # +2 because the range starts from 2

    # Fit the model with the optimal number of breaks using ElasticNetCV for regularization
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    breaks = my_pwlf.fit(optimal_breaks)
    A = my_pwlf.assemble_regression_matrix(breaks, x)
    en_model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False, max_iter=1000, n_jobs=-1)
    en_model.fit(A, y)

    # Calculate the residuals
    y_hat = my_pwlf.predict(x)
    residuals = y - y_hat

    # Identify outliers as points that are more than 'outlier_threshold' standard deviations from the mean residual
    outlier_mask = np.abs(residuals - np.mean(residuals)) > outlier_threshold * np.std(residuals)
    outliers = df[outlier_mask]

    print("Plotting piecewise linear fit with optimal number of breaks: ", optimal_breaks)
    # plot interpolation
    plt.figure(figsize=(10,5))
    plt.scatter(x, y, c='r')
    plt.plot(x, y_hat)
    plt.scatter(x[outlier_mask], y[outlier_mask], c='b')
    plt.ylabel(val_col)
    plt.xlabel('Index')
    plt.title('Piecewise Linear Fit of ' + val_col + " for well " + df['parent.id'].unique()[0])
    plt.legend(['Data','Piecewise Linear Fit', 'Outliers'])

    return optimal_breaks, my_pwlf, outliers