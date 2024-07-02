"""
This script is used to test the Piecewise_Linear_Fit script.
It reads the SP500 data from a csv file and adds outlier points.
"""

__date__ = "2024-06-24"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"


# %% --------------------------------------------------------------------------
# Import Modules
import pandas as pd
import matplotlib.pyplot as plt
from Piecewise_Linear_Fit.PiecewiseLinearFit import find_optimal_pwlf

# %%
data = pd.read_csv("Improve_Efficiency/data/SP500.csv")
val_col = "SP500"
# add outlier points at index 300 and 1100
data.loc[300, val_col] = 1500
data.loc[1100, val_col] = 3000

plt.scatter(data.index, data[val_col])
plt.title("SP500 data with outliers")
plt.xlabel("Index")
plt.ylabel(val_col)

find_optimal_pwlf(
    data, 
    'Date', 
    val_col,
    complexity_penalty=20, 
    max_breaks=8, 
    outlier_threshold=8
    )
