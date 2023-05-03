import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('salary_data.csv')
x = data['YearsExperience']
y = data['Salary']
print(data.head())
def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)
def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {} \\nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x, y, b)
if __name__ == "__main__":
    main()
    
    
"""NumPy: NumPy (short for "Numerical Python") is a Python library that provides support for arrays and matrices, as well as mathematical functions that can be applied to them. NumPy is widely used for scientific computing and data analysis.
NumPy import statement: import numpy as np

Matplotlib: Matplotlib is a Python library for creating data visualizations. It provides support for a wide range of plot types, including line plots, scatter plots, bar plots, and more. Matplotlib is often used for exploratory data analysis and creating publication-quality graphics.

Matplotlib import statement: import matplotlib.pyplot as plt

Pandas: Pandas is a Python library for data manipulation and analysis. It provides support for two main data structures: Series (1-dimensional) and DataFrame (2-dimensional). Pandas also provides functions for data cleaning, merging, and visualization.

Pandas import statement: import pandas as pd

In all three import statements, we use the as keyword to alias the library names to shorter, more convenient names (np for NumPy, plt for Matplotlib, and pd for Pandas)."""
    