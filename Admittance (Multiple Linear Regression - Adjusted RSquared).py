#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()


# Load the data
data = pd.read_csv('1.02. Multiple linear regression.csv')
data

# Create a linear regression
data.describe()

y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()

results.summary()


# Reviewing Adj. R-squared against the simple linear regression model
# we have added more variables but have lost value

# Therefore, we drop 'Rand 1,2,3' for it is insignificant 

# Having 'b0' reduced only lessens the power of the model

# F test is used to measures the overall significance of a model
