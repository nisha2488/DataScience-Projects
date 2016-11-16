import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
#
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

# from sklearn.datasets import load_boston
# from sklearn.ensemble import RandomForestRegressor

# %matplotlib inline

# print "Testing various imports"

train = pd.read_csv("train_complete.csv")
test = pd.read_csv("test_complete.csv")

# print train.head()

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

print "All data columns:"
print all_data.columns
print train.columns
print "Printing Index below"
print train.index
print train['Alley']

num_data = all_data.dtypes[all_data.dtypes != "object"].index
print num_data

cat_data = all_data.dtypes[all_data.dtypes == "object"].index
print "Cat_data below:"
print cat_data

# Does the same thing as above code for cat_data
# cats = []
# print "In loop"
# for col in all_data.columns.values:
#     # print col
#     if all_data[col].dtype == 'object':
#         print col
#         cats.append(col)

