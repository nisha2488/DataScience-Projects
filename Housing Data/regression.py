import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib
#
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import uniform as sp_rand
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score

# from sklearn.datasets import load_boston
# from sklearn.ensemble import RandomForestRegressor

# %matplotlib inline
# plt.interactive(True)
debug = 0

# print "Testing various imports"

train = pd.read_csv("train_complete.csv", index_col=0)
test = pd.read_csv("test_complete.csv", index_col=0)
# print "Null values in test"
# print test.isnull().sum().sum()
# print train.head()
# print len(sale_price)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# print all_data.isnull().sum().sum()

if debug == 1:
    print "All data columns:"
    print all_data.columns
    print train.columns
    print "Printing Index below"
    print train.index
    print train['Alley']

# Convert data type for MSSubCLass to categorical from numerical
if debug == 1:
    print "Data type for MSSubClass:"
    print all_data["MSSubClass"].dtype
all_data['MSSubClass'] = all_data['MSSubClass'].astype(object)
# all_data['GarageYrBlt'] = all_data['GarageYrBlt'].convert_objects(convert_numeric=True)
if debug == 1:
    print all_data["MSSubClass"].dtype

# print all_data.isnull().sum().sum()

# Split variables into numerical and categorical
# num_data = all_data.dtypes[all_data.dtypes != "object"].index
num_data = all_data.columns[all_data.dtypes != 'object']
if debug == 1:
    print "First num data"
    print num_data

# cat_data = all_data.dtypes[all_data.dtypes == "object"].index
cat_data = all_data.columns[all_data.dtypes == 'object']
if debug == 1:
    print "Cat_data below:"
    print cat_data

# Does the same thing as above code for cat_data
# cats = []
# if debug == 1:
#     print "In loop"
# for col in all_data.columns.values:
#     print col, all_data[col].dtype
#     if all_data[col].dtype == 'object':
#         # print col
#         cats.append(col)


# num_data = all_data.columns[all_data.dtypes != 'object']
# if debug == 1:
#     print "Second num data"
#     print num_data

# Remove skewness from target variable as well as all other numerical features
# Using log(value+1) transformation since numerical data may have 0/-ve values after scaling
train["SalePrice"] = np.log(train["SalePrice"])
sale_price = train["SalePrice"].values
all_data[num_data] = np.log1p(all_data[num_data])

# Scale numeric variables
# scale = StandardScaler()
# all_data[num_data] = scale.fit_transform(all_data[num_data].as_matrix())
# Above two LOCs and below give the same end result
all_data[num_data] = preprocessing.scale(all_data[num_data])


# Convert categorical variables to dummy variables
all_data_dummy = pd.get_dummies(all_data)
# Tried normalizing categorical data. Didn't really work
# try:
# all_data_dummy[all_data_dummy.columns] = preprocessing.scale(all_data_dummy.columns)
# except ValueError, e:

# all_data_dummy.to_csv('output_dummy.csv', index=False)
# print all_data_dummy.head()
# print all_data_dummy.columns
# print all_data_dummy["SaleCondition_Partial"].values

# print all_data_dummy.isnull().sum().sum()

# num_data = all_data.columns[all_data.dtypes != 'object']
# if debug == 1:
#     print num_data
#
# # Remove skewness from target variable as well as all other numerical features
# # Using log(value+1) transformation since numerical data may have 0/-ve values after scaling
# train["SalePrice"] = np.log1p(train["SalePrice"])
# all_data_dummy[num_data] = np.log1p(all_data_dummy[num_data])
#
# # Scale numeric variables
# scale = StandardScaler()
# all_data_dummy[num_data] = scale.fit_transform(all_data[num_data].as_matrix())


# Remove skewness from target variable as well as all other numerical features
# Using log(value+1) transformation since numerical data may have 0/-ve values after scaling
# train["SalePrice"] = np.log1p(train["SalePrice"])
# all_data_dummy[num_data] = np.log1p(all_data_dummy[num_data])

# print all_data_dummy.isnull().sum().sum()

# all_data_dummy[all_data_dummy.columns] = preprocessing.scale(all_data_dummy.columns)

# Split the final pre-processed data back to training and test data sets
final_train = all_data_dummy.loc[train.index]
final_test = all_data_dummy.loc[test.index]
# print final_train.columns
if debug == 1:
    print len(final_train.index)
    print len(final_test.index)
    print final_train.shape, final_test.shape
    print len(train.index)
    print len(test.index)
    print len(all_data.index)
    print len(all_data_dummy.index)

# # Trying PCA for feature selection and reducing multi-collinearity
# pca = PCA(n_components=200)
# pca.fit(final_train.values)
# data_reduced = pca.fit_transform(final_train.values)

# var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# print data_reduced
# plt.plot(var1)
# print len(data_reduced)
# plt.show()

if debug == 1:
    print np.any(np.isnan(final_train))
    print np.all(np.isfinite(final_train))
    print final_train.isnull().sum().sum()
    print train.isnull().sum().sum()
    print test.isnull().sum().sum()
    # print np.any(np.isnan(test))

# # Trying RandomForestClassifier
# X = final_train.values
# model = RandomForestClassifier()
# model.fit(X, sale_price)
# feature_imp = pd.DataFrame(model.feature_importances_, index=final_train.columns, columns = ["importance"])
# print feature_imp.sort_values("importance", ascending=False)
# imp_features = feature_imp[feature_imp.values > 0]
# print imp_features

# imp_features = []
# for feature in feature_imp.columns.():
#     # for col in all_data.columns.values:
#     if feature_imp["importance"] > 0:
#         imp_features.append(feature_imp.)


# Parameter tuning (alpha) for Ridge regression - method 1
# alphas = np.logspace(-3, 2, 50)
X_train = final_train.values
X_test = final_test.values
# # print alphas
# test_scores = []
# for alpha in alphas:
#     clf = Ridge(alpha)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, sale_price, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(alphas, test_scores)
# plt.title("Alpha vs CV Error")
# plt.show()


# # Parameter tuning (alpha) for Ridge regression - method 2 (Performed worse than method 1)
# alpha_grid = {'alpha':sp_rand()}
# model = Ridge()
# alpha_search = RandomizedSearchCV(estimator=model, param_distributions=alpha_grid, n_iter=100)
# alpha_search.fit(X_train, sale_price)
# print alpha_search
# print alpha_search.best_score_
# best_alpha = alpha_search.best_estimator_.alpha
# print best_alpha

# # Parameter tuning (alpha) for Ridge regression - method 2 (Performed worse than method 1)
# alpha_grid = {'alpha':np.logspace(-3, 2, 500)}
# model = Ridge(normalize=True)
# alpha_search = GridSearchCV(estimator=model, param_grid=alpha_grid, scoring='neg_mean_squared_error', cv=10)
# alpha_search.fit(X_train, sale_price)
# print alpha_search
# print alpha_search.best_score_
# best_alpha = alpha_search.best_estimator_.alpha
# print best_alpha
# Returned 0.281176869797

# As evident from the plot, lowest error obtained between 10 to 20, so setting alpha = 15
ridge = Ridge(alpha=15)
ridge.fit(X_train, sale_price)
# Since the model is trained on log of sale price, back transforming it for output
y_ridge = np.exp(ridge.predict(X_test))
print y_ridge

# Parameter tuning (alpha) for Lasso regression - method 1
# alphas = np.logspace(-5, 1, 50)
# # print alphas
# test_scores = []
# for alpha in alphas:
#     clf = Lasso(alpha)
#     test_score = np.sqrt(-cross_val_score(clf, X_train, sale_price, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
# plt.plot(alphas, test_scores)
# plt.title("Alpha vs CV Error")
# plt.show()

# Parameter tuning (alpha) for Lasso regression - method 2
# Re-write this part (inspired from kernel)
lscv = LassoCV(alphas=None, copy_X=True, cv=10, eps=0.001, fit_intercept=True,
               max_iter=5000, n_alphas=1000, n_jobs=1, normalize=False, positive=False,
               precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
               verbose=False)

lscv.fit(X_train, sale_price)
# the best alpha is lscv.alpha_
# print (lscv.alpha_)
# gives 0.000482366054317

lasso = Lasso(alpha=lscv.alpha_, max_iter=5000)
lasso.fit(X_train, sale_price)
# Since the model is trained on log of sale price, back transforming it for output
y_lasso = np.exp(lasso.predict(X_test))
print y_lasso

y_final = (0.7 * y_ridge) + (0.3 * y_lasso)

final_output = pd.DataFrame(data= {'Id' : test.index, 'SalePrice': y_final})
print final_output.head(10)
final_output.to_csv('output_ridge.csv', index=False)
