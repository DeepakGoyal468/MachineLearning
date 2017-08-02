from sklearn import preprocessing as p

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from keras.models import Sequential
from keras.layers import Dense, Activation
# from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import xlrd


import numpy as np

# Load the dataset
worksheet = xlrd.open_workbook('data_carsmall.xlsx')
sheetName = worksheet.sheet_by_name('Sheet1')
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
y = []
num_rows = sheetName.nrows
num_cols = sheetName.ncols

for row in range(2, num_rows):
    value = sheetName.cell(row,5).value
    if value != 'NaN':
        x1.append(sheetName.cell(row, 0).value)
        x2.append(sheetName.cell(row, 1).value)
        x3.append(sheetName.cell(row, 2).value)
        x4.append(sheetName.cell(row, 3).value)
        x5.append(sheetName.cell(row, 4).value)
        y.append(sheetName.cell(row, 5).value)


x1 = p.scale(x1)
x2 = p.scale(x2)
x3 = p.scale(x3)
x4 = p.scale(x4)
x5 = p.scale(x5)

features = [x1, x2, x3, x4, x5]
output = [y]
#print(x2)
features = np.array(features).transpose()
y = np.array(y)


x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
for row in range(2,num_rows):
    value = sheetName.cell(row,5).value
    if value == 'NaN':
        x1.append(sheetName.cell(row, 0).value)
        x2.append(sheetName.cell(row, 1).value)
        x3.append(sheetName.cell(row, 2).value)
        x4.append(sheetName.cell(row, 3).value)
        x5.append(sheetName.cell(row, 4).value)


x1 = p.scale(x1)
x2 = p.scale(x2)
x3 = p.scale(x3)
x4 = p.scale(x4)
x5 = p.scale(x5)
test_data = [x1, x2, x3, x4, x5]
test_data = np.array(test_data).transpose()

# generate a model of polynomial features
poly = PolynomialFeatures(degree=2)
# transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
newX = poly.fit_transform(features)

# transform the prediction to fit the model type
Test_data = poly.fit_transform(test_data)

# generate the regression object
clf = linear_model.LinearRegression()

# preform the actual regression
clf.fit(newX, y)
weights = clf.coef_
print(weights)
predicted_values = clf.predict(Test_data)
print "predictions"
print predicted_values




