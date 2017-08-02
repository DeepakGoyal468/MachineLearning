import numpy
from sklearn import preprocessing as p

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from keras.models import Sequential
from keras.layers import Dense, Activation
# from pylab import scatter, show, title, xlabel, ylabel, plot, contour

import xlrd


def baseline_model():
    model = Sequential()
    model.add(Dense(6, input_dim=5))
    layer2 = Dense(1)
    model.add(layer2)
    model.compile(loss='mean_squared_error', optimizer='adam')
    #c = layer2.get_weights()
    return model


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
model = Sequential()
model.add(Dense(6, input_dim=5))
layer2 = Dense(1)
model.add(layer2)
model.compile(loss='mean_squared_error', optimizer='adam')
c = layer2.get_weights()
print(c)

model.fit(features, y, nb_epoch=2000)
#print(c)
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
print(test_data)
#check = [test_data]
test_data = np.array(test_data).transpose()
print(model.predict(test_data))
#scores = model.evaluate(features, y)




