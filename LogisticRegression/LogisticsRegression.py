from sklearn import preprocessing as p

import xlrd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

worksheet = xlrd.open_workbook('ex2data1-logistic.xls')
sheetName1 = worksheet.sheet_by_name('ex2data1')
worksheet = xlrd.open_workbook('ex2data2-logistic.xls')
sheetName2 = worksheet.sheet_by_name('ex2data2')

seed = 7
np.random.seed(seed)


def baseline_model():
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='sigmoid'))
    layer2 = Dense(1, activation='sigmoid')
    model.add(layer2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    c = layer2.get_weights()
    print(c)
    return model

x1 = []
x2 = []
y = []
num_rows1 = (sheetName1.nrows)*.90
num_rows1 = int(num_rows1)
print(num_rows1)
num_cols1 = sheetName1.ncols

num_rows2 = (sheetName2.nrows)*.90
num_rows2 = int(num_rows2)
print(num_rows2)
num_cols2 = sheetName2.ncols


for row in range(1, num_rows1):
    x1.append(sheetName1.cell(row, 0).value)
    x2.append(sheetName1.cell(row, 1).value)
    y.append(sheetName1.cell(row,2).value)

for row in range(1, num_rows2):
    x1.append(sheetName2.cell(row, 0).value)
    x2.append(sheetName2.cell(row, 1).value)
    y.append(sheetName2.cell(row,2).value)

#x1 = p.scale(x1)
#x2 = p.scale(x2)

features = [x1, x2]
#print(features)
features = np.array(features).transpose()
y = np.array(y).transpose()

model = baseline_model()
model.fit(features, y, nb_epoch=270, verbose=1)

x1 = []
x2 = []
y=[]

for row in range(num_rows1, sheetName1.nrows):
    x1.append(sheetName1.cell(row,0).value)
    x2.append(sheetName1.cell(row,1).value)
    y.append(sheetName1.cell(row,2).value)


for row in range(num_rows2, sheetName2.nrows):
    x1.append(sheetName2.cell(row,0).value)
    x2.append(sheetName2.cell(row,1).value)
    y.append(sheetName2.cell(row,2).value)

#x1 = p.scale(x1)
#x2 = p.scale(x2)
test_data = [x1, x2]
print(test_data)
print("Actual ")
print(y)
test_data = np.array(test_data).transpose()
#scores = model.evaluate(features, y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
prediction = model.predict(test_data)
rounded = [round(x[0]) for x in prediction]
print("Predicted")
print(rounded)




