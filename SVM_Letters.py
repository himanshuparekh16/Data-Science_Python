import pandas as pd 
import numpy as np 
import seaborn as sns

letters = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Support Vector Machines\\letters.csv")
letters.head()
letters.describe()
letters.columns

sns.boxplot(x="lettr",y="x-box",data=letters,palette = "hls")
sns.boxplot(x="y-box",y="lettr",data=letters,palette = "hls")
sns.pairplot(data=letters)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(letters,test_size = 0.3)
test.head()
train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 85.233

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 94.499

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 97.016


