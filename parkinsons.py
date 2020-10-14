# importing basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# exrtracting data
dataset = pd.read_csv('parkinsons.csv')
X = dataset.iloc[: ,~dataset.columns.isin(['status', 'name','Index'])].values
y = dataset['status'].values

# spliting data for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# scaling the data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#importing libraries for artificial neural network
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# adding layers in model
classifier.add(Dense(units=22,input_dim = 22,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(units=22,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# fitting the model
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# predicting the result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)

# accuracy of the model
acc = accuracy_score(y_test,y_pred)
print("Accuracy: {:.2f} %".format(acc*100))
