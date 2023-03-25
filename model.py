import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
df=pd.read_csv("Iris.csv")
df.sample(frac=1,random_state=42)
print(df.head())
x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df[['Species']]
print(x.columns)
print(y.columns)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
x_train=min_max.fit_transform(x_train)
x_test=min_max.fit_transform(x_test)

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
y_pred=dt_model.predict(x_test)
print(y_pred)
print("Accuracy on Decision Tree Model: ", accuracy_score(y_test, y_pred))
import joblib
joblib.dump(dt_model,"dt_iris")
