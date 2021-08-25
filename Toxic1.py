import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RForest
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB



#importing dataset and preparation
data=pd.read_csv('qsar_oral_toxicity.csv')
xlist= []
ylist = []

datalist = data.values.tolist()
for i in datalist:
    for j in i:
        xlist.append(j.split(';')[:-1])
        ylist.append(j.split(';')[-1])

x = pd.DataFrame(xlist)
y = pd.DataFrame(ylist)


#Dimensionality Reduction 
pca = PCA(n_components=512)
principalComponents = pca.fit_transform(x)


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_calibration, x_validation, y_calibration, y_validation = train_test_split(principalComponents, y, test_size = 0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_calibration = sc.fit_transform(x_calibration)
x_validation = sc.transform(x_validation)



#Classifying with KNN
clf = KNN(n_neighbors = 3)
clf.fit(x_calibration, y_calibration)
y_pred = clf.predict(x_validation)
print("\n========KNN classifying results=======")
print("Confusion Matrix : ")
print(confusion_matrix(y_validation, y_pred))
print("Accuracy :",accuracy_score(y_validation, y_pred))
print("Classification report :")
print(classification_report(y_validation, y_pred))


#Classifying with Random Forest
clf = RForest(max_depth = 300)
clf.fit(x_calibration, y_calibration)
y_pred = clf.predict(x_validation)
print("\n========Random Forest classifying results=======")
print("Confusion Matrix : ")
print(confusion_matrix(y_validation, y_pred))
print("Accuracy :",accuracy_score(y_validation, y_pred))
print("Classification report :")
print(classification_report(y_validation, y_pred))


#Classifying with Gradient Boosting
clf = GradientBoostingClassifier(random_state = 0)
clf.fit(x_calibration, y_calibration)
y_pred = clf.predict(x_validation)
print("\n========Gradient Boosting classifying results=======")
print("Confusion Matrix : ")
print(confusion_matrix(y_validation, y_pred))
print("Accuracy :",accuracy_score(y_validation, y_pred))
print("Classification report :")
print(classification_report(y_validation, y_pred))


#Classifying with Ada Boosting
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(x_calibration, y_calibration)
y_pred = clf.predict(x_validation)
print("\n========Ada Boosting classifying results=======")
print("Confusion Matrix : ")
print(confusion_matrix(y_validation, y_pred))
print("Accuracy :",accuracy_score(y_validation, y_pred))
print("Classification report :")
print(classification_report(y_validation, y_pred))


#Classifying with MLP
clf = MLPClassifier(random_state=1, max_iter=300)
clf.fit(x_calibration, y_calibration)
y_pred = clf.predict(x_validation)
print("\n========MLP classifying results=======")
print("Confusion Matrix : ")
print(confusion_matrix(y_validation, y_pred))
print("Accuracy :",accuracy_score(y_validation, y_pred))
print("Classification report :")
print(classification_report(y_validation, y_pred))


#Classifying with NB
clf = GaussianNB()
clf.fit(x_calibration, y_calibration)
y_pred = clf.predict(x_validation)
print("\n========NB classifying results=======")
print("Confusion Matrix : ")
print(confusion_matrix(y_validation, y_pred))
print("Accuracy :",accuracy_score(y_validation, y_pred))
print("Classification report :")
print(classification_report(y_validation, y_pred))

