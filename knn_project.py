'''
K-Nearest Neighbors Project for Udemy Data Science Course
anonymous data source
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#Get the Data
df = pd.read_csv('KNN_Project_Data')
#check out the head of the data frame
print(df.head())

'''
sns.pairplot(df, hue = 'TARGET CLASS', diag_kind='histogram')
plt.show()
'''

#Standardize the Variables 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Fit scaler to the features
scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])

#print(df_feat.head())

X = df_feat
y = df['TARGET CLASS']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


#Using KNN 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#Predictions and Evaluations 
pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print('\n')
print(confusion_matrix(y_test, pred))

'''
Elbowing technique to choose an appropriate K value
'''

#Choosing a K Value 
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors= i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

'''
Creating a plot to show which K value to choose
'''
plt.figure(figsize=(10,7))
plt.title('K-values vs. Error Rate')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.plot(range(1,40), error_rate)
plt.show()

#Retrain with the new K value 
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

'''
Print out the reports on the newly trained model
'''
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))