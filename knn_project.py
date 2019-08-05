'''
K-Nearest Neighbors Project for Udemy Data Science Course
anonymous data source
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def gather_data(file):
    df = pd.read_csv(file)
    print(df.head())
    return df

def visualize_data(dataframe):
    sns.pairplot(dataframe, hue='TARGET CLASS', diag_kind='histogram')
    plt.show()

def standardize_variables(dataframe):
    scaler = StandardScaler()
    scaler.fit(dataframe.drop('TARGET CLASS', axis=1))
    scaled_features = scaler.transform(dataframe.drop('TARGET CLASS', axis=1))
    df_feat = pd.DataFrame(scaled_features, columns = dataframe.columns[:-1])
    return df_feat

def split_data(dataframe, df_feat):
    X = df_feat
    y = dataframe['TARGET CLASS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def create_model(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    return knn


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))
    print('\n')
    print(confusion_matrix(y_test, pred))

def choose_k_value(X_train,X_test, y_train, y_test):
    '''
    Elbowing technique to choose an appropriate K value
    '''
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

    new_neighbors = int(input('What new value would you like to use for k?'))
    if new_neighbors > 50 or type(new_neighbors) != int:
        new_neighbors = int(input('Invalid! What new value would you like to use for k?'))
    #Retrain with the new K value
    knn = KNeighborsClassifier(n_neighbors=new_neighbors)
    knn.fit(X_train, y_train)
    evaluate_model(knn, X_test, y_test)
    return knn

if __name__ == '__main__':
    filename = 'KNN_Project_Data'
    df = gather_data(filename)
    explore = input('Do you want to visualize the data (takes a lot of processing power)')
    if explore == 'yes':
        visualize_data(df)
    df_feat = standardize_variables(df)
    data = split_data(df, df_feat)
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    knn = create_model(X_train, y_train)
    evaluate_model(knn, X_test, y_test)
    choose_k_value(X_train, X_test, y_train, y_test)
