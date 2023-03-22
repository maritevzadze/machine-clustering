# machine learning


#Clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



data = pd.read_csv("https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv")
data.drop(["CustomerID", "Gender", "Age"], axis=1, inplace=True)
print(data.head())

my_cluster = KMeans(n_clusters=5, max_iter=1000)
my_cluster.fit(data)

KMeans(max_iter=1000, n_clusters=5)
Y_predicted = my_cluster.predict(data)
plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"], c=Y_predicted)
print(plt.show())
print(silhouette_score(data, Y_predicted)


#task1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('emails.csv')
print(data.head())

X = data.drop('Prediction',axis=1).values
y = data['Prediction'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)

algo=LogisticRegression(max_iter=500000, C=0.7)
algo.fit(X_train,y_train)
print(algo.score(X_test,y_test))


#task2
import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("emails.csv")
print(data.head())


X = data.drop('Prediction', axis=1).values
y = data['Prediction'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)


selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X, y)

algo1 = SVC()
algo1.fit(X_train, y_train)
print(algo1.score(X_test, y_test))
print(algo1.score(X_train, y_train))


#task3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('regression.csv')
data = data.dropna()
X = data.drop('x', axis=1)
y = data['x']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

scaler = StandardScaler()
scaler.fit_transform(X_train, y_train)
scaler.fit_transform(X_test, y_test)

lasso = Lasso()
lasso.fit(X_train, y_train)

print(lasso.score(X_test, y_test))
print(lasso.score(X_train, y_train))

