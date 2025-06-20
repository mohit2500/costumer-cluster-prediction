#Prepare a cluster of customer to predict the purchase power based on their income and soending
#importing Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans

#Loading the Dataset inot Dataframe
df = pd.read_csv("Mall_Customers.csv")
# print(df.info())
X = df[["Annual Income (k$)" , "Spending Score (1-100)"]]
wcss_list = []
for i in range(1,11):
    model = KMeans(n_clusters = i , init = "k-means++" , random_state = 1)
    model.fit(X)
    wcss_list.append(model.inertia_) #a method which is trying to find out the sum of squares within the cluster , distance from data point to each cluster then append that cluster into wcss_lis
#Visualize the results
#plt.plot(range(1,11),wcss_list)
#plt.title("Elbow Method Graph")
#plt.xlabel("Number of clusters ")
#plt.ylabel("WCSS List")
#plt.show()


# training the model 
model = KMeans(n_clusters=4,init="k-means++",random_state=1)
y_predict = model.fit_predict(X)

print(y_predict)

#converting the dataframe X into numpy array 
X_array = X.values

#plotting the graph of clusters
plt.scatter(X_array[y_predict == 0,0],X_array[y_predict ==0,1],s =100,color="Green")
plt.scatter(X_array[y_predict == 1,0],X_array[y_predict ==1,1],s =100,color="Red")
plt.scatter(X_array[y_predict == 2,0],X_array[y_predict ==2,1],s =100,color="Yellow")
plt.scatter(X_array[y_predict == 3,0],X_array[y_predict ==3,1],s =100,color="Blue")
plt.scatter(X_array[y_predict == 4,0],X_array[y_predict ==4,1],s =100,color="Brown")


plt.title("Customer segmentation graph")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()