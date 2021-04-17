#Import all the required libraries including KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#reading the dataset
df = pd.read_csv('Iris.csv')
df.head(10)

x = df.iloc[:, [0,1,2,3]].values
print(x)

# Now, we are choose number of cluster random=5 and then fit the value of x into that
kmeans = KMeans(n_clusters=5)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

print(kmeans.cluster_centers_)

plt.scatter(x[:,0], x[:,1],c=y_kmeans,cmap='rainbow')
plt.show()


Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

# Now, we are choose number of cluster random=4 and then fit the value of x into that
kmeans = KMeans(n_clusters=4)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)

print(kmeans.cluster_centers_)
plt.scatter(x[:,0], x[:,1],c=y_kmeans,cmap='rainbow')
plt.show()

