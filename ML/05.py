
# Practical 10: K-Means Clustering on Sales Data

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 2: Load Dataset
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

print("Dataset Loaded Successfully")
print(df.head())

# Step 3: Select Numerical Features
data = df.select_dtypes(include=['int64', 'float64'])

# Remove constant columns
data = data.loc[:, data.nunique() > 1]

# Handle Missing Values
data = data.fillna(data.median())

print("\nSelected Numerical Columns:")
print(data.columns)

# Step 4: Standardize the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 5: Find Optimal Number of Clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Step 6: Plot Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid()
plt.show()

# Step 7: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_data)

print("\nK-Means Cluster Results:")
print(df[['Cluster']].head(10))

# Step 8: Display Number of Records in Each Cluster
print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# Step 9: Calculate Silhouette Score
score = silhouette_score(scaled_data, df['Cluster'])
print("\nSilhouette Score:", score)

# Step 10: Visualize Clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 5))
plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=df['Cluster'],
    cmap='viridis'
)
plt.title("K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Step 11: Display Cluster-wise Average Values
df['Cluster'] = df['Cluster'].astype(int)

print("\nCluster-wise Average:")
print(df.groupby('Cluster')[data.columns].mean())

print("\nK-Means Clustering Completed Successfully!")
