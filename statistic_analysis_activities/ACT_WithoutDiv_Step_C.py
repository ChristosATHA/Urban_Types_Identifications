import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the PCA data from the Excel file
pca_data_path = r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/final_combined_with_pca_Activities_Without_F_Div.xlsx"
pca_data = pd.read_excel(pca_data_path)

# Dynamically select available PCA columns
pca_columns = [col for col in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5'] if col in pca_data.columns]
pca_data_selected = pca_data[pca_columns]

# Define a range of possible cluster numbers (k) to evaluate
cluster_range = range(1, 11)  # Testing k from 1 to 10 clusters
silhouette_avg_scores = []  # Start with an empty list for silhouette scores
wcss = []  # Start with an empty list for WCSS (within-cluster sum of squares)

# Loop through each k value
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)  # Set n_init to suppress the warning
    cluster_labels = kmeans.fit_predict(pca_data_selected)
    
    # Calculate the WCSS for the Elbow method (can be done for k=1 to 10)
    wcss.append(kmeans.inertia_)
    
    # Silhouette score calculation can only start from k=2
    if k > 1:
        silhouette_avg = silhouette_score(pca_data_selected, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)
    else:
        silhouette_avg_scores.append(float('nan'))  # Append NaN for k=1 since silhouette cannot be calculated

# Save Silhouette and Elbow Data to Excel
cluster_values = list(cluster_range)
silhouette_data_df = pd.DataFrame({
    'Number of Clusters (k)': cluster_values,
    'Average Silhouette Score': silhouette_avg_scores
})
silhouette_data_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/silhouette_data_Activities_Without_F_Div.xlsx", index=False)

elbow_data_df = pd.DataFrame({
    'Number of Clusters (k)': cluster_values,
    'WCSS': wcss
})
elbow_data_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/elbow_data_Activities_Without_F_Div.xlsx", index=False)

# Plot silhouette score vs. number of clusters (k)
plt.figure(figsize=(12, 6))

# Silhouette plot
plt.subplot(1, 2, 1)
plt.plot(cluster_values, silhouette_avg_scores, marker='o', linestyle='--')
plt.title('Silhouette Method: Optimal number of clusters')
plt.xlabel('Number of clusters k')
plt.ylabel('Average silhouette score')
plt.xticks(cluster_values)  # Show k from 1 to 10 on x-axis
plt.grid(True)

# Elbow plot (WCSS vs. number of clusters)
plt.subplot(1, 2, 2)
plt.plot(cluster_values, wcss, marker='o', linestyle='--')
plt.title('Elbow Method: Optimal number of clusters')
plt.xlabel('Number of clusters k')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.xticks(cluster_values)  # Show k from 1 to 10 on x-axis
plt.grid(True)

# Display both plots
plt.tight_layout()
plt.savefig(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/Optical_Num_of_Clusters_Silhouette+Elbow_Activities_Without_F_Div.png")
plt.show()