import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Load the PCA data (ensure you're selecting only the relevant columns)
pca_data_path = r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/final_combined_with_pca_Activities_Without_F_Div.xlsx"
pca_data = pd.read_excel(pca_data_path)

# Check the columns in the loaded DataFrame
print(pca_data.columns)

# Strip whitespace from column names if necessary
pca_data.columns = pca_data.columns.str.strip()

# Step 2: Select only the PCA columns (ensure you're working with numeric data)
# Update pca_columns based on the actual columns present
pca_columns = ['PC1', 'PC2', 'PC3']  # Adjust as needed
pca_data_selected = pca_data[pca_columns]

# Step 3: Perform KMeans clustering with the SELECTED number of clusters
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=1, n_init=10)

# Step 4: Fit and predict clusters (using the numeric PCA data)
clusters = kmeans.fit_predict(pca_data_selected)

# Step 5: Append the cluster labels to the original DataFrame
pca_df = pca_data.copy()  # Copy the original DataFrame to avoid overwriting it
pca_df['cluster'] = clusters  # Add the cluster labels

# Determine the number of unique clusters for filename suffix
num_clusters = pca_df['cluster'].nunique()
suffix = f"{num_clusters}clusters"  # e.g., "5clusters"

# Step 6: Save the final data with clusters to an Excel file
pca_df.to_excel(fr"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/clusters_Activities_Without_F_Div_{suffix}.xlsx", index=False)

print("Clustering complete. Results saved to Excel.")