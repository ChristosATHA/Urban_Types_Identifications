import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the combined data (including original and normalized columns)
data_combined = pd.read_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/normalized_network_data_combined_new.xlsx")

# Step 2: Ensure the first 5 columns ("OBJECTID", "fid_1", "id", "class", "networkGrp") are included
first_5_columns = ["OBJECTID", "fid_1", "id", "class", "networkGrp"]
# Select all columns (excluding the first 5) and columns ending with '_norm'
normalized_columns = [col for col in data_combined.columns if col.endswith('_norm')]
selected_columns = first_5_columns + list(data_combined.columns) + normalized_columns

# Step 3: Perform PCA on the normalized data (excluding the first 5 columns)
normalized_data_only = data_combined[normalized_columns]
pca = PCA(n_components=22)  # perform for all components (equal the number of metrics)
pca_data = pca.fit_transform(normalized_data_only)

# Step 4: Explained variance plot (to show the variance explained by the number of components)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xticks(range(0, 23))  # Set the x-axis to start from 0 up to 22
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Step 5: Create an Excel table for the Importance of Components
# Calculating standard deviation, proportion of variance, and cumulative proportion
std_dev = np.sqrt(pca.explained_variance_)
proportion_variance = pca.explained_variance_ratio_
cumulative_proportion = np.cumsum(pca.explained_variance_ratio_)

# Creating a DataFrame for the importance of components
importance_df = pd.DataFrame({
    'Standard Deviation': std_dev,
    'Proportion of Variance': proportion_variance,
    'Cumulative Proportion': cumulative_proportion
}, index=[f'PC{i+1}' for i in range(len(std_dev))])

# Save the importance of components to an Excel file
importance_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/importance_of_components.xlsx")

# Step 6: Create a DataFrame for all principal components
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(22)])

# Step 7: Concatenate the original dataset (including all columns) with the PCA results
# Use all columns from data_combined along with the PCA results
final_df = pd.concat([data_combined, pca_df], axis=1)

# Step 8: Save the final data (original + PCA components) to an Excel file (.xlsx)
final_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/final_combined_with_pca.xlsx", index=False)

# Step 9: Prepare PCA loadings (eigenvectors) for the variables
original_columns = [col for col in data_combined.columns if col.startswith('mn_')]  # Replace with the relevant column names
pca_loadings = pd.DataFrame(pca.components_.T, index=original_columns[:22], columns=[f'PC{i+1}' for i in range(22)])

# Save the PCA loadings to Excel
pca_loadings.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/pca_loadings.xlsx", index=True)

print("PCA analysis complete. Results saved to final_combined_with_pca.xlsx, pca_loadings.xlsx, and importance_of_components.xlsx.")
