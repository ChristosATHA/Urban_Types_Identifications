import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the combined data (including original and normalized columns)
data_combined = pd.read_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/BUILD_DENSITY/normalized_Build_Density.xlsx")

# Step 2: Ensure the following columns ("OBJECTID", "Block_Area", "Block_ID", "GSI", "networkGrp") are included
first_columns = ["OBJECTID", "Block_Area", "Block_ID", "Block_Area_str_gr", "Block_Area_ha", "Block_Area", "COUNT_BHeight", "MEAN_BHeight", "FIRST_Block_Area"]
# Select first columns and columns ending with '_norm'
normalized_columns = ["GSI_norm", "FSI_norm"]
selected_columns = first_columns + normalized_columns

# Step 3: Perform PCA on the normalized data (excluding the first columns)
normalized_data_only = data_combined[normalized_columns]
pca = PCA(n_components=2)  # Specify 2 components
pca_data = pca.fit_transform(normalized_data_only)

# Step 4: Explained variance plot (to show the variance explained by the number of components)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))  # Adjusted x-axis to match the number of components
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
importance_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/BUILD_DENSITY/importance_of_components_Build_Density.xlsx")

# Step 6: Create a DataFrame for the 2 principal components
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(2)])

# Step 7: Concatenate the original dataset (including all columns) with the PCA results
final_df = pd.concat([data_combined, pca_df], axis=1)

# Step 8: Save the final data (original + PCA components) to an Excel file (.xlsx)
final_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/BUILD_DENSITY/final_combined_with_pca_Build_Density.xlsx", index=False)

# Step 9: Prepare PCA loadings (eigenvectors) for the variables
pca_loadings = pd.DataFrame(pca.components_.T, index=normalized_columns, columns=[f'PC{i+1}' for i in range(2)])

# Save the PCA loadings to Excel
pca_loadings.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/BUILD_DENSITY/pca_loadings_Build_Density.xlsx", index=True)

print("PCA analysis complete. Results saved to final_combined_with_pca_Build_Density.xlsx, pca_loadings_Build_Density.xlsx, and importance_of_components_Build_Density.xlsx.")
