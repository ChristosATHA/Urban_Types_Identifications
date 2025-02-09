import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the combined data (including original and normalized columns)
data_combined = pd.read_excel(r"C:\Users\athan\Documents\DATA STOKHOLM\ANALYSIS\Python_exports\ACTIVIΤΙΕΣ_WITHOUT_F_DIV/normalized_Activities_Without_F_Div.xlsx")

# Step 2: Ensure the following columns ("OBJECTID", "Block_Area", "Block_ID", "GSI", "networkGrp") are included
first_columns = ["OBJECTID", "Block_ID", "Block_Area_str_gr", "Block_Area_ha", "Block_Area", "SUM_Pop2018", "SUM_Initial_UAPoly_Area", "MEAN_Ratio_Area", "SUM_Adj_Pop2018", "COUNT_osm_id", "UNIQUE_General_Category" ]
# Select first columns and columns ending with '_norm'
normalized_columns = ["POP_DENS_ha_norm", "F_DENS_ha_norm", "POS_Influence_ha_norm"]
selected_columns = first_columns + normalized_columns

# Step 3: Perform PCA on the normalized data (excluding the first columns)
normalized_data_only = data_combined[normalized_columns]
pca = PCA(n_components=3)  # Specify 3 components
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
importance_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/importance_of_components_Activities_Without_F_Div.xlsx")

# Step 6: Create a DataFrame for the 3 principal components
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(3)])

# Step 7: Concatenate the original dataset (including all columns) with the PCA results
final_df = pd.concat([data_combined, pca_df], axis=1)

# Step 8: Save the final data (original + PCA components) to an Excel file (.xlsx)
final_df.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/final_combined_with_pca_Activities_Without_F_Div.xlsx", index=False)

# Step 9: Prepare PCA loadings (eigenvectors) for the variables
pca_loadings = pd.DataFrame(pca.components_.T, index=normalized_columns, columns=[f'PC{i+1}' for i in range(3)])

# Save the PCA loadings to Excel
pca_loadings.to_excel(r"C:/Users/athan/Documents/DATA STOKHOLM/ANALYSIS/Python_exports/ACTIVIΤΙΕΣ_WITHOUT_F_DIV/pca_loadings_Activities_Without_F_Div.xlsx", index=True)

print("PCA analysis complete. Results saved to final_combined_with_pca_Activities_Without_F_Div.xlsx, pca_loadings_Activities_Without_F_Div.xlsx, and importance_of_components_Activities_Without_F_Div.xlsx.")
